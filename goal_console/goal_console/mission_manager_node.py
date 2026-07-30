#!/usr/bin/env python3
"""
mission_manager_node — 比赛任务状态机节点

状态机驱动，自主执行一区识别→抓取→对接→路线→三区动作 全流程。
使用 Nav2 NavigateToPose action，所有动态目标基于 TF (map→base_link) 计算。

订阅:
  /camera/yolo/y_offset  std_msgs/Float32 — 一区视觉 y 偏移 (m)，左正右负
  /camera/view_sig       std_msgs/UInt8   — 抓取完成信号 (10)
  /camera/view_cmd       std_msgs/UInt8   — 对接完成 (2) / 一区结束 (3)
  /camera4               std_msgs/Float32 — 三区视觉 y 偏移 (m)
  /chassis_action_done   std_msgs/Int32   — 下位机动作完成 (1)

发布:
  /camera/yolo/request   std_msgs/UInt8   — 到达识别点 (1)，三连发
  /chassis_action_cmd    std_msgs/Int32   — 下位机动作指令 (1)
"""

import math
import os
import sys
import time
import traceback
import numpy as np
from enum import IntEnum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import ExternalShutdownException

from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32, Float64, Int32, String, UInt8, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration as DurationMsg
from rclpy.duration import Duration as RclpyDuration
import tf2_ros
from tf2_ros import TransformException
import yaml


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def normalize_angle(a: float) -> float:
    """将角度归一化到 [-pi, pi]"""
    a = math.fmod(a, 2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_to_quaternion(yaw: float):
    """弧度 yaw → (qz, qw)"""
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(qz: float, qw: float) -> float:
    """(qz, qw) → yaw (弧度)"""
    siny_cosp = 2.0 * (qw * qz)
    cosy_cosp = 1.0 - 2.0 * (qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_dynamic_goal(current_x: float, current_y: float, current_yaw: float,
                         dx: float, dy: float, dyaw: float):
    """
    根据当前 map 下位姿和 base_link 相对偏移，计算 map 坐标系下的目标位姿。

    target_x = current_x + cos(yaw)*dx - sin(yaw)*dy
    target_y = current_y + sin(yaw)*dx + cos(yaw)*dy
    target_yaw = normalize(yaw + dyaw)
    """
    target_x = current_x + math.cos(current_yaw) * dx - math.sin(current_yaw) * dy
    target_y = current_y + math.sin(current_yaw) * dx + math.cos(current_yaw) * dy
    target_yaw = normalize_angle(current_yaw + dyaw)
    return target_x, target_y, target_yaw


def compute_goal_from_ref(ref_x: float, ref_y: float, ref_yaw: float,
                          dx: float, dy: float, dyaw: float):
    """
    基于参考位姿（非当前位姿）和相对偏移计算目标位姿。
    用于 last_camera1_goal map X -0.20m 等场景。
    """
    target_x = ref_x + math.cos(ref_yaw) * dx - math.sin(ref_yaw) * dy
    target_y = ref_y + math.sin(ref_yaw) * dx + math.cos(ref_yaw) * dy
    target_yaw = normalize_angle(ref_yaw + dyaw)
    return target_x, target_y, target_yaw


# ---------------------------------------------------------------------------
# 状态枚举
# ---------------------------------------------------------------------------

class State(IntEnum):
    START = 0
    WAIT_MISSION_START = 1
    GO_ZONE1_START = 2
    PUBLISH_CAMERA0_START = 3
    WAIT_CAMERA1_OFFSET = 4
    MOVE_TO_CAMERA1_DYNAMIC_GOAL = 5
    WAIT_CAMERA2_GRAB_DONE = 6
    TURN_AROUND_DYNAMIC = 7
    WAIT_CAMERA3_DOCK_DONE = 8
    TURN_BACK_FOR_NEXT_ROUND = 9
    MOVE_MAP_X_MINUS_0_2 = 10
    RUN_FIXED_ROUTE = 11
    WAIT_ZONE2_GRID_DONE = 12
    ENTER_ZONE3 = 13
    WAIT_CAMERA4_OFFSET = 14
    MOVE_TO_ZONE3_DYNAMIC_GOAL = 15
    PUBLISH_CHASSIS_ACTION_CMD = 16
    WAIT_CHASSIS_ACTION_DONE = 17
    TURN_AROUND_ZONE3 = 18
    MOVE_FORWARD_0_5_DYNAMIC = 19
    MISSION_DONE = 20
    FAILED = 21
    WAIT_ZONE2_COOLDOWN = 22  # data=3 后 4 秒冷却


STATE_NAMES = {v: k for k, v in State.__members__.items()}


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------

class MissionManager(Node):
    """比赛任务状态机节点"""

    def __init__(self):
        super().__init__('mission_manager')

        # ==================== 参数声明 ====================
        # 简单类型参数：声明 + 默认值
        self.declare_parameter('selected_route_id', 1)
        self.declare_parameter('zone1_continue_code', 2)
        self.declare_parameter('zone1_finish_code', 3)
        self.declare_parameter('route_select_mode', 'startup_param')
        self.declare_parameter('camera_start_topic', '/camera/yolo/request')
        self.declare_parameter('zone1_offset_topic', '/camera/yolo/y_offset')
        self.declare_parameter('grab_done_topic', '/camera/view_sig')
        self.declare_parameter('dock_done_topic', '/camera/view_cmd')
        self.declare_parameter('camera_start_code', 1)
        self.declare_parameter('grab_done_code', 10)
        self.declare_parameter('zone1_continue_map_x_offset', -0.20)
        self.declare_parameter('zone3_forward_x_offset', 0.3)
        self.declare_parameter('final_forward_x_offset', 0.5)
        self.declare_parameter('max_vision_y_offset', 2.0)
        self.declare_parameter('vision_timeout_sec', 5.0)
        self.declare_parameter('action_timeout_sec', 10.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('action_name', '/navigate_to_pose')
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('bench_test_mode', False)
        self.declare_parameter('dry_run_nav', False)
        self.declare_parameter('route_arrive_then_rotate', True)
        self.declare_parameter('route_only_mode', False)
        self.declare_parameter('zone2_executor', 'grid_gui')
        self.declare_parameter('zone2_grid_done_timeout_sec', 60.0)
        self.declare_parameter('bench_timeout_sec', 60.0)
        self.declare_parameter('enable_debug_visualization', True)
        self.declare_parameter('debug_goal_topic', '/mission_manager/debug_goal')
        self.declare_parameter('debug_marker_topic', '/mission_manager/debug_marker')
        self.declare_parameter('turn_cmd_topic', '/cmd_vel_nav')
        self.declare_parameter('turn_kp', 1.0)
        self.declare_parameter('turn_max_wz', 0.8)
        self.declare_parameter('turn_min_wz', 0.1)
        self.declare_parameter('turn_yaw_tolerance', 0.05)
        self.declare_parameter('turn_timeout_sec', 10.0)
        self.declare_parameter('turn_direction', -1)

        # ==================== 复杂参数：声明 + 稳健解析 ====================
        # zone1_start: [x, y, yaw] list，ROS2 可正确加载
        try:
            self.declare_parameter('zone1_start', [0.0, 0.0, 0.0])
        except Exception:
            pass
        z1_raw = self.get_parameter_or(
            'zone1_start', Parameter('zone1_start', value=[0.0, 0.0, 0.0])).value
        if isinstance(z1_raw, list) and len(z1_raw) == 3:
            self._zone1_start = {
                'x': float(z1_raw[0]),
                'y': float(z1_raw[1]),
                'yaw': float(z1_raw[2]),
            }
        else:
            self.get_logger().error(
                f'zone1_start 格式错误，期望 [x, y, yaw]，实际: {z1_raw}')
            self._zone1_start = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

        # 阵营特定 zone1_start
        for alliance_name, param_name, flag_name in [
            ('blue', 'zone1_start_blue', None),
            ('red', 'zone1_start_red', 'zone1_start_red_configured'),
        ]:
            try:
                self.declare_parameter(param_name, [0.0, 0.0, 0.0])
            except Exception:
                pass
            raw = self.get_parameter_or(
                param_name, Parameter(param_name, value=[0.0, 0.0, 0.0])).value
            key = f'_zone1_start_{alliance_name}'
            if isinstance(raw, list) and len(raw) == 3:
                if flag_name:
                    try:
                        self.declare_parameter(flag_name, False)
                    except Exception:
                        pass
                    configured = self.get_parameter_or(
                        flag_name, Parameter(flag_name, value=False)).value
                else:
                    configured = True  # 蓝方始终已配置
                if configured:
                    setattr(self, key, {'x': float(raw[0]), 'y': float(raw[1]), 'yaw': float(raw[2])})
                else:
                    setattr(self, key, None)
            else:
                setattr(self, key, None)

        # 路线参数：空列表在 YAML 中可能不被 ROS2 初始化，用 get_parameter_or 兜底
        try:
            self.declare_parameter('route_1', [])
        except Exception:
            pass
        self._route_1 = self._parse_route_flat(
            self.get_parameter_or('route_1', Parameter('route_1', value=[])).value,
            'route_1')

        try:
            self.declare_parameter('route_2', [])
        except Exception:
            pass
        self._route_2 = self._parse_route_flat(
            self.get_parameter_or('route_2', Parameter('route_2', value=[])).value,
            'route_2')

        try:
            self.declare_parameter('route_3', [])
        except Exception:
            pass
        self._route_3 = self._parse_route_flat(
            self.get_parameter_or('route_3', Parameter('route_3', value=[])).value,
            'route_3')

        try:
            self.declare_parameter('route_4', [])
        except Exception:
            pass
        self._route_4 = self._parse_route_flat(
            self.get_parameter_or('route_4', Parameter('route_4', value=[])).value,
            'route_4')

        # 读取简单参数
        self._selected_route_id = self.get_parameter('selected_route_id').value
        self._zone1_continue_code = self.get_parameter('zone1_continue_code').value
        self._zone1_finish_code = self.get_parameter('zone1_finish_code').value
        self._route_select_mode = self.get_parameter('route_select_mode').value
        self._camera_start_topic = self.get_parameter('camera_start_topic').value
        self._zone1_offset_topic = self.get_parameter('zone1_offset_topic').value
        self._grab_done_topic = self.get_parameter('grab_done_topic').value
        self._dock_done_topic = self.get_parameter('dock_done_topic').value
        self._camera_start_code = self.get_parameter('camera_start_code').value
        self._grab_done_code = self.get_parameter('grab_done_code').value
        self._zone1_continue_map_x_offset = self.get_parameter('zone1_continue_map_x_offset').value
        self._zone3_forward_x_offset = self.get_parameter('zone3_forward_x_offset').value
        self._final_forward_x_offset = self.get_parameter('final_forward_x_offset').value
        self._max_vision_y_offset = self.get_parameter('max_vision_y_offset').value
        self._vision_timeout_sec = self.get_parameter('vision_timeout_sec').value
        self._action_timeout_sec = self.get_parameter('action_timeout_sec').value
        self._frame_id = self.get_parameter('frame_id').value
        self._base_frame = self.get_parameter('base_frame').value
        self._action_name = self.get_parameter('action_name').value
        self._goal_tolerance = self.get_parameter('goal_tolerance').value
        self._bench_test_mode = self.get_parameter('bench_test_mode').value
        self._dry_run_nav = self.get_parameter('dry_run_nav').value
        self._route_arrive_then_rotate = self.get_parameter('route_arrive_then_rotate').value
        self._route_only_mode = self.get_parameter('route_only_mode').value
        self._bench_timeout_sec = self.get_parameter('bench_timeout_sec').value
        self._zone2_grid_done_timeout = self.get_parameter('zone2_grid_done_timeout_sec').value
        self._enable_debug_visualization = self.get_parameter('enable_debug_visualization').value
        self._debug_goal_topic = self.get_parameter('debug_goal_topic').value
        self._debug_marker_topic = self.get_parameter('debug_marker_topic').value
        self._turn_cmd_topic = self.get_parameter('turn_cmd_topic').value
        self._turn_kp = self.get_parameter('turn_kp').value
        self._turn_max_wz = self.get_parameter('turn_max_wz').value
        self._turn_min_wz = self.get_parameter('turn_min_wz').value
        self._turn_yaw_tolerance = self.get_parameter('turn_yaw_tolerance').value
        self._turn_timeout_sec = self.get_parameter('turn_timeout_sec').value
        self._turn_direction = self.get_parameter('turn_direction').value

        # 校验 selected_route_id 合法性
        if self._selected_route_id not in (1, 2, 3, 4):
            self.get_logger().fatal(
                f'selected_route_id = {self._selected_route_id}，无效！只能是 1/2/3/4')
            raise ValueError(f'selected_route_id 无效: {self._selected_route_id}')

        # 根据 selected_route_id 构建路线名→数据映射
        self._route_map = {
            1: ('route_1', self._route_1),
            2: ('route_2', self._route_2),
            3: ('route_3', self._route_3),
            4: ('route_4', self._route_4),
        }

        # 校验必须参数
        self._validate_params()

        # ==================== 状态变量 ====================
        self._state: State = State.START
        self._state_entry_time: float = 0.0
        self._last_camera1_goal: Optional[tuple] = None  # (x, y, yaw)
        self._route_name: str = ''
        self._selected_route: list = []
        self._route_index: int = 0
        self._route_sub_state: int = 0   # 0=MOVE_XY, 1=ROTATE_YAW (route_arrive_then_rotate)
        self._zone1_cycle_count: int = 0
        self._zone1_round: int = 1  # 当前轮次 (1/2/3)
        self._zone1_cooldown_active: bool = False
        self._zone1_cooldown_start: float = 0.0
        self._zone1_reference_y: Optional[float] = None
        self._zone1_reference_yaw: Optional[float] = None

        # debug 可视化
        self._debug_marker_id: int = 0

        # 转身闭环控制
        self._turn_target_yaw: Optional[float] = None
        self._turn_start_yaw: Optional[float] = None
        self._turn_last_yaw: Optional[float] = None
        self._turn_accum_yaw: float = 0.0
        self._turn_target_delta: Optional[float] = None
        self._turn_cmd_to_yaw_sign: Optional[float] = None
        self._turn_initial_cmd_sign: float = -1.0
        self._turn_zero_count: int = 0
        self._turn_initialized: bool = False
        self._turn_absolute_target_yaw: Optional[float] = None

        # 视觉数据缓存（仅在对应状态有效）
        self._zone1_offset: Optional[float] = None
        self._grab_done_data: Optional[int] = None
        self._dock_done_data: Optional[int] = None
        self._camera4_offset: Optional[float] = None
        self._chassis_action_done_received: bool = False

        # Nav2 action 状态
        self._action_client: Optional[ActionClient] = None
        self._goal_handle = None
        self._goal_active: bool = False
        self._goal_succeeded: bool = False
        self._goal_failed: bool = False
        self._goal_result_ready: bool = False

        # ==================== 回调组 ====================
        self._cb_group = MutuallyExclusiveCallbackGroup()

        # ==================== 初始化组件 ====================
        self._init_action_client()
        self._init_tf()
        self._init_publishers()
        self._init_subscribers()

        # ==================== 主循环定时器 (10Hz) ====================
        self._timer = self.create_timer(0.1, self._state_machine_tick)

        self.get_logger().info('mission_manager 已启动')
        self.get_logger().info(f'  zone1_start: {self._zone1_start}')
        self.get_logger().info(f'  路线选择模式: {self._route_select_mode}')
        self.get_logger().info(f'  selected_route_id = {self._selected_route_id}')
        self.get_logger().info(f'  selected_route_name = route_{self._selected_route_id}')
        self.get_logger().info(f'  bench_test_mode = {self._bench_test_mode}')
        self.get_logger().info(f'  dry_run_nav = {self._dry_run_nav}')
        self.get_logger().info(f'  route_arrive_then_rotate = {self._route_arrive_then_rotate}')
        self.get_logger().info(f'  route_only_mode = {self._route_only_mode}')
        self.get_logger().info(f'  vision_timeout_sec = {self._vision_timeout_sec}')
        self.get_logger().info(f'  bench_timeout_sec = {self._bench_timeout_sec}')
        self.get_logger().info(f'  effective_vision_timeout = {self._effective_vision_timeout()}')
        self.get_logger().info(f'  enable_debug_visualization = {self._enable_debug_visualization}')
        self.get_logger().info(f'  debug_goal_topic = {self._debug_goal_topic}')
        self.get_logger().info(f'  debug_marker_topic = {self._debug_marker_topic}')
        self.get_logger().info(f'  frame: {self._frame_id} → {self._base_frame}')

    # =================================================================
    # 参数校验
    # =================================================================

    def _validate_params(self):
        """校验关键参数，缺失则报错"""
        errors = []

        # 校验 zone1_start
        z1 = self._zone1_start
        if not isinstance(z1, dict) or 'x' not in z1 or 'y' not in z1:
            errors.append('zone1_start 格式错误，需要 {x, y, yaw} 字典')

        # 校验路线参数（空列表只 WARN，不 ERROR；进入路线时再检查）
        for route_name, route_data in [('route_1', self._route_1),
                                        ('route_2', self._route_2),
                                        ('route_3', self._route_3),
                                        ('route_4', self._route_4)]:
            if not route_data or len(route_data) == 0:
                self.get_logger().warn(
                    f'{route_name} 未配置或为空，路线将在一区结束后才需要')
            elif not isinstance(route_data, list):
                errors.append(f'{route_name} 格式错误，需要列表')
            else:
                for i, wp in enumerate(route_data):
                    if not isinstance(wp, dict) or 'x' not in wp or 'y' not in wp:
                        errors.append(f'{route_name}[{i}] 格式错误，需要 {{x, y, yaw}}')

        if errors:
            for e in errors:
                self.get_logger().error(f'[PARAM ERROR] {e}')
            # 不 raise，允许启动但会在进入对应状态时报错

    def _parse_route_flat(self, values, route_name):
        """将 flat double 数组 [x1,y1,yaw1, x2,y2,yaw2, ...] 转为 dict 列表。

        兼容旧格式 [{x:..., y:..., yaw:...}, ...] 直接透传。
        """
        if values is None or len(values) == 0:
            return []
        # 兼容旧格式：已为 dict 列表则直接返回
        if isinstance(values[0], dict):
            return values
        if len(values) % 3 != 0:
            self.get_logger().error(
                f'{route_name} 长度必须是 3 的倍数，当前 len={len(values)}')
            return []
        route = []
        for i in range(0, len(values), 3):
            route.append({
                'x': float(values[i]),
                'y': float(values[i + 1]),
                'yaw': float(values[i + 2]),
            })
        return route

    # =================================================================
    # 初始化
    # =================================================================

    def _init_action_client(self):
        self._action_client = ActionClient(self, NavigateToPose, self._action_name)

    def _init_tf(self):
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _init_publishers(self):
        self._camera_start_pub = self.create_publisher(
            UInt8, self._camera_start_topic, 10)
        self._chassis_action_cmd_pub = self.create_publisher(
            Int32, '/chassis_action_cmd', 10)
        # debug 可视化
        if self._enable_debug_visualization:
            self._debug_goal_pub = self.create_publisher(
                PoseStamped, self._debug_goal_topic, 10)
            self._debug_marker_pub = self.create_publisher(
                MarkerArray, self._debug_marker_topic, 10)
        else:
            self._debug_goal_pub = None
            self._debug_marker_pub = None
        # 转身 cmd_vel
        self._turn_cmd_pub = self.create_publisher(
            Twist, self._turn_cmd_topic, 10)

    def _init_subscribers(self):
        self._zone1_offset_sub = self.create_subscription(
            Float32, self._zone1_offset_topic,
            self._zone1_offset_callback, 10,
            callback_group=self._cb_group)
        self._grab_done_sub = self.create_subscription(
            UInt8, self._grab_done_topic,
            self._grab_done_callback, 10,
            callback_group=self._cb_group)
        self._dock_done_sub = self.create_subscription(
            UInt8, self._dock_done_topic,
            self._dock_done_callback, 10,
            callback_group=self._cb_group)
        self._camera4_sub = self.create_subscription(
            Float32, '/camera4',
            self._camera4_callback, 10,
            callback_group=self._cb_group)
        self._chassis_action_done_sub = self.create_subscription(
            Int32, '/chassis_action_done',
            self._chassis_action_done_callback, 10,
            callback_group=self._cb_group)

        self._alliance: Optional[int] = None
        self._zone2_ready: bool = False
        self._alliance_sub = self.create_subscription(
            UInt8, '/competition/alliance',
            self._alliance_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self._cb_group)
        self._mission_start_sub = self.create_subscription(
            UInt8, '/mission/start',
            self._mission_start_callback, 10,
            callback_group=self._cb_group)
        self._zone2_ready_sub = self.create_subscription(
            UInt8, '/zone2_grid/ready',
            self._zone2_ready_callback,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=self._cb_group)
        self._mission_status_pub = self.create_publisher(
            String, '/mission/status', 10)

        self._zone2_start_pub = self.create_publisher(
            UInt8, '/zone2_grid/start', 10)
        self._startup_yaw_pub = self.create_publisher(
            Float64, '/zone2_grid/startup_yaw', 10)
        self._startup_yaw: Optional[float] = None  # 启动区 yaw，任务开始时捕获
        self._zone2_done_data: Optional[int] = None
        self._zone2_done_sub = self.create_subscription(
            UInt8, '/zone2_grid/done',
            self._zone2_done_callback, 10,
            callback_group=self._cb_group)

    # =================================================================
    # Topic 回调
    # =================================================================

    def _zone1_offset_callback(self, msg: Float32):
        """一区 y 偏移回调：/camera/yolo/y_offset → Float32，左正右负，限幅"""
        if self._state == State.WAIT_CAMERA1_OFFSET:
            offset = float(msg.data)
            offset = max(-self._max_vision_y_offset, min(self._max_vision_y_offset, offset))
            self._zone1_offset = offset
            self.get_logger().info(
                f'[ZONE1_OFFSET] 收到偏移: {offset:.3f}m (已限幅 ±{self._max_vision_y_offset:.1f})')

    def _grab_done_callback(self, msg: UInt8):
        """抓取完成回调：/camera/view_sig → UInt8，data==grab_done_code 时转身"""
        if self._state in (State.WAIT_CAMERA1_OFFSET, State.WAIT_CAMERA2_GRAB_DONE):
            self._grab_done_data = int(msg.data)
            self.get_logger().info(f'[GRAB_DONE] 收到数据: {self._grab_done_data}')
    def _zone2_done_callback(self, msg: UInt8):
        if self._state == State.WAIT_ZONE2_GRID_DONE:
            self._zone2_done_data = msg.data

    def _dock_done_callback(self, msg: UInt8):
        """dock_done_topic 回调"""
        # data=3 冷却期间忽略重复消息
        if self._state == State.WAIT_ZONE2_COOLDOWN:
            if msg.data == 3:
                self.get_logger().debug(
                    '[MISSION] duplicate data=3 ignored during zone2 cooldown')
                return

        if self._state == State.WAIT_CAMERA3_DOCK_DONE:
            self._dock_done_data = int(msg.data)
            self.get_logger().info(f'[DOCK_DONE] 收到数据: {self._dock_done_data}')

    def _camera4_callback(self, msg: Float32):
        """三区 y 偏移回调"""
        if self._state == State.WAIT_CAMERA4_OFFSET:
            offset = float(msg.data)
            offset = max(-self._max_vision_y_offset, min(self._max_vision_y_offset, offset))
            self._camera4_offset = offset
            self.get_logger().info(f'[CAMERA4] 收到偏移: {offset:.3f}m')

    def _alliance_callback(self, msg: UInt8):
        if msg.data in (1, 2):
            self._alliance = msg.data
            name = 'BLUE' if msg.data == 1 else 'RED'
            self.get_logger().info(f'[MISSION] alliance received: {name} (data={msg.data})')

    def _mission_start_callback(self, msg: UInt8):
        if msg.data == 1 and self._state == State.WAIT_MISSION_START:
            self.get_logger().info('[MISSION] /mission/start data=1 received')
            self._mission_started = True

    def _zone2_ready_callback(self, msg: UInt8):
        if msg.data == 1:
            self._zone2_ready = True
            self.get_logger().info('[MISSION] zone2_grid ready')

    def _chassis_action_done_callback(self, msg: Int32):
        if self._state == State.WAIT_CHASSIS_ACTION_DONE and int(msg.data) == 1:
            self._chassis_action_done_received = True
            self.get_logger().info('[CHASSIS] 动作完成信号收到')

    # =================================================================
    # TF 查询
    # =================================================================

    def _get_current_pose(self) -> Optional[tuple]:
        """返回 (x, y, yaw) 或 None"""
        try:
            if not self._tf_buffer.can_transform(
                self._frame_id, self._base_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=2.0),
            ):
                return None
            t = self._tf_buffer.lookup_transform(
                self._frame_id, self._base_frame,
                rclpy.time.Time(),
                timeout=RclpyDuration(seconds=2.0),
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            yaw = quaternion_to_yaw(qz, qw)
            return (x, y, yaw)
        except TransformException as e:
            self.get_logger().warn(f'TF 查询失败: {e}')
            return None

    # =================================================================
    # Nav2 Action 操作
    # =================================================================

    def _cancel_current_goal(self):
        """取消当前导航目标"""
        if self._goal_handle is not None:
            self.get_logger().info('取消当前导航目标...')
            try:
                self._goal_handle.cancel_goal_async()
            except Exception as e:
                self.get_logger().warn(f'取消失败: {e}')
        self._reset_goal_state()

    def _reset_goal_state(self):
        self._goal_handle = None
        self._goal_active = False
        self._goal_succeeded = False
        self._goal_failed = False
        self._goal_result_ready = False

    def _publish_debug_goal(self, x: float, y: float, yaw: float, label: str):
        """发布 debug PoseStamped + MarkerArray (sphere/arrow/text)"""
        if not self._enable_debug_visualization:
            return

        now = self.get_clock().now().to_msg()
        qz, qw = yaw_to_quaternion(yaw)

        # ----- PoseStamped -----
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self._frame_id
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        if self._debug_goal_pub is not None:
            self._debug_goal_pub.publish(pose_msg)

        # ----- MarkerArray -----
        mid = self._debug_marker_id
        self._debug_marker_id += 3  # 每次 3 个 marker

        markers = []

        # A. sphere
        sphere = Marker()
        sphere.header.stamp = now
        sphere.header.frame_id = self._frame_id
        sphere.ns = "mission_goal"
        sphere.id = mid
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = x
        sphere.pose.position.y = y
        sphere.pose.position.z = 0.15
        sphere.scale.x = 0.25
        sphere.scale.y = 0.25
        sphere.scale.z = 0.25
        sphere.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.9)
        sphere.lifetime = DurationMsg(sec=0, nanosec=0)
        markers.append(sphere)

        # B. arrow
        arrow = Marker()
        arrow.header.stamp = now
        arrow.header.frame_id = self._frame_id
        arrow.ns = "mission_goal_yaw"
        arrow.id = mid + 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = x
        arrow.pose.position.y = y
        arrow.pose.position.z = 0.2
        arrow.pose.orientation.z = qz
        arrow.pose.orientation.w = qw
        arrow.scale.x = 0.5
        arrow.scale.y = 0.08
        arrow.scale.z = 0.08
        arrow.color = ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.9)
        arrow.lifetime = DurationMsg(sec=0, nanosec=0)
        markers.append(arrow)

        # C. text
        text = Marker()
        text.header.stamp = now
        text.header.frame_id = self._frame_id
        text.ns = "mission_goal_label"
        text.id = mid + 2
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = x
        text.pose.position.y = y
        text.pose.position.z = 0.6
        text.scale.z = 0.2
        text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        text.text = label
        text.lifetime = DurationMsg(sec=0, nanosec=0)
        markers.append(text)

        if self._debug_marker_pub is not None:
            self._debug_marker_pub.publish(MarkerArray(markers=markers))

    def _send_nav_goal(self, target_x: float, target_y: float, target_yaw: float,
                        label: str = ""):
        """发送 NavigateToPose 目标，带 action server 等待重试；dry_run_nav 时跳过"""
        # debug 可视化：dry_run 和正常路径都发布
        self._publish_debug_goal(target_x, target_y, target_yaw, label)

        # dry_run_nav: 只打印目标，模拟 Nav2 接受并成功
        if self._dry_run_nav:
            self.get_logger().info(
                f'[DRY RUN] 目标: x={target_x:.3f}, y={target_y:.3f}, yaw={target_yaw:.3f} rad')
            self._goal_active = False
            self._goal_succeeded = True
            self._goal_failed = False
            self._goal_result_ready = True
            return

        # 循环等待 action server 就绪（最多 30s）
        for i in range(30):
            if self._action_client.wait_for_server(timeout_sec=1.0):
                break
            self.get_logger().warn(
                f'等待 {self._action_name} action server... {i + 1}/30')
        else:
            self.get_logger().error(
                f'Action server {self._action_name} 30s 内不可用，'
                f'请确认 /home/lyu/rc.sh 已启动且 Nav2 已 active')
            self._goal_failed = True
            self._goal_result_ready = True
            return

        self._cancel_current_goal()

        qz, qw = yaw_to_quaternion(target_yaw)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self._frame_id
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(target_x)
        goal_msg.pose.pose.position.y = float(target_y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self._goal_active = True
        self._goal_succeeded = False
        self._goal_failed = False
        self._goal_result_ready = False

        self.get_logger().info(
            f'发送目标: x={target_x:.3f}, y={target_y:.3f}, yaw={target_yaw:.3f} rad')

        send_future = self._action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('目标被 Nav2 拒绝')
            self._goal_active = False
            self._goal_failed = True
            self._goal_result_ready = True
            return

        self.get_logger().info('Nav2 已接受目标')
        self._goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result()
        status = result.status

        self._goal_handle = None
        self._goal_active = False
        self._goal_result_ready = True

        if status == 4:  # SUCCEEDED
            self.get_logger().info('目标已到达')
            self._goal_succeeded = True
            self._goal_failed = False
        elif status == 5:  # CANCELED
            self.get_logger().info('目标已取消')
            self._goal_succeeded = False
            self._goal_failed = True
        elif status == 6:  # ABORTED
            self.get_logger().error('目标被 Nav2 中止')
            self._goal_succeeded = False
            self._goal_failed = True
        else:
            self.get_logger().warn(f'目标结果: status={status}')
            self._goal_succeeded = False
            self._goal_failed = True

    # =================================================================
    # 状态切换
    # =================================================================

    def _transition_to(self, new_state: State):
        old_name = STATE_NAMES.get(self._state, '?')
        new_name = STATE_NAMES.get(new_state, '?')
        self.get_logger().info(f'[状态] {old_name} → {new_name}')
        self._state = new_state
        self._state_entry_time = time.time()
        self._goal_result_ready = False
        # mission 结束/失败时清理 reference pose
        if new_state in (State.MISSION_DONE, State.FAILED):
            if self._zone1_reference_y is not None or self._zone1_reference_yaw is not None:
                self.get_logger().info(
                    '[MISSION] clearing zone1 reference pose after mission end')
                self._zone1_reference_y = None
                self._zone1_reference_yaw = None

    def _time_in_state(self) -> float:
        return time.time() - self._state_entry_time

    def _effective_vision_timeout(self) -> float:
        """工作台模式使用 bench_timeout_sec，否则使用 vision_timeout_sec"""
        return self._bench_timeout_sec if self._bench_test_mode else self._vision_timeout_sec

    def _stop_turn_cmd(self):
        """发布零速度到 turn_cmd_topic"""
        stop = Twist()
        if self._turn_cmd_pub is not None:
            self._turn_cmd_pub.publish(stop)

    def _cleanup_turn_state(self):
        """清理转身状态变量"""
        self._stop_turn_cmd()
        self._turn_initialized = False
        self._turn_absolute_target_yaw = None
        self._turn_target_yaw = None
        self._turn_start_yaw = None
        self._turn_last_yaw = None
        self._turn_accum_yaw = 0.0
        self._turn_target_delta = None
        self._turn_cmd_to_yaw_sign = None
        self._turn_zero_count = 0

    def _execute_turn_step(self, target_yaw: float) -> Optional[bool]:
        """
        转身闭环控制。
        两种模式：
          - 累计模式（_turn_absolute_target_yaw is None）：方向学习 + 累计 π
          - 绝对模式（_turn_absolute_target_yaw is not None）：P 闭环到绝对 yaw
        返回值: True=完成, False=超时/异常, None=继续
        target_yaw 参数仅用于 dry_run 日志，实际控制不使用。
        """
        # dry_run_nav: 模拟转身
        if self._dry_run_nav:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().warn('[DRY RUN TURN] TF 不可用，跳过')
                return True
            _, _, cyaw = pose
            self.get_logger().info(
                f'[DRY RUN TURN] start_yaw={self._turn_start_yaw:.3f}, '
                f'target_yaw={target_yaw:.3f} → 模拟成功')
            return True

        # 获取当前 yaw
        pose = self._get_current_pose()
        if pose is None:
            self.get_logger().warn('转身中 TF 不可用')
            return None  # 继续等 TF

        _, _, cyaw = pose

        # ══════════════════════════════════════════════════════
        #  绝对模式：直接 P 闭环到 _turn_absolute_target_yaw
        # ══════════════════════════════════════════════════════
        if self._turn_absolute_target_yaw is not None:
            error = normalize_angle(self._turn_absolute_target_yaw - cyaw)
            abs_error = abs(error)

            # 完成判定
            if abs_error <= self._turn_yaw_tolerance:
                self._turn_zero_count += 1
                self._stop_turn_cmd()
                if self._turn_zero_count >= 3:
                    self.get_logger().info(
                        f'[TURN ABS] 转身完成: target_yaw={self._turn_absolute_target_yaw:.4f}, '
                        f'current={cyaw:.4f}, error={error:.4f}')
                    return True
                return None

            self._turn_zero_count = 0

            # 超时
            if self._time_in_state() > self._turn_timeout_sec:
                self.get_logger().error(
                    f'[TURN ABS] 转身超时 ({self._turn_timeout_sec}s): '
                    f'target={self._turn_absolute_target_yaw:.4f}, '
                    f'current={cyaw:.4f}, error={error:.4f}')
                return False

            # P 闭环
            speed = self._turn_kp * abs_error
            speed = max(self._turn_min_wz, min(self._turn_max_wz, speed))
            wz = np.sign(error) * speed if error != 0 else 0.0

            cmd = Twist()
            cmd.angular.z = wz
            if self._turn_cmd_pub is not None:
                self._turn_cmd_pub.publish(cmd)

            self.get_logger().info(
                f'[TURN ABS] 转身中: target={self._turn_absolute_target_yaw:.4f}, '
                f'current={cyaw:.4f}, error={error:.4f}, wz={wz:.3f}')
            return None

        # ══════════════════════════════════════════════════════
        #  累计模式：方向学习 + 累计 yaw → 目标 π
        # ══════════════════════════════════════════════════════
        # 累计 yaw 增量
        if self._turn_last_yaw is not None:
            delta_yaw = normalize_angle(cyaw - self._turn_last_yaw)
            self._turn_accum_yaw += delta_yaw
        self._turn_last_yaw = cyaw

        # ========== 阶段 A：方向学习 ==========
        if self._turn_cmd_to_yaw_sign is None:
            if abs(self._turn_accum_yaw) > math.pi + 1.5:
                self.get_logger().error(
                    f'转身异常：accum={self._turn_accum_yaw:.3f} 远超 π')
                return False

            if abs(self._turn_accum_yaw) > 0.05:
                observed_sign = np.sign(self._turn_accum_yaw)
                cmd_sign = self._turn_initial_cmd_sign
                self._turn_cmd_to_yaw_sign = observed_sign / cmd_sign
                self._turn_target_delta = observed_sign * math.pi
                self.get_logger().info(
                    f'转身方向映射已学习: accum={self._turn_accum_yaw:.4f}, '
                    f'cmd_sign={cmd_sign}, cmd_to_yaw_sign={self._turn_cmd_to_yaw_sign}, '
                    f'target_delta={self._turn_target_delta:.4f}')
            else:
                if self._time_in_state() > self._turn_timeout_sec:
                    self.get_logger().error(
                        f'转身方向学习超时 ({self._turn_timeout_sec}s)')
                    return False
                speed = max(self._turn_min_wz, min(self._turn_max_wz, 0.3))
                wz = self._turn_initial_cmd_sign * speed
                cmd = Twist()
                cmd.angular.z = wz
                if self._turn_cmd_pub is not None:
                    self._turn_cmd_pub.publish(cmd)
                self.get_logger().info(
                    f'转身方向学习中: accum={self._turn_accum_yaw:.4f}, '
                    f'initial_cmd_sign={self._turn_initial_cmd_sign}, wz={wz:.3f}')
                return None

        # ========== 阶段 B：误差闭环 ==========
        error = self._turn_target_delta - self._turn_accum_yaw

        if abs(error) <= self._turn_yaw_tolerance:
            self._turn_zero_count += 1
            self._stop_turn_cmd()
            if self._turn_zero_count >= 3:
                self.get_logger().info(
                    f'转身完成: accum={self._turn_accum_yaw:.4f}, '
                    f'target_delta={self._turn_target_delta:.4f}, '
                    f'error={error:.4f}')
                return True
            return None

        self._turn_zero_count = 0

        if self._time_in_state() > self._turn_timeout_sec:
            self.get_logger().error(
                f'转身超时 ({self._turn_timeout_sec}s): '
                f'accum={self._turn_accum_yaw:.4f}, error={error:.4f}')
            return False

        if abs(self._turn_accum_yaw) > math.pi + 1.5 and abs(error) > 0.5:
            self.get_logger().error(
                f'转身过冲异常: accum={self._turn_accum_yaw:.4f}')
            return False

        desired_sign = np.sign(error) if error != 0 else 1.0
        cmd_sign = desired_sign * self._turn_cmd_to_yaw_sign
        speed = self._turn_kp * abs(error)
        speed = max(self._turn_min_wz, min(self._turn_max_wz, speed))
        wz = cmd_sign * speed

        cmd = Twist()
        cmd.angular.z = wz
        if self._turn_cmd_pub is not None:
            self._turn_cmd_pub.publish(cmd)

        self.get_logger().info(
            f'转身中: start={self._turn_start_yaw:.3f}, current={cyaw:.3f}, '
            f'accum={self._turn_accum_yaw:.4f}, target_delta={self._turn_target_delta:.4f}, '
            f'error={error:.4f}, cmd_to_yaw_sign={self._turn_cmd_to_yaw_sign}, '
            f'wz={wz:.3f}')
        return None
    # =================================================================

    def _state_machine_tick(self):
        """10Hz 主循环，根据当前状态执行对应逻辑"""
        try:
            if self._state == State.START:
                self._handle_start()
            elif self._state == State.WAIT_MISSION_START:
                self._handle_wait_mission_start()
            elif self._state == State.GO_ZONE1_START:
                self._handle_go_zone1_start()
            elif self._state == State.PUBLISH_CAMERA0_START:
                self._handle_publish_camera0_start()
            elif self._state == State.WAIT_CAMERA1_OFFSET:
                self._handle_wait_camera1_offset()
            elif self._state == State.MOVE_TO_CAMERA1_DYNAMIC_GOAL:
                self._handle_move_to_camera1_dynamic_goal()
            elif self._state == State.WAIT_CAMERA2_GRAB_DONE:
                self._handle_wait_camera2_grab_done()
            elif self._state == State.TURN_AROUND_DYNAMIC:
                self._handle_turn_around_dynamic()
            elif self._state == State.WAIT_CAMERA3_DOCK_DONE:
                self._handle_wait_camera3_dock_done()
            elif self._state == State.TURN_BACK_FOR_NEXT_ROUND:
                self._handle_turn_back_for_next_round()
            elif self._state == State.MOVE_MAP_X_MINUS_0_2:
                self._handle_move_map_x_minus_02()
            elif self._state == State.WAIT_ZONE2_GRID_DONE:
                self._handle_wait_zone2_grid_done()
            elif self._state == State.RUN_FIXED_ROUTE:
                self._handle_run_fixed_route()
            elif self._state == State.ENTER_ZONE3:
                self._handle_enter_zone3()
            elif self._state == State.WAIT_CAMERA4_OFFSET:
                self._handle_wait_camera4_offset()
            elif self._state == State.MOVE_TO_ZONE3_DYNAMIC_GOAL:
                self._handle_move_zone3_dynamic_goal()
            elif self._state == State.PUBLISH_CHASSIS_ACTION_CMD:
                self._handle_publish_chassis_action_cmd()
            elif self._state == State.WAIT_CHASSIS_ACTION_DONE:
                self._handle_wait_chassis_action_done()
            elif self._state == State.TURN_AROUND_ZONE3:
                self._handle_turn_around_zone3()
            elif self._state == State.MOVE_FORWARD_0_5_DYNAMIC:
                self._handle_move_forward_05()
            elif self._state == State.WAIT_ZONE2_COOLDOWN:
                self._handle_wait_zone2_cooldown()
            elif self._state == State.MISSION_DONE:
                pass  # 完成，不再动作
            elif self._state == State.FAILED:
                pass  # 失败，停止
        except Exception as e:
            self.get_logger().error(f'状态机异常: {e}\n{traceback.format_exc()}')
            self._transition_to(State.FAILED)

    # =================================================================
    # 各状态处理函数
    # =================================================================

    def _handle_start(self):
        """START → 根据模式分发"""
        if self._route_only_mode:
            self._route_name, self._selected_route = self._route_map[self._selected_route_id]
            self._route_index = 0
            self._route_sub_state = 0
            if not self._selected_route or len(self._selected_route) == 0:
                self.get_logger().error(
                    f'{self._route_name} 未配置或为空，请在 mission_manager.yaml 中设置路线点')
                self._transition_to(State.FAILED)
                return
            self.get_logger().info(
                f'任务启动 (route_only_mode)，跳过一区，直接执行 {self._route_name} '
                f'(共 {len(self._selected_route)} 个点)')
            self._transition_to(State.RUN_FIXED_ROUTE)
        elif self._bench_test_mode:
            self.get_logger().info('任务启动 (工作台模式)，跳过 zone1_start，直接进入视觉偏移等待')
            self._transition_to(State.PUBLISH_CAMERA0_START)
        else:
            self.get_logger().info('任务启动，进入 WAIT_MISSION_START')
            self._transition_to(State.WAIT_MISSION_START)

    def _handle_wait_mission_start(self):
        """等待 alliance + zone2 ready + /mission/start → GO_ZONE1_START"""
        # 发布状态
        self._mission_status_pub.publish(String(data='等待开始'))
        if self._alliance is None:
            return
        if not hasattr(self, '_zone2_ready_logged') or not self._zone2_ready_logged:
            self.get_logger().info('[MISSION] alliance received, waiting zone2_grid ready + /mission/start')
            self._zone2_ready_logged = True
        if not self._zone2_ready:
            return
        if not hasattr(self, '_mission_started'):
            return
        if not self._mission_started:
            return
        # 确认配置可用
        name = 'BLUE' if self._alliance == 1 else 'RED'
        z1_cfg = getattr(self, f'_zone1_start_{name.lower()}', None)
        if z1_cfg is None:
            self.get_logger().error(
                f'[MISSION] {name} zone1_start is not configured')
            return
        self._zone1_start = z1_cfg
        self._zone1_reference_y = z1_cfg['y']
        self._zone1_reference_yaw = z1_cfg['yaw']
        self.get_logger().info(
            f'[MISSION] Zone1 reference pose: alliance={name}, '
            f'zone1_start=({z1_cfg["x"]:.3f}, {z1_cfg["y"]:.3f}, {z1_cfg["yaw"]:.4f}), '
            f'reference_y={self._zone1_reference_y:.4f} '
            f'reference_yaw={self._zone1_reference_yaw:.4f}')
        # 捕获启动区 yaw
        pose = self._get_current_pose()
        if pose is not None:
            self._startup_yaw = pose[2]
            self.get_logger().info(
                f'[MISSION] startup yaw captured: {self._startup_yaw:.3f} rad / '
                f'{math.degrees(self._startup_yaw):.1f} deg')
        else:
            self.get_logger().warn('[MISSION] TF unavailable at startup, startup_yaw not captured')
            self._startup_yaw = 0.0
        self.get_logger().info(
            f'[MISSION] full mission starting: alliance={name}, '
            f'zone1_start=({z1_cfg["x"]:.3f}, {z1_cfg["y"]:.3f}, {z1_cfg["yaw"]:.3f})')
        self._mission_status_pub.publish(String(data='一区执行中'))
        self._transition_to(State.GO_ZONE1_START)

    def _handle_go_zone1_start(self):
        """导航到 zone1_start → PUBLISH_CAMERA0_START"""
        if not self._goal_active and not self._goal_result_ready:
            z1 = self._zone1_start
            # 安全检查：禁止 YAML 未加载时默认 0,0,0 发车
            if z1['x'] == 0.0 and z1['y'] == 0.0 and z1['yaw'] == 0.0:
                self.get_logger().error(
                    'zone1_start 为 (0, 0, 0)，YAML 可能未正确加载！不允许发车。'
                    '如需强制使用原点，请设置 allow_zero_zone1_start: true')
                self._transition_to(State.FAILED)
                return
            self._send_nav_goal(float(z1['x']), float(z1['y']), float(z1.get('yaw', 0.0)),
                                label="zone1_start")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                pose = self._get_current_pose()
                if pose is not None:
                    self._last_camera1_goal = pose
                self._transition_to(State.PUBLISH_CAMERA0_START)
            else:
                self.get_logger().error('前往 zone1_start 失败')
                self._transition_to(State.FAILED)

    def _publish_camera0_triple(self):
        """连续发布 3 次 camera_start_topic = camera_start_code，间隔 0.1s，避免视觉节点漏收"""
        for i in range(3):
            msg = UInt8()
            msg.data = self._camera_start_code
            self._camera_start_pub.publish(msg)
            self.get_logger().info(
                f'发布 {self._camera_start_topic} = {self._camera_start_code} ({i + 1}/3)')
            if i < 2:
                time.sleep(0.1)

    def _handle_publish_camera0_start(self):
        """发布 camera_start_topic (单发) → WAIT_CAMERA1_OFFSET（无次数上限）"""
        self._zone1_cycle_count += 1
        msg = UInt8()
        msg.data = self._camera_start_code
        self._camera_start_pub.publish(msg)
        self.get_logger().info(
            f'发布 {self._camera_start_topic} = {self._camera_start_code} '
            f'(第 {self._zone1_cycle_count} 次)')
        self._zone1_offset = None
        self._grab_done_data = None
        self._transition_to(State.WAIT_CAMERA1_OFFSET)

    def _handle_wait_camera1_offset(self):
        """等待 zone1_offset_topic 或 grab_done → MOVE_TO_CAMERA1_DYNAMIC_GOAL / TURN_AROUND / FAILED"""
        if self._zone1_offset is not None:
            self._transition_to(State.MOVE_TO_CAMERA1_DYNAMIC_GOAL)
        elif self._grab_done_data == self._grab_done_code:
            if self._last_camera1_goal is None:
                pose = self._get_current_pose()
                if pose is not None:
                    self._last_camera1_goal = pose
                    self.get_logger().info(
                        f'[MISSION] grab_done received before camera1 offset; '
                        f'use current pose as fallback last_camera1_goal: '
                        f'x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={pose[2]:.3f}')
                else:
                    self.get_logger().warn(
                        '[MISSION] grab_done received before camera1 offset, '
                        'but current pose unavailable; last_camera1_goal remains None')
            self._transition_to(State.TURN_AROUND_DYNAMIC)
        elif self._time_in_state() > self._effective_vision_timeout():
            self.get_logger().warn(
                f'等待 {self._zone1_offset_topic} 超时 ({self._effective_vision_timeout()}s)')
            self._transition_to(State.FAILED)

    def _handle_move_to_camera1_dynamic_goal(self):
        """基于 zone1_offset_topic y_offset 移动 → WAIT_CAMERA2_GRAB_DONE"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().error('无法获取当前位姿')
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            offset = self._zone1_offset
            tx, ty, tyaw = compute_dynamic_goal(cx, cy, cyaw, dx=0.0, dy=offset, dyaw=0.0)
            self._last_camera1_goal = (tx, ty, tyaw)
            self.get_logger().info(
                f'[MISSION] camera1 dynamic goal saved as last_camera1_goal: '
                f'x={tx:.3f}, y={ty:.3f}, yaw={tyaw:.3f}')
            self.get_logger().info(
                f'动态目标 (zone1_offset): dy={offset:.3f}m → map({tx:.3f}, {ty:.3f})')
            self._send_nav_goal(tx, ty, tyaw,
                                label=f"zone1_offset dy={offset:.3f}")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self.get_logger().info('平移到位，重新请求相机检测')
                self._grab_done_data = None
                self._transition_to(State.PUBLISH_CAMERA0_START)
            else:
                self.get_logger().error('动态目标导航失败')
                self._transition_to(State.FAILED)

    def _handle_wait_camera2_grab_done(self):
        """等待 grab_done_topic == grab_done_code → TURN_AROUND_DYNAMIC"""
        if self._grab_done_data == self._grab_done_code:
            self._transition_to(State.TURN_AROUND_DYNAMIC)
        elif self._time_in_state() > self._effective_vision_timeout():
            self.get_logger().warn(f'等待 {self._grab_done_topic} 超时')
            self._transition_to(State.FAILED)

    def _handle_turn_around_dynamic(self):
        """原地转身 180° (Nav2 NavigateToPose) → WAIT_CAMERA3_DOCK_DONE"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().error('转身：无法获取当前位姿')
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            tyaw = normalize_angle(cyaw + math.pi)
            self._publish_debug_goal(cx, cy, tyaw, "turn_around 180deg")
            self.get_logger().info(
                f'原地转身 180°: 保持 ({cx:.3f}, {cy:.3f}), yaw {cyaw:.3f} → {tyaw:.3f}')
            self._send_nav_goal(cx, cy, tyaw, label="turn_around 180deg")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self._dock_done_data = None
                self._transition_to(State.WAIT_CAMERA3_DOCK_DONE)
            else:
                self.get_logger().error('转身失败')
                self._transition_to(State.FAILED)

    def _handle_wait_camera3_dock_done(self):
        """等待 dock_done_topic → TURN_BACK_FOR_NEXT_ROUND 或 zone2 流程"""
        if self._dock_done_data is None:
            return

        code = self._dock_done_data
        if code == self._zone1_continue_code:
            # code=2: 对接完成，转身回任务朝向，继续下一轮
            if self._zone1_round >= 3:
                self.get_logger().error(
                    f'[MISSION] view_cmd=2 ignored: round={self._zone1_round}, only round 1/2 accept data=2')
                self._dock_done_data = None
                return
            self._dock_done_data = None  # 消费消息，防止重复
            self.get_logger().info(
                f'[MISSION] view_cmd=2 accepted: round={self._zone1_round}, '
                'starting turn-back to task orientation')
            self._transition_to(State.TURN_BACK_FOR_NEXT_ROUND)
        elif code == self._zone1_finish_code:
            # code=3: 一区任务完成，进入 4 秒冷却状态
            if self._zone1_round != 3:
                self.get_logger().warn(
                    f'[MISSION] view_cmd=3 at round={self._zone1_round} (expected round=3), '
                    f'accepting anyway')
            self._dock_done_data = None
            self.get_logger().info(
                f'[MISSION] view_cmd=3, zone1 finished (round={self._zone1_round}), '
                f'starting 4.0s cooldown')
            self._zone1_cycle_count = 0
            self._zone1_round = 1
            self._zone1_cooldown_start = time.monotonic()
            self._transition_to(State.WAIT_ZONE2_COOLDOWN)
        else:
            self.get_logger().warn(f'未知 {self._dock_done_topic} code: {code}，忽略')

    def _handle_wait_zone2_cooldown(self):
        """data=3 后 4 秒冷却，然后发布 /zone2_grid/start"""
        elapsed = time.monotonic() - self._zone1_cooldown_start
        if elapsed < 4.0:
            if int(elapsed) != int(elapsed - 0.1):  # 每秒打印一次
                self.get_logger().info(
                    f'[MISSION] zone2 cooldown: elapsed={elapsed:.1f}s '
                    f'remaining={4.0 - elapsed:.1f}s')
            return

        self.get_logger().info('[MISSION] 4.0s cooldown complete')
        self._zone1_cooldown_start = 0.0

        zone2_exec = self.get_parameter('zone2_executor').value
        if zone2_exec == 'grid_gui':
            self.get_logger().info('[MISSION] zone2 executor=grid_gui')
            self.get_logger().info(
                f'[MISSION] waiting /zone2_grid/done '
                f'timeout={self._zone2_grid_done_timeout:.1f}s')
            self.get_logger().info('[MISSION] publish /zone2_grid/start data=1')
            if self._startup_yaw is not None:
                self._startup_yaw_pub.publish(Float64(data=self._startup_yaw))
                self.get_logger().info(
                    f'[MISSION] publish /zone2_grid/startup_yaw '
                    f'{math.degrees(self._startup_yaw):.1f}°')
            self._zone2_start_pub.publish(UInt8(data=1))
            self._zone2_done_data = None
            self._zone2_start_retried_secs = set()
            self._transition_to(State.WAIT_ZONE2_GRID_DONE)
        else:
            self._route_name, self._selected_route = self._route_map[self._selected_route_id]
            self._route_index = 0
            if not self._selected_route or len(self._selected_route) == 0:
                self.get_logger().error(
                    f'{self._route_name} 未配置或为空')
                self._transition_to(State.FAILED)
                return
            self.get_logger().info(
                f'zone1 finished, running {self._route_name} '
                f'(共 {len(self._selected_route)} 个点)')
            self._route_sub_state = 0
            self._transition_to(State.RUN_FIXED_ROUTE)

    def _handle_wait_zone2_grid_done(self):
        """等待 /zone2_grid/done → MISSION_DONE 或 FAILED"""
        elapsed = self._time_in_state()
        timeout = self._zone2_grid_done_timeout

        # timeout
        if elapsed > timeout:
            self.get_logger().error(
                f'[MISSION] zone2 grid timeout waiting done after {timeout:.1f}s')
            self._transition_to(State.FAILED)
            return

        # 第 0/1/2 秒各重发一次 start
        retry_sec = int(elapsed)
        if retry_sec < 3 and retry_sec not in self._zone2_start_retried_secs:
            self._zone2_start_retried_secs.add(retry_sec)
            self.get_logger().info(
                f'[MISSION] publish /zone2_grid/start data=1 '
                f'retry={len(self._zone2_start_retried_secs)} elapsed={elapsed:.1f}s')
            # 每次重试前先重发 startup_yaw
            if self._startup_yaw is not None:
                self._startup_yaw_pub.publish(Float64(data=self._startup_yaw))
            self._zone2_start_pub.publish(UInt8(data=1))

        if self._zone2_done_data is None:
            return
        code = self._zone2_done_data
        if code == 1:
            self.get_logger().info('[MISSION] zone2 grid done data=1')
            self._transition_to(State.MISSION_DONE)
        elif code == 2:
            self.get_logger().error('[MISSION] zone2 grid failed (data=2)')
            self._transition_to(State.FAILED)
        elif code == 3:
            self.get_logger().error('[MISSION] zone2 grid: no cells selected (data=3)')
            self._transition_to(State.FAILED)
        elif code == 4:
            self.get_logger().warn('[MISSION] zone2 grid already running (data=4), ignoring')
            self._zone2_done_data = None  # 继续等待
        else:
            self.get_logger().warn(f'[MISSION] unknown /zone2_grid/done code: {code}')

    def _handle_turn_back_for_next_round(self):
        """data=2 后第一步：原地转身 180° 回到任务朝向 → MOVE_MAP_X_MINUS_0_2"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().error('[TURN_BACK] 无法获取当前位姿')
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            tyaw = normalize_angle(cyaw + math.pi)
            self.get_logger().info(
                f'[TURN_BACK] round={self._zone1_round} turn-back source: '
                f'x={cx:.3f}, y={cy:.3f}, yaw={math.degrees(cyaw):.1f}°')
            self.get_logger().info(
                f'[TURN_BACK] round={self._zone1_round} turn-back goal: '
                f'x={cx:.3f}, y={cy:.3f}, yaw={math.degrees(tyaw):.1f}° '
                f'(delta_x=0, delta_y=0, delta_yaw=180°)')
            self._publish_debug_goal(cx, cy, tyaw, f"turn_back_r{self._zone1_round}")
            self._send_nav_goal(cx, cy, tyaw, label=f"turn_back_r{self._zone1_round}")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                pose = self._get_current_pose()
                if pose is not None:
                    self.get_logger().info(
                        f'[TURN_BACK] round={self._zone1_round} turn-back completed: '
                        f'x={pose[0]:.3f}, y={pose[1]:.3f}, yaw={math.degrees(pose[2]):.1f}°')
                self._transition_to(State.MOVE_MAP_X_MINUS_0_2)
            else:
                self.get_logger().error(f'[TURN_BACK] round={self._zone1_round} 转身失败')
                self._transition_to(State.FAILED)

    def _handle_move_map_x_minus_02(self):
        """data=2 后第二步：向 map X 负 0.20m，Y 回到 zone1 基准线 → 下一轮"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().error('[MOVE_MAP_X] 无法获取当前位姿')
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            ref_y = self._zone1_reference_y
            ref_yaw = self._zone1_reference_yaw
            if ref_y is None or ref_yaw is None:
                self.get_logger().error(
                    '[MOVE_MAP_X] zone1 reference pose not set, cannot compute target')
                self._transition_to(State.FAILED)
                return
            tx = cx + self._zone1_continue_map_x_offset
            ty = ref_y
            tyaw = ref_yaw
            y_correction = ref_y - cy
            yaw_correction = math.degrees(normalize_angle(ref_yaw - cyaw))
            self.get_logger().info(
                f'[MOVE_MAP_X] round={self._zone1_round} map -X 0.20m: '
                f'post_turn=({cx:.3f}, {cy:.3f}, {math.degrees(cyaw):.1f}°) '
                f'ref_y={ref_y:.4f} ref_yaw={math.degrees(ref_yaw):.1f}° '
                f'y_corr={y_correction:.4f} yaw_corr={yaw_correction:.1f}° '
                f'goal=({tx:.3f}, {ty:.3f}, {math.degrees(tyaw):.1f}°)')
            self._publish_debug_goal(tx, ty, tyaw, f"move_map_minus_x_r{self._zone1_round}")
            self._send_nav_goal(tx, ty, tyaw, label=f"move_map_X-0.20_r{self._zone1_round}")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self._grab_done_data = None
                self.get_logger().info(
                    f'[MOVE_MAP_X] round {self._zone1_round} complete, next camera detection')
                self._zone1_round += 1  # 移动成功后才递增轮次
                self._transition_to(State.PUBLISH_CAMERA0_START)
            else:
                self.get_logger().error(f'[MOVE_MAP_X] round={self._zone1_round} 移动失败')
                self._transition_to(State.FAILED)

    def _handle_run_fixed_route(self):
        """运行固定路线 → ENTER_ZONE3。

        当 route_arrive_then_rotate=true 时，每个 route 点拆为两段：
          MOVE_XY:  移动到 (x, y)，保持当前车头方向
          ROTATE_YAW: 原地转到目标 yaw
        当 route_arrive_then_rotate=false 时，直接发送 (x, y, yaw)。
        """
        if self._route_index >= len(self._selected_route):
            self.get_logger().info('路线完成，进入三区')
            self._route_index = 0
            self._transition_to(State.ENTER_ZONE3)
            return

        wp = self._selected_route[self._route_index]
        route_idx = self._route_index + 1
        route_total = len(self._selected_route)
        target_x = float(wp['x'])
        target_y = float(wp['y'])
        target_yaw = float(wp.get('yaw', 0.0))

        # 旧逻辑：直接发送完整目标
        if not self._route_arrive_then_rotate:
            if not self._goal_active and not self._goal_result_ready:
                self.get_logger().info(f'路线点 [{route_idx}/{route_total}]: '
                                       f'({target_x:.3f}, {target_y:.3f}, yaw={target_yaw:.3f})')
                self._send_nav_goal(target_x, target_y, target_yaw,
                                    label=f"route[{route_idx}/{route_total}]")
            elif self._goal_result_ready:
                if self._goal_succeeded:
                    self._route_index += 1
                    self._goal_result_ready = False
                else:
                    self.get_logger().error(f'路线点 [{route_idx}] 失败')
                    self._transition_to(State.FAILED)
            return

        # 新逻辑：两段式 arrive → then → rotate
        if not self._goal_active and not self._goal_result_ready:
            if self._route_sub_state == 0:
                # —— MOVE_XY ——
                pose = self._get_current_pose()
                if pose is None:
                    self.get_logger().error('无法获取当前位姿，跳过 MOVE_XY 直接发完整目标')
                    self.get_logger().info(f'路线点 [{route_idx}/{route_total}]: '
                                           f'({target_x:.3f}, {target_y:.3f}, yaw={target_yaw:.3f})')
                    self._send_nav_goal(target_x, target_y, target_yaw,
                                        label=f"route[{route_idx}/{route_total}]")
                    self._route_sub_state = 1  # 跳过 MOVE_XY 结果直接走 ROTATE_YAW 逻辑
                    return
                _, _, current_yaw = pose
                self.get_logger().info(
                    f'[ROUTE] point {route_idx}/{route_total} MOVE_XY: '
                    f'goal=({target_x:.3f}, {target_y:.3f}, {current_yaw:.3f})')
                self._send_nav_goal(target_x, target_y, current_yaw,
                                    label=f"route[{route_idx}]_move_xy")
            else:
                # —— ROTATE_YAW ——
                pose = self._get_current_pose()
                current_yaw = pose[2] if pose is not None else 0.0
                yaw_error = normalize_angle(target_yaw - current_yaw)
                if abs(yaw_error) < 0.03:
                    self.get_logger().info(
                        f'[ROUTE] point {route_idx}/{route_total} yaw diff={abs(yaw_error):.4f} < 0.03, skip ROTATE_YAW')
                    self._route_index += 1
                    self._route_sub_state = 0
                    self._goal_result_ready = False
                    return
                self.get_logger().info(
                    f'[ROUTE] point {route_idx}/{route_total} ROTATE_YAW: '
                    f'goal=({target_x:.3f}, {target_y:.3f}, {target_yaw:.3f})')
                self._send_nav_goal(target_x, target_y, target_yaw,
                                    label=f"route[{route_idx}]_rotate")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                if self._route_sub_state == 0:
                    # MOVE_XY 成功 → 进入 ROTATE_YAW
                    self._route_sub_state = 1
                    self._goal_result_ready = False
                else:
                    # ROTATE_YAW 成功 → 下一点
                    self.get_logger().info(f'[ROUTE] point {route_idx}/{route_total} done')
                    self._route_index += 1
                    self._route_sub_state = 0
                    self._goal_result_ready = False
            else:
                phase = 'MOVE_XY' if self._route_sub_state == 0 else 'ROTATE_YAW'
                self.get_logger().error(f'路线点 [{route_idx}] {phase} 失败')
                self._transition_to(State.FAILED)

    def _handle_enter_zone3(self):
        """进入三区 → WAIT_CAMERA4_OFFSET"""
        self.get_logger().info('进入三区')
        self._camera4_offset = None
        self._transition_to(State.WAIT_CAMERA4_OFFSET)

    def _handle_wait_camera4_offset(self):
        """等待 /camera4 → MOVE_TO_ZONE3_DYNAMIC_GOAL"""
        if self._camera4_offset is not None:
            self._transition_to(State.MOVE_TO_ZONE3_DYNAMIC_GOAL)
        elif self._time_in_state() > self._vision_timeout_sec:
            self.get_logger().warn(f'等待 /camera4 超时')
            self._transition_to(State.FAILED)

    def _handle_move_zone3_dynamic_goal(self):
        """基于 /camera4 y_offset 移动 → PUBLISH_CHASSIS_ACTION_CMD"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            offset = self._camera4_offset
            tx, ty, tyaw = compute_dynamic_goal(
                cx, cy, cyaw, dx=self._zone3_forward_x_offset, dy=offset, dyaw=0.0)
            self.get_logger().info(f'三区动态目标: dx=+{self._zone3_forward_x_offset:.1f}, '
                                   f'dy={offset:.3f} → map({tx:.3f}, {ty:.3f})')
            self._send_nav_goal(tx, ty, tyaw,
                                label=f"zone3 dx+0.3 dy={offset:.3f}")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self._transition_to(State.PUBLISH_CHASSIS_ACTION_CMD)
            else:
                self.get_logger().error('三区动态目标失败')
                self._transition_to(State.FAILED)

    def _handle_publish_chassis_action_cmd(self):
        """发布 /chassis_action_cmd = 1 → WAIT_CHASSIS_ACTION_DONE"""
        msg = Int32()
        msg.data = 1
        self._chassis_action_cmd_pub.publish(msg)
        self.get_logger().info('发布 /chassis_action_cmd = 1')
        self._chassis_action_done_received = False
        self._transition_to(State.WAIT_CHASSIS_ACTION_DONE)

    def _handle_wait_chassis_action_done(self):
        """等待 /chassis_action_done = 1 → TURN_AROUND_ZONE3"""
        if self._chassis_action_done_received:
            self._transition_to(State.TURN_AROUND_ZONE3)
        elif self._time_in_state() > self._action_timeout_sec:
            self.get_logger().warn(f'等待 chassis_action_done 超时')
            self._transition_to(State.FAILED)

    def _handle_turn_around_zone3(self):
        """原地转身 180° (Nav2 NavigateToPose) → MOVE_FORWARD_0_5_DYNAMIC"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self.get_logger().error('三区转身：无法获取当前位姿')
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            tyaw = normalize_angle(cyaw + math.pi)
            self._publish_debug_goal(cx, cy, tyaw, "zone3 turn_around")
            self.get_logger().info(
                f'三区转身 180°: 保持 ({cx:.3f}, {cy:.3f}), yaw {cyaw:.3f} → {tyaw:.3f}')
            self._send_nav_goal(cx, cy, tyaw, label="zone3 turn_around")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self._transition_to(State.MOVE_FORWARD_0_5_DYNAMIC)
            else:
                self.get_logger().error('三区转身失败')
                self._transition_to(State.FAILED)

    def _handle_move_forward_05(self):
        """前进 0.5m → MISSION_DONE"""
        if not self._goal_active and not self._goal_result_ready:
            pose = self._get_current_pose()
            if pose is None:
                self._transition_to(State.FAILED)
                return
            cx, cy, cyaw = pose
            tx, ty, tyaw = compute_dynamic_goal(
                cx, cy, cyaw, dx=self._final_forward_x_offset, dy=0.0, dyaw=0.0)
            self.get_logger().info(f'最后前进 {self._final_forward_x_offset:.1f}m: '
                                   f'map({tx:.3f}, {ty:.3f})')
            self._send_nav_goal(tx, ty, tyaw, label=f"forward {self._final_forward_x_offset:.1f}m")
        elif self._goal_result_ready:
            if self._goal_succeeded:
                self.get_logger().info('===== MISSION COMPLETE =====')
                self._transition_to(State.MISSION_DONE)
            else:
                self.get_logger().error('最后前进失败')
                self._transition_to(State.FAILED)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
