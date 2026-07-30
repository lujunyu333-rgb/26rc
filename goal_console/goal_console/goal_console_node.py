#!/usr/bin/env python3
"""
goal_console_node — 终端交互式 Nav2 目标点导航控制台

功能:
  ls / list      — 列出所有预存点位
  pose           — 显示当前 map->base_link 位姿
  record / rec   — 保存当前机器人位置为新点位
  click          — 从 RViz /clicked_point 保存点位
  add            — 手动添加点位
  edit           — 编辑点位
  rm / delete    — 删除点位
  go             — 发送导航目标（二次确认）
  go!            — 快速发送导航目标（不确认）
  cancel / stop  — 取消当前导航目标
  status         — 显示系统状态
  reload         — 重新从 YAML 加载点位
  save           — 手动保存点位到 YAML
  help           — 显示命令帮助
  quit / exit    — 退出控制台
"""

import math
import os
import shlex
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, PoseStamped
import tf2_ros
from tf2_ros import TransformException
import yaml


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def yaw_to_quaternion(yaw: float):
    """弧度 yaw → (qz, qw)，平面旋转（绕Z轴）"""
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(qz: float, qw: float) -> float:
    """(qz, qw) → yaw (弧度)"""
    # 只提取偏航角，假设 roll=pitch=0（平面2D）
    siny_cosp = 2.0 * (qw * qz)
    cosy_cosp = 1.0 - 2.0 * (qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _looks_like_int(s: str) -> bool:
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------

class GoalConsoleNode(Node):
    """终端交互式 Nav2 目标点导航控制台"""

    def __init__(self):
        super().__init__('goal_console_node')

        # -------------------- 参数 --------------------
        self.declare_parameter('points_yaml', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('action_name', '/navigate_to_pose')

        self._points_yaml_path: str = self.get_parameter('points_yaml').value
        self._map_frame: str = self.get_parameter('map_frame').value
        self._base_frame: str = self.get_parameter('base_frame').value
        self._action_name: str = self.get_parameter('action_name').value

        # 如果 launch 没传路径，用包内默认路径
        if not self._points_yaml_path:
            import ament_index_python
            try:
                pkg_share = ament_index_python.get_package_share_directory('goal_console')
                self._points_yaml_path = os.path.join(pkg_share, 'config', 'goal_points.yaml')
            except Exception:
                self._points_yaml_path = os.path.join(
                    os.path.dirname(__file__), '..', 'config', 'goal_points.yaml')

        # -------------------- 状态 --------------------
        self._points: list[dict] = []          # 点位列表
        self._clicked_point: Optional[PointStamped] = None
        self._clicked_point_event = threading.Event()

        # Action 相关
        self._action_client: Optional[ActionClient] = None
        self._goal_handle = None
        self._goal_future = None
        self._active_goal_name: str = ''       # 当前正在导航的目标名

        # 退出标志
        self._running = True

        # -------------------- 回调组 --------------------
        self._cb_group = MutuallyExclusiveCallbackGroup()

        # -------------------- 初始化组件 --------------------
        self._init_action_client()
        self._init_tf()
        self._init_clicked_subscriber()

        # -------------------- 加载点位 --------------------
        self._load_points()

        self.get_logger().info('goal_console_node 已启动')
        self.get_logger().info(f'点位文件: {self._points_yaml_path}')
        self.get_logger().info(f'地图坐标系: {self._map_frame} → 基座: {self._base_frame}')
        self.get_logger().info(f'Action: {self._action_name}')
        self.get_logger().info(f'已加载 {len(self._points)} 个点位')

    # =================================================================
    # 初始化
    # =================================================================

    def _init_action_client(self):
        """初始化 NavigateToPose Action 客户端"""
        self._action_client = ActionClient(self, NavigateToPose, self._action_name)

    def _init_tf(self):
        """初始化 TF 监听器"""
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _init_clicked_subscriber(self):
        """订阅 RViz /clicked_point"""
        self._clicked_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self._clicked_callback,
            10,
            callback_group=self._cb_group,
        )

    def _clicked_callback(self, msg: PointStamped):
        """收到 /clicked_point 消息"""
        self._clicked_point = msg
        self._clicked_point_event.set()

    # =================================================================
    # YAML 读写
    # =================================================================

    def _load_points(self):
        """从 YAML 文件加载点位"""
        try:
            with open(self._points_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None:
                self._points = []
                return
            raw = data.get('points', [])
            self._points = []
            for p in raw:
                if not isinstance(p, dict):
                    continue
                self._points.append({
                    'name': str(p.get('name', '')),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'yaw': float(p.get('yaw', 0.0)),
                    'yaw_deg': float(p.get('yaw_deg', 0.0)),
                    'description': str(p.get('description', '')),
                })
        except FileNotFoundError:
            self.get_logger().warn(f'点位文件不存在: {self._points_yaml_path}，将使用空列表')
            self._points = []
        except Exception as e:
            self.get_logger().error(f'加载点位文件失败: {e}')
            self._points = []

    def _save_points(self):
        """保存点位到 YAML 文件"""
        try:
            data = {'points': self._points}
            with open(self._points_yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f'点位已保存到 {self._points_yaml_path}')
        except Exception as e:
            print(f'[ERROR] 保存点位失败: {e}')

    def _find_point(self, identifier: str) -> tuple[Optional[int], Optional[dict]]:
        """根据名称或 1-based 编号查找点位，返回 (index, point_dict)"""
        if _looks_like_int(identifier):
            idx = int(identifier) - 1
            if 0 <= idx < len(self._points):
                return idx, self._points[idx]
            else:
                return None, None
        # 按名称匹配（精确匹配）
        for i, p in enumerate(self._points):
            if p['name'] == identifier:
                return i, p
        return None, None

    # =================================================================
    # TF 查询
    # =================================================================

    def _get_current_pose(self) -> Optional[tuple[float, float, float]]:
        """查询当前 map->base_link 变换，返回 (x, y, yaw_rad) 或 None"""
        try:
            if not self._tf_buffer.can_transform(
                self._map_frame,
                self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            ):
                print(f'[ERROR] TF 不可用: {self._map_frame} → {self._base_frame}')
                return None

            t = self._tf_buffer.lookup_transform(
                self._map_frame,
                self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0),
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            yaw = quaternion_to_yaw(qz, qw)
            return (x, y, yaw)
        except TransformException as e:
            print(f'[ERROR] TF 查询失败: {e}')
            return None
        except Exception as e:
            print(f'[ERROR] TF 查询异常: {e}')
            return None

    # =================================================================
    # 命令实现
    # =================================================================

    def cmd_help(self):
        """显示帮助信息"""
        print("""
══════════════════════════════════════════════════════════════
  goal_console — Nav2 终端导航控制台
══════════════════════════════════════════════════════════════

  help                  — 显示本帮助
  ls | list             — 列出所有点位（编号、名称、坐标、角度、描述）
  pose                  — 显示当前 map→base_link 位姿
  record <name> [desc]  — 保存当前机器人位置为点位
  rec  <name> [desc]    — 同上（简写）
  click <name> [yaw_deg] [desc]
                        — 等待 RViz Publish Point，保存点位
  add <name> <x> <y> <yaw_deg> [desc]
                        — 手动添加点位
  edit <name> yaw_deg <value>
                        — 编辑点位信息
  edit <name> x <value>
  edit <name> y <value>
  edit <name> desc <text>
  rm  <name_or_index>   — 删除点位
  delete <name_or_index> — 同上
  go  <name_or_index>   — 发送导航目标（需确认）
  go! <name_or_index>   — 快速发送导航目标（不确认，谨慎使用）
  cancel | stop         — 取消当前导航目标
  status                — 显示系统状态
  reload                — 重新从 YAML 加载点位
  save                  — 手动保存点位到 YAML
  quit | exit           — 退出控制台

  编号为 1-based 列表序号。
  角度 yaw_deg: 0=朝东, 90=朝北, ±180=朝西, -90=朝南。
══════════════════════════════════════════════════════════════
""")

    def cmd_ls(self):
        """列出所有点位"""
        if not self._points:
            print('(无预存点位)')
            return
        print(f'{"#":>3s}  {"name":<20s}  {"x":>10s}  {"y":>10s}  {"yaw_deg":>8s}  description')
        print('-' * 75)
        for i, p in enumerate(self._points, 1):
            desc = p.get('description', '')
            print(f'{i:3d}  {p["name"]:<20s}  {p["x"]:10.3f}  {p["y"]:10.3f}  {p["yaw_deg"]:8.1f}  {desc}')

    def cmd_pose(self):
        """显示当前机器人位姿"""
        result = self._get_current_pose()
        if result is None:
            return
        x, y, yaw = result
        deg = rad_to_deg(yaw)
        print(f'pose (map→{self._base_frame}):')
        print(f'  x      = {x:.4f}')
        print(f'  y      = {y:.4f}')
        print(f'  yaw    = {yaw:.6f} rad')
        print(f'  yaw_deg= {deg:.2f}°')

    def cmd_record(self, name: str, description: str = ''):
        """保存当前机器人位置"""
        if not name:
            print('[ERROR] 用法: record <name> [description]')
            return

        # 检查是否已存在
        idx, existing = self._find_point(name)
        if existing is not None:
            ans = input(f'点位 "{name}" 已存在，是否覆盖? [y/N] ').strip().lower()
            if ans != 'y':
                print('已取消。')
                return

        result = self._get_current_pose()
        if result is None:
            return
        x, y, yaw = result
        deg = rad_to_deg(yaw)

        point = {
            'name': name,
            'x': round(x, 4),
            'y': round(y, 4),
            'yaw': round(yaw, 6),
            'yaw_deg': round(deg, 2),
            'description': description,
        }

        if existing is not None:
            # 覆盖
            self._points[idx] = point
            print(f'已覆盖点位 #{idx + 1}: {name}')
        else:
            self._points.append(point)
            print(f'已添加点位 #{len(self._points)}: {name}')

        self._save_points()

    def cmd_click(self, name: str, yaw_deg_str: str = '0', description: str = ''):
        """等待 RViz /clicked_point 并保存点位"""
        if not name:
            print('[ERROR] 用法: click <name> [yaw_deg] [description]')
            return

        # 尝试解析 yaw_deg，如果失败则认为是 description 的一部分
        try:
            yaw_deg = float(yaw_deg_str)
        except (ValueError, TypeError):
            yaw_deg = 0.0
            description = (yaw_deg_str + ' ' + description).strip()

        # 检查是否已存在
        idx, existing = self._find_point(name)
        if existing is not None:
            ans = input(f'点位 "{name}" 已存在，是否覆盖? [y/N] ').strip().lower()
            if ans != 'y':
                print('已取消。')
                return

        self._clicked_point = None
        self._clicked_point_event.clear()

        print(f'请在 RViz 中使用 "Publish Point" 工具点击地图上的目标位置...')
        print(f'(等待 /clicked_point，frame_id 必须为 {self._map_frame}，超时 120 秒)')

        # 等待 clicked_point
        if not self._clicked_point_event.wait(120.0):
            print('[ERROR] 等待 /clicked_point 超时（120秒）')
            return

        cp = self._clicked_point
        if cp is None:
            print('[ERROR] 未收到 /clicked_point')
            return

        # 检查 frame_id
        if cp.header.frame_id != self._map_frame:
            print(f'[ERROR] /clicked_point 的 frame_id={cp.header.frame_id}，'
                  f'需要 {self._map_frame}。请确认 RViz 中 Fixed Frame 为 {self._map_frame}。')
            return

        x = cp.point.x
        y = cp.point.y
        yaw = deg_to_rad(yaw_deg)

        point = {
            'name': name,
            'x': round(float(x), 4),
            'y': round(float(y), 4),
            'yaw': round(yaw, 6),
            'yaw_deg': round(yaw_deg, 2),
            'description': description,
        }

        if existing is not None:
            self._points[idx] = point
            print(f'已覆盖点位 #{idx + 1}: {name}')
        else:
            self._points.append(point)
            print(f'已添加点位 #{len(self._points)}: {name}')

        self._save_points()

    def cmd_add(self, name: str, x: float, y: float, yaw_deg: float, description: str = ''):
        """手动添加点位"""
        if not name:
            print('[ERROR] 用法: add <name> <x> <y> <yaw_deg> [description]')
            return

        idx, existing = self._find_point(name)
        if existing is not None:
            ans = input(f'点位 "{name}" 已存在，是否覆盖? [y/N] ').strip().lower()
            if ans != 'y':
                print('已取消。')
                return

        yaw = deg_to_rad(yaw_deg)
        point = {
            'name': name,
            'x': round(float(x), 4),
            'y': round(float(y), 4),
            'yaw': round(yaw, 6),
            'yaw_deg': round(float(yaw_deg), 2),
            'description': description,
        }

        if existing is not None:
            self._points[idx] = point
            print(f'已覆盖点位 #{idx + 1}: {name}')
        else:
            self._points.append(point)
            print(f'已添加点位 #{len(self._points)}: {name}')

        self._save_points()

    def cmd_edit(self, identifier: str, field: str, value: str):
        """编辑点位"""
        idx, point = self._find_point(identifier)
        if point is None:
            print(f'[ERROR] 未找到点位: {identifier}')
            return

        field = field.lower()
        try:
            if field == 'yaw_deg':
                new_val = float(value)
                point['yaw_deg'] = round(new_val, 2)
                point['yaw'] = round(deg_to_rad(new_val), 6)
            elif field == 'x':
                point['x'] = round(float(value), 4)
            elif field == 'y':
                point['y'] = round(float(value), 4)
            elif field in ('desc', 'description'):
                point['description'] = value
            else:
                print(f'[ERROR] 不支持的字段: {field}，支持: x, y, yaw_deg, desc')
                return

            print(f'已更新点位 #{idx + 1} "{point["name"]}": {field} = {value}')
            self._save_points()
        except ValueError:
            print(f'[ERROR] 无效的值: {value}')

    def cmd_delete(self, identifier: str):
        """删除点位"""
        idx, point = self._find_point(identifier)
        if point is None:
            print(f'[ERROR] 未找到点位: {identifier}')
            return

        name = point['name']
        del self._points[idx]
        print(f'已删除点位: {name}')
        self._save_points()

    def cmd_go(self, identifier: str, skip_confirm: bool = False):
        """发送 Nav2 导航目标"""
        idx, point = self._find_point(identifier)
        if point is None:
            print(f'[ERROR] 未找到点位: {identifier}')
            return

        # 检查 action server
        if not self._is_action_server_ready():
            print(f'[ERROR] Action server "{self._action_name}" 不可用，无法发送目标')
            return

        # 二次确认
        if not skip_confirm:
            print(f'目标点位: "{point["name"]}"')
            print(f'  x={point["x"]:.4f}, y={point["y"]:.4f}, yaw={point["yaw_deg"]:.2f}°')
            ans = input('Send goal? [y/N] ').strip().lower()
            if ans != 'y':
                print('已取消。')
                return

        self._send_goal(point)

    def _is_action_server_ready(self) -> bool:
        """检查 action server 是否可用"""
        if self._action_client is None:
            return False
        # 非阻塞检查
        return self._action_client.server_is_ready()

    def _send_goal(self, point: dict):
        """实际发送 NavigateToPose 目标"""
        qz, qw = yaw_to_quaternion(point['yaw'])

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self._map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(point['x'])
        goal_msg.pose.pose.position.y = float(point['y'])
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        name = point['name']
        self._active_goal_name = name

        print(f'→ 发送目标: "{name}" '
              f'x={point["x"]:.4f} y={point["y"]:.4f} yaw={point["yaw_deg"]:.2f}°')

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        """Action goal 被 server 接受/拒绝后的回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            print(f'[FAILED] 目标被 Nav2 拒绝: "{self._active_goal_name}"')
            self._active_goal_name = ''
            return

        print(f'[ACCEPTED] Nav2 已接受目标: "{self._active_goal_name}"，开始规划...')
        self._goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _feedback_callback(self, feedback_msg):
        """导航过程中的 feedback"""
        fb = feedback_msg.feedback
        dist = fb.distance_remaining
        nav_time = fb.navigation_time
        est_remain = fb.estimated_time_remaining

        nav_sec = nav_time.sec + nav_time.nanosec * 1e-9 if hasattr(nav_time, 'sec') else 0
        est_sec = est_remain.sec + est_remain.nanosec * 1e-9 if hasattr(est_remain, 'sec') else 0

        parts = [f'"{self._active_goal_name}"']
        parts.append(f'剩余距离={dist:.2f}m' if dist > 0 else f'剩余距离={dist:.2f}m')
        if nav_sec > 0:
            parts.append(f'已用={nav_sec:.1f}s')
        if est_sec > 0:
            parts.append(f'预计剩余={est_sec:.1f}s')

        print(f'  [FB] ' + ' | '.join(parts))

    def _goal_result_callback(self, future):
        """导航完成后的回调"""
        result = future.result()
        status = result.status

        name = self._active_goal_name
        self._active_goal_name = ''
        self._goal_handle = None

        # 根据 action status 判断
        # status: 0=UNKNOWN, 1=ACCEPTED, 2=EXECUTING, 3=CANCELING, 4=SUCCEEDED,
        #          5=CANCELED, 6=ABORTED
        status_map = {
            0: 'UNKNOWN',
            1: 'ACCEPTED',
            2: 'EXECUTING',
            3: 'CANCELING',
            4: 'SUCCEEDED',
            5: 'CANCELED',
            6: 'ABORTED',
        }
        status_str = status_map.get(status, f'CODE_{status}')

        if status == 4:
            print(f'[SUCCESS] 目标 "{name}" 已到达！')
        elif status == 5:
            print(f'[CANCELED] 目标 "{name}" 已被取消。')
        elif status == 6:
            print(f'[ABORTED] 目标 "{name}" 被 Nav2 中止。')
        else:
            print(f'[RESULT] 目标 "{name}": {status_str}')

        # 打印 result 中的错误码（如果有）
        if hasattr(result.result, 'error_code') and result.result.error_code != 0:
            print(f'  error_code={result.result.error_code}')

    def cmd_cancel(self):
        """取消当前导航目标"""
        if self._goal_handle is None:
            print('当前没有 active goal。')
            return

        name = self._active_goal_name
        print(f'正在取消目标 "{name}"...')
        try:
            cancel_future = self._goal_handle.cancel_goal_async()
            # 不阻塞主线程：取消请求已发出，
            # _goal_result_callback 会在目标最终完成/取消时打印结果
            print(f'[OK] 目标 "{name}" 取消请求已提交。')
        except Exception as e:
            print(f'[ERROR] 取消失败: {e}')

    def cmd_status(self):
        """显示系统状态"""
        # Action server
        ready = self._is_action_server_ready()
        print(f'Action Server ({self._action_name}): {"✓ 可用" if ready else "✗ 不可用"}')

        # Active goal
        if self._goal_handle is not None and self._active_goal_name:
            print(f'Active Goal: "{self._active_goal_name}"')
        else:
            print('Active Goal: 无')

        # TF
        result = self._get_current_pose()
        if result is not None:
            x, y, yaw = result
            print(f'TF ({self._map_frame}→{self._base_frame}): ✓ 可用')
            print(f'  x={x:.4f}, y={y:.4f}, yaw={rad_to_deg(yaw):.2f}°')
        else:
            print(f'TF ({self._map_frame}→{self._base_frame}): ✗ 不可用')

        # 点位
        print(f'已加载点位: {len(self._points)} 个')

    def cmd_reload(self):
        """重新加载 YAML"""
        self._load_points()
        print(f'已重新加载 {len(self._points)} 个点位。')

    def cmd_save(self):
        """手动保存"""
        self._save_points()
        print('点位已手动保存。')

    # =================================================================
    # 交互循环
    # =================================================================

    def _parse_command(self, line: str):
        """解析命令行输入"""
        line = line.strip()
        if not line:
            return

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f'[ERROR] 命令解析失败: {e}')
            return

        if not parts:
            return
        cmd = parts[0].lower()

        try:
            if cmd in ('help', 'h', '?'):
                self.cmd_help()

            elif cmd in ('ls', 'list'):
                self.cmd_ls()

            elif cmd == 'pose':
                self.cmd_pose()

            elif cmd in ('record', 'rec'):
                name = parts[1] if len(parts) > 1 else ''
                desc = ' '.join(parts[2:]) if len(parts) > 2 else ''
                self.cmd_record(name, desc)

            elif cmd == 'click':
                if len(parts) < 2:
                    print('[ERROR] 用法: click <name> [yaw_deg] [description]')
                    return
                name = parts[1]
                yaw_deg = parts[2] if len(parts) > 2 else '0'
                desc = ' '.join(parts[3:]) if len(parts) > 3 else ''
                self.cmd_click(name, yaw_deg, desc)

            elif cmd == 'add':
                if len(parts) < 5:
                    print('[ERROR] 用法: add <name> <x> <y> <yaw_deg> [description]')
                    return
                name = parts[1]
                x = float(parts[2])
                y = float(parts[3])
                yaw_deg = float(parts[4])
                desc = ' '.join(parts[5:]) if len(parts) > 5 else ''
                self.cmd_add(name, x, y, yaw_deg, desc)

            elif cmd == 'edit':
                if len(parts) < 4:
                    print('[ERROR] 用法: edit <name> <field> <value>')
                    print('  field: x, y, yaw_deg, desc')
                    return
                identifier = parts[1]
                field = parts[2]
                # desc 字段值可能含空格，合并剩余参数
                if field.lower() in ('desc', 'description'):
                    value = ' '.join(parts[3:])
                else:
                    value = parts[3] if len(parts) > 3 else ''
                self.cmd_edit(identifier, field, value)

            elif cmd in ('rm', 'delete'):
                if len(parts) < 2:
                    print('[ERROR] 用法: rm <name_or_index>')
                    return
                self.cmd_delete(parts[1])

            elif cmd == 'go':
                if len(parts) < 2:
                    print('[ERROR] 用法: go <name_or_index>')
                    return
                self.cmd_go(parts[1], skip_confirm=False)

            elif cmd == 'go!':
                if len(parts) < 2:
                    print('[ERROR] 用法: go! <name_or_index>')
                    return
                self.cmd_go(parts[1], skip_confirm=True)

            elif cmd in ('cancel', 'stop'):
                self.cmd_cancel()

            elif cmd == 'status':
                self.cmd_status()

            elif cmd == 'reload':
                self.cmd_reload()

            elif cmd == 'save':
                self.cmd_save()

            elif cmd in ('quit', 'exit', 'q'):
                print('正在退出 goal_console...')
                self._running = False

            else:
                print(f'未知命令: {cmd}，输入 help 查看帮助。')

        except Exception as e:
            print(f'[ERROR] 命令执行异常: {e}')

    def run(self):
        """主交互循环（在后台线程中运行）"""
        print()
        print('╔════════════════════════════════════════════════╗')
        print('║       goal_console — Nav2 终端导航控制台       ║')
        print('╠════════════════════════════════════════════════╣')
        print('║  输入 help 查看命令    输入 quit 退出          ║')
        print('╚════════════════════════════════════════════════╝')
        print()

        while self._running and rclpy.ok():
            try:
                line = input('goal> ')
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                print('按 Ctrl+C，输入 quit 或 exit 退出。')
                continue

            self._parse_command(line)

        self._running = False


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = GoalConsoleNode()

    # 使用 MultiThreadedExecutor：一个线程 spin，主线程运行交互循环
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    # 在后台线程运行 executor
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        node.run()
    finally:
        node._running = False
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)


if __name__ == '__main__':
    main()
