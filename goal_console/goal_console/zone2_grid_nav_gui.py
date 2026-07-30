#!/usr/bin/env python3
"""
zone2_grid_nav_gui — 二区 3×5 格子导航 GUI (含前置点)

独立工具，不依赖 mission_manager / route_1~4 / goal_gui。
使用 Nav2 NavigateToPose action，按 segment 执行：

到达 R0C1 之前（pre-grid）：
  所有点（F/R/OUT）使用一次直接 NavigateToPose，边走边转。

到达 R0C1 之后（grid mode）：
  R/OUT 点：先转向 segment_yaw，再走到目标 (SEGMENT_ROTATE → SEGMENT_MOVE)

F 点始终使用一次直接导航 + stair_action_cmd + delay。
"""

import math
import signal
import time
import tkinter as tk
import tkinter.ttk as ttk
from enum import IntEnum
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import ExternalShutdownException

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float64, String, UInt8
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter as RclParameter
from rcl_interfaces.msg import ParameterValue

import tf2_ros
from tf2_ros import TransformException


# ══════════════════════════════════════════════════════════════
#  网格常量
# ══════════════════════════════════════════════════════════════

GRID_ROWS = 5
GRID_COLS = 3
GRID_SIZE = 1.2

# ══════════════════════════════════════════════════════════════
#  阵营坐标配置
# ══════════════════════════════════════════════════════════════

class Alliance(IntEnum):
    BLUE = 1
    RED = 2

ALLIANCE_GRID = {
    Alliance.BLUE: {
        'R0C1_X': 2.7222,
        'R0C1_Y': -2.3241,
        'F0_POINTS': {
            0: (1.8616, -1.0934),
            1: (1.4891, -2.3241),
            2: (1.8616, -3.5065),
        },
        'F00_POINTS': {
            0: (1.4616, -1.0934),
            2: (1.4616, -3.5065),
        },
    },
    Alliance.RED: {
        'R0C1_X': 2.7522,
        'R0C1_Y': -8.3741,
        'F0_POINTS': {
            0: (1.8616, -7.1434),
            1: (1.5091, -8.3741),
            2: (1.8616, -9.5565),
        },
        'F00_POINTS': {
            0: (1.4616, -7.1434),
            2: (1.4616, -9.5565),
        },
    },
}

ALLIANCE_LABELS = {Alliance.BLUE: '蓝方', Alliance.RED: '红方'}

# 默认蓝方值（用于 compute_grid_center 未传 alliance 时的兼容）
R0C1_X = 2.7522
R0C1_Y = -2.2741

ZONE2_STAIR_CMD_TOPIC = "/stair_action_cmd"
ZONE2_STAIR_CMD_VALUE = 1

YAW_SKIP_THRESHOLD = 0.03
AXIS_STEP_EPS = 0.03

# ══════════════════════════════════════════════════════════════
#  层高表 (cm)
# ══════════════════════════════════════════════════════════════
BLUE_HEIGHT_CM = {
    'R0C0': 40, 'R0C1': 20, 'R0C2': 40,
    'R1C0': 20, 'R1C1': 40, 'R1C2': 60,
    'R2C0': 40, 'R2C1': 60, 'R2C2': 40,
    'R3C0': 20, 'R3C1': 40, 'R3C2': 20,
    'OUT0': 0,  'OUT2': 0,
}

RED_HEIGHT_CM = {
    'R0C0': 40, 'R0C1': 20, 'R0C2': 40,
    'R1C0': 60, 'R1C1': 40, 'R1C2': 20,
    'R2C0': 40, 'R2C1': 60, 'R2C2': 40,
    'R3C0': 20, 'R3C1': 40, 'R3C2': 20,
    'OUT0': 0,  'OUT2': 0,
}

DOWNHILL_SPEED = 0.55  # m/s


# ══════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════

def normalize_angle(a: float) -> float:
    a = math.fmod(a, 2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_diff(a: float, b: float) -> float:
    return abs(normalize_angle(b - a))


def compute_grid_center(row: int, col: int, alliance: Optional[Alliance] = None) -> Optional[Tuple[float, float]]:
    """返回 (x, y) 或 None（该点未配置）"""
    cfg = ALLIANCE_GRID.get(alliance)
    if cfg is None:
        return None
    rx0 = cfg['R0C1_X']
    ry0 = cfg['R0C1_Y']
    if row == 4:
        if rx0 is None or ry0 is None or col == 1:
            return None
        rx = rx0 + 3 * GRID_SIZE
        ry = ry0 + (1 - col) * GRID_SIZE
        return rx + GRID_SIZE, ry
    if row >= 0:
        if rx0 is None or ry0 is None:
            return None
        x = rx0 + row * GRID_SIZE
        y = ry0 + (1 - col) * GRID_SIZE
        return x, y
    if row == -2:
        fp = cfg.get('F00_POINTS', {}).get(col)
        if fp is None:
            return None
        return fp
    fp = cfg['F0_POINTS'].get(col)
    if fp is None:
        return None
    return fp


def compute_segment_yaw(sx, sy, nx, ny):
    return math.atan2(ny - sy, nx - sx)


def grid_label(row: int, col: int) -> str:
    if row == -1:
        return f"F0C{col}"
    if row == -2:
        return f"F00C{col}"
    if row == 4:
        return f"OUT{col}"
    return f"R{row}C{col}"


# ══════════════════════════════════════════════════════════════
#  执行子状态
# ══════════════════════════════════════════════════════════════

class ExecPhase(IntEnum):
    IDLE = 0
    SEGMENT_ROTATE = 1
    SEGMENT_MOVE = 2
    GRAB_DELAY = 3
    F_ACTION_DELAY = 13           # F 点 stair action 等待 (保留 value=13 兼容旧日志)
    # 以下 phase 已停用，保留枚举值以兼容：
    # F_ROTATE_ZERO = 10
    # F_MOVE_Y = 11
    # F_MOVE_X = 12


# ══════════════════════════════════════════════════════════════
#  主节点
# ══════════════════════════════════════════════════════════════

class Zone2GridNav(Node):
    """二区格子导航 GUI"""

    def __init__(self):
        super().__init__('zone2_grid_nav_gui')

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._action_client = ActionClient(
            self, NavigateToPose, '/navigate_to_pose',
            callback_group=ReentrantCallbackGroup())

        self._stair_action_pub = self.create_publisher(
            UInt8, ZONE2_STAIR_CMD_TOPIC, 10)
        self._cmd_vel_nav_pub = self.create_publisher(
            Twist, '/cmd_vel_nav', 10)
        self._cmd_vel_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)
        self._zone2_done_pub = self.create_publisher(
            UInt8, '/zone2_grid/done', 10)
        self._zone2_start_sub = self.create_subscription(
            UInt8, '/zone2_grid/start', self._on_zone2_start, 10)
        self._zone2_startup_yaw_sub = self.create_subscription(
            Float64, '/zone2_grid/startup_yaw', self._on_startup_yaw, 10)
        self._alliance_pub = self.create_publisher(
            UInt8, '/competition/alliance',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._alliance_sub = self.create_subscription(
            UInt8, '/competition/alliance', self._on_alliance_msg,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # ==================== 状态字段（必须在任何 publish/subscribe 前初始化） ====================
        self._alliance: Optional[Alliance] = None
        self._startup_ready: bool = False
        self._startup_pose_status: int = 0
        self._mission_status: str = '等待开始'
        self._shutting_down: bool = False
        self._executing: bool = False
        self._selected: List[Tuple[int, int]] = []
        self._grid_targets: List[Tuple[float, float]] = []
        self._seg_index: int = 0
        self._stop_requested: bool = False
        self._segment_yaw: float = 0.0
        self._grab_delay_start: float = 0.0
        self._action_delay: float = 0.0
        self._current_start: Tuple[float, float, float] = (0., 0., 0.)
        self._f_target_x: float = 0.0
        self._f_target_y: float = 0.0
        self._f_cur_x: float = 0.0
        self._f_cur_y: float = 0.0
        self._f_delay: float = 0.0
        self._f_stair_active: bool = False
        self._grid_mode_active: bool = False
        self._zone1_end_yaw: Optional[float] = None
        self._startup_yaw: Optional[float] = None
        self._saved_max_accel: Optional[List[float]] = None
        self._zone2_accel_applied: bool = False
        self._saved_max_vel: Optional[List[float]] = None
        self._saved_min_vel: Optional[List[float]] = None

        # ==================== 创建 publisher/subscriber ====================
        self._ready_pub = self.create_publisher(
            UInt8, '/zone2_grid/ready',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._mission_start_pub = self.create_publisher(
            UInt8, '/mission/start', 10)
        self._mission_status_sub = self.create_subscription(
            String, '/mission/status', self._on_mission_status, 10)
        self._startup_status_sub = self.create_subscription(
            UInt8, '/competition/startup_pose_status',
            self._on_startup_status,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # SIGTERM 优雅退出
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        # 默认发布 ready=0
        self._publish_ready()

        self._exec_phase: ExecPhase = ExecPhase.IDLE
        self._goal_handle = None
        self._goal_result_ready: bool = False
        self._goal_succeeded: bool = False

        self._param_get_client = self.create_client(
            GetParameters, '/velocity_smoother/get_parameters')
        self._param_set_client = self.create_client(
            SetParameters, '/velocity_smoother/set_parameters')

        self._root = tk.Tk()
        self._root.title("二区格子导航 (Zone2 Grid Nav) — 3×5")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._cell_buttons: List[List[tk.Button]] = []
        self._setup_gui()

        self._spin_period_ms = 50
        self._root.after(self._spin_period_ms, self._ros_spin_once)
        self._root.after(100, self._exec_tick)
        self.get_logger().info('zone2_grid_nav_gui 已启动 (3×5)')

    # ══════════════════════════════════════════════════════════
    #  GUI
    # ══════════════════════════════════════════════════════════

    def _setup_gui(self):
        main_frame = ttk.Frame(self._root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(main_frame, text="二区 3×5 格子导航 (含前置点)",
                  font=("", 14, "bold")).pack(pady=(0, 10))
        grid_frame = ttk.Frame(main_frame)
        grid_frame.pack(pady=5)
        display_rows = [4, 3, 2, 1, 0, -1, -2]
        for drow in display_rows:
            row_frame = ttk.Frame(grid_frame)
            row_frame.pack()
            btn_row = []
            for col in range(GRID_COLS):
                if drow == 4 and col == 1:
                    lbl = tk.Label(row_frame, text="", width=8, height=2)
                    lbl.pack(side=tk.LEFT, padx=2, pady=2)
                    btn_row.append(None)
                    continue
                text = grid_label(drow, col)
                btn = tk.Button(
                    row_frame, text=text, width=8, height=2,
                    bg="lightgray",
                    command=lambda r=drow, c=col: self._on_cell_click(r, c))
                btn.pack(side=tk.LEFT, padx=2, pady=2)
                btn_row.append(btn)
            self._cell_buttons.append(btn_row)

        # 阵营选择
        alliance_frame = ttk.LabelFrame(main_frame, text="阵营 (Alliance)", padding=5)
        alliance_frame.pack(pady=(10, 0))
        self._alliance_var = tk.IntVar(value=0)  # 0=未选
        self._alliance_rb_blue = tk.Radiobutton(
            alliance_frame, text="蓝方 (BLUE)", variable=self._alliance_var,
            value=1, command=self._on_alliance_change)
        self._alliance_rb_blue.pack(side=tk.LEFT, padx=8)
        self._alliance_rb_red = tk.Radiobutton(
            alliance_frame, text="红方 (RED)", variable=self._alliance_var,
            value=2, command=self._on_alliance_change)
        self._alliance_rb_red.pack(side=tk.LEFT, padx=8)

        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(pady=10)
        self._btn_clear = tk.Button(ctrl_frame, text="清空",
                                     command=self._clear_selection, width=10)
        self._btn_clear.pack(side=tk.LEFT, padx=4)
        self._btn_reverse = tk.Button(ctrl_frame, text="反向",
                                       command=self._reverse_selection, width=10)
        self._btn_reverse.pack(side=tk.LEFT, padx=4)
        self._btn_preview = tk.Button(ctrl_frame, text="预览目标点",
                                       command=self._preview, width=12)
        self._btn_preview.pack(side=tk.LEFT, padx=4)
        self._btn_full_mission = tk.Button(ctrl_frame, text="开始完整任务",
                                          command=self._start_full_mission,
                                          bg="blue", fg="white", width=14)
        self._btn_full_mission.pack(side=tk.LEFT, padx=4)
        self._btn_start = tk.Button(ctrl_frame, text="单独测试二区",
                                     command=self._start_execution,
                                     bg="green", width=12)
        self._btn_start.pack(side=tk.LEFT, padx=4)
        self._btn_stop = tk.Button(ctrl_frame, text="停止",
                                    command=self._stop_execution,
                                    bg="red", fg="white", width=10,
                                    state=tk.DISABLED)
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self._status_var = tk.StringVar(value="就绪 — 点击格子开始选择")
        ttk.Label(status_frame, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W, padding=4).pack(fill=tk.X)

        seq_frame = ttk.LabelFrame(main_frame, text="执行序列", padding=5)
        seq_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self._seq_text = tk.Text(seq_frame, height=5, width=45,
                                  state=tk.DISABLED, font=("monospace", 10))
        self._seq_text.pack(fill=tk.BOTH, expand=True)

    # ══════════════════════════════════════════════════════════
    #  ROS2 spin / cell select / preview
    # ══════════════════════════════════════════════════════════

    def _ros_spin_once(self):
        try:
            rclpy.spin_once(self, timeout_sec=0.001)
        except Exception:
            pass
        self._root.after(self._spin_period_ms, self._ros_spin_once)

    def _on_cell_click(self, row: int, col: int):
        if self._executing:
            return
        if row == 4 and col == 1:
            return  # OUT 空白列不可点击
        key = (row, col)
        if key in self._selected:
            self._selected.remove(key)
        else:
            self._selected.append(key)
        self._refresh_buttons()
        self._update_seq_display()

    def _clear_selection(self):
        if self._executing:
            return
        self._selected.clear()
        self._refresh_buttons()
        self._update_seq_display()
        self._set_status("已清空")

    def _reverse_selection(self):
        if self._executing:
            return
        self._selected.reverse()
        self._refresh_buttons()
        self._update_seq_display()
        self._set_status("已反向")

    def _refresh_buttons(self):
        idx_map = {0: 4, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2}
        for idx_row, btn_row in enumerate(self._cell_buttons):
            drow = idx_map.get(idx_row, idx_row)
            for col in range(GRID_COLS):
                btn = btn_row[col]
                if btn is None:
                    continue
                key = (drow, col)
                if key in self._selected:
                    pos = self._selected.index(key) + 1
                    btn.config(text=f"{pos}.{grid_label(drow, col)}", bg="lightblue")
                else:
                    btn.config(text=grid_label(drow, col), bg="lightgray")

    def _update_seq_display(self):
        self._seq_text.config(state=tk.NORMAL)
        self._seq_text.delete("1.0", tk.END)
        if not self._selected:
            self._seq_text.insert(tk.END, "（未选择格子）")
        else:
            for i, (r, c) in enumerate(self._selected):
                pt = compute_grid_center(r, c, self._alliance)
                if pt is None:
                    self._seq_text.insert(tk.END, f"{i + 1}. {grid_label(r, c)}: [NOT CONFIGURED]\n")
                    continue
                x, y = pt
                tag = ""
                if self._is_stair_action_point(r, c):
                    delay = self._get_action_delay(r, c)
                    cmd = self._get_stair_action_cmd(r, c) or 1
                    tag = f" [STAIR: cmd={cmd}, delay={delay:.1f}s]"
                self._seq_text.insert(tk.END, f"{i + 1}. {grid_label(r, c)}: x={x:.4f} y={y:.4f}{tag}\n")
        self._seq_text.config(state=tk.DISABLED)
        self._publish_ready()

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _preview(self):
        if not self._selected:
            self._set_status("预览：未选择格子")
            return
        lines = ["=== 预览目标点 ==="]
        for i, (r, c) in enumerate(self._selected):
            pt = compute_grid_center(r, c, self._alliance)
            if pt is None:
                lines.append(f"{i + 1}. {grid_label(r, c)}: [NOT CONFIGURED for this alliance]")
                continue
            x, y = pt
            tag = ""
            if self._is_stair_action_point(r, c):
                delay = self._get_action_delay(r, c)
                cmd = self._get_stair_action_cmd(r, c) or 1
                tag = f" [STAIR: cmd={cmd}, delay={delay:.1f}s]"
            lines.append(f"{i + 1}. {grid_label(r, c)}: x={x:.4f} y={y:.4f}{tag}")
        self.get_logger().info("\n" + "\n".join(lines))
        self._set_status(f"预览已打印 — 共 {len(self._selected)} 个点")

    # ══════════════════════════════════════════════════════════
    #  判断方法
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _is_stair_action_point(row: int, col: int) -> bool:
        return row == -1 and col in (0, 1, 2)

    @staticmethod
    def _is_r0c1(row: int, col: int) -> bool:
        return row == 0 and col == 1

    @staticmethod
    def _is_c1_cell(row: int, col: int) -> bool:
        """C1 列：F0C1, R0C1, R1C1, R2C1, R3C1（OUT1 不存在）"""
        return col == 1 and row in (-1, 0, 1, 2, 3)

    @staticmethod
    def _is_f_cell(row: int, col: int) -> bool:
        """F 点：F0C0, F0C1, F0C2 / F00C0, F00C2"""
        return (row == -1 and col in (0, 1, 2)) or (row == -2 and col in (0, 2))

    @staticmethod
    def _get_action_delay(row: int, col: int) -> float:
        if row == -1:
            if col == 1:
                return 2.0
            if col in (0, 2):
                return 3.0
        return 0.0

    @staticmethod
    def _get_stair_action_cmd(row: int, col: int) -> Optional[int]:
        if row == -1 and col == 1:
            return 2
        if row == -1 and col in (0, 2):
            return 1
        return None

    # ══════════════════════════════════════════════════════════
    #  TF / Nav Action / 停车
    # ══════════════════════════════════════════════════════════

    def _get_current_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            tf = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            q = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return x, y, math.atan2(siny, cosy)
        except TransformException as e:
            self.get_logger().warn(f'TF: {e}')
            return None

    def _send_nav_goal(self, x: float, y: float, yaw: float):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self._goal_result_ready = False
        self._goal_succeeded = False
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('/navigate_to_pose 不可用')
            self._goal_result_ready = True
            self._goal_succeeded = False
            return
        send_future = self._action_client.send_goal_async(goal)
        send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().error('NavigateToPose 被拒绝')
            self._goal_result_ready = True
            self._goal_succeeded = False
            return
        self._goal_handle = gh
        gh.get_result_async().add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        self._goal_result_ready = True
        self._goal_succeeded = (future.result().status == 4)

    def _cancel_current_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None

    def _publish_zero_velocity(self):
        z = Twist()
        self._cmd_vel_nav_pub.publish(z)
        self._cmd_vel_pub.publish(z)

    def _publish_stair_action(self, value: int = ZONE2_STAIR_CMD_VALUE):
        msg = UInt8(data=value)
        self._stair_action_pub.publish(msg)
        self.get_logger().info(
            f'[GRID_NAV] {ZONE2_STAIR_CMD_TOPIC} data={value}')

    # ══════════════════════════════════════════════════════════
    #  运行时加速度限制
    # ══════════════════════════════════════════════════════════

    def _apply_zone2_accel_limit(self):
        if self._saved_max_accel is not None:
            return  # already applied
        if not self._param_get_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                '[GRID_NAV] failed to apply zone2 accel limit: '
                '/velocity_smoother parameter service unavailable')
            return
        try:
            # get current
            get_req = GetParameters.Request()
            get_req.names = ['max_accel']
            get_fut = self._param_get_client.call_async(get_req)
            rclpy.spin_until_future_complete(self, get_fut, timeout_sec=2.0)
            if get_fut.result() and get_fut.result().values:
                raw = get_fut.result().values[0].double_array_value
                old = [float(v) for v in raw] if raw else [6.0, 6.0, 3.0]
            else:
                old = [6.0, 6.0, 3.0]
            self._saved_max_accel = old

            # set
            pv = ParameterValue()
            pv.type = 8  # PARAMETER_DOUBLE_ARRAY
            pv.double_array_value = [3.0, 3.0, 3.0]
            param = RclParameter()
            param.name = 'max_accel'
            param.value = pv
            set_req = SetParameters.Request()
            set_req.parameters = [param]
            self._param_set_client.call_async(set_req)

            self.get_logger().info(
                f'[GRID_NAV] zone2 accel limit applied: '
                f'max_accel {old} -> [3.0, 3.0, 3.0]')
        except Exception as e:
            self.get_logger().warn(
                f'[GRID_NAV] zone2 accel limit failed: {e}')
            self._saved_max_accel = None

    def _restore_zone2_accel_limit(self):
        if self._saved_max_accel is None:
            return
        try:
            pv = ParameterValue()
            pv.type = 8
            pv.double_array_value = self._saved_max_accel
            param = RclParameter()
            param.name = 'max_accel'
            param.value = pv
            set_req = SetParameters.Request()
            set_req.parameters = [param]
            self._param_set_client.call_async(set_req)
            self.get_logger().info(
                f'[GRID_NAV] zone2 accel limit restored: '
                f'max_accel [3.0, 3.0, 3.0] -> {self._saved_max_accel}')
        except Exception as e:
            self.get_logger().warn(
                f'[GRID_NAV] zone2 accel restore failed: {e}')
        self._saved_max_accel = None
        self._zone2_accel_applied = False

    # ══════════════════════════════════════════════════════════════
    #  下台阶速度限制
    # ══════════════════════════════════════════════════════════════

    def _get_height_table(self):
        if self._alliance == Alliance.BLUE:
            return BLUE_HEIGHT_CM
        if self._alliance == Alliance.RED:
            return RED_HEIGHT_CM
        return None

    def _is_descending_segment(self, from_cell: str, to_cell: str) -> bool:
        tbl = self._get_height_table()
        if tbl is None:
            return False
        h_from = tbl.get(from_cell)
        h_to = tbl.get(to_cell)
        if h_from is None or h_to is None:
            return False  # F 点或缺失层高 → 不变速
        return h_to < h_from

    def _apply_downhill_speed(self):
        """下台阶：max_velocity 和 min_velocity 同步限制为 ±0.5 m/s"""
        if self._saved_max_vel is not None:
            return  # 已经限制
        if not self._param_get_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('[GRID_NAV] downhill speed: param service unavailable')
            return
        try:
            # 读取当前 max_velocity 和 min_velocity
            get_req = GetParameters.Request()
            get_req.names = ['max_velocity', 'min_velocity']
            get_fut = self._param_get_client.call_async(get_req)
            rclpy.spin_until_future_complete(self, get_fut, timeout_sec=2.0)
            result = get_fut.result()
            if result and len(result.values) >= 2:
                raw_max = result.values[0].double_array_value
                raw_min = result.values[1].double_array_value
                old_max = [float(v) for v in raw_max] if raw_max else [2.0, 2.0, 1.8]
                old_min = [float(v) for v in raw_min] if raw_min else [-2.0, -2.0, -1.8]
            else:
                old_max = [2.0, 2.0, 1.8]
                old_min = [-2.0, -2.0, -1.8]

            self._saved_max_vel = old_max
            self._saved_min_vel = old_min

            # 设置下台阶限制（线速度 ±0.5，角速度不变）
            downhill_max = [DOWNHILL_SPEED, DOWNHILL_SPEED, old_max[2]]
            downhill_min = [-DOWNHILL_SPEED, -DOWNHILL_SPEED, old_min[2]]

            pv_max = ParameterValue()
            pv_max.type = 8
            pv_max.double_array_value = downhill_max
            param_max = RclParameter()
            param_max.name = 'max_velocity'
            param_max.value = pv_max

            pv_min = ParameterValue()
            pv_min.type = 8
            pv_min.double_array_value = downhill_min
            param_min = RclParameter()
            param_min.name = 'min_velocity'
            param_min.value = pv_min

            set_req = SetParameters.Request()
            set_req.parameters = [param_max, param_min]
            set_fut = self._param_set_client.call_async(set_req)
            rclpy.spin_until_future_complete(self, set_fut, timeout_sec=2.0)
            set_result = set_fut.result()
            if set_result is None:
                raise RuntimeError('set_parameters service call failed')
            ok = all(r.successful for r in set_result.results)
            if not ok:
                reasons = '; '.join(r.reason for r in set_result.results if not r.successful)
                raise RuntimeError(f'set_parameters rejected: {reasons}')

            self.get_logger().info(
                f'[GRID_NAV] downhill speed applied: '
                f'max_vel {old_max} -> {downhill_max}, '
                f'min_vel {old_min} -> {downhill_min}')
        except Exception as e:
            self.get_logger().warn(f'[GRID_NAV] downhill speed apply failed: {e}')
            self._saved_max_vel = None
            self._saved_min_vel = None

    def _restore_cruise_speed(self):
        """恢复巡航速度（max_velocity + min_velocity）"""
        if self._saved_max_vel is None and self._saved_min_vel is None:
            return
        try:
            params = []
            if self._saved_max_vel is not None:
                pv = ParameterValue()
                pv.type = 8
                pv.double_array_value = self._saved_max_vel
                param = RclParameter()
                param.name = 'max_velocity'
                param.value = pv
                params.append(param)
            if self._saved_min_vel is not None:
                pv = ParameterValue()
                pv.type = 8
                pv.double_array_value = self._saved_min_vel
                param = RclParameter()
                param.name = 'min_velocity'
                param.value = pv
                params.append(param)
            if params:
                set_req = SetParameters.Request()
                set_req.parameters = params
                self._param_set_client.call_async(set_req)
            self.get_logger().info(
                f'[GRID_NAV] cruise speed restored: '
                f'max={self._saved_max_vel}, min={self._saved_min_vel}')
        except Exception as e:
            self.get_logger().warn(f'[GRID_NAV] cruise speed restore failed: {e}')
        self._saved_max_vel = None
        self._saved_min_vel = None

    # ══════════════════════════════════════════════════════════════
    #  执行入口
    # ══════════════════════════════════════════════════════════════

    def _on_zone2_start(self, msg: UInt8):
        if msg.data == 1:
            self.get_logger().info(
                f'[GRID_NAV] received /zone2_grid/start data=1, '
                f'startup_yaw={"%.1f" % math.degrees(self._startup_yaw) if self._startup_yaw is not None else "NONE"}°')
            self._start_execution(trigger="mission_manager")

    def _on_alliance_msg(self, msg: UInt8):
        """接收 alliance topic（避免与 radio button 回调形成循环）"""
        if msg.data in (1, 2):
            a = Alliance(msg.data)
            if self._alliance != a:
                self._alliance = a
                self._alliance_var.set(msg.data)
                self.get_logger().info(
                    f'[GRID_NAV] alliance synced from topic: {ALLIANCE_LABELS[a]}')
                self._set_status(f'{ALLIANCE_LABELS[a]} — 点击格子选择路线')

    def _on_alliance_change(self):
        if self._executing:
            self.get_logger().warn('[GRID_NAV] cannot change alliance while route is executing')
            self._alliance_var.set(1 if self._alliance == Alliance.BLUE else
                                   2 if self._alliance == Alliance.RED else 0)
            return
        val = self._alliance_var.get()
        if val == 1:
            self._alliance = Alliance.BLUE
        elif val == 2:
            self._alliance = Alliance.RED
        else:
            self._alliance = None
            return
        self._startup_ready = False  # 切换阵营后等待新定位
        self._alliance_pub.publish(UInt8(data=val))
        self.get_logger().info(
            f'[GRID_NAV] alliance selected: {ALLIANCE_LABELS[self._alliance]}')
        self._set_status(f'{ALLIANCE_LABELS[self._alliance]} — 点击格子选择路线')
        self._publish_ready()

    def _on_mission_status(self, msg: String):
        self._mission_status = msg.data
        self._set_status(f'Status: {msg.data}')

    def _compute_ready_code(self) -> int:
        """计算 ready 码：1=就绪，0=未就绪"""
        alliance_ok = self._alliance is not None
        startup_match = (
            self._startup_pose_status in (1, 2) and
            self._alliance is not None and
            self._startup_pose_status == int(self._alliance)
        )
        route_count = len(self._selected)
        route_valid = True
        for r, c in self._selected:
            if compute_grid_center(r, c, self._alliance) is None:
                route_valid = False
                break
        idle_statuses = ('WAIT_MISSION_START', '等待开始', '等待开始', 'MISSION_DONE', '任务完成', '任务失败')
        mission_idle = self._mission_status in idle_statuses

        ready = bool(
            alliance_ok and startup_match and route_count > 0 and
            route_valid and not self._executing and mission_idle and
            not self._shutting_down
        )
        self.get_logger().info(
            f'[GRID_NAV] ready check: alliance={self._alliance} alliance_ok={alliance_ok} '
            f'startup_status={getattr(self, "_startup_pose_status", None)} '
            f'startup_match={startup_match} route_count={route_count} '
            f'route_valid={route_valid} executing={self._executing} '
            f'mission_status={getattr(self, "_mission_status", "?")} '
            f'mission_idle={mission_idle} → ready={ready}')
        return 1 if ready else 0

    def _publish_ready(self):
        code = self._compute_ready_code()
        self._ready_pub.publish(UInt8(data=code))
        reason = 'ready' if code else 'not ready'
        self.get_logger().info(f'[GRID_NAV] publish /zone2_grid/ready={code} ({reason})')

    def _on_startup_status(self, msg: UInt8):
        self._startup_pose_status = msg.data
        prev = self._startup_ready
        if msg.data in (1, 2) and self._alliance is not None and msg.data == int(self._alliance):
            self._startup_ready = True
        else:
            self._startup_ready = False
        if prev != self._startup_ready:
            self.get_logger().info(
                f'[GRID_NAV] startup pose {"ready" if self._startup_ready else "not ready"} '
                f'(status={msg.data}, alliance={self._alliance})')
        self._publish_ready()

    def _start_full_mission(self):
        """GUI '开始完整任务' → 发布 alliance/ready/start"""
        ready_code = self._compute_ready_code()
        self.get_logger().info(
            f'[GRID_NAV] full mission button clicked: '
            f'alliance={self._alliance}, startup_status={getattr(self, "_startup_pose_status", None)}, '
            f'startup_ready={self._startup_ready}, ready_code={ready_code}, '
            f'route_count={len(self._selected)}, executing={self._executing}, '
            f'mission_status={getattr(self, "_mission_status", "?")}')
        if self._executing:
            self.get_logger().warn('[GRID_NAV] cannot start: already executing')
            return
        if self._alliance is None:
            self.get_logger().warn('[GRID_NAV] cannot start: no alliance selected')
            self._set_status('请先选择阵营')
            return
        if not self._startup_ready:
            self.get_logger().warn('[GRID_NAV] cannot start: startup pose not ready')
            self._set_status('启动区定位未就绪，请稍候')
            return
        if not self._selected:
            self.get_logger().warn('[GRID_NAV] cannot start: no zone2 route')
            self._set_status('请先选择二区路线')
            return
        missing = []
        for r, c in self._selected:
            if compute_grid_center(r, c, self._alliance) is None:
                missing.append(grid_label(r, c))
        if missing:
            self.get_logger().warn(f'[GRID_NAV] cannot start: unconfigured={",".join(missing)}')
            self._set_status(f'未配置: {",".join(missing)}')
            return
        if ready_code != 1:
            self.get_logger().warn(f'[GRID_NAV] cannot start: ready_code={ready_code}')
            self._set_status(f'条件未满足 (code={ready_code})')
            return
        self.get_logger().info(
            f'[GRID_NAV] full mission starting: '
            f'alliance={ALLIANCE_LABELS[self._alliance]}, '
            f'route={",".join(grid_label(r,c) for r,c in self._selected)}')
        self._alliance_pub.publish(UInt8(data=int(self._alliance)))
        self.get_logger().info('[GRID_NAV] publish /competition/alliance')
        self._publish_ready()
        self._mission_start_pub.publish(UInt8(data=1))
        self.get_logger().info('[GRID_NAV] publish /mission/start=1')
        self._set_status('完整任务已启动 — 等待一区执行')

    def _on_startup_yaw(self, msg: Float64):
        self._startup_yaw = msg.data
        self.get_logger().info(
            f'[GRID_NAV] startup_yaw received: {msg.data:.3f} rad / '
            f'{math.degrees(msg.data):.1f} deg')

    def _start_execution(self, trigger: str = "manual"):
        if self._executing:
            self.get_logger().warn('[GRID_NAV] already executing, ignoring start')
            if trigger == "mission_manager":
                self._publish_zone2_done(4)
            return
        if not self._selected:
            self.get_logger().error('[GRID_NAV] start requested but no zone2 cells selected')
            self._publish_zone2_done(3)
            return
        pose = self._get_current_pose()
        if pose is None:
            self._set_status("错误：无法获取当前位姿")
            self._publish_zone2_done(2)
            return
        self._current_start = pose
        self._zone1_end_yaw = pose[2]
        # 手动模式：使用当前 map→base_link yaw 作为 startup_yaw
        if trigger == 'manual':
            self._manual_startup_yaw = pose[2]
            self.get_logger().info(
                f'[GRID_NAV] manual Zone2 test: captured startup_yaw='
                f'{pose[2]:.3f} rad / {math.degrees(pose[2]):.1f} deg from map→base_link')
        else:
            startup_str = f'{math.degrees(self._startup_yaw):.1f}°' if self._startup_yaw is not None else 'NONE'
            self.get_logger().info(
                f'[GRID_NAV] zone1_end_yaw captured: {pose[2]:.3f} rad / '
                f'{math.degrees(pose[2]):.1f} deg, '
                f'startup_yaw={startup_str}')
            if self._startup_yaw is None:
                self.get_logger().error(
                    '[GRID_NAV] cannot start: startup_yaw not received from mission_manager')
                self._publish_zone2_done(2)
                self._executing = False
                return
        if self._alliance is None:
            self.get_logger().error(
                '[GRID_NAV] cannot start: alliance not selected')
            self._publish_zone2_done(2)
            self._executing = False
            return
        # 解析并检查所有 cell 坐标
        self._grid_targets = []
        missing = []
        for r, c in self._selected:
            pt = compute_grid_center(r, c, self._alliance)
            if pt is None:
                missing.append(grid_label(r, c))
            self._grid_targets.append(pt if pt is not None else (0.0, 0.0))
        if missing:
            self.get_logger().error(
                f'[GRID_NAV] cannot start: unconfigured cells for '
                f'{ALLIANCE_LABELS.get(self._alliance, "?")}: {",".join(missing)}')
            self._publish_zone2_done(2)
            self._executing = False
            return
        self._seg_index = 0
        self._executing = True
        self._stop_requested = False
        self._exec_phase = ExecPhase.IDLE
        self._grid_mode_active = False
        self._f_stair_active = False
        self._btn_full_mission.config(state=tk.DISABLED)
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL)
        self._btn_clear.config(state=tk.DISABLED)
        self._btn_reverse.config(state=tk.DISABLED)
        self._set_status(f"开始执行 — {len(self._grid_targets)} 段 ({trigger})")
        self.get_logger().info(
            f'[GRID_NAV] start execution trigger={trigger}, '
            f'cells={len(self._grid_targets)}, '
            f'grid mode will activate after reaching R0C1')
        self._begin_segment_rotate()

    # ══════════════════════════════════════════════════════════
    #  段入口 (统一调度)
    # ══════════════════════════════════════════════════════════

    def _begin_segment_rotate(self):
        r, c = self._selected[self._seg_index]
        tx, ty = self._grid_targets[self._seg_index]

        # 层高检查：下台阶限速
        self._adjust_segment_speed(r, c)

        # F 点和 C1 列：统一使用 startup_yaw，全向平移（不旋转）
        if self._is_f_cell(r, c) or self._is_c1_cell(r, c):
            label = grid_label(r, c)
            # 手动模式用捕获 yaw，完整任务用 mission_manager yaw
            exec_yaw = self._manual_startup_yaw if hasattr(self, '_manual_startup_yaw') and self._manual_startup_yaw is not None else self._startup_yaw
            goal_yaw = exec_yaw if exec_yaw is not None else 0.0
            tag = 'F' if self._is_f_cell(r, c) else 'C1'
            yaw_src = f'source=startup_yaw={math.degrees(goal_yaw):.1f}°, no offset'
            self._segment_yaw = goal_yaw
            # 如果是 stair action point，预设 delay
            if self._is_stair_action_point(r, c):
                self._f_delay = self._get_action_delay(r, c)
            self.get_logger().info(
                f'[GRID_NAV] {tag} direct MOVE to {label}, '
                f'goal yaw={math.degrees(goal_yaw):.1f}° ({yaw_src}), no rotation')
            self._exec_phase = ExecPhase.SEGMENT_MOVE
            self._send_nav_goal(tx, ty, goal_yaw)
            return

        # 到达 R0C1 之前：直接 NavigateToPose，边走边转
        if not self._grid_mode_active:
            self._exec_direct_nav()
            return

        # === grid mode: R/OUT 点 SEGMENT_ROTATE → SEGMENT_MOVE ===
        sx, sy, syaw = self._current_start
        seg_yaw = compute_segment_yaw(sx, sy, tx, ty)
        self._segment_yaw = seg_yaw
        yaw_err = yaw_diff(syaw, seg_yaw)
        self.get_logger().info(
            f'[GRID_NAV] grid mode seg{self._seg_index} ROTATE: '
            f'({sx:.2f},{sy:.2f})→({tx:.2f},{ty:.2f}), '
            f'yaw={math.degrees(seg_yaw):.0f}° err={math.degrees(yaw_err):.1f}°')
        if yaw_err < YAW_SKIP_THRESHOLD:
            self.get_logger().info(f'[GRID_NAV] seg{self._seg_index} yaw skip')
            self._begin_segment_move()
            return
        self._exec_phase = ExecPhase.SEGMENT_ROTATE
        self._send_nav_goal(sx, sy, seg_yaw)

    def _exec_direct_nav(self):
        """pre-grid 模式：一次 NavigateToPose 直接走到目标，边走边转"""
        r, c = self._selected[self._seg_index]
        label = grid_label(r, c)
        pose = self._get_current_pose()
        if pose is None:
            self.get_logger().error('pre-grid direct nav: 无法获取当前位姿')
            self._finish_execution(stopped=False)
            return
        sx, sy, syaw = pose
        tx, ty = self._grid_targets[self._seg_index]
        seg_yaw = compute_segment_yaw(sx, sy, tx, ty)
        self._segment_yaw = seg_yaw
        self._current_start = (sx, sy, syaw)
        self.get_logger().info(
            f'[GRID_NAV] pre-grid direct MOVE to {label}: '
            f'current=({sx:.2f},{sy:.2f}), target=({tx:.2f},{ty:.2f}), '
            f'yaw={math.degrees(seg_yaw):.0f}°; Nav2 may move and rotate together')
        self._exec_phase = ExecPhase.SEGMENT_MOVE
        self._send_nav_goal(tx, ty, seg_yaw)

    def _exec_f_direct(self):
        """F 点：一次直接导航 + 到达后 stair action + delay"""
        r, c = self._selected[self._seg_index]
        label = grid_label(r, c)
        pose = self._get_current_pose()
        if pose is None:
            self.get_logger().error(f'F点 {label}: 无法获取当前位姿')
            self._finish_execution(stopped=False)
            return
        sx, sy, syaw = pose
        tx, ty = self._grid_targets[self._seg_index]
        seg_yaw = compute_segment_yaw(sx, sy, tx, ty)
        self._f_target_x = tx
        self._f_target_y = ty
        self._f_delay = self._get_action_delay(r, c)
        self._segment_yaw = seg_yaw
        self._current_start = (sx, sy, syaw)
        self.get_logger().info(
            f'[GRID_NAV] pre-grid direct MOVE to {label}: '
            f'current=({sx:.2f},{sy:.2f}), target=({tx:.2f},{ty:.2f}), '
            f'yaw={math.degrees(seg_yaw):.0f}°; Nav2 may move and rotate together')
        self._exec_phase = ExecPhase.SEGMENT_MOVE
        self._send_nav_goal(tx, ty, seg_yaw)

    def _begin_segment_move(self):
        tx, ty = self._grid_targets[self._seg_index]
        seg_yaw = self._segment_yaw
        self.get_logger().info(
            f'[GRID_NAV] grid mode seg{self._seg_index} MOVE: '
            f'({tx:.2f},{ty:.2f}), yaw={math.degrees(seg_yaw):.0f}°')
        self._exec_phase = ExecPhase.SEGMENT_MOVE
        self._send_nav_goal(tx, ty, seg_yaw)

    # ══════════════════════════════════════════════════════════
    #  段完成处理
    # ══════════════════════════════════════════════════════════

    def _on_segment_move_done(self):
        self._advance_to_next_segment()

    def _on_delay_done(self):
        self.get_logger().info(
            f'[GRID_NAV] F_ACTION_DELAY done (delay={self._action_delay:.1f}s), '
            f'advancing from seg{self._seg_index}')
        self._f_stair_active = False
        self._advance_to_next_segment()

    def _advance_to_next_segment(self):
        self.get_logger().info(
            f'[GRID_NAV] advancing: seg{self._seg_index} done, '
            f'total={len(self._grid_targets)}')
        # 检查刚完成的 segment 是否 R0C1
        r, c = self._selected[self._seg_index]
        if self._is_r0c1(r, c):
            self._grid_mode_active = True
            self.get_logger().info('[GRID_NAV] R0C1 reached, entering grid mode')
            if not self._zone2_accel_applied:
                self._apply_zone2_accel_limit()

        self._seg_index += 1
        if self._seg_index >= len(self._grid_targets):
            self._finish_execution(stopped=False)
        else:
            tx, ty = self._grid_targets[self._seg_index - 1]
            self._current_start = (tx, ty, self._segment_yaw)
            self._exec_phase = ExecPhase.IDLE
            self._begin_segment_rotate()

    # ══════════════════════════════════════════════════════════
    #  exec tick
    # ══════════════════════════════════════════════════════════

    def _exec_tick(self):
        if not self._executing:
            self._root.after(100, self._exec_tick)
            return
        if self._stop_requested:
            self._cancel_current_goal()
            self._finish_execution(stopped=True)
            self._root.after(100, self._exec_tick)
            return
        if self._exec_phase == ExecPhase.IDLE:
            self._root.after(100, self._exec_tick)
            return
        if self._exec_phase in (ExecPhase.GRAB_DELAY, ExecPhase.F_ACTION_DELAY):
            if time.time() - self._grab_delay_start >= self._action_delay:
                self._on_delay_done()
            self._root.after(100, self._exec_tick)
            return
        if not self._goal_result_ready:
            self._root.after(100, self._exec_tick)
            return
        if not self._goal_succeeded:
            self.get_logger().error(f'seg{self._seg_index} 导航失败')
            self._finish_execution(stopped=False)
            self._root.after(100, self._exec_tick)
            return

        ph = self._exec_phase
        self._goal_result_ready = False
        self._goal_handle = None

        if ph == ExecPhase.SEGMENT_ROTATE:
            self._begin_segment_move()
        elif ph == ExecPhase.SEGMENT_MOVE:
            r, c = self._selected[self._seg_index]
            if self._is_stair_action_point(r, c):
                # F 点到达 → stair action + delay
                cmd_val = self._get_stair_action_cmd(r, c) or 1
                label = grid_label(r, c)
                self._f_stair_active = True
                self.get_logger().info(
                    f'[GRID_NAV] reached {label}, '
                    f'publish {ZONE2_STAIR_CMD_TOPIC} data={cmd_val}, '
                    f'delay={self._f_delay:.1f}s')
                self._publish_zero_velocity()
                self._publish_stair_action(cmd_val)
                self._grab_delay_start = time.time()
                self._action_delay = self._f_delay
                self._exec_phase = ExecPhase.F_ACTION_DELAY
                self._root.after(100, self._exec_tick)
                return
            self.get_logger().info(f'[GRID_NAV] seg{self._seg_index} done')
            self._on_segment_move_done()
        self._root.after(100, self._exec_tick)

    # ══════════════════════════════════════════════════════════════
    #  层高速度调整
    # ══════════════════════════════════════════════════════════════

    def _adjust_segment_speed(self, r: int, c: int):
        """根据当前段和下一段层高，决定是否应用下台阶限速"""
        idx = self._seg_index
        next_label = grid_label(r, c)

        if idx == 0:
            # 第一段：巡航速度
            self._restore_cruise_speed()
            self.get_logger().info(
                f'[GRID_NAV] segment {idx} ({next_label}): first segment, cruise speed')
            return

        prev_r, prev_c = self._selected[idx - 1]
        prev_label = grid_label(prev_r, prev_c)

        descending = self._is_descending_segment(prev_label, next_label)

        h_tbl = self._get_height_table()
        h_prev = h_tbl.get(prev_label, 'N/A') if h_tbl else 'N/A'
        h_next = h_tbl.get(next_label, 'N/A') if h_tbl else 'N/A'

        self.get_logger().info(
            f'[GRID_NAV] segment height check: '
            f'alliance={ALLIANCE_LABELS.get(self._alliance, "?")}, '
            f'from={prev_label} h={h_prev}, '
            f'to={next_label} h={h_next}, '
            f'descending={descending}')

        if descending:
            self._apply_downhill_speed()
            self.get_logger().info(
                f'[GRID_NAV] applying downhill speed limit: '
                f'linear_max={DOWNHILL_SPEED:.3f} m/s')
        else:
            self._restore_cruise_speed()
            self.get_logger().info('[GRID_NAV] restoring cruise speed')

    # ══════════════════════════════════════════════════════════════
    #  停止 / 结束
    # ══════════════════════════════════════════════════════════════

    def _stop_execution(self):
        if not self._executing:
            return
        self.get_logger().info('用户停止')
        self._stop_requested = True

    def _finish_execution(self, stopped: bool):
        self._cancel_current_goal()
        self._executing = False
        self._exec_phase = ExecPhase.IDLE
        self._grid_mode_active = False
        self._zone1_end_yaw = None
        self._f_stair_active = False
        self._btn_full_mission.config(state=tk.NORMAL)
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED)
        self._btn_clear.config(state=tk.NORMAL)
        self._btn_reverse.config(state=tk.NORMAL)
        self._restore_zone2_accel_limit()
        self._restore_cruise_speed()
        if stopped:
            self._set_status("已手动停止")
            self._publish_zone2_done(2)
        else:
            self._set_status("执行完成")

    def _publish_zone2_done(self, code: int):
        self._zone2_done_pub.publish(UInt8(data=code))
        self.get_logger().info(f'[GRID_NAV] /zone2_grid/done data={code}')

    def _handle_sigterm(self, signum, frame):
        """SIGTERM：在主线程安全退出"""
        self.get_logger().info('[GRID_NAV] SIGTERM received, shutting down')
        self._root.after(0, self._shutdown)

    def _shutdown(self):
        self._shutting_down = True
        self._cancel_current_goal()
        self._restore_zone2_accel_limit()
        if self._executing:
            self._cancel_current_goal()
        self._root.quit()

    def _on_close(self):
        self.get_logger().info('GUI 关闭')
        if self._shutting_down:
            return
        self._shutdown()
        self.destroy_node()


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = Zone2GridNav()
    try:
        node._root.mainloop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
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
