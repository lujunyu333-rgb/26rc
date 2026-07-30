#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 区域检测节点 - 订阅 RealSense 彩色图像话题，
根据 ROS2 参数 target_color 决定检测颜色。

用法：
  # 终端直接运行
  ros2 run camera res_kfs --ros-args -p target_color:=blue
  ros2 run camera res_kfs --ros-args -p target_color:=red
  ros2 run camera res_kfs --ros-args -p roi_ratio:=0.5
  # 通过 launch 文件
  ros2 launch camera res_kfs_launch.py target_color:=red

target_color 可选值:
  both   - 检测蓝色+红色（默认）
  blue   - 仅检测蓝色
  red    - 仅检测红色

roi_ratio 参数:
  检测区域为图像中心的正方形，roi_ratio 控制正方形边长占短边的比例
  默认 1.0（内接正方形），范围 0.1~1.0

检测逻辑:
  1. z > 0.09m → 进入持续检测模式，z ≤ 0.09m → 退出并重置
  2. 进入检测模式时自动开启首次移动检测窗口 (10s)
  3. x/y 方向每累计移动 0.7m → 开启 10s 移动窗口
     窗口内识别到蓝/红 → 发送 flag=5，提前关闭窗口
     窗口超时 → 自动关闭，重新累计距离
     窗口期间暂停距离累计
  4. 检测模式下 z 上升 ≥ 0.10m → 启动 10s 上坡窗口
     窗口内识别到目标 → 发送 flag=4（该窗口仅一次）
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
import time
from collections import deque


# ── HSV 阈值 (蓝色) ──────────────────────────────────────
B_HUE_MIN   = 100
B_HUE_MAX   = 130
B_SATU_MIN  = 80
B_SATU_MAX  = 255
B_VAL_MIN   = 50
B_VAL_MAX   = 255
B_TRIGGER   = 0.2

# ── HSV 阈值 (红色 - 红色在 HSV 中跨越 0° 边界，需要两个区间) ─
R_HUE_MIN1  = 0
R_HUE_MAX1  = 10
R_HUE_MIN2  = 160
R_HUE_MAX2  = 180
R_SATU_MIN  = 80
R_SATU_MAX  = 255
R_VAL_MIN   = 50
R_VAL_MAX   = 255
R_TRIGGER   = 0.2

# ── 发布延迟 ───────────────────────────────────────────
PUBLISH_RISE_DELAY_S        = 0.43    # 检测到颜色后延迟发送 (s)
PUBLISH_DOWN_DELAY_S        = 0.31    # 检测到移动后延迟发送 (s)

# ── 上坡触发参数 ───────────────────────────────────────

MORPH_KERNEL_OPEN  = (5, 5)
MORPH_KERNEL_CLOSE = (7, 7)
MIN_AREA_RATIO     = 0.05
ROI_RATIO           = 0.3   # 检测区域为图像中心正方形，边长占短边的比例 (0.1~1.0)

# ── Odometry 触发参数 ───────────────────────────────────
Z_DETECT_THRESHOLD     = 0.09   # z > 此值进入持续检测模式 (m)
UPHILL_DELTA           = 0.19   # 检测模式下 z 上升量触发上坡 (m)
UPHILL_Z_FRAMES        = 10     # 上坡检测滑动窗口帧数
UPHILL_WINDOW_S        = 2.5   # 上坡触发后检测窗口时长 (s)
UPHILL_COOLDOWN_S      = 0    # 检测到目标后上坡触发冷却 (s)

# ── 移动窗口参数 ───────────────────────────────────────
MOVE_DISTANCE_THRESHOLD = 0.568    # 移动距离阈值 (m)
MOVE_WINDOW_S           = 2.0   # 移动检测窗口时长 (s)


class RegKfsNode(Node):
    """HSV 颜色检测节点，通过 target_color 参数切换检测目标"""

    def __init__(self):
        super().__init__('reg_kfs')

        # ── ROS2 参数：target_color ──────────────────────
        self.declare_parameter('target_color', 'both')
        self._target_color = self.get_parameter('target_color').value
        valid = ('both', 'blue', 'red')
        if self._target_color not in valid:
            self.get_logger().warn(
                f"无效的 target_color='{self._target_color}'，"
                f"回退为 'both' (有效值: {valid})")
            self._target_color = 'both'


        self.bridge = CvBridge()
        self.frame = None

        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_OPEN)
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_CLOSE)
        self.min_area_px = 0  # 延迟初始化
        self.total_pixels = 0

        self.sub_rs = self.create_subscription(
            Image, 'camera/color/image_raw', self.rs_callback, 10)
        self.pub_flag = self.create_publisher(UInt8, 'camera/view_cmd', 10)

        # ── Odometry 订阅 ─────────────────────────────
        self.sub_odom = self.create_subscription(
            Odometry, '/Odometry', self.odom_callback, 10)
        self._z_current = 0.0               # 最新 z 值
        self._odom_x = 0.0                  # 最新 x 值
        self._odom_y = 0.0                  # 最新 y 值
        self._detect_active = False         # 是否处于检测模式
        self._z_history = deque(maxlen=UPHILL_Z_FRAMES)  # z 值滑动窗口 (10帧)
        self._uphill_deadline = 0.0         # 上坡窗口截止时间戳 (0=未激活)
        self._flag4_sent = False            # 本次上坡窗口是否已发送 flag=4
        self._uphill_cooldown_deadline = 0.0  # 上坡触发冷却截止时间戳
        # ── 移动窗口状态 ─────────────────────────────
        self._x_ref = 0.0                   # 距离累计参考点 x
        self._y_ref = 0.0                   # 距离累计参考点 y
        self._move_window_deadline = 0.0    # 移动窗口截止时间戳 (0=未激活)
        self._move_window_flag_sent = False # 当前移动窗口是否已发送 flag=5

        # 延迟发送队列：(deadline, flag)，非阻塞延迟发送
        self._pending_results = deque()

        # Timer 驱动处理（30Hz）
        self._process_timer = self.create_timer(1.0 / 30.0, self._process_callback)

        # Timer 驱动延迟发送（20Hz 轮询队列）
        self._publish_timer = self.create_timer(0.05, self._publish_callback)

        color_map = {'both': '蓝色+红色', 'blue': '蓝色', 'red': '红色'}
        self.get_logger().info(
            f"reg_kfs 颜色检测节点已启动 "
            f"(target_color={self._target_color} → {color_map[self._target_color]})"
            f" | z>{Z_DETECT_THRESHOLD}m→检测, 默认→flag=5, "
            f"上坡≥{UPHILL_DELTA}m→flag=4")

    def rs_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.min_area_px == 0:
                h, w = self.frame.shape[:2]
                self.min_area_px = int(h * w * MIN_AREA_RATIO)
                self.total_pixels = h * w
                self.get_logger().info(
                    f"图像尺寸={w}x{h}, 最小面积={self.min_area_px}px")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 错误: {e}")

    def odom_callback(self, msg):
        """接收 /Odometry，记录最新 z、x、y 值"""
        self._z_current = msg.pose.pose.position.z
        self._odom_x = msg.pose.pose.position.x
        self._odom_y = msg.pose.pose.position.y

    def _detect_color(self, hsv_img, hue_min, hue_max, sat_min, sat_max,
                      val_min, val_max, trigger_ratio) -> bool:
        """检测一种颜色区间，返回是否超过触发比例（基于传入图像尺寸计算比例）"""
        th = cv2.inRange(hsv_img,
                         (hue_min, sat_min, val_min),
                         (hue_max, sat_max, val_max))

        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.kernel_open)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.kernel_close)

        img_h, img_w = th.shape[:2]
        total_px = img_h * img_w
        min_area_px = int(total_px * MIN_AREA_RATIO)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            th, connectivity=8)
        mask = np.zeros_like(th)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                mask[labels == i] = 255

        area = cv2.countNonZero(mask)
        proportion = area / total_px
        return proportion > trigger_ratio

    def _enqueue_result(self, flag, delay):
        """将检测结果加入延迟队列，delay 为发送 flag 前的等待秒数，flag 发送后紧跟 0 复位"""
        now = time.time()
        self._pending_results.append((now + delay, flag))
        self._pending_results.append((now + delay + 0.3, 0))

    def _publish_callback(self):
        """非阻塞延迟发送：轮询队列，发送到期结果"""
        now = time.time()
        while self._pending_results and self._pending_results[0][0] <= now:
            _, flag = self._pending_results.popleft()
            self.pub_flag.publish(UInt8(data=flag))

    def _process_callback(self):
        """Timer 回调：z 阈值驱动持续检测 + 上坡触发 flag4"""
        now = time.time()

        # ── 检测模式切换 ──────────────────────────────
        if self._z_current > Z_DETECT_THRESHOLD:
            if not self._detect_active:
                # 进入检测模式
                self._detect_active = True
                self._z_history.clear()
                self._uphill_deadline = now + UPHILL_WINDOW_S
                self._flag4_sent = False
                # 移动窗口：重置参考点，自动开启首次检测窗口
                self._x_ref = self._odom_x
                self._y_ref = self._odom_y
                self._move_window_deadline = now + MOVE_WINDOW_S
                self._move_window_flag_sent = False
                self.get_logger().info(
                    f"[odom] z={self._z_current:.3f}m > {Z_DETECT_THRESHOLD}m, "
                    f"进入检测模式, 自动开启上坡窗口 {UPHILL_WINDOW_S}s + "
                    f"移动检测窗口 {MOVE_WINDOW_S}s")
        else:
            if self._detect_active:
                # 退出检测模式
                self._detect_active = False
                self.get_logger().info(
                    f"[odom] z={self._z_current:.3f}m ≤ {Z_DETECT_THRESHOLD}m, "
                    f"退出检测模式")
            return

        # ── 检测模式下：累计 z 值到滑动窗口 ────────────
        self._z_history.append(self._z_current)

        # ── 检测模式下：上坡判断（冷却期内不触发）────────
        if (self._uphill_deadline == 0.0
                and now >= self._uphill_cooldown_deadline
                and len(self._z_history) == UPHILL_Z_FRAMES):
            z_range = max(self._z_history) - min(self._z_history)
            if z_range >= UPHILL_DELTA:
                self._uphill_deadline = now + UPHILL_WINDOW_S
                self._flag4_sent = False
                self.get_logger().info(
                    f"[odom] {UPHILL_Z_FRAMES}帧内 z 变化 {z_range:.3f}m ≥ "
                    f"{UPHILL_DELTA}m, 启动 {UPHILL_WINDOW_S}s 上坡检测窗口")

        # 上坡窗口过期
        if self._uphill_deadline > 0 and now > self._uphill_deadline:
            self._uphill_deadline = 0.0
            self.get_logger().info("[odom] 上坡检测窗口结束")

        # ── 移动窗口管理 ─────────────────────────────
        if self._move_window_deadline > 0 and now > self._move_window_deadline:
            # 窗口超时，重置参考点重新累计距离
            self._move_window_deadline = 0.0
            self._x_ref = self._odom_x
            self._y_ref = self._odom_y
            self.get_logger().info("[odom] 移动检测窗口超时结束, 重新累计距离")

        if self._move_window_deadline == 0.0:
            # 窗口未激活，累计 x、y 位移距离
            dx = self._odom_x - self._x_ref
            dy = self._odom_y - self._y_ref
            dist = math.sqrt(dx * dx + dy * dy)
            if dist >= MOVE_DISTANCE_THRESHOLD:
                self._move_window_deadline = now + MOVE_WINDOW_S
                self._move_window_flag_sent = False
                # 以当前位置为新参考点，窗口期间暂停累计
                self._x_ref = self._odom_x
                self._y_ref = self._odom_y
                self.get_logger().info(
                    f"[odom] 累计移动 {dist:.3f}m ≥ "
                    f"{MOVE_DISTANCE_THRESHOLD}m, "
                    f"开启移动检测窗口 {MOVE_WINDOW_S}s")

        if self.frame is None:
            return

        # 重新读取参数，允许运行时动态切换
        self._target_color = self.get_parameter('target_color').value

        # BGR → HSV + V 均衡化
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])

        # 裁剪图像中心正方形作为检测 ROI
        roi_ratio = max(0.1, min(1.0, ROI_RATIO))
        hsv_h, hsv_w = hsv.shape[:2]
        roi_size = int(min(hsv_h, hsv_w) * roi_ratio)
        y_start = (hsv_h - roi_size) // 2
        x_start = (hsv_w - roi_size) // 2
        hsv = hsv[y_start:y_start + roi_size, x_start:x_start + roi_size]

        # ── 颜色检测 ──────────────────────────────────
        detected = False

        if self._target_color in ('both', 'blue'):
            if self._detect_color(hsv,
                                  B_HUE_MIN, B_HUE_MAX,
                                  B_SATU_MIN, B_SATU_MAX,
                                  B_VAL_MIN, B_VAL_MAX, B_TRIGGER):
                detected = True

        if self._target_color in ('both', 'red'):
            red1 = self._detect_color(hsv,
                                      R_HUE_MIN1, R_HUE_MAX1,
                                      R_SATU_MIN, R_SATU_MAX,
                                      R_VAL_MIN, R_VAL_MAX, R_TRIGGER)
            red2 = self._detect_color(hsv,
                                      R_HUE_MIN2, R_HUE_MAX2,
                                      R_SATU_MIN, R_SATU_MAX,
                                      R_VAL_MIN, R_VAL_MAX, R_TRIGGER)
            if red1 or red2:
                detected = True

        if not detected:
            return

        # ── 上坡窗口内 → flag=4 ─────────────────────────
        if self._uphill_deadline > 0 and not self._flag4_sent:
            self._flag4_sent = True
            self._enqueue_result(4, PUBLISH_RISE_DELAY_S)
            self._uphill_cooldown_deadline = now + UPHILL_COOLDOWN_S
            self.get_logger().info(
                f"[odom] 上坡窗口检测到目标 → flag=4, "
                f"上坡冷却 {UPHILL_COOLDOWN_S}s")
            return

        # ── 移动窗口内 → flag=5 ───────────────────────
        if self._move_window_deadline > 0 and not self._move_window_flag_sent:
            self._move_window_flag_sent = True
            self._enqueue_result(5, PUBLISH_DOWN_DELAY_S)
            self._uphill_cooldown_deadline = now + UPHILL_COOLDOWN_S
            # 提前关闭窗口，以当前位置为参考点重新累计
            self._move_window_deadline = 0.0
            self._x_ref = self._odom_x
            self._y_ref = self._odom_y
            self.get_logger().info(
                f"[odom] 移动窗口检测到目标 → flag=5, "
                f"上坡冷却 {UPHILL_COOLDOWN_S}s")


def main(args=None):
    rclpy.init(args=args)
    node = RegKfsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
