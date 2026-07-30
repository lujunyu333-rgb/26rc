#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白底黑字检测节点 - 订阅 RealSense 彩色图像话题，
通过灰度直方图双峰特征判断图像中是否存在白底黑字。

检测原理：
  白底黑字图像的灰度直方图呈现明显的双峰分布——
  大量暗像素（文字）集中在低灰度区，大量亮像素（背景）集中在高灰度区，
  中间灰度像素极少。这与普通场景（渐变、彩色、单一色调）形成鲜明对比。

用法：
  ros2 run camera ws_text --ros-args -p dark_threshold:=80 -p bright_threshold:=160

flag 含义:
  0 = 未检测到白底黑字
  1 = 检测到白底黑字
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


# ── 检测参数（可通过 ROS2 param 覆盖） ──────────────────────
DARK_THRESHOLD    = 80    # 灰度值低于此视为"黑字"（0-255）
BRIGHT_THRESHOLD  = 160   # 灰度值高于此视为"白底"（0-255）
MID_MAX_RATIO     = 0.15  # 中间灰度最大占比，超过说明对比度不足
DARK_MIN_RATIO    = 0.08  # 暗像素（黑字）最小占比
BRIGHT_MIN_RATIO  = 0.25  # 亮像素（白底）最小占比
EDGE_MIN_RATIO    = 0.002 # 边缘像素最小占比（辅助确认有纹理/文字）
ENABLE_EDGE_CHECK = True  # 是否启用边缘密度辅助判断


class WhiteBgBlackTextNode(Node):
    """白底黑字检测节点 — 通过灰度直方图双峰特征识别"""

    def __init__(self):
        super().__init__('ws_text')

        # ── 声明 ROS2 参数 ──────────────────────────────
        self.declare_parameter('dark_threshold', DARK_THRESHOLD)
        self.declare_parameter('bright_threshold', BRIGHT_THRESHOLD)
        self.declare_parameter('mid_max_ratio', MID_MAX_RATIO)
        self.declare_parameter('dark_min_ratio', DARK_MIN_RATIO)
        self.declare_parameter('bright_min_ratio', BRIGHT_MIN_RATIO)
        self.declare_parameter('edge_min_ratio', EDGE_MIN_RATIO)
        self.declare_parameter('enable_edge_check', ENABLE_EDGE_CHECK)

        self.bridge = CvBridge()
        self.frame = None

        # 订阅 RealSense 彩色图像
        self.sub_rs = self.create_subscription(
            Image, 'camera/color/image_raw', self.rs_callback, 10)

        # 发布检测结果
        self.pub_flag = self.create_publisher(UInt8, '/flag', 10)

        # Timer 驱动处理（30Hz，与 res_kfs 一致）
        self._process_timer = self.create_timer(1.0 / 30.0, self._process_callback)

        self.get_logger().info(
            "ws_text 白底黑字检测节点已启动 "
            f"(暗阈值<{self.get_parameter('dark_threshold').value}, "
            f"亮阈值>{self.get_parameter('bright_threshold').value})")

    def rs_callback(self, msg):
        """接收 RealSense 图像并转为 OpenCV BGR 格式"""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge 错误: {e}")

    def _detect_white_bg_black_text(self, gray_img: np.ndarray) -> tuple:
        """
        检测白底黑字，返回 (detected: bool, dark_ratio: float,
        bright_ratio: float, mid_ratio: float, edge_ratio: float)。

        检测逻辑:
          1. 统计暗像素占比（黑字）
          2. 统计亮像素占比（白底）
          3. 统计中间灰度占比（应为低值）
          4. 可选：Canny 边缘密度辅助判断（文字区域有纹理）
          5. 三个条件同时满足 → 判定为白底黑字
        """
        # 读取运行时参数（允许动态调参）
        dark_th = self.get_parameter('dark_threshold').value
        bright_th = self.get_parameter('bright_threshold').value
        mid_max = self.get_parameter('mid_max_ratio').value
        dark_min = self.get_parameter('dark_min_ratio').value
        bright_min = self.get_parameter('bright_min_ratio').value
        edge_min = self.get_parameter('edge_min_ratio').value
        enable_edge = self.get_parameter('enable_edge_check').value

        total = gray_img.size

        # 三区域统计
        dark_mask = gray_img < dark_th
        bright_mask = gray_img > bright_th
        mid_mask = ~(dark_mask | bright_mask)

        dark_ratio = np.count_nonzero(dark_mask) / total
        bright_ratio = np.count_nonzero(bright_mask) / total
        mid_ratio = np.count_nonzero(mid_mask) / total

        # 条件判断
        cond_dark = dark_ratio > dark_min
        cond_bright = bright_ratio > bright_min
        cond_mid = mid_ratio < mid_max

        # 边缘密度（可选）
        edge_ratio = 0.0
        cond_edge = True
        if enable_edge:
            edges = cv2.Canny(gray_img, 50, 150)
            edge_ratio = np.count_nonzero(edges) / total
            cond_edge = edge_ratio > edge_min

        detected = cond_dark and cond_bright and cond_mid and cond_edge

        return detected, dark_ratio, bright_ratio, mid_ratio, edge_ratio

    def _process_callback(self):
        """Timer 回调：对最新帧做白底黑字检测"""
        if self.frame is None:
            self.pub_flag.publish(UInt8(data=0))
            return

        # BGR → 灰度
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        detected, dark_r, bright_r, mid_r, edge_r = \
            self._detect_white_bg_black_text(gray)

        flag = 1 if detected else 0
        self.pub_flag.publish(UInt8(data=flag))

        self.get_logger().info(
            f"dark={dark_r:.3f} bright={bright_r:.3f} mid={mid_r:.3f} "
            f"edge={edge_r:.4f} → flag={flag}",
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = WhiteBgBlackTextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
