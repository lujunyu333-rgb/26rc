#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
move_base_of_yolo.py — 基于 D435i + YOLO 的视觉伺服节点

功能：
  1. 订阅 RealSense D435i 的彩色图、深度图、深度因子、相机内参
  2. 后台线程运行 YOLO 推理，检测画面中的目标物体
  3. 优先级：class_id == 3 优先；若无 id=3 则回退到其他任意检测目标
  4. 多目标时选择检测框中心最靠近图像中心的那个
  5. 计算目标中心与图像中心的 Y 方向偏移量（左为正），单位米
  6. 下位机发送 /camera/yolo/request=1 → 校准一次（px→m），分支：
     a) 偏移量 > 死区(m) → 仅发布 y_offset (Float32, 米)
     b) 偏移量 ≤ 死区(m) → 发布 view_cmd=1 (Int8)，抓取 +1，
        启动画面亮度监测，实际检测到变黑后延时 0.5s 发 view_sig=1
  7. 抓取计数 > max_grasp_count 时，收到下一个 request 后自动退出

发布话题：
  /camera/yolo/y_offset    (std_msgs/Float32) — Y 方向偏移量（米，左为正）
  /camera/yolo/view_cmd    (std_msgs/Int8)    — 抓取信号（1=抓取）
  /camera/yolo/view_sig    (std_msgs/Int8)    — 遮挡信号（1=遮挡）

订阅话题：
  /camera/color/image_raw   (sensor_msgs/Image)
  /camera/depth/image_raw   (sensor_msgs/Image)
  /camera/depth/scale       (std_msgs/Float32)
  /camera/color/camera_info (sensor_msgs/CameraInfo)
  /camera/yolo/request      (std_msgs/Int8)    — 下位机触发校准
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import UInt8, Float32
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging
import threading
import time
from collections import deque


# 压制 ultralytics 的 INFO/DEBUG 日志
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class GraspLimitReached(Exception):
    """抓取次数达到上限时抛出的异常，用于安全退出 spin 循环"""


# ══════════════════════════════════════════════════════════════
#  默认参数
# ══════════════════════════════════════════════════════════════
DEFAULT_MODEL_PATH = "/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt"
DEFAULT_CONFIDENCE  = 0.50
DEFAULT_IOU         = 0.45
DEFAULT_MIN_BOX_AREA  = 500
DEFAULT_PRIORITY_CLASS = 3        # 优先跟踪的类别 ID
DEFAULT_DEAD_ZONE_M    = 0.04     # 死区（米），偏移量在此范围内视为已居中
DEFAULT_DARK_THRESHOLD = 60       # 遮挡判定：画面平均亮度低于此值视为变黑
DEFAULT_OBSTRUCTION_DELAY = 0.6   # 遮挡后等待秒数再发完成信号
DEFAULT_OBSTRUCTION_TIMEOUT = 20.0  # 遮挡检测超时 (秒)，超时后自动解锁

# ── 遮挡检测 ROI（物体被抓取后出现的区域）──────────── 
DEFAULT_DARK_ROI_X1 = 0.650      # ROI 左边界（距图像左侧的比例，1-1/3≈0.667）
DEFAULT_DARK_ROI_X2 = 0.930      # ROI 右边界（距图像左侧的比例，1-1/5=0.800）
DEFAULT_DARK_ROI_Y1 = 0.500      # ROI 上边界（距图像顶部的比例，1/2=0.500）
DEFAULT_DARK_ROI_Y2 = 1.000      # ROI 下边界（图像底部）

# ── 相机安装偏置 ────────────────────────────────────
DEFAULT_CAMERA_OFFSET_M = 0.01   # RGB镜头在机械臂中心左侧的偏移量（米，正值=左）
DEFAULT_MAX_GRASP_COUNT  = 6      # 最大抓取次数，超过后退出
DEFAULT_FRAME_ID = "camera_color_optical_frame"
DEFAULT_MIN_DEPTH_M = 0.15             # 深度过滤下限（米），低于此值视为误报
DEFAULT_MAX_DEPTH_M = 0.35             # 深度过滤上限（米），高于此值视为误报
DEFAULT_DEPTH_FILTER_ENABLED = True    # 是否启用深度误报过滤


# ══════════════════════════════════════════════════════════════
#  全局抓取计数器
# ══════════════════════════════════════════════════════════════
_grasp_count = 0
_grasp_count_lock = threading.Lock()


def _increment_grasp_count():
    global _grasp_count
    with _grasp_count_lock:
        _grasp_count += 1
        return _grasp_count


def _get_grasp_count():
    with _grasp_count_lock:
        return _grasp_count


class MoveBaseOfYolo(Node):
    """YOLO 视觉伺服节点：检测目标 → 计算 Y 偏移 → 抓取决策 → 遮挡检测"""

    def __init__(self):
        super().__init__('move_base_of_yolo')

        # ── ROS2 参数 ──────────────────────────────────
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE)
        self.declare_parameter('iou_threshold', DEFAULT_IOU)
        self.declare_parameter('min_box_area', DEFAULT_MIN_BOX_AREA)
        self.declare_parameter('priority_class', DEFAULT_PRIORITY_CLASS)
        self.declare_parameter('dead_zone_m', DEFAULT_DEAD_ZONE_M)
        self.declare_parameter('dark_threshold', DEFAULT_DARK_THRESHOLD)
        self.declare_parameter('obstruction_delay', DEFAULT_OBSTRUCTION_DELAY)
        self.declare_parameter('obstruction_timeout', DEFAULT_OBSTRUCTION_TIMEOUT)
        self.declare_parameter('max_grasp_count', DEFAULT_MAX_GRASP_COUNT)
        self.declare_parameter('frame_id', DEFAULT_FRAME_ID)
        self.declare_parameter('min_depth_m', DEFAULT_MIN_DEPTH_M)
        self.declare_parameter('max_depth_m', DEFAULT_MAX_DEPTH_M)
        self.declare_parameter('depth_filter_enabled', DEFAULT_DEPTH_FILTER_ENABLED)
        self.declare_parameter('dark_roi_x1', DEFAULT_DARK_ROI_X1)
        self.declare_parameter('dark_roi_x2', DEFAULT_DARK_ROI_X2)
        self.declare_parameter('dark_roi_y1', DEFAULT_DARK_ROI_Y1)
        self.declare_parameter('dark_roi_y2', DEFAULT_DARK_ROI_Y2)
        self.declare_parameter('camera_offset_m', DEFAULT_CAMERA_OFFSET_M)

        self.model_path       = self.get_parameter('model_path').value
        self.conf_thres       = self.get_parameter('confidence_threshold').value
        self.iou_thres        = self.get_parameter('iou_threshold').value
        self.min_box_area     = self.get_parameter('min_box_area').value
        self.priority_class   = self.get_parameter('priority_class').value
        self.dead_zone_m      = self.get_parameter('dead_zone_m').value
        self.dark_threshold   = self.get_parameter('dark_threshold').value
        self.obstruction_delay = self.get_parameter('obstruction_delay').value
        self.obstruction_timeout = self.get_parameter('obstruction_timeout').value
        self.max_grasp_count  = self.get_parameter('max_grasp_count').value
        self.frame_id            = self.get_parameter('frame_id').value
        self.min_depth_m        = self.get_parameter('min_depth_m').value
        self.max_depth_m        = self.get_parameter('max_depth_m').value
        self.depth_filter_enabled = self.get_parameter('depth_filter_enabled').value
        self.dark_roi_x1         = self.get_parameter('dark_roi_x1').value
        self.dark_roi_x2         = self.get_parameter('dark_roi_x2').value
        self.dark_roi_y1         = self.get_parameter('dark_roi_y1').value
        self.dark_roi_y2         = self.get_parameter('dark_roi_y2').value
        self.camera_offset_m     = self.get_parameter('camera_offset_m').value

        # ── 状态变量 ───────────────────────────────────
        self.bridge = CvBridge()
        self.color_frame = None
        self.depth_frame = None
        self.camera_info = None
        self.depth_scale = None
        self._info_received = False
        self._scale_received = False

        # ── 线程同步 ───────────────────────────────────
        self._frame_lock    = threading.Lock()       # 保护 color/depth_frame
        self._result_lock   = threading.Lock()       # 保护推理结果
        self._latest_offset = 0.0                    # 最新 Y 偏移量（像素，左为正）
        self._latest_has_target = False              # 是否有有效目标
        self._latest_best_box  = None                # 最佳检测框 [x1,y1,x2,y2,conf,cls]
        self._infer_ready   = threading.Event()      # 新帧就绪信号
        self._stop_event    = threading.Event()      # 停止信号

        # ── 遮挡检测状态 ─────────────────────────────
        self._obstruction_monitoring = False     # 是否正在等待画面变黑
        self._obstruction_dark_start = None       # 画面首次变黑的时间戳
        self._obstruction_monitor_timer = None    # 遮挡检测定时器
        self._obstruction_monitoring_start = None # 遮挡监测启动时间戳（用于超时）

        # ── 任务锁 ─────────────────────────────────
        self._task_locked = False                # True=正在执行任务，拒绝新请求

        # ── 非阻塞延迟发送队列（参考 res_kfs.py）────
        self._pending_results = deque()
        self._publish_timer = self.create_timer(0.05, self._publish_callback)

        # ── 加载 YOLO 模型 ────────────────────────────
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO 模型已加载: {self.model_path}")

        # ── 订阅 ───────────────────────────────────────
        self.sub_color = self.create_subscription(
            Image, "/camera/color/image_raw",
            self._color_callback, 60)
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw",
            self._depth_callback, 60)
        self.sub_scale = self.create_subscription(
            Float32, "/camera/depth/scale",
            self._scale_callback, 1)
        self.sub_info = self.create_subscription(
            CameraInfo, "/camera/color/camera_info",
            self._info_callback, 1)

        # ── 下位机触发：收到 Int8=1 → 校准一次并发偏移量 ──
        self.sub_request = self.create_subscription(
            UInt8, "/camera/yolo/request",
            self._request_callback, 30)        #校准请求

        # ── 发布 ───────────────────────────────────────
        self.pub_offset = self.create_publisher(
            Float32, "/camera/yolo/y_offset", 30)       # Y 偏移量（单位：m ，左为正）
        self.pub_grasp = self.create_publisher(
            UInt8, "/camera/view_cmd", 30)             #抓取命令
        self.pub_obstruction = self.create_publisher(
            UInt8, "/camera/view_sig", 30)       #抓取完成信号



        # ── 启动后台推理线程 ──────────────────────────
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="yolo-infer")
        self._infer_thread.start()

        self.get_logger().info(
            f"move_base_of_yolo 就绪 | 优先类别={self.priority_class} | "
            f"死区=±{self.dead_zone_m:.3f}m | 遮挡延时={self.obstruction_delay}s | "
            f"最大抓取次数={self.max_grasp_count}")

    def _publish_callback(self):
        """非阻塞延迟发送：轮询队列，发送到期结果（参考 res_kfs.py）"""
        now = time.time()
        while self._pending_results and self._pending_results[0][0] <= now:
            _, flag = self._pending_results.popleft()
            self.pub_grasp.publish(UInt8(data=flag))

    # ══════════════════════════════════════════════════════
    #  订阅回调
    # ══════════════════════════════════════════════════════

    def _info_callback(self, msg: CameraInfo):
        if not self._info_received:
            self.camera_info = msg
            self._info_received = True
            self.get_logger().info(
                f"已接收相机内参 | 分辨率={msg.width}x{msg.height} | "
                f"fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}")

    def _scale_callback(self, msg: Float32):
        if not self._scale_received:
            self.depth_scale = msg.data
            self._scale_received = True
            self.get_logger().info(f"已接收深度缩放因子: {self.depth_scale}")

    def _color_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._frame_lock:
                self.color_frame = frame
            self._infer_ready.set()
            if not hasattr(self, '_color_count'):
                self._color_count = 0
                self.get_logger().info("✓ 首次收到彩色图像")
            self._color_count += 1
        except CvBridgeError as e:
            self.get_logger().error(f"彩色图转换失败: {e}")

    def _depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            with self._frame_lock:
                self.depth_frame = depth
            if not hasattr(self, '_depth_count'):
                self._depth_count = 0
                self.get_logger().info("✓ 首次收到深度图像")
            self._depth_count += 1
        except CvBridgeError as e:
            self.get_logger().error(f"深度图转换失败: {e}")

    # ══════════════════════════════════════════════════════
    #  候选框过滤 & 最佳目标选择
    # ══════════════════════════════════════════════════════

    def _filter_boxes(self, boxes, allowed_classes=None):
        """从 YOLO 检测结果中筛选候选框。"""
        candidates = []
        if boxes is None:
            return candidates

        for box in boxes:
            cls_id = int(box.cls[0])
            if allowed_classes is not None and cls_id not in allowed_classes:
                continue

            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            if self.color_frame is not None:
                h, w = self.color_frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

            candidates.append([x1, y1, x2, y2, conf, cls_id])
        return candidates

    def _pick_closest_to_center(self, candidates):
        """选取检测框中心最靠近图像中心的那个。"""
        if not candidates:
            return None
        if self.color_frame is None:
            return candidates[0]

        h, w = self.color_frame.shape[:2]
        cx_img = w / 2.0
        cy_img = h / 2.0

        def dist2(box):
            bx = (box[0] + box[2]) / 2.0
            by = (box[1] + box[3]) / 2.0
            return (bx - cx_img) ** 2 + (by - cy_img) ** 2

        candidates.sort(key=dist2)
        return candidates[0]

    def _compute_y_offset(self, box):
        """
        计算 Y 方向偏移量：左为正。

        公式：offset = 图像中心 X - 检测框中心 X
        目标在画面左侧 → offset > 0（左为正）
        目标在画面右侧 → offset < 0
        """
        if self.color_frame is None:
            return 0.0

        h, w = self.color_frame.shape[:2]
        cx_img = w / 2.0
        box_cx = (box[0] + box[2]) / 2.0
        raw_offset = cx_img - box_cx  # 左为正

        return raw_offset

    def _get_box_depth_m(self, box, depth_frame):
        """
        读取检测框中心点的深度值（米）。

        Args:
            box: [x1, y1, x2, y2, conf, cls_id]
            depth_frame: np.ndarray (16UC1)

        Returns:
            float | None: 深度值（米），读取失败返回 None
        """
        if depth_frame is None:
            return None

        box_cx = int((box[0] + box[2]) / 2.0)
        box_cy = int((box[1] + box[3]) / 2.0)

        h, w = depth_frame.shape[:2]
        if box_cx < 0 or box_cx >= w or box_cy < 0 or box_cy >= h:
            return None

        depth_raw = float(depth_frame[box_cy, box_cx])
        if depth_raw <= 0:
            return None

        return depth_raw * self.depth_scale if self.depth_scale else depth_raw / 1000.0

    # ══════════════════════════════════════════════════════
    #  画面遮挡检测
    # ══════════════════════════════════════════════════════

    def _is_frame_dark(self, frame):
        """判断画面是否被遮挡（变黑）—— 只统计右下特定区域（抓取后物体出现的区域）。"""
        if frame is None:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 裁取指定 ROI 区域（默认：右侧1/5~1/3，底部~1/2高度）
        y1 = int(h * self.dark_roi_y1)
        y2 = int(h * self.dark_roi_y2)
        x1 = int(w * self.dark_roi_x1)
        x2 = int(w * self.dark_roi_x2)

        # 边界保护
        y1 = max(0, y1); y2 = min(h, y2)
        x1 = max(0, x1); x2 = min(w, x2)

        if y2 <= y1 or x2 <= x1:
            return False

        roi = gray[y1:y2, x1:x2]
        mean_val = np.mean(roi)
        return mean_val < self.dark_threshold

    # ══════════════════════════════════════════════════════
    #  像素 → 米 转换
    # ══════════════════════════════════════════════════════

    def _pixel_to_world_offset(self, offset_px, box, depth_frame):
        """将像素偏移量转换为世界坐标偏移量（米），保持左为正的符号约定。"""
        if self.camera_info is None or depth_frame is None or offset_px == 0.0:
            return 0.0

        fx = self.camera_info.k[0]

        # 读取目标中心点的深度值
        box_cx = int((box[0] + box[2]) / 2.0)
        box_cy = int((box[1] + box[3]) / 2.0)

        h, w = depth_frame.shape[:2]
        box_cx = max(0, min(box_cx, w - 1))
        box_cy = max(0, min(box_cy, h - 1))

        depth_raw = float(depth_frame[box_cy, box_cx])
        depth_m = depth_raw * self.depth_scale if self.depth_scale else depth_raw / 1000.0

        if depth_m <= 0.0 or fx <= 0.0:
            return 0.0

        return offset_px * depth_m / fx

    # ══════════════════════════════════════════════════════
    #  后台推理线程
    # ══════════════════════════════════════════════════════

    def _infer_loop(self):
        """
        后台线程：YOLO 推理 → 优先 class_id=3 → 选离中心最近 →
        计算 Y 偏移（左为正）→ 写入共享变量。
        """
        while not self._stop_event.is_set():
            self._infer_ready.wait(timeout=0.1)
            self._infer_ready.clear()

            with self._frame_lock:
                if self.color_frame is None:
                    continue
                frame = self.color_frame.copy()

            try:
                results = self.model(
                    frame,
                    conf=self.conf_thres,
                    iou=self.iou_thres,
                    verbose=False,
                )
            except Exception as e:
                self.get_logger().warn(f"YOLO 推理异常: {e}")
                continue

            best_box = None
            has_target = False

            if results and len(results) > 0:
                boxes = results[0].boxes

                # 第一步：优先 priority_class
                priority_candidates = self._filter_boxes(
                    boxes, allowed_classes=[self.priority_class])
                best_box = self._pick_closest_to_center(priority_candidates)

                # 第二步：回退到所有类别
                if best_box is None:
                    all_candidates = self._filter_boxes(boxes, allowed_classes=None)
                    best_box = self._pick_closest_to_center(all_candidates)

            if best_box is not None:
                # ── 深度误报过滤：深度不在正常区间 → 视为误报 ──
                if self.depth_filter_enabled:
                    with self._frame_lock:
                        dframe = self.depth_frame.copy() if self.depth_frame is not None else None
                    if dframe is not None:
                        depth_m = self._get_box_depth_m(best_box, dframe)
                        if depth_m is not None and \
                           (depth_m < self.min_depth_m or depth_m > self.max_depth_m):
                            # self.get_logger().info(
                            #     f"⚠ 深度 {depth_m:.3f}m 不在正常区间 "
                            #     f"[{self.min_depth_m:.3f}, {self.max_depth_m:.3f}]m → 视为误报，丢弃")
                            best_box = None

                if best_box is not None:
                    has_target = True
                    offset = self._compute_y_offset(best_box)
                else:
                    has_target = False
                    offset = 0.0
                    best_box = None
            else:
                has_target = False
                offset = 0.0
                best_box = None

            with self._result_lock:
                self._latest_offset = offset
                self._latest_has_target = has_target
                self._latest_best_box = best_box

    # ══════════════════════════════════════════════════════
    #  下位机请求回调 — 校准 → 偏移量 → 抓取 → 遮挡
    # ══════════════════════════════════════════════════════

    def _request_callback(self, msg: UInt8):
        """收到 Int8=1 → 校准偏移量(m) → 死区外发偏移量 / 死区内发抓取+遮挡"""
        if msg.data != 1:
            return

        # ── 任务锁：正在执行任务时忽略新请求 ──────────
        if self._task_locked:
            self.get_logger().warn("任务锁已持有 → 忽略重复的校准请求", throttle_duration_sec=2.0)
            return
        self._task_locked = True

        # ── 检查抓取次数 ──────────────────────────────
        if _get_grasp_count() > self.max_grasp_count:
            self.get_logger().info(
                f"抓取次数已达 {_get_grasp_count()} (>{self.max_grasp_count})，拒绝请求，退出进程")
            self._stop_event.set()
            raise GraspLimitReached()

        # 取消之前未完成的遮挡监测
        if self._obstruction_monitor_timer is not None:
            self.destroy_timer(self._obstruction_monitor_timer)
            self._obstruction_monitor_timer = None
        self._obstruction_monitoring = False
        self._obstruction_dark_start = None

        # ── 读取最新推理结果 ─────────────────────────
        with self._result_lock:
            offset_px = self._latest_offset
            has_target = self._latest_has_target
            best_box = self._latest_best_box

        with self._frame_lock:
            dframe = self.depth_frame.copy() if self.depth_frame is not None else None

        # ── 像素 → 米 ────────────────────────────────
        if has_target and best_box is not None and dframe is not None:
            offset_m = self._pixel_to_world_offset(offset_px, best_box, dframe)
        else:
            offset_m = 0.0

        # ── 相机偏置补偿 ──────────────────────────────
        # 相机在机械臂中心左侧，需将偏移量向左修正（左为正）
        offset_m += self.camera_offset_m

        # ── 死区判断 ─────────────────────────────────
        in_dead_zone = abs(offset_m) <= self.dead_zone_m

        if in_dead_zone:
            # 目标已居中 → 发送抓取，启动遮挡监测
            grasp_msg = UInt8()
            grasp_msg.data = 1
            self.pub_grasp.publish(grasp_msg)
            # 非阻塞延时 0.3s 后发送 0 复位 view_cmd
            self._pending_results.append((time.time() + 0.3, 0))

            cnt = _increment_grasp_count()

            self.get_logger().info(
                f"← 收到校准请求(1) → 目标在死区内 (|{offset_m:.3f}|≤{self.dead_zone_m:.3f}m)"
                f" → grasp=1 | 抓取计数={cnt} | 开始监测遮挡...")

            # 启动遮挡检测（10Hz 轮询画面亮度）
            self._obstruction_monitoring = True
            self._obstruction_dark_start = None
            self._obstruction_monitoring_start = time.time()
            if self._obstruction_monitor_timer is not None:
                self.destroy_timer(self._obstruction_monitor_timer)
            self._obstruction_monitor_timer = self.create_timer(
                0.1, self._check_obstruction)
        else:
            # 目标偏离中心 → 停止遮挡监测，仅发送偏移量 → 解锁
            if self._obstruction_monitor_timer is not None:
                self.destroy_timer(self._obstruction_monitor_timer)
                self._obstruction_monitor_timer = None
            self._obstruction_monitoring = False
            self._obstruction_dark_start = None

            resp = Float32()
            resp.data = float(offset_m)
            self.pub_offset.publish(resp)

            self._task_locked = False

            self.get_logger().info(
                f"← 收到校准请求(1) → 目标偏离中心"
                f" → y_offset={offset_m:+.3f}m (死区={self.dead_zone_m:.3f}m) → 已解锁")

    def _check_obstruction(self):
        """遮挡监测回调（10Hz）：画面持续变黑 → 发送 obstruction=1；超时自动解锁"""
        if not self._obstruction_monitoring:
            return

        with self._frame_lock:
            cframe = self.color_frame.copy() if self.color_frame is not None else None

        # ── 计算 ROI 平均亮度并打印（1Hz 节流） ──────
        roi_mean = None
        if cframe is not None:
            gray = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            y1 = int(h * self.dark_roi_y1)
            y2 = int(h * self.dark_roi_y2)
            x1 = int(w * self.dark_roi_x1)
            x2 = int(w * self.dark_roi_x2)
            y1 = max(0, y1); y2 = min(h, y2)
            x1 = max(0, x1); x2 = min(w, x2)
            if y2 > y1 and x2 > x1:
                roi = gray[y1:y2, x1:x2]
                roi_mean = float(np.mean(roi))

        if roi_mean is not None:
            self.get_logger().info(
                f"[遮挡监测] ROI平均亮度={roi_mean:.1f} "
                f"(阈值={self.dark_threshold}) 监测中...",
                throttle_duration_sec=1.0)

        is_dark = self._is_frame_dark(cframe)

        # ── 超时检查 ──────────────────────────────────
        now = time.time()
        if self._obstruction_monitoring_start is not None:
            elapsed_total = now - self._obstruction_monitoring_start
            if elapsed_total > self.obstruction_timeout:
                self.get_logger().warn(
                    f"遮挡检测超时 ({elapsed_total:.1f}s > {self.obstruction_timeout}s) → 解锁")
                self._stop_monitoring()
                self._task_locked = False
                return

        if is_dark:
            if self._obstruction_dark_start is None:
                # 首次检测到变黑 → 记录起始时间
                self._obstruction_dark_start = now
                self.get_logger().info(f"检测到画面变黑 (亮度={roi_mean:.1f})，开始计时...")
            elif (now - self._obstruction_dark_start) >= self.obstruction_delay:
                # 持续变黑 ≥ 延时 → 发送遮挡信号 → 解锁
                obs_msg = UInt8()
                obs_msg.data = 1
                self.pub_obstruction.publish(obs_msg)
                self.get_logger().info(
                    f"画面持续变黑 {self.obstruction_delay}s → obstruction=1 → 已解锁")
                self._stop_monitoring()
                self._task_locked = False
        else:
            # 画面恢复明亮 → 重置计时
            if self._obstruction_dark_start is not None:
                self.get_logger().info(f"画面恢复明亮 (亮度={roi_mean:.1f})，重置遮挡计时")
                self._obstruction_dark_start = None

    def _stop_monitoring(self):
        """停止遮挡监测并销毁定时器"""
        self._obstruction_monitoring = False
        self._obstruction_dark_start = None
        self._obstruction_monitoring_start = None
        if self._obstruction_monitor_timer is not None:
            self.destroy_timer(self._obstruction_monitor_timer)
            self._obstruction_monitor_timer = None

    # ══════════════════════════════════════════════════════
    #  生命周期
    # ══════════════════════════════════════════════════════

    def shutdown(self):
        """清理：停止推理线程"""
        self._stop_event.set()
        if hasattr(self, '_infer_thread') and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=2.0)
            self.get_logger().info("推理线程已停止")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = MoveBaseOfYolo()
    except FileNotFoundError as e:
        rclpy.logging.get_logger("move_base_of_yolo").error(str(e))
        rclpy.shutdown()
        return

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, GraspLimitReached):
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
