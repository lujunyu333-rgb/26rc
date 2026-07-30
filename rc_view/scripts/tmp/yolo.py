#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 检测节点 - 订阅 RealSense D435i 彩色+深度图像话题，
运行 YOLOv8 目标检测，发布检测框中心的 3D 坐标（PointStamped）。

架构说明：
  后台推理线程 —— 持续取最新彩色帧执行 YOLO 推理（耗时操作），
  推理结果写入共享变量；Timer 回调（30Hz）只做轻量的可视化绘制、
  cv2.imshow 刷新和话题发布，确保图形显示不被推理阻塞。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2
import numpy as np
import os
import logging
import threading

logging.getLogger("ultralytics").setLevel(logging.ERROR)


# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_MODEL_PATH       = "/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt"
DEFAULT_CONFIDENCE       = 0.45
DEFAULT_MIN_BOX_AREA     = 500
DEFAULT_TARGET_CLASS     = 3       # COCO class ID: 3=car, 只检测这个类别
DEFAULT_SELECT_MODE      = "highest_conf"
DEFAULT_FRAME_ID         = "camera_color_optical_frame"
DEFAULT_FALLBACK_CLASSES  = [0, 1, 2]   # COCO: 0=person, 1=bicycle, 2=car


class RsQiyuNode(Node):
    """YOLO 检测 + 深度估计 ROS2 节点，只发 3D 坐标点"""

    def __init__(self):
        super().__init__('yolo_detection_node')

        # ── 参数 ──────────────────────────────────────
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE)
        self.declare_parameter('min_box_area', DEFAULT_MIN_BOX_AREA)
        self.declare_parameter('target_class', DEFAULT_TARGET_CLASS)
        self.declare_parameter('select_mode', DEFAULT_SELECT_MODE)
        self.declare_parameter('frame_id', DEFAULT_FRAME_ID)
        self.declare_parameter('fallback_classes', DEFAULT_FALLBACK_CLASSES)
        self.model_path      = self.get_parameter('model_path').value
        self.conf_thres      = self.get_parameter('confidence_threshold').value
        self.min_box_area    = self.get_parameter('min_box_area').value
        self.iou_thres       = 0.45               # NMS IoU 阈值
        self.target_class    = self.get_parameter('target_class').value
        self.select_mode     = self.get_parameter('select_mode').value
        self.frame_id        = self.get_parameter('frame_id').value
        self.fallback_classes = self.get_parameter('fallback_classes').value

        # ── 状态 ─────────────────────────────────────
        self.bridge = CvBridge()
        self.color_frame = None
        self.depth_frame = None
        self.camera_info = None
        self.depth_scale = None
        self.info_received = False
        self.scale_received = False

        # ── 多线程同步 ──────────────────────────────
        self._frame_lock   = threading.Lock()       # 保护 color/depth_frame
        self._result_lock  = threading.Lock()       # 保护推理结果
        self._latest_point_msg = None               # 最新 3D 坐标点
        self._latest_candidates = []                # 最新候选框
        self._infer_frame_ready = threading.Event() # 新帧就绪信号
        self._stop_event   = threading.Event()      # 停止信号

        # ── 可视化数据 ────────────────────────────────
        self._vis_candidates = []      # 当前帧所有候选框 [[x1,y1,x2,y2,conf], ...]

        # ── 丢失目标时的回退坐标 ──────────────────────
        self._last_valid_point_msg = None   # 最后一帧有效检测的 PointStamped

        # ── 模型 ─────────────────────────────────────
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"模型文件不存在: {self.model_path}")
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO 模型已加载: {self.model_path}")

        # ── 订阅 ─────────────────────────────────────
        self.sub_cam_info = self.create_subscription(
            CameraInfo, "/camera/color/camera_info",
            self.camera_info_callback, 1)
        self.sub_depth_scale = self.create_subscription(
            Float32, "/camera/depth/scale",
            self.depth_scale_callback, 1)
        self.sub_color = self.create_subscription(
            Image, "/camera/color/image_raw",
            self.color_callback, 60)
        self.sub_depth = self.create_subscription(
            Image, "/camera/depth/image_raw",
            self.depth_callback, 60)

        # ── 发布：PointStamped（只发 3D 坐标） ──────
        self.pub_point = self.create_publisher(
            PointStamped, "camera/yolo/point", 10)

        # ── 发布：可视化图像（检测框标注图） ────────
        self.pub_vis = self.create_publisher(
            Image, "camera/yolo/vis_image", 10)

        # ── 处理 Timer ─────────────────────────────────
        self._has_gui   = "DISPLAY" in os.environ
        self._first_warn = True
        self._vis_frame_count = 0          # 可视化发布计数器
        self._vis_pub_interval = 3         # 每 N 帧发布一次 vis_image
        self._proc_timer = self.create_timer(1.0 / 30.0, self._timer_callback)

        # ── 启动后台推理线程 ─────────────────────────
        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True, name="yolo-infer")
        self._infer_thread.start()
        self.get_logger().info("后台推理线程已启动")
        self.get_logger().info("YOLO 检测节点就绪（Timer @ 30Hz），等待图像...")

    # ── 回调 ────────────────────────────────────────

    def camera_info_callback(self, msg):
        if not self.info_received:
            self.camera_info = msg
            self.info_received = True
            self.get_logger().info("已接收相机内参")

    def depth_scale_callback(self, msg):
        if not self.scale_received:
            self.depth_scale = msg.data
            self.scale_received = True
            self.get_logger().info(f"已接收深度缩放因子: {self.depth_scale}")

    def color_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
            with self._frame_lock:
                self.color_frame = frame
            self._infer_frame_ready.set()      # 通知推理线程有新帧
            # 诊断：首次收到彩色图像时打印
            if not hasattr(self, '_color_msg_count'):
                self._color_msg_count = 0
                self.get_logger().info("✓ 首次收到彩色图像话题")
            self._color_msg_count += 1
            if self._color_msg_count % 100 == 0:
                self.get_logger().info(
                    f"已收到 {self._color_msg_count} 帧彩色图像",
                    throttle_duration_sec=5)
        except CvBridgeError as e:
            self.get_logger().error(f"彩色图转换失败: {e}")

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='16UC1')
            with self._frame_lock:
                self.depth_frame = depth
            if not hasattr(self, '_depth_msg_count'):
                self._depth_msg_count = 0
                self.get_logger().info("✓ 首次收到深度图像话题")
            self._depth_msg_count += 1
        except CvBridgeError as e:
            self.get_logger().error(f"深度图转换失败: {e}")

    # ── 坐标转换 ────────────────────────────────────

    def pixel_to_3d(self, u, v, depth_meters):
        """像素坐标 → 相机坐标系 3D 坐标 (x, y, z)"""
        if depth_meters <= 0 or self.camera_info is None:
            return None
        k = self.camera_info.k
        x = (u - k[2]) * depth_meters / k[0]
        y = (v - k[5]) * depth_meters / k[4]
        return (x, y, depth_meters)

    def get_center_depth(self, dep_img, x1, y1, x2, y2):
        """检测框中心深度（米），失效时用 ROI 中值回退"""
        if self.depth_scale is None:
            return -1.0

        h_img, w_img = dep_img.shape[:2]
        cx = np.clip((x1 + x2) // 2, 0, w_img - 1)
        cy = np.clip((y1 + y2) // 2, 0, h_img - 1)

        depth_m = dep_img[cy, cx] * self.depth_scale
        if depth_m > 0:
            return float(depth_m)

        # 回退：ROI 中值
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_img, x2), min(h_img, y2)
        if x2c <= x1c or y2c <= y1c:
            return -1.0

        roi = dep_img[y1c:y2c, x1c:x2c] * self.depth_scale
        valid = roi[roi > 0]
        return float(np.median(valid)) if len(valid) > 0 else -1.0

    # ── 候选过滤 ────────────────────────────────────

    def _filter_candidates(self, boxes, target_classes=None):
        """从所有检测框中筛选出指定类别 + 面积过线 + 边界裁剪的候选"""
        if target_classes is None:
            target_classes = [self.target_class]
        candidates = []
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in target_classes:
                continue

            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < self.min_box_area:
                continue

            if self.color_frame is not None:
                h, w = self.color_frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1:
                    continue

            candidates.append([x1, y1, x2, y2, conf])

        return candidates

    def _pick_best(self, candidates):
        """无锁定时按 select_mode 选最佳"""
        if not candidates:
            return None
        if self.select_mode == "leftmost":
            candidates.sort(key=lambda c: c[0])
        else:
            candidates.sort(key=lambda c: c[4], reverse=True)
        return candidates[0]

    def _pick_center(self, candidates):
        """从候选中选择检测框中心最靠近图像中心的框"""
        if not candidates:
            return None
        if self.color_frame is None:
            return candidates[0]
        h, w = self.color_frame.shape[:2]
        cx_img, cy_img = w / 2.0, h / 2.0

        def dist2(c):
            bx = (c[0] + c[2]) / 2.0
            by = (c[1] + c[3]) / 2.0
            return (bx - cx_img) ** 2 + (by - cy_img) ** 2

        candidates.sort(key=dist2)
        return candidates[0]

    def _box_to_point(self, x1, y1, x2, y2):
        """检测框 → PointStamped 消息（返回 None 表示深度无效）"""
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if (self.depth_frame is None or not self.scale_received
                or self.camera_info is None):
            return None

        depth_m = self.get_center_depth(self.depth_frame, x1, y1, x2, y2)
        if depth_m <= 0:
            return None

        pt3d = self.pixel_to_3d(cx, cy, depth_m)
        if pt3d is None:
            return None

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.point.x = float(pt3d[0])
        msg.point.y = float(pt3d[1])
        msg.point.z = float(pt3d[2])
        return msg

    # ── 后台推理线程 ────────────────────────────────

    def _infer_loop(self):
        """后台线程：持续取最新彩色帧做 YOLO 推理，结果写入共享变量"""
        while not self._stop_event.is_set():
            # 等待新帧就绪信号，超时 100ms 防止死等
            signaled = self._infer_frame_ready.wait(timeout=0.1)
            self._infer_frame_ready.clear()

            # 取最新帧（线程安全拷贝）
            with self._frame_lock:
                if self.color_frame is None:
                    continue
                frame = self.color_frame.copy()

            if frame is None:
                continue

            # ── YOLO 推理（耗时操作，在后台线程中执行） ──
            try:
                results = self.model(frame, conf=self.conf_thres,
                                     iou=self.iou_thres, verbose=False)
            except Exception as e:
                self.get_logger().warn(f"YOLO 推理异常: {e}")
                continue

            candidates = []
            if results and len(results) > 0 and results[0].boxes is not None:
                candidates = self._filter_candidates(results[0].boxes)

            # 选取最佳并计算 3D 坐标（统一按靠近图像中心优先）
            best = self._pick_center(candidates)
            point_msg = None

            # ── 目标类别(id=3)未检测到 → 回退到 fallback_classes ──
            if best is None and self.fallback_classes:
                if results and len(results) > 0 and results[0].boxes is not None:
                    fallback_candidates = self._filter_candidates(
                        results[0].boxes, target_classes=self.fallback_classes)
                    best = self._pick_center(fallback_candidates)

            if best is not None:
                with self._frame_lock:
                    dep = self.depth_frame
                if dep is not None:
                    depth_m = self.get_center_depth(dep, *best[:4])
                    if depth_m > 0:
                        cx = (best[0] + best[2]) // 2
                        cy = (best[1] + best[3]) // 2
                        pt3d = self.pixel_to_3d(cx, cy, depth_m)
                        if pt3d is not None:
                            point_msg = PointStamped()
                            point_msg.header.stamp = self.get_clock().now().to_msg()
                            point_msg.header.frame_id = self.frame_id
                            point_msg.point.x = float(pt3d[0])
                            point_msg.point.y = float(pt3d[1])
                            point_msg.point.z = float(pt3d[2])

            # 原子更新共享结果
            with self._result_lock:
                self._latest_point_msg = point_msg
                self._latest_candidates = candidates

    # ── Timer 回调（高频轻量：可视化 + 发布）───────

    def _timer_callback(self):
        """ROS2 Timer 回调：轻量操作 —— 只做可视化绘制 + 发布（~30Hz）

        YOLO 推理已移到后台线程，这里不再执行耗时操作。
        """
        try:
            # ── 获取最新帧（线程安全拷贝） ────────────
            with self._frame_lock:
                if self.color_frame is None:
                    if self._first_warn:
                        self.get_logger().warn(
                            "尚未收到彩色图像，等待 RealSense 发布节点...")
                        self._first_warn = False
                    return
                disp_frame = self.color_frame.copy()
            self._first_warn = True

            # ── 获取最新推理结果（线程安全） ──────────
            with self._result_lock:
                point_msg = self._latest_point_msg
                vis_candidates = list(self._latest_candidates)

            # ── 发布 3D 坐标点 ──────────────────────────
            if point_msg is not None:
                point_msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_point.publish(point_msg)
                self._last_valid_point_msg = point_msg
                self.get_logger().info(
                    f"3D点: ({point_msg.point.x:.3f}, "
                    f"{point_msg.point.y:.3f}, {point_msg.point.z:.3f}) "
                    f"frame={point_msg.header.frame_id}",
                    throttle_duration_sec=1.0)
            elif self._last_valid_point_msg is not None:
                # 目标丢失，重发最后一帧有效坐标
                last_msg = self._last_valid_point_msg
                last_msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_point.publish(last_msg)
                self.get_logger().info(
                    "目标丢失，发送最后坐标: "
                    f"({last_msg.point.x:.3f}, "
                    f"{last_msg.point.y:.3f}, {last_msg.point.z:.3f})",
                    throttle_duration_sec=1.0)
            else:
                # 从未检测到目标时的回退零点
                zero_msg = PointStamped()
                zero_msg.header.stamp = self.get_clock().now().to_msg()
                zero_msg.header.frame_id = self.frame_id
                zero_msg.point.x = 0.0
                zero_msg.point.y = 0.0
                zero_msg.point.z = 0.0
                self.pub_point.publish(zero_msg)

            # ── 可视化绘制 ────────────────────────────
            self._vis_candidates = vis_candidates
            vis = self._draw_visualization(disp_frame)

            # ── 发布可视化图像（降频：每 N 帧发一次）──
            self._vis_frame_count += 1
            if self._vis_frame_count % self._vis_pub_interval == 0:
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
                    vis_msg.header.stamp = self.get_clock().now().to_msg()
                    vis_msg.header.frame_id = self.frame_id
                    self.pub_vis.publish(vis_msg)
                except Exception as e:
                    self.get_logger().warn(f"可视化图像发布失败: {e}",
                                           throttle_duration_sec=5)

            # ── 本地窗口显示（高频刷新，不阻塞）───────
            if self._has_gui:
                cv2.imshow("yolo_detect", vis)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Timer 回调异常: {e}")

    def shutdown(self):
        """清理：停止推理线程并释放资源"""
        self._stop_event.set()
        if hasattr(self, '_infer_thread') and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=2.0)
            self.get_logger().info("推理线程已停止")

    # ── 可视化 ───────────────────────────────────────

    def _draw_visualization(self, frame):
        """在帧上绘制检测框、标签和3D坐标信息"""
        if frame is None:
            return frame

        disp = frame.copy()
        h, w = disp.shape[:2]

        # ── 绘制所有候选框（绿色） ────────────────────
        for cand in self._vis_candidates:
            x1, y1, x2, y2, conf = cand
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(disp, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return disp


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RsQiyuNode()
    except FileNotFoundError as e:
        rclpy.logging.get_logger("yolo_detect").error(str(e))
        rclpy.shutdown()
        return

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        if node._has_gui:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
