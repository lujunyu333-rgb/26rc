#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QR码检测节点 - 从USB摄像头采集图像，检测画面中的QR码，
发布 occupancy flag。边沿触发：仅在QR码首次出现时发送一次，避免重复发布。

flag 含义: 仅发布 2=检测到QR码(内容为"2") 或 3=检测到QR码(内容为"3")；0/1/其他均过滤不发布
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from rc_view.scripts.cam_of_ros2.allcamera import uvc_cam
import cv2
import numpy as np
import os
import time
from collections import deque


# ── 可调参数（宏定义） ────────────────────────────────
# 检测窗口
DEFAULT_DETECTION_WINDOW_S = 120.0   # 收到遮挡信号后持续检测的时长（秒）

# ROI（占全图比例，0.0~1.0）
ROI_X_RATIO  = 0   # 左边界
ROI_Y_RATIO  = 0   # 上边界
ROI_W_RATIO  = 1   # 宽度
ROI_H_RATIO  = 1   # 高度

# 曝光控制
GAMMA = 2.0            # Gamma校正：>1降低曝光，<1提高曝光，1.0=不变

# 二值化
BLUR_KSIZE        = 5   # 高斯模糊核大小（奇数，1=不模糊）
THRESH_BLOCK_SIZE = 11  # 自适应阈值邻域大小（奇数）
THRESH_C          = 2   # 自适应阈值常数

# 显示
ENABLE_DISPLAY = True   # 是否开启可视化窗口（False=无头模式）
DISPLAY_SCALE  = 0.3    # imshow 缩放比例（<1缩小窗口）


class QiyuQRNode(Node):
    """通过QR码检测器识别QR码并发布flag"""

    def __init__(self, node_name: str, topic_name: str):
        super().__init__(node_name)

        # ── ROS2 参数（默认值引用顶部宏定义） ──────
        self.declare_parameter('device_path', 6)
        device_path = self.get_parameter('device_path').value

        self.flag_pub = self.create_publisher(UInt8, topic_name, 10)
        self.qr_flag_pub = self.create_publisher(UInt8, "/camera/qr_flag", 10)

        self.declare_parameter('detection_window_s', DEFAULT_DETECTION_WINDOW_S)
        self.detection_window_s = self.get_parameter('detection_window_s').value

        self.declare_parameter('roi_x_ratio', ROI_X_RATIO)
        self.declare_parameter('roi_y_ratio', ROI_Y_RATIO)
        self.declare_parameter('roi_w_ratio', ROI_W_RATIO)
        self.declare_parameter('roi_h_ratio', ROI_H_RATIO)
        self.roi_x_ratio = self.get_parameter('roi_x_ratio').value
        self.roi_y_ratio = self.get_parameter('roi_y_ratio').value
        self.roi_w_ratio = self.get_parameter('roi_w_ratio').value
        self.roi_h_ratio = self.get_parameter('roi_h_ratio').value

        self.declare_parameter('gamma', GAMMA)
        self.gamma = self.get_parameter('gamma').value

        self.declare_parameter('blur_ksize', BLUR_KSIZE)
        self.declare_parameter('thresh_block_size', THRESH_BLOCK_SIZE)
        self.declare_parameter('thresh_C', THRESH_C)
        self.blur_ksize = self.get_parameter('blur_ksize').value
        self.thresh_block_size = self.get_parameter('thresh_block_size').value
        self.thresh_C = self.get_parameter('thresh_C').value

        self.declare_parameter('enable_display', ENABLE_DISPLAY)
        self.declare_parameter('display_scale', DISPLAY_SCALE)
        self.enable_display = self.get_parameter('enable_display').value
        self.display_scale = self.get_parameter('display_scale').value

        # ── 屏蔽 libjpeg 的 "Corrupt JPEG data" C 层 stderr 输出（一次性设置）──
        self._stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)

        self.cam = uvc_cam(
            device_path=device_path,
            name=node_name,
            funcation=0,
            width=2560,
            height=1440,
            fps=30
        )

        if not self.cam.isOpened():
            self.get_logger().error(f"相机 {node_name} 打开失败，设备索引={device_path}")
            raise RuntimeError(f"Camera open failed: {node_name}")

        # ── QR码检测器 ──────────────────────────────────
        self.qr_detector = cv2.QRCodeDetector()

        # ── 启动预热：丢弃前若干帧，等待传感器稳定 ──
        self.get_logger().info("⏳ 相机预热中…")
        warmup_ok = False
        for i in range(50):
            ret, f = self.cam.read_one_frame()
            if ret and f is not None and f.shape[0] > 0:
                if not warmup_ok:
                    self.get_logger().info(f"✅ 相机预热完成（第{i+1}帧起稳定）")
                    warmup_ok = True
                break
            time.sleep(0.15)
        if not warmup_ok:
            self.get_logger().warn("⚠️ 相机预热超时，将以冷启动状态运行")

        # ── 边沿触发：仅在QR码首次出现时发送一次 ────
        self._prev_flag = 0
        self._consecutive_failures = 0  # 连续读取失败计数

        # ── 遮挡触发检测窗口 ──────────────────────────
        self._detection_active = False          # 当前是否在检测窗口内
        self._detection_start_time = None       # 窗口起始时间戳

        # 订阅 move_base_of_yolo 的遮挡信号
        self.sub_obstruction = self.create_subscription(
            UInt8, "/camera/view_sig",
            self._obstruction_callback, 10)

        # ── 非阻塞延迟发送队列（参考 res_kfs.py）────
        self._pending_results = deque()
        self._publish_timer = self.create_timer(0.05, self._publish_callback)

        self.get_logger().info(
            f"{node_name} 启动 | QR码检测模式 | 边沿触发 | "
            f"检测窗口={self.detection_window_s:.0f}s（由遮挡信号触发）")

    def _publish_callback(self):
        """非阻塞延迟发送：轮询队列，发送到期结果（参考 res_kfs.py）"""
        now = time.time()
        while self._pending_results and self._pending_results[0][0] <= now:
            _, flag = self._pending_results.popleft()
            self.flag_pub.publish(UInt8(data=flag))

    def _preprocess_for_qr(self, bgr_img):
        """
        QR码检测预处理管线：
        1. ROI 裁剪（缩小检测范围，聚焦目标区域）
        2. 灰度化
        3. Gamma 校正（降低过曝）
        4. 高斯模糊降噪
        5. 自适应二值化（应对不均匀光照）
        返回 (processed_img, roi_rect) 其中 roi_rect = (x, y, w, h)
        """
        h, w = bgr_img.shape[:2]

        # ── ROI 裁剪 ──────────────────────────────
        rx = int(w * self.roi_x_ratio)
        ry = int(h * self.roi_y_ratio)
        rw = int(w * self.roi_w_ratio)
        rh = int(h * self.roi_h_ratio)
        # 边界保护
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = max(10, min(rw, w - rx))
        rh = max(10, min(rh, h - ry))

        roi = bgr_img[ry:ry + rh, rx:rx + rw]

        # ── 灰度化 ──────────────────────────────
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ── Gamma 校正（降低曝光） ──────────
        if abs(self.gamma - 1.0) > 1e-3:
            # LUT 查表法：T(x) = ((x/255)^(1/gamma)) * 255
            lut = np.array([
                ((i / 255.0) ** (1.0 / self.gamma)) * 255.0
                for i in range(256)
            ], dtype=np.uint8)
            gray = cv2.LUT(gray, lut)

        # ── 高斯模糊降噪 ──────────────────────
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        # ── 自适应二值化 ──────────────────────
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.thresh_block_size,
            self.thresh_C
        )

        return binary, (rx, ry, rw, rh)

    def _detect_qr_code(self, bgr_img):
        """检测QR码（含预处理），返回 (detected, data, bbox, flag_value)"""
        # ── 路径1: 预处理后检测 ──────────────────
        processed, roi_rect = self._preprocess_for_qr(bgr_img)
        data, bbox, _ = self.qr_detector.detectAndDecode(processed)

        if len(data) > 0:
            # 将ROI坐标映射回原图坐标
            if bbox is not None:
                rx, ry, _, _ = roi_rect
                bbox = bbox.astype(np.float32)
                if bbox.ndim == 3:
                    bbox = bbox.reshape(4, 2)
                bbox[:, 0] += rx
                bbox[:, 1] += ry
            data = data.strip()
            try:
                flag_val = int(data)
            except ValueError:
                flag_val = 0
            return True, data, bbox, flag_val

        # ── 路径2: 原始灰度图兜底 ──────────────
        gray_full = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        data, bbox, _ = self.qr_detector.detectAndDecode(gray_full)

        detected = (len(data) > 0)
        if not detected:
            return False, "", None, 0

        data = data.strip()
        try:
            flag_val = int(data)
        except ValueError:
            flag_val = 0

        return True, data, bbox, flag_val

    def _obstruction_callback(self, msg: UInt8):
        """收到遮挡信号 → 启动检测窗口"""
        if msg.data == 1 and not self._detection_active:
            self._detection_active = True
            self._detection_start_time = time.time()
            self._prev_flag = 0  # 重置边沿状态，确保窗口内首次检测到时能触发
            self.get_logger().info(
                f"← 收到遮挡信号 → 启动检测窗口 "
                f"({self.detection_window_s:.0f}s)")

    def process_frame(self, bgr_frame: np.ndarray) -> tuple:
        """对一帧进行处理，返回 (detected, flag, qr_data, bbox)"""
        detected, qr_data, bbox, flag = self._detect_qr_code(bgr_frame)
        return detected, flag, qr_data, bbox

    def _draw_visualization(self, frame, flag, qr_data, bbox):
        """在图像上绘制QR码检测结果"""
        status_map = {
            0: "NO QR",
            1: "QR DETECTED",
            2: "QR DETECTED",
            3: "QR DETECTED",

        }
        status_text = status_map.get(flag, f"QR: {flag}")
        flag_color = (
            (0, 255, 0) if flag > 0 else
            (128, 128, 128)
        )

        result = frame.copy()

        # 绘制ROI区域（虚线框，蓝绿色）
        h, w = frame.shape[:2]
        rx = int(w * self.roi_x_ratio)
        ry = int(h * self.roi_y_ratio)
        rw = int(w * self.roi_w_ratio)
        rh = int(h * self.roi_h_ratio)
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)
        cv2.putText(result, "ROI", (rx + 4, ry + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 绘制QR码边界框
        if bbox is not None and flag != 0:
            bbox_int = bbox.astype(np.int32)
            # bbox 可能是 (4,1,2) 或 (1,4,2) 形状
            if bbox_int.ndim == 3:
                bbox_int = bbox_int.reshape(4, 2)
            cv2.polylines(result, [bbox_int], True, (0, 255, 0), 2)
            # 在QR码上方显示解码内容
            cx = int(np.mean(bbox_int[:, 0]))
            cy = int(np.mean(bbox_int[:, 1]))
            cv2.putText(result, f"Data: {qr_data}", (cx - 60, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 状态信息
        cv2.putText(result, f"Flag: {flag} ({status_text})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, flag_color, 2)
        cv2.putText(result, f"QR Data: {qr_data if qr_data else 'None'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return result

    def run(self):
        """主循环"""
        while rclpy.ok():
            ret, frame = self.cam.read_one_frame()
            # ── 帧有效性校验 ──────────────────────────
            ok = (ret and frame is not None
                  and frame.shape[0] > 0 and frame.shape[1] > 0)

            if not ok:
                self._consecutive_failures += 1
                backoff = min(0.1 * self._consecutive_failures, 2.0)
                self.get_logger().warn(
                    f"{self.cam.cam_name} 读取失败 "
                    f"(连续{self._consecutive_failures}次, 退避{backoff:.1f}s)",
                    throttle_duration_sec=5)
                time.sleep(backoff)
                rclpy.spin_once(self, timeout_sec=0.01)
                continue

            self._consecutive_failures = 0  # 成功读取，重置计数

            # ── 检测窗口管理 ──────────────────────────────
            if self._detection_active:
                elapsed = time.time() - self._detection_start_time
                if elapsed > self.detection_window_s:
                    self._detection_active = False
                    self._detection_start_time = None
                    self.get_logger().info(
                        f"⏰ 检测窗口结束（{elapsed:.0f}s > {self.detection_window_s:.0f}s）")

            if self._detection_active:
                # 窗口内：正常检测 + 发布
                detected, flag, qr_data, bbox = self.process_frame(frame)

                # 边沿触发：仅在QR码从未出现变为出现时发布一次
                if detected and flag >= 2 and self._prev_flag != flag:
                    self.flag_pub.publish(UInt8(data=flag))
                    self.qr_flag_pub.publish(UInt8(data=flag))
                    # 非阻塞延时 0.3s 后发送 0 复位 view_cmd（仅复位主话题）
                    self._pending_results.append((time.time() + 0.3, 0))
                    self.get_logger().info(
                        f"↑ QR码出现！ 内容=\"{qr_data}\" → 发布 flag={flag}")
                    # 检测到一次后立即关闭窗口，回到等待状态
                    self._detection_active = False
                    self._detection_start_time = None
                    self.get_logger().info("✅ 已检测到QR码 → 关闭检测窗口，等待下次遮挡信号")

                self._prev_flag = flag

                if self.enable_display:
                    result = self._draw_visualization(frame, flag, qr_data, bbox)
                    display = cv2.resize(result, None, fx=self.display_scale, fy=self.display_scale)
                    cv2.imshow(f"{self.cam.cam_name}_result", display)
            else:
                if self.enable_display:
                    display = cv2.resize(frame, None, fx=self.display_scale, fy=self.display_scale)
                    cv2.imshow(f"{self.cam.cam_name}_result", display)

            if self.enable_display and cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

            rclpy.spin_once(self, timeout_sec=0.01)

    def shutdown(self):
        if hasattr(self, '_stderr_fd'):
            os.dup2(self._stderr_fd, 2)
            os.close(self._stderr_fd)
        if hasattr(self, '_devnull'):
            os.close(self._devnull)
        self.cam.close_uvc_camera()
        if self.enable_display:
            cv2.destroyAllWindows()
        self.get_logger().info(f"{self.cam.cam_name} 已关闭")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = QiyuQRNode(
            node_name="qiyuQR_cam",
            topic_name="/camera/view_cmd"
        )
        node.run()
    except RuntimeError as e:
        rclpy.logging.get_logger("qiyuQR").error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
