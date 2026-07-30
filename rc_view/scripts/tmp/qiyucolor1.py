#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
亮度条纹检测节点 - 从USB摄像头采集图像，检测画面中高亮横向/纵向灯带，
发布 occupancy flag。边沿触发：仅在条纹出现时发送一次，避免重复发布。

flag 含义: 0=无目标, 2=横向亮带, 3=纵向亮带
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from allcamera import uvc_cam
import cv2
import numpy as np
import os
import time


# ── 亮度条纹检测参数 ──────────────────────────────────
BRIGHTNESS_THRESHOLD = 200   # 亮度阈值 (0-255)
MIN_STRIPE_LENGTH    = 180  # 最小条纹长度 (pixels)
ASPECT_RATIO         = 1.70  # 宽高比阈值

# ── Gamma 校正参数 (降低曝光度) ─────────────────────────
GAMMA_DEFAULT = 1.8   # gamma > 1 压暗画面，降低曝光度；=1 不变；<1 提亮

# ── 通用参数 ─────────────────────────────────────────────
MORPH_KERNEL_OPEN  = (5, 5)
MORPH_KERNEL_CLOSE = (7, 7)
MIN_AREA_RATIO     = 0.05

# ── 遮挡触发检测窗口参数 ──────────────────────────────
DEFAULT_DETECTION_WINDOW_S = 60.0   # 收到遮挡信号后持续检测的时长（秒）


class QiyuColorNode(Node):
    """通过亮度阈值检测横向/纵向亮带并发布flag"""

    def __init__(self, device_path: int, node_name: str, topic_name: str):
        super().__init__(node_name)

        self.flag_pub = self.create_publisher(UInt8, topic_name, 10)

        # ── ROS2 参数 ──────────────────────────────────
        self.declare_parameter('detection_window_s', DEFAULT_DETECTION_WINDOW_S)
        self.detection_window_s = self.get_parameter('detection_window_s').value

        self.declare_parameter('gamma', GAMMA_DEFAULT)
        self.gamma = self.get_parameter('gamma').value
        self._gamma_lut = self._build_gamma_lut(self.gamma)

        self.cam = uvc_cam(
            device_path=device_path,
            name=node_name,
            funcation=0,
            width=640,
            height=480,
            fps=30
        )

        if not self.cam.isOpened():
            self.get_logger().error(f"相机 {node_name} 打开失败，设备索引={device_path}")
            raise RuntimeError(f"Camera open failed: {node_name}")

        self.morph_kernel_open  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_OPEN)
        self.morph_kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_CLOSE)

        self.min_area_pixels = int(
            self.cam.cam_height * self.cam.cam_width * MIN_AREA_RATIO)
        self.total_pixels = self.cam.cam_height * self.cam.cam_width

        # ── 边沿触发：仅在条纹首次出现时发送一次 ────
        self._prev_flag = 0

        # ── 遮挡触发检测窗口 ──────────────────────────
        self._detection_active = False          # 当前是否在检测窗口内
        self._detection_start_time = None       # 窗口起始时间戳

        # 订阅 move_base_of_yolo 的遮挡信号
        self.sub_obstruction = self.create_subscription(
            UInt8, "/camera/view_sig",
            self._obstruction_callback, 10)

        self.get_logger().info(
            f"{node_name} 启动 | 亮度>{BRIGHTNESS_THRESHOLD} "
            f"最小长度>{MIN_STRIPE_LENGTH}px | Gamma={self.gamma:.2f} | 边沿触发模式 | "
            f"检测窗口={self.detection_window_s:.0f}s（由遮挡信号触发）")

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        """构建 Gamma 校正查找表。

        Gamma > 1: 压暗画面（降低曝光度）
        Gamma = 1: 不变
        Gamma < 1: 提亮画面

        公式: output = 255 * (input/255) ^ (1/gamma)
        """
        inv_gamma = 1.0 / gamma
        lut = np.array([
            np.clip(pow(i / 255.0, inv_gamma) * 255.0, 0, 255)
            for i in range(256)
        ], dtype=np.uint8)
        return lut

    def _detect_bright_stripes(self, gray_img):
        """检测亮度条纹，返回 (h_triggered, v_triggered, h_area, v_area, h_mask, v_mask)"""
        _, th = cv2.threshold(gray_img, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.morph_kernel_open)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.morph_kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            th, connectivity=8)

        h_mask = np.zeros_like(th)
        v_mask = np.zeros_like(th)

        for i in range(1, num_labels):
            x, y, w, h_box, area = stats[i]
            if area < self.min_area_pixels:
                continue
            # 横向条纹: 宽度 > MIN_STRIPE_LENGTH 且 宽高比 > ASPECT_RATIO
            if w > MIN_STRIPE_LENGTH and w > h_box * ASPECT_RATIO:
                h_mask[labels == i] = 255
            # 纵向条纹: 高度 > MIN_STRIPE_LENGTH 且 高宽比 > ASPECT_RATIO
            elif h_box > MIN_STRIPE_LENGTH and h_box > w * ASPECT_RATIO:
                v_mask[labels == i] = 255

        h_area = cv2.countNonZero(h_mask)
        v_area = cv2.countNonZero(v_mask)
        return h_area > 0, v_area > 0, h_area, v_area, h_mask, v_mask

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
        """对一帧进行处理，返回 (flag, h_area, v_area, h_mask, v_mask, gamma_frame)"""
        # Gamma 校正：降低曝光度（gamma > 1 压暗高光区域）
        gamma_frame = cv2.LUT(bgr_frame, self._gamma_lut)
        gray = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2GRAY)

        h_ok, v_ok, h_area, v_area, h_mask, v_mask = \
            self._detect_bright_stripes(gray)

        # 合并: 0=无, 2=横向亮带, 3=纵向亮带 (同时存在时纵向优先)
        if v_ok:
            flag = 3
        elif h_ok:
            flag = 2
        else:
            flag = 0

        return flag, h_area, v_area, h_mask, v_mask, gamma_frame

    def _draw_visualization(self, frame, h_mask, v_mask, flag, h_area, v_area):
        """结果图(原图+轮廓+标注) + 掩码图(横向=青, 纵向=洋红)"""
        status_map = {0: "NONE", 2: "H-STRIPE", 3: "V-STRIPE"}
        status_text = status_map.get(flag, "?")
        flag_color = (
            (0, 255, 255) if flag == 2 else
            (255, 0, 255) if flag == 3 else
            (128, 128, 128)
        )

        # ── 结果图：原图 + 轮廓 + 标注 ──
        result = frame.copy()
        h_cnts, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, h_cnts, -1, (255, 255, 0), 2)
        v_cnts, _ = cv2.findContours(v_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, v_cnts, -1, (255, 0, 255), 2)

        cv2.putText(result, f"Flag: {flag} ({status_text})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, flag_color, 2)
        cv2.putText(result, f"Gamma: {self.gamma:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        cv2.putText(result, f"H-Stripe: {h_area} px", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(result, f"V-Stripe: {v_area} px", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        # ── 掩码图：横向=青，纵向=洋红 ──
        mask_bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        mask_bgr[h_mask > 0] = (0, 255, 255)
        mask_bgr[v_mask > 0] = (255, 0, 255)

        return result, mask_bgr

    def run(self):
        """主循环"""
        while rclpy.ok():
            # 屏蔽 libjpeg 的 "Corrupt JPEG data" C 层 stderr 输出
            _stderr_fd = os.dup(2)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, 2)
            os.close(_devnull)
            try:
                ret, frame = self.cam.read_one_frame()
            finally:
                os.dup2(_stderr_fd, 2)
                os.close(_stderr_fd)
            if not ret:
                self.get_logger().warn(
                    f"{self.cam.cam_name} 读取失败",
                    throttle_duration_sec=5)
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

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
                flag, h_area, v_area, h_mask, v_mask, gamma_frame = self.process_frame(frame)

                # 边沿触发：仅在条纹从未出现变为出现时发布一次
                if flag in (2, 3) and self._prev_flag != flag:
                    self.flag_pub.publish(UInt8(data=flag))
                    direction = "横向亮带" if flag == 2 else "纵向亮带"
                    area = h_area if flag == 2 else v_area
                    self.get_logger().info(
                        f"↑ {direction}出现！ area={area}px → 发布 flag={flag}")
                    # 检测到一次后立即关闭窗口，回到等待状态
                    self._detection_active = False
                    self._detection_start_time = None
                    self.get_logger().info("✅ 已检测到条纹 → 关闭检测窗口，等待下次遮挡信号")

                self._prev_flag = flag

                # 可视化显示（使用 gamma 校正后的帧）
                result, mask_bgr = self._draw_visualization(
                    gamma_frame, h_mask, v_mask, flag, h_area, v_area)
                cv2.imshow(f"{self.cam.cam_name}_result", result)
                cv2.imshow(f"{self.cam.cam_name}_mask", mask_bgr)
            else:
                # 窗口外：显示 gamma 校正后的画面，不检测不发布
                gamma_frame = cv2.LUT(frame, self._gamma_lut)
                cv2.imshow(f"{self.cam.cam_name}_result", gamma_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

            rclpy.spin_once(self, timeout_sec=0.01)

    def shutdown(self):
        self.cam.close_uvc_camera()
        cv2.destroyAllWindows()
        self.get_logger().info(f"{self.cam.cam_name} 已关闭")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = QiyuColorNode(
            device_path=2,
            node_name="qiyu2_publisher",
            topic_name="/camera/view_cmd"
        )
        node.run()
    except RuntimeError as e:
        rclpy.logging.get_logger("qiyucolor2").error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
