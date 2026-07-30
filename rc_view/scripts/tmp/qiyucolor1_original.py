#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域颜色检测节点 - 从USB摄像头采集图像，通过HSV颜色过滤同时检测黄色和紫色，
发布 occupancy flag。边沿触发：仅在颜色出现时发送一次，避免重复发布。

flag 含义: 0=无目标, 2=黄色, 3=紫色
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from rc_view.scripts.cam_of_ros2.allcamera import uvc_cam
import cv2
import numpy as np


# ── HSV 阈值 (黄色) ──────────────────────────────────────
Y_HUE_MIN   = 10
Y_HUE_MAX   = 40
Y_SATU_MIN  = 90
Y_SATU_MAX  = 255
Y_VAL_MIN   = 1
Y_VAL_MAX   = 255
Y_TRIGGER   = 0.3   # 触发比例

# ── HSV 阈值 (紫色) ──────────────────────────────────────
P_HUE_MIN   = 125
P_HUE_MAX   = 155
P_SATU_MIN  = 50
P_SATU_MAX  = 255
P_VAL_MIN   = 50
P_VAL_MAX   = 255
P_TRIGGER   = 0.3

# ── 通用参数 ─────────────────────────────────────────────
MORPH_KERNEL_OPEN  = (5, 5)
MORPH_KERNEL_CLOSE = (7, 7)
MIN_AREA_RATIO     = 0.05


class QiyuColorNode(Node):
    """通过HSV颜色过滤检测区域并发布flag"""

    def __init__(self, device_path: int, node_name: str, topic_name: str):
        super().__init__(node_name)

        self.flag_pub = self.create_publisher(UInt8, topic_name, 10)

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

        # ── 边沿触发：仅在颜色首次出现时发送一次 ────
        self._prev_flag = 0

        self.get_logger().info(
            f"{node_name} 启动 | 最小面积={self.min_area_pixels}px | "
            f"黄色触发={Y_TRIGGER} 紫色触发={P_TRIGGER} | 边沿触发模式")

    def _detect_color(self, hsv_img, hue_min, hue_max, sat_min, sat_max,
                      val_min, val_max, trigger_ratio) -> tuple:
        """检测一种颜色，返回 (is_triggered, area, mask)"""
        th = cv2.inRange(hsv_img,
                         (hue_min, sat_min, val_min),
                         (hue_max, sat_max, val_max))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.morph_kernel_open)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.morph_kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            th, connectivity=8)
        mask = np.zeros_like(th)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area_pixels:
                mask[labels == i] = 255

        area = cv2.countNonZero(mask)
        triggered = (area / self.total_pixels) > trigger_ratio
        return triggered, area, mask

    def process_frame(self, bgr_frame: np.ndarray) -> tuple:
        """对一帧进行处理，返回 (flag, y_area, p_area, y_mask, p_mask)"""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv = cv2.merge([h, s, v])

        y_ok, y_area, y_mask = self._detect_color(
            hsv, Y_HUE_MIN, Y_HUE_MAX, Y_SATU_MIN, Y_SATU_MAX,
            Y_VAL_MIN, Y_VAL_MAX, Y_TRIGGER)

        p_ok, p_area, p_mask = self._detect_color(
            hsv, P_HUE_MIN, P_HUE_MAX, P_SATU_MIN, P_SATU_MAX,
            P_VAL_MIN, P_VAL_MAX, P_TRIGGER)

        # 合并: 0=无, 2=黄色, 3=紫色 (同时存在时紫色优先)
        if p_ok:
            flag = 3
        elif y_ok:
            flag = 2
        else:
            flag = 0

        return flag, y_area, p_area, y_mask, p_mask

    def _draw_visualization(self, frame, y_mask, p_mask, flag, y_area, p_area):
        """结果图(原图+轮廓+标注) + 掩码图(黄色=青, 紫色=洋红)"""
        status_map = {0: "NONE", 2: "YELLOW", 3: "PURPLE"}
        status_text = status_map.get(flag, "?")
        flag_color = (
            (0, 255, 255) if flag == 2 else
            (255, 0, 255) if flag == 3 else
            (128, 128, 128)
        )

        # ── 结果图：原图 + 轮廓 + 标注 ──
        result = frame.copy()
        y_cnts, _ = cv2.findContours(y_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, y_cnts, -1, (255, 255, 0), 2)
        p_cnts, _ = cv2.findContours(p_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, p_cnts, -1, (255, 0, 255), 2)

        cv2.putText(result, f"Flag: {flag} ({status_text})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, flag_color, 2)
        cv2.putText(result, f"Yellow: {y_area} px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(result, f"Purple: {p_area} px", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        # ── 掩码图：黄=青，紫=洋红 ──
        mask_bgr = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        mask_bgr[y_mask > 0] = (0, 255, 255)
        mask_bgr[p_mask > 0] = (255, 0, 255)

        return result, mask_bgr

    def run(self):
        """主循环"""
        while rclpy.ok():
            ret, frame = self.cam.read_one_frame()
            if not ret:
                self.get_logger().warn(
                    f"{self.cam.cam_name} 读取失败",
                    throttle_duration_sec=5)
                rclpy.spin_once(self, timeout_sec=0.05)
                continue

            flag, y_area, p_area, y_mask, p_mask = self.process_frame(frame)

            # 边沿触发：仅在颜色从未出现变为出现时发布一次
            if flag in (2, 3) and self._prev_flag != flag:
                self.flag_pub.publish(UInt8(data=flag))
                color = "黄色" if flag == 2 else "紫色"
                area = y_area if flag == 2 else p_area
                ratio = area / self.total_pixels
                self.get_logger().info(
                    f"↑ {color}出现！ area={area}px ratio={ratio:.3f} → 发布 flag={flag}")
            elif flag != self._prev_flag:
                self.get_logger().info(
                    f"flag 变化: {self._prev_flag}→{flag} "
                    f"(黄: {y_area}px 紫: {p_area}px)",
                    throttle_duration_sec=2.0)

            self._prev_flag = flag

            # 可视化显示
            result, mask_bgr = self._draw_visualization(
                frame, y_mask, p_mask, flag, y_area, p_area)
            cv2.imshow(f"{self.cam.cam_name}_result", result)
            cv2.imshow(f"{self.cam.cam_name}_mask", mask_bgr)
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
