#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
颜色检测节点 — 从 USB 单目相机采集图像，在各自 ROI 内同时检测绿色/紫色/蓝色，
超过阈值时向 /camera/view_cmd 发布对应指令，边沿触发（仅发送一次，
延时后补发 0）。多色同时超过阈值时只发送面积（ratio）最大者。

绿色 → 6, 紫色 → 7, 蓝色 → 8；补发 → 0

用法：
  ros2 run rc_view green_detector
  ros2 run rc_view green_detector --ros-args -p device_path:=0
  ros2 run rc_view green_detector --ros-args -p green_ratio:=0.3 purple_ratio:=0.3
  ros2 run rc_view green_detector --ros-args -p blue_enable:=false
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import UInt8
from rc_view.scripts.cam_of_ros2.allcamera import uvc_cam
from nav_msgs.msg import Odometry
import cv2
import numpy as np


# ── 默认 HSV 阈值 ─────────────────────────────────────
_GREEN_HSV  = (40, 85, 50, 255, 50, 255)   # H_min, H_max, S_min, S_max, V_min, V_max
_PURPLE_HSV = (120, 155, 50, 255, 50, 255)
_BLUE_HSV = (95, 120, 60, 255, 80, 255)

# ── 默认触发参数 ─────────────────────────────────────
_GREEN_RATIO  = 0.60
_PURPLE_RATIO = 0.60
_BLUE_RATIO = 0.60
_DEFAULT_RADAR_Z_THRESHOLD = 0.50
_ODOM_TOPIC = '/Odometry'
_DEFAULT_FOLLOW_DELAY = 0.3
_DEFAULT_COOLDOWN_S   = 3.0  # 同种颜色再次触发的冷却时间

# ── 形态学参数 ───────────────────────────────────────
MORPH_KERNEL_OPEN  = (3, 3)
MORPH_KERNEL_CLOSE = (5, 5)
MIN_AREA_RATIO     = 0.001


# ── 每个颜色的配置模板 ────────────────────────────────
_COLOR_CONFIGS = {
    'green':  {'cmd': 6, 'hsv': _GREEN_HSV,  'ratio': _GREEN_RATIO,  'label': '绿色'},
    'purple': {'cmd': 7, 'hsv': _PURPLE_HSV, 'ratio': _PURPLE_RATIO, 'label': '紫色'},
    'blue': {'cmd': 8, 'hsv': _BLUE_HSV, 'ratio': _BLUE_RATIO, 'label': '蓝色'},
}

# GUI 显示色（BGR）
_DRAW_COLORS = {
    'green':  (0, 255, 0),
    'purple': (255, 0, 255),
    'blue': (255, 0, 0),
}


class ColorDetectorNode(Node):
    """在各自 ROI 内检测多种颜色，边沿触发发布指令到 /camera/view_cmd"""

    def __init__(self):
        super().__init__('color_detector')

        # ── 相机参数 ──────────────────────────────────
        self.declare_parameter('device_path', 6)
        self.declare_parameter('cam_width', 640)
        self.declare_parameter('cam_height', 480)
        self.declare_parameter('cam_fps', 30)

        # ── 延时参数 ──────────────────────────────────
        self.declare_parameter('follow_delay', _DEFAULT_FOLLOW_DELAY)
        self.declare_parameter('cooldown_s', _DEFAULT_COOLDOWN_S)
        self.declare_parameter('min_area_ratio', MIN_AREA_RATIO)
        self.declare_parameter('show_gui', True)

        # ── 雷达门控参数 ──────────────────────────────
        self.declare_parameter('radar_z_threshold', _DEFAULT_RADAR_Z_THRESHOLD)

        # ── 每种颜色的参数 ────────────────────────────
        for name, cfg in _COLOR_CONFIGS.items():
            h_min, h_max, s_min, s_max, v_min, v_max = cfg['hsv']
            self.declare_parameter(f'{name}_enable', True)
            self.declare_parameter(f'{name}_hue_min', h_min)
            self.declare_parameter(f'{name}_hue_max', h_max)
            self.declare_parameter(f'{name}_satu_min', s_min)
            self.declare_parameter(f'{name}_satu_max', s_max)
            self.declare_parameter(f'{name}_val_min',  v_min)
            self.declare_parameter(f'{name}_val_max',  v_max)
            self.declare_parameter(f'{name}_roi_x', 0.0)
            self.declare_parameter(f'{name}_roi_y', 0.0)
            self.declare_parameter(f'{name}_roi_w', 1.0)
            self.declare_parameter(f'{name}_roi_h', 1.0)
            self.declare_parameter(f'{name}_ratio', cfg['ratio'])

        # ── 读取相机参数 ──────────────────────────────
        device_path = self.get_parameter('device_path').value
        cam_w = self.get_parameter('cam_width').value
        cam_h = self.get_parameter('cam_height').value
        cam_fps = self.get_parameter('cam_fps').value
        self.follow_delay = self.get_parameter('follow_delay').value
        self.cooldown_s = self.get_parameter('cooldown_s').value
        self.min_area_ratio = self.get_parameter('min_area_ratio').value
        self.show_gui = self.get_parameter('show_gui').value

        # ── 发布者：/camera/view_cmd ──────────────────
        self.cmd_pub = self.create_publisher(UInt8, '/camera/view_cmd', 10)

        # ── 订阅雷达里程计，获取机器人 z 坐标 ──────────
        self._radar_z = -999.0
        self._odom_sub = self.create_subscription(
            Odometry, _ODOM_TOPIC, self._odom_callback, 10)

        # ── 打开相机 ──────────────────────────────────
        self.cam = uvc_cam(
            device_path=device_path,
            name='color_detector',
            funcation=0,
            width=cam_w,
            height=cam_h,
            fps=cam_fps,
        )

        if not self.cam.isOpened():
            self.get_logger().error(f'无法打开摄像头 device={device_path}')
            raise RuntimeError(f'Camera open failed: {device_path}')

        self.cam_w = int(self.cam.cam_width)
        self.cam_h = int(self.cam.cam_height)

        # ── 形态学核 ──────────────────────────────────
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_OPEN)
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, MORPH_KERNEL_CLOSE)

        # ── 雷达门控 ──────────────────────────────────
        self._radar_z_threshold = self.get_parameter('radar_z_threshold').value

        # ── 每颜色独立边沿触发状态 ──────────────────────
        self._armed = {name: True for name in _COLOR_CONFIGS}  # 每种颜色是否可触发
        self._cooldown_until = {name: None for name in _COLOR_CONFIGS}  # 冷却截止时间
        self._last_winner = None          # 上一帧的胜出颜色名，None = 无
        self._zero_deadline = None        # 补发 0 的截止时间，None = 无

        self.get_logger().info(
            f'颜色检测节点启动 | device={device_path} {self.cam_w}x{self.cam_h}@{cam_fps}Hz\n'
            f'  绿色→6, 紫色→7, 蓝色→8, 补发→0  '
            f'延时={self.follow_delay:.1f}s  冷却={self.cooldown_s:.0f}s\n'
            f'  多色同时触发时 → 面积最大者胜出\n'
            f'  雷达门控: z>{self._radar_z_threshold}m 才启动颜色识别 '
            f'(话题 {_ODOM_TOPIC})'
        )

        # ── Timer 驱动主循环 ──────────────────────────
        period = 1.0 / max(cam_fps, 1)
        self._timer = self.create_timer(period, self._process)

    # ═══════════════════════════════════════════════════
    #  参数读取（每帧动态刷新）
    # ═══════════════════════════════════════════════════

    def _read_color_params(self, name):
        """读取一种颜色的所有参数，返回 dict"""
        enabled = self.get_parameter(f'{name}_enable').value
        hsv = (
            self.get_parameter(f'{name}_hue_min').value,
            self.get_parameter(f'{name}_hue_max').value,
            self.get_parameter(f'{name}_satu_min').value,
            self.get_parameter(f'{name}_satu_max').value,
            self.get_parameter(f'{name}_val_min').value,
            self.get_parameter(f'{name}_val_max').value,
        )
        roi = (
            self.get_parameter(f'{name}_roi_x').value,
            self.get_parameter(f'{name}_roi_y').value,
            self.get_parameter(f'{name}_roi_w').value,
            self.get_parameter(f'{name}_roi_h').value,
        )
        ratio_th = self.get_parameter(f'{name}_ratio').value
        return enabled, hsv, roi, ratio_th

    @staticmethod
    def _roi_pixels(roi, cam_w, cam_h):
        """归一化 ROI → 像素坐标 + 像素数"""
        rx, ry, rw, rh = roi
        x1 = int(rx * cam_w)
        y1 = int(ry * cam_h)
        x2 = int((rx + rw) * cam_w)
        y2 = int((ry + rh) * cam_h)
        pixels = max((x2 - x1) * (y2 - y1), 1)
        return x1, y1, x2, y2, pixels

    # ═══════════════════════════════════════════════════
    #  颜色检测
    # ═══════════════════════════════════════════════════

    def _detect_color(self, hsv_roi, hsv_thresholds, min_area_px):
        """在 ROI 的 HSV 图像中检测颜色，返回 (ratio, mask)"""
        h_min, h_max, s_min, s_max, v_min, v_max = hsv_thresholds
        mask = cv2.inRange(hsv_roi, (h_min, s_min, v_min), (h_max, s_max, v_max))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                clean_mask[labels == i] = 255

        px = cv2.countNonZero(clean_mask)
        roi_px = hsv_roi.shape[0] * hsv_roi.shape[1]
        ratio = px / max(roi_px, 1)
        return ratio, clean_mask

    # ═══════════════════════════════════════════════════
    #  雷达里程计回调 & 门控
    # ═══════════════════════════════════════════════════

    def _odom_callback(self, msg: Odometry):
        """更新雷达 z 坐标"""
        self._radar_z = msg.pose.pose.position.z

    @property
    def _detection_enabled(self) -> bool:
        """颜色识别是否允许：雷达 z > threshold"""
        return self._radar_z > self._radar_z_threshold

    def _check_detection_gate(self, now) -> bool:
        """
        检测雷达门控条件。

        若 z 跌落至 <=threshold，清空挂起的补发 0。
        返回当前是否允许颜色识别。
        """
        enabled = self._detection_enabled
        if not enabled and self._zero_deadline is not None:
            self._zero_deadline = None
        return enabled

    # ═══════════════════════════════════════════════════
    #  主循环
    # ═══════════════════════════════════════════════════

    def _process(self):
        """Timer 回调：读取帧 → 检测三种颜色 → 发布"""
        # 动态参数
        self.follow_delay = self.get_parameter('follow_delay').value
        self.cooldown_s = self.get_parameter('cooldown_s').value
        self.min_area_ratio = self.get_parameter('min_area_ratio').value
        self.show_gui = self.get_parameter('show_gui').value
        self._radar_z_threshold = self.get_parameter('radar_z_threshold').value

        now = self.get_clock().now()
        if not self._check_detection_gate(now):
            return  # 雷达 z 不足，跳过本帧

        ret, frame = self.cam.read_one_frame()
        if not ret:
            self.get_logger().warning('相机读取失败', throttle_duration_sec=5)
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 第一遍：检测所有颜色，收集候选 ──────────────
        gui_data = []             # [(name, x1,y1,x2,y2, ratio, mask, triggered)]
        candidates = []           # [(name, ratio, cmd)] 超过阈值的颜色

        for name in _COLOR_CONFIGS:
            enabled, hsv_th, roi, ratio_th = self._read_color_params(name)
            if not enabled:
                continue

            x1, y1, x2, y2, roi_px = self._roi_pixels(roi, self.cam_w, self.cam_h)
            min_area_px = int(roi_px * self.min_area_ratio)

            roi_hsv = hsv[y1:y2, x1:x2]
            ratio, mask = self._detect_color(roi_hsv, hsv_th, min_area_px)

            triggered = ratio > ratio_th
            gui_data.append((name, x1, y1, x2, y2, ratio, mask, triggered))

            if triggered:
                candidates.append((name, ratio, _COLOR_CONFIGS[name]['cmd']))
            else:
                # 颜色消失 → 重新装填该颜色
                if not self._armed[name]:
                    self.get_logger().info(
                        f'[{_COLOR_CONFIGS[name]["label"]}] ↓ 消失 → 重新装填')
                self._armed[name] = True

        # ── 补发 0 的延迟检查 ──────────────────────────
        if self._zero_deadline is not None and now >= self._zero_deadline:
            self.cmd_pub.publish(UInt8(data=0))
            self.get_logger().info('→ 补发 view_cmd=0')
            self._zero_deadline = None

        # ── 确定当前胜者 ──────────────────────────────
        winner = None   # 当前帧胜者 (name or None)
        if candidates:
            winner = max(candidates, key=lambda x: x[1])[0]

        # ── 边沿触发：胜者变化 + 已装填 + 冷却已过 ──────
        if (winner is not None
                and winner != self._last_winner
                and self._armed[winner]
                and self._check_cooldown(winner, now)):
            winner_cmd = _COLOR_CONFIGS[winner]['cmd']
            self.cmd_pub.publish(UInt8(data=winner_cmd))
            self._armed[winner] = False
            self._cooldown_until[winner] = now + Duration(seconds=self.cooldown_s)
            self._zero_deadline = now + Duration(seconds=self.follow_delay)

            labels = ', '.join(
                f'{_COLOR_CONFIGS[n]["label"]}={r:.3f}' for n, r, _ in candidates)
            cd = self.cooldown_s
            self.get_logger().info(
                f'↑ 胜者切换: {self._last_winner or "无"} → {_COLOR_CONFIGS[winner]["label"]}  '
                f'候选: [{labels}]  cmd={winner_cmd} '
                f'(补发0={self.follow_delay:.1f}s  冷却={cd:.0f}s)')

        self._last_winner = winner

        # ── GUI 显示 ──────────────────────────────────
        if self.show_gui:
            self._draw_gui(frame, gui_data, winner)

    # ═══════════════════════════════════════════════════
    #  GUI
    # ═══════════════════════════════════════════════════

    def _check_cooldown(self, name, now):
        """返回 True 表示该颜色冷却已过（可触发）"""
        deadline = self._cooldown_until.get(name)
        return deadline is None or now >= deadline

    def _draw_gui(self, frame, gui_data, winner=None):
        """可视化：画面 + ROI 框 + 合成掩码图"""
        display = frame.copy()
        y_offset = 30

        for (name, x1, y1, x2, y2, ratio, mask, triggered) in gui_data:
            color = _DRAW_COLORS[name]
            label = _COLOR_CONFIGS[name]['label']
            cmd = _COLOR_CONFIGS[name]['cmd']

            # ROI 框 — 胜出颜色用其色调，其他触发候选用灰白，未触发用暗红
            if name == winner:
                box_color = color
                thickness = 3
            elif triggered:
                box_color = (200, 200, 200)
                thickness = 2
            else:
                box_color = (0, 0, 200)
                thickness = 1
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, thickness)

            # 状态文字
            if name == winner:
                status = 'WINNER'
                st_color = (0, 255, 255)
            elif triggered:
                status = 'CANDIDATE'
                st_color = (200, 200, 200)
            else:
                status = 'ARMED' if self._armed.get(name, True) else 'WAIT'
                st_color = color
            cv2.putText(display, f'{label}({cmd}): {status}  ratio={ratio:.3f}',
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, st_color, 2)
            y_offset += 22

        # ── 合成掩码图：每种颜色对应其 BGR 色调 ──────
        mask_canvas = np.zeros((self.cam_h, self.cam_w, 3), dtype=np.uint8)
        # 叠加顺序：绿 → 紫 → 黄（后面的可覆盖前面的）
        for name in ['blue', 'purple', 'green']:
            item = next((d for d in gui_data if d[0] == name), None)
            if item is None:
                continue
            _, x1, y1, x2, y2, _, mask, _ = item
            color = _DRAW_COLORS[name]
            # mask 是 ROI 尺寸，放入全帧 ROI 位置
            roi_h, roi_w = y2 - y1, x2 - x1
            if mask.shape[0] != roi_h or mask.shape[1] != roi_w:
                mask = cv2.resize(mask, (roi_w, roi_h))
            mask_canvas[y1:y2, x1:x2][mask > 0] = color

        cv2.imshow('Masks', mask_canvas)

        cv2.imshow('Color Detector', display)
        cv2.waitKey(1)

    def shutdown(self):
        self.cam.close_uvc_camera()
        cv2.destroyAllWindows()
        self.get_logger().info('颜色检测节点已关闭')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ColorDetectorNode()
        rclpy.spin(node)
    except RuntimeError as e:
        rclpy.logging.get_logger('color_detector').error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
