#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qiyuQR_debug.py — QR码检测调试版（无 ROS2 依赖 / 720p + 抗过曝管线）

针对"亮屏黑码白底、距离0.35m、画面全白"的问题：
  - 硬件曝光猛烈压低（默认 exposure=3，即 300µs 极短曝光）
  - 4 种自动曝光关闭策略（兼容不同 UVC 摄像头）
  - 实时饱和像素比例监测 + 直方图
  - 按 'e' 一键自动扫描最佳曝光值（降到饱和率<5% 为止）

预处理管线（软件层，用于微调对比度）：
  1. 🔧 硬件曝光    — MANUAL 模式 + 极短曝光时间
  2. 🔆 Gamma 校正  — γ>1 压缩高光
  3. 📉 亮度压缩    — 线性缩放
  4. 🎨 CLAHE       — LAB L 通道自适应直方图均衡化
  5. 🗜️  色调映射    — Drago/Reinhard/Mantiuk

操作:
  键盘:
    's'         — 模拟遮挡信号，启动检测窗口
    'e'         — 自动扫描最佳硬件曝光值
    'r'         — 重置曝光为 3（极短）
    'h'         — 切换直方图窗口
    'd'         — 诊断：打印当前帧饱和率 + 曝光实际值
    'p'         — 切换全部预处理
    '1'-'4'     — 切换各预处理模块
    '←' '→'     — 手动微调硬件曝光 (±1, 加 Shift=±10)
    '↑' '↓'     — 调整当前焦点参数
    '`'         — 切换调整焦点
    'ESC' / 'q' — 退出
"""

import cv2
import numpy as np
import time
import argparse
import sys
import subprocess
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════════
#  默认参数
# ═══════════════════════════════════════════════════════════
DEFAULT_DEVICE         = 2
DEFAULT_WIDTH          = 1280
DEFAULT_HEIGHT         = 720
DEFAULT_FPS            = 30
DEFAULT_DETECTION_WINDOW_S = 60.0

# ── 硬件曝光（关键！亮屏必须极低） ──────────────────
# V4L2 的 V4L2_CID_EXPOSURE_ABSOLUTE 单位 = 100µs
#  值 1   = 100µs  (极短，适合拍摄亮屏)
#  值 10  = 1ms
#  值 100 = 10ms
#  值 1000= 100ms
# 多数 UVC 摄像头最小值=1~3，最大值=10000+
# 亮屏QR码建议从 3 开始，按需上调
DEFAULT_EXPOSURE_VALUE = 3              # 极短曝光（300µs），拍摄亮屏

# ── 曝光自动扫描 ────────────────────────────────────
AUTOEXP_SWEEP_START    = 1              # 扫描起点（最短曝光）
AUTOEXP_SWEEP_END      = 200            # 扫描终点（如果一直过曝也不用再往上）
AUTOEXP_TARGET_SAT     = 0.05           # 目标：饱和像素 < 5%
AUTOEXP_CAPTURE_FRAMES = 2              # 每个曝光值采几帧取平均

# ── Gamma 校正 ────────────────────────────────────────
DEFAULT_GAMMA          = 2.2
GAMMA_STEP             = 0.2
GAMMA_MIN, GAMMA_MAX   = 0.4, 5.0

# ── 亮度线性压缩 ──────────────────────────────────────
DEFAULT_BRIGHTNESS_SCALE = 0.70
BRIGHTNESS_STEP        = 0.05
BRIGHTNESS_MIN, BRIGHTNESS_MAX = 0.10, 1.00

# ── CLAHE ─────────────────────────────────────────────
DEFAULT_CLAHE_CLIP     = 2.0
DEFAULT_CLAHE_GRID     = 8
CLAHE_CLIP_STEP        = 0.5
CLAHE_CLIP_MIN, CLAHE_CLIP_MAX = 0.5, 10.0

# ── 色调映射 ──────────────────────────────────────────
class TonemapMethod(Enum):
    DRAGO    = "drago"
    REINHARD = "reinhard"
    MANTIUK  = "mantiuk"

DEFAULT_TONEMAP_METHOD = TonemapMethod.DRAGO
DEFAULT_TONEMAP_GAMMA  = 1.0
TONEMAP_GAMMA_STEP     = 0.1
TONEMAP_GAMMA_MIN, TONEMAP_GAMMA_MAX = 0.5, 3.0

# ── 直方图 ────────────────────────────────────────────
HIST_BINS = 256
HIST_H    = 200    # 直方图显示高度（像素）


def _timestamp():
    return datetime.now().strftime("[%H:%M:%S]")


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ═══════════════════════════════════════════════════════════
#  QiyuQRDebug
# ═══════════════════════════════════════════════════════════

class QiyuQRDebug:
    """无 ROS2 依赖的 QR 码检测器，硬件曝光优先 + 软件管线辅助"""

    def __init__(self,
                 device_index: int = DEFAULT_DEVICE,
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS,
                 detection_window_s: float = DEFAULT_DETECTION_WINDOW_S,
                 name: str = "qiyuQR_debug",
                 # 预处理开关
                 enable_preprocess: bool = True,
                 enable_gamma: bool = True,
                 enable_brightness: bool = True,
                 enable_clahe: bool = True,
                 enable_tonemap: bool = True,
                 # 预处理参数
                 exposure_value: int = DEFAULT_EXPOSURE_VALUE,
                 gamma: float = DEFAULT_GAMMA,
                 brightness_scale: float = DEFAULT_BRIGHTNESS_SCALE,
                 clahe_clip: float = DEFAULT_CLAHE_CLIP,
                 clahe_grid: int = DEFAULT_CLAHE_GRID,
                 tonemap_method: TonemapMethod = DEFAULT_TONEMAP_METHOD,
                 tonemap_gamma: float = DEFAULT_TONEMAP_GAMMA,
                 ):
        self.name = name
        self.device_index = device_index
        self.detection_window_s = detection_window_s
        self._exposure_available = False  # 运行后诊断确定

        # ── 预处理开关 ──────────────────────────────
        self.enable_preprocess = enable_preprocess
        self.enable_gamma      = enable_gamma
        self.enable_brightness = enable_brightness
        self.enable_clahe      = enable_clahe
        self.enable_tonemap    = enable_tonemap

        # ── 预处理参数 ──────────────────────────────
        self.exposure_value    = exposure_value
        self.gamma             = gamma
        self.brightness_scale  = brightness_scale
        self.clahe_clip        = clahe_clip
        self.clahe_grid        = clahe_grid
        self.tonemap_method    = tonemap_method
        self.tonemap_gamma     = tonemap_gamma

        # ── 参数调整焦点 ────────────────────────────
        self._adjust_focus = "gamma"

        # ── 直方图窗口 ──────────────────────────────
        self._show_histogram = False

        # ── 打开摄像头 ──────────────────────────────
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"摄像头打开失败！device_index={device_index} "
                f"(尝试改为 0 或 1 如果使用内置摄像头)")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"{_timestamp()} {name} 摄像头已打开 | "
              f"分辨率={actual_w:.0f}x{actual_h:.0f} | FPS={actual_fps:.0f}")

        # ── 诊断摄像头能力 ──────────────────────────
        self._diagnose_camera()

        # ── 硬件曝光控制（关键步骤） ────────────────
        self._configure_hardware_exposure()

        # ── QR码检测器 ──────────────────────────────
        self.qr_detector = cv2.QRCodeDetector()

        # ── CLAHE 实例 ──────────────────────────────
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid))

        # ── 色调映射器 ──────────────────────────────
        self._tonemap_instance = None
        self._tonemap_key = None

        # ── Gamma LUT 缓存 ──────────────────────────
        self._gamma_lut = None
        self._gamma_cached_for = None

        # ── 饱和统计（持续更新） ────────────────────
        self._sat_pct = 0.0        # 饱和像素比例 (>250)
        self._mean_val = 0.0       # 灰度均值
        self._gray_hist = None     # 最新直方图

        # ── 边沿触发 ────────────────────────────────
        self._prev_flag = 0

        # ── 遮挡触发检测窗口 ────────────────────────
        self._detection_active = False
        self._detection_start_time = None

        # ── FPS 统计 ────────────────────────────────
        self._fps_t0 = time.time()
        self._fps_counter = 0
        self._fps_display = 0.0

        self._print_status()

    # ═════════════════════════════════════════════════════
    #  摄像头诊断
    # ═════════════════════════════════════════════════════

    def _diagnose_camera(self):
        """打印所有可读的摄像头属性，确认曝光控制能力"""
        props = {
            "CAP_PROP_AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE,
            "CAP_PROP_EXPOSURE":       cv2.CAP_PROP_EXPOSURE,
            "CAP_PROP_GAIN":           cv2.CAP_PROP_GAIN,
            "CAP_PROP_BRIGHTNESS":     cv2.CAP_PROP_BRIGHTNESS,
            "CAP_PROP_CONTRAST":       cv2.CAP_PROP_CONTRAST,
            "CAP_PROP_SATURATION":     cv2.CAP_PROP_SATURATION,
            "CAP_PROP_AUTO_WB":        cv2.CAP_PROP_AUTO_WB,
            "CAP_PROP_WB_TEMPERATURE": cv2.CAP_PROP_WB_TEMPERATURE,
            "CAP_PROP_BACKLIGHT":      cv2.CAP_PROP_BACKLIGHT,
        }
        print(f"{_timestamp()} ── 摄像头属性诊断 ──")
        for name, prop_id in props.items():
            val = self.cap.get(prop_id)
            print(f"{_timestamp()}   {name:30s} = {val:.2f}")

        # 额外尝试 v4l2-ctl 读取（如果可用）
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", str(self.device_index), "--list-ctrls"],
                capture_output=True, text=True, timeout=3)
            if result.returncode == 0 and result.stdout.strip():
                print(f"{_timestamp()} ── v4l2-ctl 可用参数 ──")
                for line in result.stdout.strip().splitlines():
                    if any(kw in line.lower() for kw in
                           ["exposure", "gain", "bright", "auto"]):
                        print(f"{_timestamp()}   {line.strip()}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        print(f"{_timestamp()} ── 诊断结束 ──")

    # ═════════════════════════════════════════════════════
    #  硬件曝光控制 — 亮屏QR核心
    # ═════════════════════════════════════════════════════

    def _configure_hardware_exposure(self):
        """
        关闭自动曝光 + 设置极短曝光。

        策略：多种方式尝试关闭自动曝光（不同摄像头驱动对
        CAP_PROP_AUTO_EXPOSURE 的解读不同），然后设置绝对曝光值。
        """
        print(f"{_timestamp()} 🔧 配置硬件曝光...")

        # ── 策略 1：CAP_PROP_AUTO_EXPOSURE = 0 (manual) ──
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        time.sleep(0.15)

        # ── 策略 2：CAP_PROP_AUTO_EXPOSURE = 1 (也常被解读为manual) ──
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        time.sleep(0.15)

        # ── 策略 3：如果上述不生效，通过 V4L2 直接关闭 ──
        try:
            subprocess.run(
                ["v4l2-ctl", "-d", str(self.device_index),
                 "-c", "auto_exposure=1"],
                capture_output=True, timeout=3)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # ── 设置极短曝光值 ──────────────────────────
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_value)
        time.sleep(0.15)

        # ── 验证 ────────────────────────────────────
        actual_auto = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exp  = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"{_timestamp()}   AUTO_EXPOSURE 实际值 = {actual_auto:.1f}  "
              f"(期望 0 或 1 = 手动)")
        print(f"{_timestamp()}   EXPOSURE 实际值 = {actual_exp:.1f}  "
              f"(设定值 = {self.exposure_value})")

        if actual_exp == self.exposure_value:
            print(f"{_timestamp()}   ✅ 曝光值设置成功")
            self._exposure_available = True
        else:
            print(f"{_timestamp()}   ⚠️ 曝光值未生效！"
                  f" 设定={self.exposure_value} 实际={actual_exp:.0f}")
            print(f"{_timestamp()}   → 可能原因：AUTO_EXPOSURE 未关闭，"
                  f"或驱动不支持手动曝光")
            print(f"{_timestamp()}   → 尝试在终端执行: "
                  f"v4l2-ctl -d {self.device_index} -c auto_exposure=1 "
                  f"-c exposure_absolute={self.exposure_value}")
            self._exposure_available = False

    def set_hardware_exposure(self, value: int):
        """设置绝对曝光值并验证"""
        self.exposure_value = value
        self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
        actual = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        return int(actual)

    def update_hardware_exposure(self, delta: int):
        """微调硬件曝光"""
        new_val = max(1, min(10000, self.exposure_value + delta))
        actual = self.set_hardware_exposure(new_val)
        print(f"{_timestamp()} 🔧 硬件曝光: {self.exposure_value} → {new_val}  "
              f"(实际={actual})")
        self.exposure_value = new_val

    # ═════════════════════════════════════════════════════
    #  自动曝光校准 — 按 'e' 触发
    # ═════════════════════════════════════════════════════

    def auto_calibrate_exposure(self):
        """
        自动扫描曝光值，找到饱和像素 < 5% 的最低曝光值。

        从 exposure=1 开始向上扫描，每次采 N 帧取平均饱和率，
        一旦饱和率 < TARGET 就停止。
        """
        print(f"{_timestamp()} 🔍 开始自动曝光校准...")
        print(f"{_timestamp()}   目标: 饱和像素 < {AUTOEXP_TARGET_SAT*100:.0f}%")
        print(f"{_timestamp()}   扫描范围: {AUTOEXP_SWEEP_START} ~ {AUTOEXP_SWEEP_END}")
        print(f"{_timestamp()}   (ESC 中断扫描)")

        # 记录原始曝光，扫描完后恢复（或保留最佳值）
        original_exp = self.exposure_value
        best_exp = None
        best_sat = 1.0

        for exp_val in range(AUTOEXP_SWEEP_START, AUTOEXP_SWEEP_END + 1):
            # 检查 ESC 中断
            if cv2.waitKey(1) & 0xFF == 27:
                print(f"{_timestamp()} ⚠ 用户中断扫描")
                break

            self.set_hardware_exposure(exp_val)

            # 丢几帧让曝光生效
            for _ in range(3):
                self.cap.read()

            # 采 N 帧取平均饱和率
            sat_samples = []
            for _ in range(AUTOEXP_CAPTURE_FRAMES):
                ret, frame = self.cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sat = np.mean(gray > 250)
                    sat_samples.append(sat)

            avg_sat = np.mean(sat_samples) if sat_samples else 1.0

            if exp_val <= 10 or exp_val % 10 == 0:
                # 每步打印（低频时每步，高频时每10步）
                print(f"{_timestamp()}   exp={exp_val:4d}  →  "
                      f"饱和率={avg_sat*100:5.1f}%  "
                      f"{'✅' if avg_sat < AUTOEXP_TARGET_SAT else ''}")

            if avg_sat < best_sat:
                best_sat = avg_sat
                best_exp = exp_val

            if avg_sat < AUTOEXP_TARGET_SAT:
                best_exp = exp_val
                break

        # ── 应用最佳值 ──────────────────────────────
        if best_exp is not None and best_sat < AUTOEXP_TARGET_SAT:
            actual = self.set_hardware_exposure(best_exp)
            self.exposure_value = best_exp
            print(f"{_timestamp()} ✅ 校准完成！")
            print(f"{_timestamp()}   最佳曝光值 = {best_exp}  (实际={actual})")
            print(f"{_timestamp()}   饱和率 = {best_sat*100:.1f}%")
        else:
            # 没找到合适的，恢复原值
            self.set_hardware_exposure(original_exp)
            self.exposure_value = original_exp
            print(f"{_timestamp()} ⚠ 未找到合适曝光值（全扫描范围饱和率均 > "
                  f"{AUTOEXP_TARGET_SAT*100:.0f}%）")
            print(f"{_timestamp()}   最低饱和率 = {best_sat*100:.1f}%  "
                  f"在 exposure={best_exp}")
            print(f"{_timestamp()}   → 可能原因：自动曝光未关闭 / "
                  f"摄像头不支持手动曝光")
            print(f"{_timestamp()}   → 请尝试: "
                  f"v4l2-ctl -d {self.device_index} "
                  f"-c auto_exposure=1 -c exposure_absolute=1")

    # ═════════════════════════════════════════════════════
    #  饱和分析（每帧调用，轻量）
    # ═════════════════════════════════════════════════════

    def _analyze_frame(self, raw_bgr: np.ndarray):
        """分析原始帧的曝光状态：饱和率、均值、直方图"""
        gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        self._mean_val = float(np.mean(gray))
        self._sat_pct = float(np.mean(gray > 250))

        if self._show_histogram:
            self._gray_hist = cv2.calcHist(
                [gray], [0], None, [HIST_BINS], [0, 256])

    def _get_saturation_status(self) -> str:
        """返回饱和度状态的彩色标签"""
        if self._sat_pct > 0.30:
            return "OVEREXPOSED"   # 严重过曝
        elif self._sat_pct > 0.10:
            return "BRIGHT"        # 偏亮
        elif self._sat_pct < 0.01 and self._mean_val < 50:
            return "UNDEREXPOSED"  # 偏暗
        else:
            return "OK"

    def _get_sat_color(self) -> tuple:
        """饱和度标签颜色 (BGR)"""
        status = self._get_saturation_status()
        if status == "OVEREXPOSED":
            return (0, 0, 255)     # 红
        elif status == "BRIGHT":
            return (0, 200, 255)   # 橙
        elif status == "UNDEREXPOSED":
            return (255, 0, 0)     # 蓝
        else:
            return (0, 255, 0)     # 绿

    def _draw_histogram(self) -> np.ndarray:
        """生成直方图可视化图像"""
        if self._gray_hist is None:
            canvas = np.zeros((HIST_H, HIST_BINS, 3), dtype=np.uint8)
            cv2.putText(canvas, "No histogram yet (press 'h')",
                        (20, HIST_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            return canvas

        canvas = np.zeros((HIST_H, HIST_BINS, 3), dtype=np.uint8)

        # 归一化
        hist = self._gray_hist.flatten()
        hist_max = hist.max()
        if hist_max > 0:
            hist_norm = hist / hist_max
        else:
            hist_norm = hist

        # 绘制柱状图
        for i in range(HIST_BINS):
            bar_h = int(hist_norm[i] * (HIST_H - 10))
            cv2.line(canvas,
                     (i, HIST_H - 1), (i, HIST_H - 1 - bar_h),
                     (100, 200, 100), 1)

        # 标记饱和区域 (>250)
        cv2.rectangle(canvas, (250, 0), (255, HIST_H - 1), (0, 0, 255), -1)
        cv2.putText(canvas, "SAT", (240, HIST_H - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # 均值线
        mean_x = int(_clamp(self._mean_val, 0, 255))
        cv2.line(canvas, (mean_x, 0), (mean_x, HIST_H - 1), (255, 255, 0), 1)
        cv2.putText(canvas, f"mean={self._mean_val:.0f}",
                    (mean_x + 2, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # 标题
        cv2.putText(canvas,
                    f"Histogram | Sat:{self._sat_pct*100:.1f}% "
                    f"| {self._get_saturation_status()}",
                    (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return canvas

    # ═════════════════════════════════════════════════════
    #  Gamma 校正
    # ═════════════════════════════════════════════════════

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        return np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8)

    def _apply_gamma(self, bgr: np.ndarray, gamma: float) -> np.ndarray:
        if self._gamma_lut is None or self._gamma_cached_for != gamma:
            self._gamma_lut = self._build_gamma_lut(gamma)
            self._gamma_cached_for = gamma
        return cv2.LUT(bgr, self._gamma_lut)

    # ═════════════════════════════════════════════════════
    #  亮度线性压缩
    # ═════════════════════════════════════════════════════

    def _apply_brightness_compress(self, bgr: np.ndarray, scale: float) -> np.ndarray:
        scaled = (bgr.astype(np.float32) * scale).clip(0, 255)
        return scaled.astype(np.uint8)

    # ═════════════════════════════════════════════════════
    #  CLAHE
    # ═════════════════════════════════════════════════════

    def _apply_clahe(self, bgr: np.ndarray,
                     clip_limit: float, grid_size: int) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        self._clahe.setClipLimit(clip_limit)
        self._clahe.setTilesGridSize((grid_size, grid_size))
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ═════════════════════════════════════════════════════
    #  色调映射
    # ═════════════════════════════════════════════════════

    def _get_tonemap(self):
        key = (self.tonemap_method, self.tonemap_gamma)
        if self._tonemap_instance is None or self._tonemap_key != key:
            if self.tonemap_method == TonemapMethod.DRAGO:
                tm = cv2.createTonemapDrago(
                    gamma=self.tonemap_gamma, saturation=1.0, bias=0.85)
            elif self.tonemap_method == TonemapMethod.REINHARD:
                tm = cv2.createTonemapReinhard(
                    gamma=self.tonemap_gamma, intensity=0.0,
                    light_adapt=1.0, color_adapt=0.0)
            else:
                tm = cv2.createTonemapMantiuk(
                    gamma=self.tonemap_gamma, scale=0.85, saturation=1.0)
            self._tonemap_instance = tm
            self._tonemap_key = key
        return self._tonemap_instance

    def _apply_tonemap(self, bgr: np.ndarray) -> np.ndarray:
        f32 = bgr.astype(np.float32) / 255.0
        mapped = self._get_tonemap().process(f32)
        mapped = np.clip(mapped, 0, 1)
        return (mapped * 255).astype(np.uint8)

    # ═════════════════════════════════════════════════════
    #  预处理管线
    # ═════════════════════════════════════════════════════

    def preprocess(self, raw_frame: np.ndarray) -> np.ndarray:
        if not self.enable_preprocess:
            return raw_frame
        result = raw_frame
        if self.enable_gamma:
            result = self._apply_gamma(result, self.gamma)
        if self.enable_brightness:
            result = self._apply_brightness_compress(result, self.brightness_scale)
        if self.enable_clahe:
            result = self._apply_clahe(result, self.clahe_clip, self.clahe_grid)
        if self.enable_tonemap:
            result = self._apply_tonemap(result)
        return result

    # ═════════════════════════════════════════════════════
    #  QR 检测
    # ═════════════════════════════════════════════════════

    def _detect_qr_code(self, bgr_img: np.ndarray):
        data, bbox, _ = self.qr_detector.detectAndDecode(bgr_img)
        detected = (len(data) > 0)
        if not detected:
            return False, "", None, 0
        data = data.strip()
        try:
            flag_val = int(data)
        except ValueError:
            flag_val = 0
        return True, data, bbox, flag_val

    def _trigger_detection_window(self):
        if not self._detection_active:
            self._detection_active = True
            self._detection_start_time = time.time()
            self._prev_flag = 0
            print(f"{_timestamp()} ← 收到遮挡信号(键盘's') → "
                  f"启动检测窗口 ({self.detection_window_s:.0f}s)")

    # ═════════════════════════════════════════════════════
    #  可视化
    # ═════════════════════════════════════════════════════

    def _draw_visualization(self, raw_frame, processed_frame, flag, qr_data, bbox):
        status_map = {0: "NO QR", 1: "QR (1)", 2: "QR (2)", 3: "QR (3)"}
        status_text = status_map.get(flag, f"QR: {flag}")
        flag_color = (0, 255, 0) if flag > 0 else (128, 128, 128)

        result = processed_frame.copy()

        if bbox is not None and flag != 0:
            bbox_int = bbox.astype(np.int32)
            if bbox_int.ndim == 3:
                bbox_int = bbox_int.reshape(4, 2)
            cv2.polylines(result, [bbox_int], True, (0, 255, 0), 2)
            cx = int(np.mean(bbox_int[:, 0]))
            cy = int(np.mean(bbox_int[:, 1]))
            cv2.putText(result, f"Data: {qr_data}", (cx - 60, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        h, w = result.shape[:2]

        # ── 左上：检测 + FPS + 饱和状态 ─────────────
        sat_color = self._get_sat_color()
        cv2.putText(result,
                    f"Flag: {flag}  {status_text}  |  FPS: {self._fps_display:.1f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, flag_color, 2)
        cv2.putText(result, f"QR Data: {qr_data if qr_data else 'None'}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── 饱和状态（醒目） ────────────────────────
        sat_text = (f"SAT: {self._sat_pct*100:.1f}%  "
                    f"mean={self._mean_val:.0f}  [{self._get_saturation_status()}]")
        cv2.putText(result, sat_text, (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sat_color, 2)

        # ── 底部：管线状态栏 ────────────────────────
        pre = "ON" if self.enable_preprocess else "OFF"
        gfx = f"G={self.gamma:.1f}" if self.enable_gamma else "G=OFF"
        bri = f"B={self.brightness_scale:.2f}" if self.enable_brightness else "B=OFF"
        cla = (f"CLAHE(c={self.clahe_clip:.1f},g={self.clahe_grid})"
               if self.enable_clahe else "CLAHE=OFF")
        tmm = (f"TM({self.tonemap_method.value},g={self.tonemap_gamma:.1f})"
               if self.enable_tonemap else "TM=OFF")
        exp_text = f"Exp={self.exposure_value}"

        pipeline_text = f"PREPROCESS [{pre}] | {exp_text} | {gfx} | {bri} | {cla} | {tmm}"
        y0 = h - 8
        cv2.putText(result, pipeline_text, (8, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        # ── 右上：调整焦点 ─────────────────────────
        focus_label = f"[adjust: {self._adjust_focus}]"
        (tw, th), _ = cv2.getTextSize(focus_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(result, focus_label, (w - tw - 8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)

        # ── 检测窗口状态 ───────────────────────────
        if self._detection_active:
            elapsed = time.time() - self._detection_start_time
            remaining = max(0, self.detection_window_s - elapsed)
            cv2.putText(result, f"🔍 DETECTION  {remaining:.1f}s",
                        (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        else:
            cv2.putText(result,
                        "⏸  IDLE  [e]=calib  [d]=diag  [h]=hist  [s]=trig",
                        (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        return result

    def _draw_raw_overlay(self, raw_frame):
        """在原始画面上叠加饱和状态"""
        vis = raw_frame.copy()
        h = vis.shape[0]

        sat_color = self._get_sat_color()
        sat_text = (f"RAW | SAT: {self._sat_pct*100:.1f}%  "
                    f"[{self._get_saturation_status()}]  "
                    f"mean={self._mean_val:.0f}  Exp={self.exposure_value}")
        cv2.putText(vis, sat_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sat_color, 2)

        # 底部快捷键提示
        cv2.putText(vis,
                    "[e]=auto-calib  [←→]=exp±1  [Shift+←→]=exp±10  [r]=reset(3)",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return vis

    # ═════════════════════════════════════════════════════
    #  参数调整
    # ═════════════════════════════════════════════════════

    def _adjust_param(self, delta: float):
        if self._adjust_focus == "gamma":
            self.gamma = _clamp(self.gamma + delta * GAMMA_STEP, GAMMA_MIN, GAMMA_MAX)
            print(f"{_timestamp()} 🔆 Gamma → {self.gamma:.1f}")
        elif self._adjust_focus == "brightness":
            self.brightness_scale = _clamp(
                self.brightness_scale + delta * BRIGHTNESS_STEP,
                BRIGHTNESS_MIN, BRIGHTNESS_MAX)
            print(f"{_timestamp()} 📉 亮度压缩 → {self.brightness_scale:.2f}")
        elif self._adjust_focus == "clahe":
            self.clahe_clip = _clamp(
                self.clahe_clip + delta * CLAHE_CLIP_STEP,
                CLAHE_CLIP_MIN, CLAHE_CLIP_MAX)
            print(f"{_timestamp()} 🎨 CLAHE clipLimit → {self.clahe_clip:.1f}")
        elif self._adjust_focus == "tonemap":
            self.tonemap_gamma = _clamp(
                self.tonemap_gamma + delta * TONEMAP_GAMMA_STEP,
                TONEMAP_GAMMA_MIN, TONEMAP_GAMMA_MAX)
            print(f"{_timestamp()} 🗜️  Tonemap gamma → {self.tonemap_gamma:.1f}")

    def _toggle_preprocess(self):
        self.enable_preprocess = not self.enable_preprocess
        state = "启用" if self.enable_preprocess else "禁用"
        print(f"{_timestamp()} ⚙️  预处理管线: {state}")

    def _print_frame_diagnosis(self):
        """打印当前帧的曝光诊断"""
        actual_exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        actual_auto = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_gain = self.cap.get(cv2.CAP_PROP_GAIN)
        print(f"{_timestamp()} 📊 帧诊断 ──────────────────────")
        print(f"{_timestamp()}   曝光设定值       = {self.exposure_value}")
        print(f"{_timestamp()}   曝光实际值       = {actual_exp:.1f}")
        print(f"{_timestamp()}   自动曝光模式     = {actual_auto:.1f}")
        print(f"{_timestamp()}   增益 (Gain)      = {actual_gain:.1f}")
        print(f"{_timestamp()}   灰度均值         = {self._mean_val:.1f}")
        print(f"{_timestamp()}   饱和像素比例     = {self._sat_pct*100:.1f}%")
        print(f"{_timestamp()}   过曝状态         = {self._get_saturation_status()}")
        print(f"{_timestamp()} ──────────────────────────────────")

    # ═════════════════════════════════════════════════════
    #  键盘处理
    # ═════════════════════════════════════════════════════

    def _handle_key(self, key: int, modifiers: int = 0) -> bool:
        """返回 True=继续, False=退出"""
        shift = (modifiers & 1) != 0  # cv2.EVENT_FLAG_SHIFTKEY

        if key == 27 or key == ord('q'):
            print(f"{_timestamp()} 用户退出")
            return False

        elif key == ord('s'):
            self._trigger_detection_window()

        elif key == ord('e'):
            self.auto_calibrate_exposure()

        elif key == ord('r'):
            actual = self.set_hardware_exposure(3)
            self.exposure_value = 3
            print(f"{_timestamp()} 🔧 曝光重置 → 3 (极短)  实际={actual}")

        elif key == ord('h'):
            self._show_histogram = not self._show_histogram
            state = "ON" if self._show_histogram else "OFF"
            print(f"{_timestamp()} 📊 直方图窗口: {state}")

        elif key == ord('d'):
            self._print_frame_diagnosis()

        elif key == ord('p'):
            self._toggle_preprocess()

        # ── 模块开关 ────────────────────────────────
        elif key == ord('1'):
            self.enable_gamma = not self.enable_gamma
            self._adjust_focus = "gamma"
            print(f"{_timestamp()} 🔆 Gamma: {'ON' if self.enable_gamma else 'OFF'} "
                  f"(γ={self.gamma:.1f})")
        elif key == ord('2'):
            self.enable_brightness = not self.enable_brightness
            self._adjust_focus = "brightness"
            print(f"{_timestamp()} 📉 亮度压缩: {'ON' if self.enable_brightness else 'OFF'} "
                  f"(scale={self.brightness_scale:.2f})")
        elif key == ord('3'):
            self.enable_clahe = not self.enable_clahe
            self._adjust_focus = "clahe"
            print(f"{_timestamp()} 🎨 CLAHE: {'ON' if self.enable_clahe else 'OFF'} "
                  f"(clip={self.clahe_clip:.1f}, grid={self.clahe_grid})")
        elif key == ord('4'):
            self.enable_tonemap = not self.enable_tonemap
            self._adjust_focus = "tonemap"
            print(f"{_timestamp()} 🗜️  色调映射: {'ON' if self.enable_tonemap else 'OFF'} "
                  f"({self.tonemap_method.value}, γ={self.tonemap_gamma:.1f})")

        elif key == ord('5'):
            methods = list(TonemapMethod)
            idx = methods.index(self.tonemap_method)
            self.tonemap_method = methods[(idx + 1) % len(methods)]
            self._tonemap_instance = None
            print(f"{_timestamp()} 🗜️  色调映射算法 → {self.tonemap_method.value}")

        elif key == ord('`') or key == ord('~'):
            foci = ["gamma", "brightness", "clahe", "tonemap"]
            idx = foci.index(self._adjust_focus)
            self._adjust_focus = foci[(idx + 1) % len(foci)]
            print(f"{_timestamp()} 🎯 调整焦点 → {self._adjust_focus}")

        # ── 方向键 ──────────────────────────────────
        elif key == 82 or key == ord('w'):
            self._adjust_param(+1)
        elif key == 84 or key == ord('x'):
            self._adjust_param(-1)

        # ── 左右：硬件曝光微调 ──────────────────────
        elif key == 81:    # ←
            self.update_hardware_exposure(-10 if shift else -1)
        elif key == 83:    # →
            self.update_hardware_exposure(+10 if shift else +1)

        # ── CLAHE 网格 ──────────────────────────────
        elif key == ord('['):
            self.clahe_grid = max(2, self.clahe_grid - 2)
            print(f"{_timestamp()} 🎨 CLAHE grid → {self.clahe_grid}")
        elif key == ord(']'):
            self.clahe_grid = min(32, self.clahe_grid + 2)
            print(f"{_timestamp()} 🎨 CLAHE grid → {self.clahe_grid}")

        return True

    # ═════════════════════════════════════════════════════
    #  主循环
    # ═════════════════════════════════════════════════════

    def run(self):
        win_raw    = f"{self.name}_raw"
        win_result = f"{self.name}_result"
        win_hist   = f"{self.name}_histogram"

        cv2.namedWindow(win_raw, cv2.WINDOW_NORMAL)
        cv2.namedWindow(win_result, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win_raw, 0, 0)
        cv2.moveWindow(win_result, 660, 0)

        while True:
            ret, raw_frame = self.cap.read()
            if not ret:
                print(f"{_timestamp()} ⚠ 读取摄像头帧失败", flush=True)
                time.sleep(0.05)
                continue

            # ── 帧分析（饱和率 + 直方图） ────────────
            self._analyze_frame(raw_frame)

            # ── 预处理 ──────────────────────────────
            processed = self.preprocess(raw_frame)

            # ── 检测窗口管理 ────────────────────────
            if self._detection_active:
                elapsed = time.time() - self._detection_start_time
                if elapsed > self.detection_window_s:
                    self._detection_active = False
                    self._detection_start_time = None
                    print(f"{_timestamp()} ⏰ 检测窗口结束 "
                          f"({elapsed:.0f}s > {self.detection_window_s:.0f}s)")

            if self._detection_active:
                detected, flag, qr_data, bbox = self._detect_qr_code(processed)
                if detected and self._prev_flag != flag:
                    print(f"{_timestamp()} ↑ QR码出现！ "
                          f"内容=\"{qr_data}\" → 发布 flag={flag}")
                    self._detection_active = False
                    self._detection_start_time = None
                    print(f"{_timestamp()} ✅ 已检测到QR码 → 关闭检测窗口")
                self._prev_flag = flag
            else:
                self._prev_flag = 0
                flag, qr_data, bbox = 0, "", None

            # ── FPS ─────────────────────────────────
            self._fps_counter += 1
            now = time.time()
            if now - self._fps_t0 >= 1.0:
                self._fps_display = self._fps_counter / (now - self._fps_t0)
                self._fps_counter = 0
                self._fps_t0 = now

            # ── 可视化 ──────────────────────────────
            vis_result = self._draw_visualization(
                raw_frame, processed, flag, qr_data, bbox)
            raw_vis = self._draw_raw_overlay(raw_frame)

            cv2.imshow(win_raw, raw_vis)
            cv2.imshow(win_result, vis_result)

            # ── 直方图窗口（可选） ───────────────────
            if self._show_histogram:
                hist_img = self._draw_histogram()
                cv2.imshow(win_hist, hist_img)
            else:
                try:
                    cv2.destroyWindow(win_hist)
                except cv2.error:
                    pass

            # ── 键盘 ────────────────────────────────
            raw_key = cv2.waitKeyEx(1)  # 用 waitKeyEx 捕获方向键+修饰键
            key = raw_key & 0xFF
            # 提取修饰键状态（waitKeyEx 返回值的低16位之外）
            # OpenCV 没有直接返回修饰键，使用 getWindowProperty 代替
            # 这里我们简化为：检查 raw_key 本身的方向键码
            if key == 0:
                # 可能是方向键等扩展键
                key = raw_key

            # 同时也用 waitKey 获取修饰键（轮询方式不可靠，用替代方案）
            if not self._handle_key(key):
                break

    def shutdown(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"{_timestamp()} {self.name} 已关闭")

    def _print_status(self):
        print(f"{_timestamp()} {self.name} 启动 | QR码检测 | 边沿触发 | "
              f"窗口={self.detection_window_s:.0f}s")
        print(f"{_timestamp()} 硬件曝光 = {self.exposure_value} "
              f"({'✅ 手动控制' if self._exposure_available else '⚠ 可能未生效'})")
        print(f"{_timestamp()} 预处理: "
              f"Gamma={self.gamma:.1f} | Bright={self.brightness_scale:.2f} | "
              f"CLAHE(clip={self.clahe_clip:.1f},grid={self.clahe_grid}) | "
              f"Tonemap({self.tonemap_method.value},γ={self.tonemap_gamma:.1f})")
        print(f"{_timestamp()} ──── 关键操作 ────")
        print(f"{_timestamp()}   [e] 自动扫描最佳曝光  ← 亮屏必用！")
        print(f"{_timestamp()}   [r] 重置曝光为3(极短)")
        print(f"{_timestamp()}   [d] 诊断当前帧状态")
        print(f"{_timestamp()}   [h] 打开直方图窗口")
        print(f"{_timestamp()}   [←→] 微调曝光 ±1  [Shift+←→] ±10")
        print(f"{_timestamp()}   [s] 触发检测窗口  [1-4] 切换预处理模块")
        print(f"{_timestamp()}   [ESC/q] 退出")


# ═══════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QR码检测调试版 — 硬件曝光优先 + 软件管线辅助",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python qiyuQR_debug.py                                   # 默认启动（曝光=3）
  python qiyuQR_debug.py --exposure 1                       # 最短曝光
  python qiyuQR_debug.py --device 0                         # 指定摄像头
  python qiyuQR_debug.py --no-preprocess --exposure 5       # 仅硬件曝光
""")
    parser.add_argument("--device", type=int, default=DEFAULT_DEVICE)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--window", type=float, default=DEFAULT_DETECTION_WINDOW_S)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--no-gamma", action="store_true")
    parser.add_argument("--no-brightness", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-tonemap", action="store_true")
    parser.add_argument("--exposure", type=int, default=DEFAULT_EXPOSURE_VALUE,
                        help=f"硬件曝光值，单位100µs，亮屏建议1~5 (默认: {DEFAULT_EXPOSURE_VALUE})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--brightness", type=float, default=DEFAULT_BRIGHTNESS_SCALE)
    parser.add_argument("--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP)
    parser.add_argument("--clahe-grid", type=int, default=DEFAULT_CLAHE_GRID)
    parser.add_argument("--tonemap-method", type=str, default=DEFAULT_TONEMAP_METHOD.value,
                        choices=["drago", "reinhard", "mantiuk"])
    parser.add_argument("--tonemap-gamma", type=float, default=DEFAULT_TONEMAP_GAMMA)

    args = parser.parse_args()

    try:
        node = QiyuQRDebug(
            device_index=args.device,
            width=args.width, height=args.height, fps=args.fps,
            detection_window_s=args.window,
            enable_preprocess=not args.no_preprocess,
            enable_gamma=not args.no_gamma,
            enable_brightness=not args.no_brightness,
            enable_clahe=not args.no_clahe,
            enable_tonemap=not args.no_tonemap,
            exposure_value=args.exposure,
            gamma=args.gamma,
            brightness_scale=args.brightness,
            clahe_clip=args.clahe_clip,
            clahe_grid=args.clahe_grid,
            tonemap_method=TonemapMethod(args.tonemap_method),
            tonemap_gamma=args.tonemap_gamma,
        )
        node.run()
    except RuntimeError as e:
        print(f"{_timestamp()} ❌ 错误: {e}")
        print(f"{_timestamp()} 提示: 尝试 --device 0 或 --device 1")
    except KeyboardInterrupt:
        print(f"\n{_timestamp()} 收到 Ctrl+C，正在退出...")
    finally:
        if 'node' in locals():
            node.shutdown()


if __name__ == "__main__":
    main()
