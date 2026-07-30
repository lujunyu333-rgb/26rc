#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白底黑字检测验证脚本 — 不依赖 ROS2，直接从 RealSense 或图片文件获取图像，
应用 ws_text 检测算法并可视化结果。

用法：
  python3 camera/scripts/test_ws_text.py                    # RealSense 实时检测
  python3 camera/scripts/test_ws_text.py --image xxx.jpg    # 单张图片检测
  python3 camera/scripts/test_ws_text.py --save             # 保存结果图
  python3 camera/scripts/test_ws_text.py --image xxx.jpg --save  # 图片检测+保存

按键：
  ESC / q — 退出
  s     — 保存当前帧
"""

import argparse
import sys
import os
import numpy as np

# ── 1. 检查 OpenCV ───────────────────────────────────
try:
    import cv2
    print("[PASS] OpenCV 导入成功")
except ImportError as e:
    print(f"[FAIL] OpenCV 未安装: {e}")
    sys.exit(1)

# ── 2. 检查 RealSense（实时模式需要） ─────────────────
try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
    print("[PASS] pyrealsense2 导入成功")
except ImportError:
    HAS_REALSENSE = False
    print("[WARN] pyrealsense2 未安装，仅支持 --image 模式")

# ── 检测参数（与 ws_text.py 保持一致） ────────────────
DARK_THRESHOLD    = 80
BRIGHT_THRESHOLD  = 160
MID_MAX_RATIO     = 0.15
DARK_MIN_RATIO    = 0.08
BRIGHT_MIN_RATIO  = 0.25
EDGE_MIN_RATIO    = 0.002


def detect_white_bg_black_text(gray: np.ndarray) -> dict:
    """
    检测白底黑字，返回 dict 包含所有指标和最终判定。

    原理：白底黑字图像的灰度直方图呈双峰分布——
    暗像素（文字）集中在低灰度区，亮像素（背景）集中在高灰度区，
    中间灰度极少。普通场景不具备此特征。
    """
    total = gray.size

    # 三区域统计
    dark_mask = gray < DARK_THRESHOLD
    bright_mask = gray > BRIGHT_THRESHOLD

    dark_ratio = np.count_nonzero(dark_mask) / total
    bright_ratio = np.count_nonzero(bright_mask) / total
    mid_ratio = 1.0 - dark_ratio - bright_ratio

    # 边缘密度（文字区域有丰富的边缘）
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.count_nonzero(edges) / total

    # 四条件判断
    cond_dark = dark_ratio > DARK_MIN_RATIO
    cond_bright = bright_ratio > BRIGHT_MIN_RATIO
    cond_mid = mid_ratio < MID_MAX_RATIO
    cond_edge = edge_ratio > EDGE_MIN_RATIO

    detected = cond_dark and cond_bright and cond_mid and cond_edge

    return {
        "detected": detected,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "mid_ratio": mid_ratio,
        "edge_ratio": edge_ratio,
        "dark_mask": dark_mask,
        "bright_mask": bright_mask,
        "edges": edges,
        "checks": {
            "dark": cond_dark,
            "bright": cond_bright,
            "mid": cond_mid,
            "edge": cond_edge,
        },
    }


def draw_overlay(frame: np.ndarray, result: dict) -> np.ndarray:
    """在原图上绘制检测信息叠加层"""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    detected = result["detected"]
    checks = result["checks"]

    # ── 标题栏 ──
    status_text = "WHITE-BG BLACK-TEXT DETECTED" if detected else "NO TEXT DETECTED"
    bar_color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.rectangle(overlay, (0, 0), (w, 80), (40, 40, 40), -1)
    cv2.putText(overlay, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)

    # ── 四个条件状态 ──
    y = 70
    conditions = [
        ("dark",   f"dark_ratio  = {result['dark_ratio']:.3f}  (> {DARK_MIN_RATIO})"),
        ("bright", f"bright_ratio= {result['bright_ratio']:.3f}  (> {BRIGHT_MIN_RATIO})"),
        ("mid",    f"mid_ratio   = {result['mid_ratio']:.3f}  (< {MID_MAX_RATIO})"),
        ("edge",   f"edge_ratio  = {result['edge_ratio']:.4f}  (> {EDGE_MIN_RATIO})"),
    ]
    for key, label in conditions:
        color = (0, 255, 0) if checks[key] else (0, 0, 255)
        cv2.putText(overlay, f"[{'OK' if checks[key] else 'NG'}] {label}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 18

    return overlay


def draw_masks(gray: np.ndarray, result: dict) -> np.ndarray:
    """
    生成三通道掩码可视化图：
    - 青色 = 暗像素区域（黑字）
    - 黄色 = 亮像素区域（白底）
    - 品红 = 边缘
    """
    dark_mask = result["dark_mask"]
    bright_mask = result["bright_mask"]
    edges = result["edges"]

    # 灰度图转 BGR 底图（半透明叠加）
    mask_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 暗像素 → 青色
    mask_bgr[dark_mask] = (255, 255, 0)
    # 亮像素 → 黄色（在亮区上叠加）
    mask_bgr[bright_mask] = (0, 255, 255)
    # 边缘 → 品红
    mask_bgr[edges > 0] = (255, 0, 255)

    return mask_bgr


def get_image_from_realsense():
    """从 RealSense 捕获一帧（用于 --image 未指定时的回退）"""
    if not HAS_REALSENSE:
        return None

    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        for _ in range(5):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        if img.ndim == 1:
            img = img.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        pipe.stop()
        print(f"[INFO] 从 RealSense 获取测试图像: {img.shape}")
        return img
    except Exception as e:
        print(f"[WARN] 无法从 RealSense 获取图像: {e}")
        return None


def run_live():
    """实时模式：通过 rs_cam（后台线程捕获）持续检测"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from allcamera import rs_cam

    cam = rs_cam(width=640, height=480, fps=30)
    print("[INFO] RealSense 已启动，开始白底黑字检测...")
    print("       按 ESC 或 q 退出，按 s 保存当前帧")

    try:
        while True:
            color, depth = cam.read_one_frame()
            if color is None:
                continue

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            result = detect_white_bg_black_text(gray)

            # 叠加图：原图 + 检测信息
            overlay = draw_overlay(color, result)
            # 掩码图：暗/亮/边缘分区
            mask_viz = draw_masks(gray, result)

            # 左右拼接
            combined = np.hstack([overlay, mask_viz])

            # 如果宽度太大则缩放
            h_disp, w_disp = combined.shape[:2]
            if w_disp > 1600:
                scale = 1600 / w_disp
                combined = cv2.resize(combined, (1600, int(h_disp * scale)))

            cv2.imshow("ws_text — overlay (L) | mask (R)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(__import__('time').time())
                out_path = f"/tmp/ws_text_{ts}.jpg"
                cv2.imwrite(out_path, combined)
                print(f"[SAVE] {out_path}")

    finally:
        cam.close_rscam()
        cv2.destroyAllWindows()
        print("[INFO] 实时检测已停止")


def run_single_image(image_path: str, save: bool = False, no_gui: bool = False):
    """单张图片模式"""
    if not os.path.exists(image_path):
        print(f"[FAIL] 图片不存在: {image_path}")
        sys.exit(1)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[FAIL] 无法读取图片: {image_path}")
        sys.exit(1)

    print(f"[INFO] 读取图片: {image_path} → {img.shape}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = detect_white_bg_black_text(gray)

    # ── 终端输出 ──
    print()
    print("=" * 55)
    print(f"  白底黑字检测结果: {'✓ 检测到' if result['detected'] else '✗ 未检测到'}")
    print("-" * 55)
    checks = result["checks"]
    print(f"  [{'OK' if checks['dark'] else 'NG'}] dark_ratio   = {result['dark_ratio']:.4f}  "
          f"(阈值 > {DARK_MIN_RATIO})")
    print(f"  [{'OK' if checks['bright'] else 'NG'}] bright_ratio = {result['bright_ratio']:.4f}  "
          f"(阈值 > {BRIGHT_MIN_RATIO})")
    print(f"  [{'OK' if checks['mid'] else 'NG'}] mid_ratio    = {result['mid_ratio']:.4f}  "
          f"(阈值 < {MID_MAX_RATIO})")
    print(f"  [{'OK' if checks['edge'] else 'NG'}] edge_ratio   = {result['edge_ratio']:.4f}  "
          f"(阈值 > {EDGE_MIN_RATIO})")
    print("=" * 55)

    # 生成结果图（可能用于保存）
    overlay = draw_overlay(img, result)
    mask_viz = draw_masks(gray, result)
    combined = np.hstack([overlay, mask_viz])

    if save:
        out_path = "/tmp/ws_text_result.jpg"
        cv2.imwrite(out_path, combined)
        print(f"[SAVE] 结果已保存: {out_path}")

    if not no_gui:
        cv2.imshow("ws_text — overlay (L) | mask (R)", combined)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="白底黑字检测验证 — RealSense 实时 / 图片文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 test_ws_text.py                        # RealSense 实时
  python3 test_ws_text.py --image example.jpg    # 单张图片
  python3 test_ws_text.py --image example.jpg --save
  python3 test_ws_text.py --image example.jpg --no-gui --save  # 无头模式
        """)
    parser.add_argument("--image", type=str, default=None,
                        help="测试图片路径（不指定则使用 RealSense 实时）")
    parser.add_argument("--save", action="store_true",
                        help="保存结果图到 /tmp/")
    parser.add_argument("--no-gui", action="store_true",
                        help="跳过 GUI 显示窗口（无头环境/SSH 使用）")
    args = parser.parse_args()

    # 自动检测无头环境
    no_gui = args.no_gui or not os.environ.get("DISPLAY")
    if no_gui:
        print("[INFO] 无 GUI 模式（--no-gui 或 DISPLAY 未设置）")

    if args.image:
        run_single_image(args.image, args.save, no_gui)
    else:
        if no_gui:
            print("[FAIL] 实时模式需要 GUI，请使用 --image 或设置 DISPLAY")
            sys.exit(1)
        if not HAS_REALSENSE:
            print("[FAIL] RealSense 不可用，请使用 --image 指定图片路径")
            sys.exit(1)
        # 先试捕获一帧确认相机可用
        test_img = get_image_from_realsense()
        if test_img is None:
            print("[FAIL] 无法从 RealSense 获取图像，请使用 --image 指定图片路径")
            sys.exit(1)
        run_live()


if __name__ == "__main__":
    main()
