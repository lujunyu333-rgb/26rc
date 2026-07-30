#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白底黑字检测 — 使用 allcamera.rs_cam 接收 RealSense 图像并检测。

与 test_rs.py 结构完全对标：rs_cam → read_one_frame → 处理 → 显示。

用法：
  python3 test_rs_ws.py              # 实时检测（需要 GUI + RealSense）
  python3 test_rs_ws.py --once       # 单帧检测（纯终端输出，不需要 GUI）
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2

# 确保能找到 allcamera 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allcamera import rs_cam

# ── 检测参数（与 ws_text.py 一致） ──────────────────────
DARK_THRESHOLD   = 80
BRIGHT_THRESHOLD = 160
MID_MAX_RATIO    = 0.15
DARK_MIN_RATIO   = 0.08
BRIGHT_MIN_RATIO = 0.25
EDGE_MIN_RATIO   = 0.002


def detect(gray: np.ndarray) -> dict:
    """检测白底黑字，返回指标字典"""
    total = gray.size
    dark_r = np.count_nonzero(gray < DARK_THRESHOLD) / total
    bright_r = np.count_nonzero(gray > BRIGHT_THRESHOLD) / total
    mid_r = 1.0 - dark_r - bright_r

    edges = cv2.Canny(gray, 50, 150)
    edge_r = np.count_nonzero(edges) / total

    hit = (dark_r > DARK_MIN_RATIO and bright_r > BRIGHT_MIN_RATIO and
           mid_r < MID_MAX_RATIO and edge_r > EDGE_MIN_RATIO)

    return {
        "flag": 1 if hit else 0,
        "dark": dark_r, "bright": bright_r, "mid": mid_r, "edge": edge_r,
        "checks": {
            "dark": dark_r > DARK_MIN_RATIO,
            "bright": bright_r > BRIGHT_MIN_RATIO,
            "mid": mid_r < MID_MAX_RATIO,
            "edge": edge_r > EDGE_MIN_RATIO,
        },
    }


def print_result(result: dict):
    """终端打印检测结果"""
    c = result["checks"]
    print(f"  flag = {result['flag']}  {'✓ 白底黑字' if result['flag'] else '✗ 未检测到'}")
    print(f"  dark  = {result['dark']:.4f}  [{'OK' if c['dark'] else 'NG'}] (> {DARK_MIN_RATIO})")
    print(f"  bright= {result['bright']:.4f}  [{'OK' if c['bright'] else 'NG'}] (> {BRIGHT_MIN_RATIO})")
    print(f"  mid   = {result['mid']:.4f}  [{'OK' if c['mid'] else 'NG'}] (< {MID_MAX_RATIO})")
    print(f"  edge  = {result['edge']:.4f}  [{'OK' if c['edge'] else 'NG'}] (> {EDGE_MIN_RATIO})")


def run_once():
    """单帧模式：从 rs_cam 取一帧，检测后退出"""
    print("[INFO] 启动 RealSense (rs_cam)...")
    try:
        cam = rs_cam(width=640, height=480, fps=30)
    except RuntimeError as e:
        print(f"[FAIL] RealSense 初始化失败: {e}")
        print("       可能原因: 设备被占用 / 未连接 / 权限不足")
        sys.exit(1)

    # 等待后台线程捕获到有效帧（最多等 3 秒）
    print("[INFO] 等待图像帧...")
    color = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        color, _ = cam.read_one_frame()
        if color is not None:
            break
        time.sleep(0.05)

    cam.close_rscam()

    if color is None:
        print("[FAIL] 3 秒内未收到 RealSense 图像，请检查相机连接")
        sys.exit(1)

    print(f"[INFO] 收到图像: {color.shape}")

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    result = detect(gray)

    print()
    print("=" * 50)
    print_result(result)
    print("=" * 50)

    # 保存结果图
    out_path = "/tmp/rs_ws_once.jpg"
    cv2.imwrite(out_path, color)
    print(f"[SAVE] 原始帧已保存: {out_path}")


def run_live():
    """实时模式：对标 test_rs.py 的 while 循环"""
    print("[INFO] 启动 RealSense (rs_cam)...")
    try:
        cam = rs_cam(width=640, height=480, fps=30)
    except RuntimeError as e:
        print(f"[FAIL] RealSense 初始化失败: {e}")
        print("       可能原因: 设备被占用 / 未连接 / 权限不足")
        sys.exit(1)
    print("[INFO] 开始实时白底黑字检测 | ESC/q 退出 | s 保存当前帧")

    try:
        while True:
            color, _ = cam.read_one_frame()
            if color is None:
                continue

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            result = detect(gray)

            # 叠加显示
            display = color.copy()
            status = f"FLAG={result['flag']}  {'WHITE-BG BLACK-TEXT' if result['flag'] else 'CLEAR'}"
            bar_color = (0, 255, 0) if result['flag'] else (0, 0, 255)
            cv2.rectangle(display, (0, 0), (color.shape[1], 40), (40, 40, 40), -1)
            cv2.putText(display, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bar_color, 2)

            y = 60
            c = result["checks"]
            for key, label in [
                ("dark",   f"dark  = {result['dark']:.3f}"),
                ("bright", f"bright= {result['bright']:.3f}"),
                ("mid",    f"mid   = {result['mid']:.3f}"),
                ("edge",   f"edge  = {result['edge']:.4f}"),
            ]:
                clr = (0, 255, 0) if c[key] else (0, 0, 255)
                cv2.putText(display, f"[{'OK' if c[key] else 'NG'}] {label}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)
                y += 18

            cv2.imshow("ws_text — rs_cam实时检测", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('s'):
                ts = int(time.time())
                path = f"/tmp/ws_text_rs_{ts}.jpg"
                cv2.imwrite(path, display)
                print(f"[SAVE] {path}")

    finally:
        cam.close_rscam()
        cv2.destroyAllWindows()
        print("[INFO] 已停止")


def main():
    parser = argparse.ArgumentParser(description="白底黑字检测 — rs_cam 直读 RealSense")
    parser.add_argument("--once", action="store_true",
                        help="单帧模式：取一帧检测后退出（纯终端，不需要 GUI）")
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_live()


if __name__ == "__main__":
    main()
