#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 环境验证脚本 — 不依赖 ROS2，直接测试模型加载 + 推理。

用法：
  python3 camera/scripts/test_yolo.py              # 自动从 RealSense 取一帧
  python3 camera/scripts/test_yolo.py --image xxx   # 指定图片路径
  python3 camera/scripts/test_yolo.py --save        # 保存标注结果图
"""

import argparse
import sys
import os
import time
import numpy as np

# ── 1. 检查 ultralytics ────────────────────────────
try:
    from ultralytics import YOLO
    print("[PASS] ultralytics 导入成功")
except ImportError as e:
    print(f"[FAIL] ultralytics 未安装: {e}")
    print("       请运行: pip install ultralytics")
    sys.exit(1)

# ── 2. 检查模型文件 ────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "best.pt")
# 回退到可能的绝对路径
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"[FAIL] 模型文件不存在: {MODEL_PATH}")
    sys.exit(1)
print(f"[PASS] 模型文件存在: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")

# ── 3. 加载模型 ────────────────────────────────────
print("[INFO] 加载 YOLO 模型...")
try:
    model = YOLO(MODEL_PATH)
    print("[PASS] 模型加载成功")
except Exception as e:
    print(f"[FAIL] 模型加载失败: {e}")
    sys.exit(1)

# ── 4. 获取测试图像 ────────────────────────────────
def get_image_from_realsense():
    """尝试从 RealSense 捕获一帧"""
    try:
        import pyrealsense2 as rs
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        # 等待几帧让自动曝光稳定
        for _ in range(5):
            pipe.wait_for_frames()
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())
        # 确保是正确的 3 通道形状
        if img.ndim == 1:
            img = img.reshape((color.get_height(), color.get_width(), 3))
        pipe.stop()
        print(f"[INFO] 从 RealSense 获取测试图像: {img.shape}")
        return img
    except Exception as e:
        print(f"[WARN] 无法从 RealSense 获取图像: {e}")
        return None

def get_fallback_image():
    """生成一张纯色测试图像"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 128  # 蓝绿色背景
    img[:, :, 1] = 200
    img[:, :, 2] = 128
    print("[INFO] 使用生成的测试图像 (640x480)")
    return img

def main():
    parser = argparse.ArgumentParser(description="YOLO 环境验证")
    parser.add_argument("--image", type=str, default=None, help="测试图片路径")
    parser.add_argument("--save", action="store_true", help="保存标注结果图")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    args = parser.parse_args()

    # 获取图像
    img = None
    if args.image:
        import cv2
        if os.path.exists(args.image):
            img = cv2.imread(args.image)
            print(f"[INFO] 从文件读取: {args.image} → {img.shape if img is not None else '失败'}")
        else:
            print(f"[FAIL] 图片不存在: {args.image}")
            sys.exit(1)

    if img is None:
        img = get_image_from_realsense()
    if img is None:
        img = get_fallback_image()

    # ── 5. 推理 ────────────────────────────────────
    print(f"[INFO] 开始推理 (conf={args.conf})...")
    t0 = time.time()
    try:
        results = model(img, conf=args.conf, verbose=False)
        elapsed = (time.time() - t0) * 1000
        print(f"[PASS] 推理完成, 耗时 {elapsed:.1f} ms")
    except Exception as e:
        print(f"[FAIL] 推理异常: {e}")
        sys.exit(1)

    # ── 6. 分析结果 ────────────────────────────────
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            print(f"[PASS] 检测到 {len(boxes)} 个目标:")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                name = model.names.get(cls_id, f"class_{cls_id}")
                print(f"  [{i}] class={name}(id={cls_id}) conf={conf:.3f} "
                      f"box=[{xyxy[0]:.0f},{xyxy[1]:.0f},{xyxy[2]:.0f},{xyxy[3]:.0f}] "
                      f"area={(xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]):.0f}px")
        else:
            print("[INFO] 未检测到目标 (可能正常 — 取决于图像内容)")

        # ── 7. 保存结果 ─────────────────────────────
        if args.save:
            import cv2
            annotated = results[0].plot()
            out_path = "/tmp/yolo_test_result.jpg"
            cv2.imwrite(out_path, annotated)
            print(f"[INFO] 标注结果已保存: {out_path}")
    else:
        print("[WARN] 推理结果为空")

    print("\n" + "=" * 50)
    print("YOLO 环境验证完成 — 模型加载、图像获取、推理均正常")
    print("=" * 50)


if __name__ == "__main__":
    main()
