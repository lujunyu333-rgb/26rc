#!/usr/bin/env python3
"""
九宫格 + RealSense 调试版（无需 ROS2）
直接读取 RealSense D435 相机，检测物体边框 → 透视变换 → 3×3网格 → 红/蓝分类
集成 tic_tac_toe_ai：识别棋局后自动推荐 AI 最优落子（约束：只能下第二行）
按 'q' 退出
"""
import pyrealsense2 as rs
import cv2
import numpy as np

from rc_view.scripts.nine_grid.tic_tac_toe_ai import NineGrid

# ------------------- 常量配置 -------------------
OUTPUT_SIZE = 600                # 最终九宫格显示尺寸
WHOLE_WIDTH = 600                # 透视变换后整个物体的宽度（像素）
WHOLE_HEIGHT = int(WHOLE_WIDTH * (200 / 160))  # 整个物体高度 = 600 * 1.25 = 750
GRID_HEIGHT = int(WHOLE_HEIGHT * (160 / 200))  # 九宫格高度 = 750 * 0.8 = 600
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 20
CELL_LENTH = 0.5

# 九宫格标签矩阵（3×3），调试打印用
jiu = np.array([["", "", ""], ["", "", ""], ["", "", ""]])


# ------------------- 辅助函数 -------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_perspective(image, corners, dst_size):
    """透视变换到指定尺寸 (width, height)"""
    ordered = order_points(corners)
    dst = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(image, M, dst_size)
    return warped, M


def split_grid_cells(warped_shape):
    """将正方形图像划分为 3×3 个单元格"""
    h, w = warped_shape[:2]
    step = h // 3
    cells = []
    for i in range(3):
        for j in range(3):
            y1 = i * step
            y2 = (i + 1) * step if i < 2 else h
            x1 = j * step
            x2 = (j + 1) * step if j < 2 else w
            cells.append((y1, y2, x1, x2))
    return cells, step


def find_whole_object_corners(edges, original_shape):
    """通过霍夫直线检测找到整个物体（200×160）的外边框角点"""
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                            threshold=80, minLineLength=MIN_LINE_LENGTH,
                            maxLineGap=MAX_LINE_GAP)
    if lines is None:
        return None

    good_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > MIN_LINE_LENGTH:
            good_lines.append((x1, y1, x2, y2))

    if len(good_lines) < 4:
        return None

    # 按角度分类
    horizontal = []
    vertical = []
    for (x1, y1, x2, y2) in good_lines:
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle) < 45 or abs(angle) > 135:
            horizontal.append((x1, y1, x2, y2))
        else:
            vertical.append((x1, y1, x2, y2))

    if len(horizontal) < 2 or len(vertical) < 2:
        return None

    # 提取水平和垂直线的中点坐标
    h_vals = [(y1 + y2) // 2 for (x1, y1, x2, y2) in horizontal]
    v_vals = [(x1 + x2) // 2 for (x1, y1, x2, y2) in vertical]

    top_y = min(h_vals)
    bottom_y = max(h_vals)
    left_x = min(v_vals)
    right_x = max(v_vals)

    corners = np.array([
        [left_x, top_y],
        [right_x, top_y],
        [right_x, bottom_y],
        [left_x, bottom_y]
    ], dtype=np.float32)

    h, w = original_shape[:2]
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


def nothing(x):
    """Trackbar 空回调"""
    pass


# ------------------- 主函数 -------------------
def main():
    # 初始化 RealSense 管线
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    # 对齐深度到彩色
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 创建 Canny 阈值调节窗口
    cv2.namedWindow('Canny Threshold')
    cv2.createTrackbar('Canny_low', 'Canny Threshold', 50, 255, nothing)
    cv2.createTrackbar('Canny_high', 'Canny Threshold', 150, 255, nothing)

    # 创建蓝色 HSV 阈值调节窗口
    cv2.namedWindow('Blue HSV')
    cv2.createTrackbar('B_H_low', 'Blue HSV', 100, 179, nothing)
    cv2.createTrackbar('B_H_high', 'Blue HSV', 124, 179, nothing)
    cv2.createTrackbar('B_S_low', 'Blue HSV', 43, 255, nothing)
    cv2.createTrackbar('B_S_high', 'Blue HSV', 255, 255, nothing)
    cv2.createTrackbar('B_V_low', 'Blue HSV', 46, 255, nothing)
    cv2.createTrackbar('B_V_high', 'Blue HSV', 255, 255, nothing)

    # 创建红色 HSV 阈值调节窗口
    cv2.namedWindow('Red HSV')
    cv2.createTrackbar('R_H_range', 'Red HSV', 10, 20, nothing)
    cv2.createTrackbar('R_S_low', 'Red HSV', 43, 255, nothing)
    cv2.createTrackbar('R_S_high', 'Red HSV', 255, 255, nothing)
    cv2.createTrackbar('R_V_low', 'Red HSV', 46, 255, nothing)
    cv2.createTrackbar('R_V_high', 'Red HSV', 255, 255, nothing)

    cv2.namedWindow("RGB")
    cv2.namedWindow("Edges")
    cv2.namedWindow("Contour")
    cv2.namedWindow("Warped Grid")

    # 初始化 #字棋 AI（执蓝方 B，对手红方 R；只能下第二行）
    ai = NineGrid(ai_label='B')

    print("[调试] 九宫格 RealSense + #字棋 AI 已启动 — 按 'q' 退出")

    try:
        while True:
            # 从 RealSense 读取帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            cv_image = np.asanyarray(color_frame.get_data())

            # 读取 Canny 阈值
            canny_low = cv2.getTrackbarPos("Canny_low", "Canny Threshold")
            canny_high = cv2.getTrackbarPos("Canny_high", "Canny Threshold")

            # 灰度 → 双边滤波 → Canny 边缘
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)   ###需要细致调节
            edges = cv2.Canny(filtered, canny_low, canny_high)

            # 检测物体角点
            corners = find_whole_object_corners(edges, cv_image.shape)

            contour_image = cv_image.copy()
            warped_grid_display = None

            if corners is not None:
                int_corners = corners.astype(int)
                cv2.polylines(contour_image, [int_corners], True,
                              (0, 0, 255), 2)

                # 透视变换
                whole_warped, _ = warp_perspective(
                    cv_image, corners, (WHOLE_WIDTH, WHOLE_HEIGHT)
                )

                # 裁剪九宫格区域
                grid_region = whole_warped[0:GRID_HEIGHT, 0:WHOLE_WIDTH]
                grid_resized = cv2.resize(grid_region,
                                          (OUTPUT_SIZE, OUTPUT_SIZE))

                # HSV 颜色检测
                warped_hsv = cv2.cvtColor(grid_resized, cv2.COLOR_BGR2HSV)

                # 读取蓝色 HSV 阈值
                b_h_low = cv2.getTrackbarPos("B_H_low", "Blue HSV")
                b_h_high = cv2.getTrackbarPos("B_H_high", "Blue HSV")
                b_s_low = cv2.getTrackbarPos("B_S_low", "Blue HSV")
                b_s_high = cv2.getTrackbarPos("B_S_high", "Blue HSV")
                b_v_low = cv2.getTrackbarPos("B_V_low", "Blue HSV")
                b_v_high = cv2.getTrackbarPos("B_V_high", "Blue HSV")

                # 读取红色 HSV 阈值
                r_h_range = cv2.getTrackbarPos("R_H_range", "Red HSV")
                r_s_low = cv2.getTrackbarPos("R_S_low", "Red HSV")
                r_s_high = cv2.getTrackbarPos("R_S_high", "Red HSV")
                r_v_low = cv2.getTrackbarPos("R_V_low", "Red HSV")
                r_v_high = cv2.getTrackbarPos("R_V_high", "Red HSV")

                # 红色掩码（双范围：0~R_H_range 和 180-R_H_range~180）
                lower_red1 = np.array([0, r_s_low, r_v_low])
                upper_red1 = np.array([r_h_range, r_s_high, r_v_high])
                lower_red2 = np.array([180 - r_h_range, r_s_low, r_v_low])
                upper_red2 = np.array([180, r_s_high, r_v_high])
                red_mask1 = cv2.inRange(warped_hsv, lower_red1, upper_red1)
                red_mask2 = cv2.inRange(warped_hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(red_mask1, red_mask2)

                # 蓝色掩码
                lower_blue = np.array([b_h_low, b_s_low, b_v_low])
                upper_blue = np.array([b_h_high, b_s_high, b_v_high])
                blue_mask = cv2.inRange(warped_hsv, lower_blue, upper_blue)

                cells, step = split_grid_cells(grid_resized.shape)

                # 绘制网格
                warped_grid_display = grid_resized.copy()
                for i in range(1, 3):
                    cv2.line(warped_grid_display,
                             (i * step, 0), (i * step, OUTPUT_SIZE),
                             (0, 255, 0), 2)
                    cv2.line(warped_grid_display,
                             (0, i * step), (OUTPUT_SIZE, i * step),
                             (0, 255, 0), 2)

                # 更新九宫格标签矩阵
                global jiu
                jiu = np.array([["", "", ""], ["", "", ""], ["", "", ""]])

                for idx, (y1, y2, x1, x2) in enumerate(cells):
                    cell_red = red_mask[y1:y2, x1:x2]
                    red_pixels = cv2.countNonZero(cell_red)
                    total_pixels = (y2 - y1) * (x2 - x1)
                    red_ratio = red_pixels / total_pixels
                    has_red = red_ratio > 0.1

                    cell_blue = blue_mask[y1:y2, x1:x2]
                    blue_pixels = cv2.countNonZero(cell_blue)
                    blue_ratio = blue_pixels / total_pixels
                    has_blue = blue_ratio > 0.1

                    if has_red:
                        label = "R"
                        color = (0, 0, 255)
                    elif has_blue:
                        label = "B"
                        color = (255, 0, 0)
                    else:
                        label = ""
                        color = (255, 255, 255)

                    if label:
                        row, col = divmod(idx, 3)
                        jiu[row, col] = label
                        cx = col * step + step // 2
                        cy = row * step + step // 2
                        cv2.putText(warped_grid_display, label,
                                    (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, color, 2)

                # 控制台打印九宫格状态
                print("\n--- 九宫格 ---")
                for r in range(3):
                    row_str = " | ".join(
                        jiu[r, c] if jiu[r, c] else "·" for c in range(3)
                    )
                    print(f"  {row_str}")
                print("--------------")

                # ---------- #字棋 AI 决策 ----------
                ai.from_array(jiu)
                best_col = ai.best_move()
                if best_col is not None:
                    print(f"[AI] 推荐落子: 第二行 第{best_col + 1}列  (cell [1, {best_col}])")
                    move_distance = (best_col - 1) * CELL_LENTH
                    # 在 Warped Grid 上高亮推荐位置（黄框 + AI 标记）
                    cx_ai = best_col * step + step // 2
                    cy_ai = 1 * step + step // 2
                    half = step // 2 - 4
                    cv2.rectangle(warped_grid_display,
                                  (cx_ai - half, cy_ai - half),
                                  (cx_ai + half, cy_ai + half),
                                  (0, 255, 255), 3)  # 黄色粗框
                    cv2.putText(warped_grid_display, "AI",
                                (cx_ai - 20, cy_ai - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
                else:
                    print("[AI] 第二行已满，无可用落子位置")
                # ------------------------------------
            else:
                print("[调试] 未检测到完整物体边框")

            # 显示窗口
            cv2.imshow("RGB", cv_image)
            cv2.imshow("Edges", edges)
            cv2.imshow("Contour", contour_image)
            if warped_grid_display is not None:
                cv2.imshow("Warped Grid", warped_grid_display)
            else:
                blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), np.uint8)
                cv2.imshow("Warped Grid", blank)

            # 按 'q' 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[调试] 用户退出")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[调试] 资源已释放")


if __name__ == "__main__":
    main()
