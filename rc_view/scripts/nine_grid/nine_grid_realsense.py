#!/usr/bin/env python3
"""
ROS2 版本：九宫格 + RealSense 相机节点
通过霍夫直线检测物体边框 → 透视变换 → 3×3网格 → 红/蓝分类
集成 tic_tac_toe_ai：识别棋局后自动推荐 AI 最优落子（约束：只能下第二行）
（保留 RealSense 管线初始化，图像源为 ROS topic）
"""
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int8, Float32
import cv2
import numpy as np

from tic_tac_toe_ai import NineGrid

# ------------------- 常量配置 -------------------
OUTPUT_SIZE = 600                # 最终九宫格显示尺寸
WHOLE_WIDTH = 600                # 透视变换后整个物体的宽度（像素）
WHOLE_HEIGHT = int(WHOLE_WIDTH * (200 / 160))  # 整个物体高度 = 600 * 1.25 = 750
GRID_HEIGHT = int(WHOLE_HEIGHT * (160 / 200))  # 九宫格高度 = 750 * 0.8 = 600
CELL_DISTANCE_M = 0.5                # 每个格子的物理边长（米），用于计算移动距离

MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 20
CELL_LENTH = 0.5              # 每个格子的物理边长（米），用于计算移动距离

# ------------------- HSV 阈值（宏定义参数）-------------------
# 红色阈值（双范围，包裹 HSV 的 H 通道两端）
RED_H_LOW1, RED_H_HIGH1 = 0, 10
RED_H_LOW2, RED_H_HIGH2 = 156, 180
RED_S_LOW, RED_S_HIGH = 43, 255
RED_V_LOW, RED_V_HIGH = 46, 255

# 蓝色阈值
BLUE_H_LOW, BLUE_H_HIGH = 100, 124
BLUE_S_LOW, BLUE_S_HIGH = 43, 255
BLUE_V_LOW, BLUE_V_HIGH = 46, 255

# ------------------- 深度有效范围（宏定义参数）-------------------
DEPTH_MIN = 1.5   # 深度有效范围下限（米）
DEPTH_MAX = 2.0   # 深度有效范围上限（米）

# ------------------- AI 执棋颜色（宏定义参数）-------------------
AI_LABEL = "B"    # AI 执子颜色: 'B' 蓝方（对手 'R' 红方）/ 'R' 红方（对手 'B' 蓝方）

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


# ------------------- 深度验证辅助函数 -------------------
def map_warped_to_original(pts_warped, M_inv):
    """将透视变换后的坐标映射回原始图像坐标"""
    pts = np.array(pts_warped, dtype=np.float32).reshape(-1, 1, 2)
    pts_orig = cv2.perspectiveTransform(pts, M_inv)
    return pts_orig.reshape(-1, 2)


def validate_depth_in_cell(depth_image, pts_orig, depth_scale):
    """检查采样点的深度是否在有效范围内。
    返回 -1 表示无深度数据（跳过验证），否则返回落在 [DEPTH_MIN, DEPTH_MAX] 内的点数。"""
    if depth_image is None or depth_scale is None:
        return -1

    h, w = depth_image.shape
    valid_count = 0
    for x, y in pts_orig:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            depth_val = depth_image[yi, xi] * depth_scale
            if DEPTH_MIN <= depth_val <= DEPTH_MAX:
                valid_count += 1
    return valid_count


# ------------------- ROS2 节点 -------------------
class NineGridRealsenseNode(Node):
    def __init__(self):
        super().__init__("nine_grid_realsense")

        self.bridge = CvBridge()

        # 声明参数
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)

        # 存储最新接收的消息（供后续深度/3D 计算使用）
        self.latest_color_image = None
        self.depth_image = None
        self.depth_scale = None
        self.camera_info = None

        # 订阅彩色图像
        self.color_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            60
        )

        # 订阅深度图像
        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            60
        )

        # 订阅深度因子
        self.scale_sub = self.create_subscription(
            Float32,
            "/camera/depth/scale",
            self.scale_callback,
            1
        )

        # 订阅彩色相机内参
        self.info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.info_callback,
            1
        )

        # 创建 Canny 阈值调节窗口
        cv2.namedWindow('Canny Threshold')
        cv2.createTrackbar('Canny_low', 'Canny Threshold', 50, 255,
                           self.nothing)
        cv2.createTrackbar('Canny_high', 'Canny Threshold', 150, 255,
                           self.nothing)

        cv2.namedWindow("RGB")
        cv2.namedWindow("Edges")
        cv2.namedWindow("Contour")
        cv2.namedWindow("Warped Grid")

        # 定时读取 trackbar
        self.timer = self.create_timer(1.0 / 30.0, self.trackbar_callback)

        # 初始化 #字棋 AI（执子颜色由宏 AI_LABEL 决定，对手为另一方；只能下第二行）
        self.ai = NineGrid(ai_label=AI_LABEL)

        # 发布位移方向 (Float32: 正值=向左, 负值=向右, 0=不动)
        self.offset_pub = self.create_publisher(Float32, '/camera/nine_grid/offset', 30)

        # 订阅判断请求（收到 1 时触发一次判断）
        self.request_sub = self.create_subscription(
            Int8,
            '/camera/nine_grid/request',
            self.request_callback,
            1
        )

        self.get_logger().info("九宫格 RealSense + #字棋 AI 节点已启动")

    def nothing(self, x):
        pass

    def depth_callback(self, msg):
        """存储最新深度图像"""
        self.depth_image = msg

    def scale_callback(self, msg):
        """存储深度因子（单位: 米/深度值）"""
        self.depth_scale = msg.data

    def info_callback(self, msg):
        """存储彩色相机内参"""
        self.camera_info = msg

    def trackbar_callback(self):
        """定时读取 Canny 阈值参数"""
        canny_low = cv2.getTrackbarPos("Canny_low", "Canny Threshold")
        canny_high = cv2.getTrackbarPos("Canny_high", "Canny Threshold")
        self.set_parameters(
            rclpy.parameter.Parameter("canny_low", value=canny_low),
            rclpy.parameter.Parameter("canny_high", value=canny_high),
        )

    def image_callback(self, msg):
        """存储最新彩色图像，并显示 RGB 窗口"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error("格式错误: %s" % e)
            return

        self.latest_color_image = cv_image
        cv2.imshow("RGB", cv_image)
        cv2.waitKey(5)

    def request_callback(self, msg):
        """收到判断请求 (data==1) 后，基于最新彩色图像进行一次棋局判断并发送位移方向"""
        if msg.data != 1:
            return

        if self.latest_color_image is None:
            self.get_logger().warn("尚未收到彩色图像，无法判断")
            return

        cv_image = self.latest_color_image
        canny_low = self.get_parameter("canny_low").value
        canny_high = self.get_parameter("canny_high").value

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(filtered, canny_low, canny_high)

        corners = find_whole_object_corners(edges, cv_image.shape)

        contour_image = cv_image.copy()
        warped_grid_display = None

        if corners is not None:
            int_corners = corners.astype(int)
            cv2.polylines(contour_image, [int_corners], True, (0, 0, 255), 2)

            whole_warped, M = warp_perspective(cv_image, corners,
                                               (WHOLE_WIDTH, WHOLE_HEIGHT))
            M_inv = np.linalg.inv(M)

            grid_region = whole_warped[0:GRID_HEIGHT, 0:WHOLE_WIDTH]
            grid_resized = cv2.resize(grid_region,
                                      (OUTPUT_SIZE, OUTPUT_SIZE))

            warped_hsv = cv2.cvtColor(grid_resized, cv2.COLOR_BGR2HSV)

            # 红色掩码（使用宏定义阈值）
            lower_red1 = np.array([RED_H_LOW1, RED_S_LOW, RED_V_LOW])
            upper_red1 = np.array([RED_H_HIGH1, RED_S_HIGH, RED_V_HIGH])
            lower_red2 = np.array([RED_H_LOW2, RED_S_LOW, RED_V_LOW])
            upper_red2 = np.array([RED_H_HIGH2, RED_S_HIGH, RED_V_HIGH])
            red_mask1 = cv2.inRange(warped_hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(warped_hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            # 蓝色掩码（使用宏定义阈值）
            lower_blue = np.array([BLUE_H_LOW, BLUE_S_LOW, BLUE_V_LOW])
            upper_blue = np.array([BLUE_H_HIGH, BLUE_S_HIGH, BLUE_V_HIGH])
            blue_mask = cv2.inRange(warped_hsv, lower_blue, upper_blue)

            cells, step = split_grid_cells(grid_resized.shape)

            # 准备深度图像（用于验证棋子是否在格子内，排除误判）
            depth_np = None
            if self.depth_image is not None:
                try:
                    depth_np = self.bridge.imgmsg_to_cv2(
                        self.depth_image, desired_encoding="mono16")
                except CvBridgeError:
                    depth_np = None

            warped_grid_display = grid_resized.copy()
            for i in range(1, 3):
                cv2.line(warped_grid_display,
                         (i * step, 0), (i * step, OUTPUT_SIZE),
                         (0, 255, 0), 2)
                cv2.line(warped_grid_display,
                         (0, i * step), (OUTPUT_SIZE, i * step),
                         (0, 255, 0), 2)

            # 每帧重置九宫格标签矩阵
            global jiu
            jiu = np.array([["", "", ""], ["", "", ""], ["", "", ""]])

            for idx, (y1, y2, x1, x2) in enumerate(cells):
                cell_red = red_mask[y1:y2, x1:x2]
                red_pixels = cv2.countNonZero(cell_red)
                total_pixels = (y2 - y1) * (x2 - x1)
                red_ratio = red_pixels / total_pixels
                has_red_color = red_ratio > 0.1

                cell_blue = blue_mask[y1:y2, x1:x2]
                blue_pixels = cv2.countNonZero(cell_blue)
                blue_ratio = blue_pixels / total_pixels
                has_blue_color = blue_ratio > 0.1

                if has_red_color or has_blue_color:
                    row, col = divmod(idx, 3)
                    # 在 cell 中心、下方 1/3、左方 1/3、右方 1/3 处采样深度
                    cx_w = col * step + step // 2
                    cy_w = row * step + step // 2
                    offset_px = step // 3
                    sample_pts_warped = [
                        (cx_w, cy_w),                    # 中心
                        (cx_w, cy_w + offset_px),        # 下方 1/3
                        (cx_w - offset_px, cy_w),        # 左方 1/3
                        (cx_w + offset_px, cy_w),        # 右方 1/3
                    ]

                    # 深度验证：至少 2 个采样点落在有效深度范围内才确认棋子
                    depth_ok = True  # 无深度数据时默认放行
                    if M_inv is not None and depth_np is not None and self.depth_scale is not None:
                        pts_orig = map_warped_to_original(sample_pts_warped, M_inv)
                        valid_n = validate_depth_in_cell(depth_np, pts_orig, self.depth_scale)
                        if valid_n >= 0:
                            depth_ok = valid_n >= 2

                    if not depth_ok:
                        label = ""
                        color = (255, 255, 255)
                    elif has_red_color:
                        label = "R"
                        color = (0, 0, 255)
                    else:
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
                                (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, color, 2)

            # ---------- #字棋 AI 决策 ----------
            self.ai.from_array(jiu)
            best_col = self.ai.best_move()
            if best_col is not None:
                # 机器人正对中间列(col=1)，计算偏移方向
                # 1=向左, -1=向右, 0=不动
                if best_col == 0:
                    offset = 1 * CELL_DISTANCE_M       # 左列，需要向左移动
                elif best_col == 1:
                    offset = 0       # 中间列，不动
                else:
                    offset = -1 * CELL_DISTANCE_M      # 右列，需要向右移动

                msg_out = Float32()
                msg_out.data = offset
                self.offset_pub.publish(msg_out)

                self.get_logger().info(
                    "AI 推荐落子: 第二行 第%d列  offset=%d"
                    % (best_col + 1, offset)
                )
                # 在 Warped Grid 上高亮推荐位置（黄框 + AI 标记）
                cx_ai = best_col * step + step // 2
                cy_ai = 1 * step + step // 2
                half = step // 2 - 4
                cv2.rectangle(warped_grid_display,
                              (cx_ai - half, cy_ai - half),
                              (cx_ai + half, cy_ai + half),
                              (0, 255, 255), 3)
                cv2.putText(warped_grid_display, "AI",
                            (cx_ai - 20, cy_ai - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)
            else:
                self.get_logger().warn("AI 第二行已满，无可用落子位置")
                # 无可落子位置，发送 0（不动）
                msg_out = Int8()
                msg_out.data = 0
                self.offset_pub.publish(msg_out)
            # ------------------------------------
        else:
            self.get_logger().warn(
                "未检测到完整物体边框", throttle_duration_sec=5
            )

        cv2.imshow("Edges", edges)
        cv2.imshow("Contour", contour_image)
        if warped_grid_display is not None:
            cv2.imshow("Warped Grid", warped_grid_display)
        else:
            blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), np.uint8)
            cv2.imshow("Warped Grid", blank)
        cv2.waitKey(5)

    def destroy_node(self):
        """重写 destroy_node 以清理资源"""
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NineGridRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
