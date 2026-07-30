#!/usr/bin/env python3
"""
ROS2 版本：九宫格 + TF 坐标变换 + 抓取目标发布节点
检测物体 → 透视变换 → 3×3 网格 → 红/蓝分类 → 相机坐标 → TF 转换到目标坐标系
"""
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseStamped
import yaml
import os

# ------------------- 常量配置 -------------------
OUTPUT_SIZE = 600                # 最终九宫格显示尺寸
WHOLE_WIDTH = 600                # 透视变换后整个物体的宽度（像素）
WHOLE_HEIGHT = int(WHOLE_WIDTH * (200 / 160))  # 整个物体高度
GRID_HEIGHT = int(WHOLE_HEIGHT * (160 / 200))  # 九宫格高度

MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 20

# 固定深度假设（米）
FIXED_DEPTH = 0.5

# TF 坐标系名称
CAMERA_FRAME = "camera_link"
TARGET_FRAME = "base_link"

# 相机内参默认值（将从 YAML 加载）
fx = 600.0
fy = 600.0
cx = 320.0
cy = 240.0
camera_mtx = None
camera_dist = None


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
    """透视变换到指定尺寸 (width, height)，返回变换后图像和变换矩阵"""
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


def pixel_to_camera_point(u, v, depth):
    """像素坐标 (u,v) + 深度 → 相机坐标系 3D 点 (x, y, z)"""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z


# ------------------- ROS2 节点 -------------------
class NineGridTFNode(Node):
    def __init__(self):
        super().__init__("nine_grid_tf")

        # 加载相机标定
        self.load_camera_calibration()

        # TF2 相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 图像订阅
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            "/usb_cam/image_raw",
            self.image_callback,
            1
        )

        # 抓取目标发布者
        self.grasp_pub = self.create_publisher(
            PoseStamped, '/grasp_target', 10
        )

        # 声明参数
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 150)

        # OpenCV 窗口
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

        self.get_logger().info("九宫格 TF 节点已启动")

    def nothing(self, x):
        pass

    def trackbar_callback(self):
        canny_low = cv2.getTrackbarPos("Canny_low", "Canny Threshold")
        canny_high = cv2.getTrackbarPos("Canny_high", "Canny Threshold")
        self.set_parameters(
            rclpy.parameter.Parameter("canny_low", value=canny_low),
            rclpy.parameter.Parameter("canny_high", value=canny_high),
        )

    def load_camera_calibration(self):
        global fx, fy, cx, cy, camera_mtx, camera_dist

        self.declare_parameter(
            "camera_calib_file",
            "/home/xiaoyu/calibration/camera_info/head_camera.yaml"
        )
        calib_file = (
            self.get_parameter("camera_calib_file")
            .get_parameter_value()
            .string_value
        )

        if not os.path.exists(calib_file):
            self.get_logger().warn(
                "相机标定文件不存在: %s，使用默认内参" % calib_file
            )
            return

        try:
            with open(calib_file, 'r') as f:
                calib_data = yaml.safe_load(f)

            camera_mtx_raw = calib_data.get('camera_matrix')
            camera_dist_raw = calib_data.get('distortion_coefficients')

            if camera_mtx_raw is None or camera_dist_raw is None:
                self.get_logger().warn(
                    "YAML 缺少 camera_matrix 或 distortion_coefficients，"
                    "使用默认内参"
                )
                return

            camera_mtx_arr = np.array(camera_mtx_raw).flatten()
            if camera_mtx_arr.size == 9:
                camera_mtx = camera_mtx_arr.reshape(3, 3)
                fx = float(camera_mtx[0, 0])
                fy = float(camera_mtx[1, 1])
                cx = float(camera_mtx[0, 2])
                cy = float(camera_mtx[1, 2])
                self.get_logger().info(
                    "成功加载相机内参: fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f"
                    % (fx, fy, cx, cy)
                )
            else:
                self.get_logger().warn("相机内参数目不是9个，使用默认值")

            camera_dist = np.array(camera_dist_raw).flatten()
        except Exception as e:
            self.get_logger().error("加载相机标定文件失败: %s" % e)

    def image_callback(self, msg):
        canny_low = self.get_parameter("canny_low").value
        canny_high = self.get_parameter("canny_high").value

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error("格式错误: %s" % e)
            return

        # 畸变校正（可选）
        if camera_mtx is not None and camera_dist is not None:
            try:
                cv_image = cv2.undistort(cv_image, camera_mtx, camera_dist)
            except Exception as e:
                self.get_logger().warn(
                    "畸变校正失败: %s" % e, throttle_duration_sec=5
                )

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
            invM = np.linalg.inv(M)

            grid_region = whole_warped[0:GRID_HEIGHT, 0:WHOLE_WIDTH]
            grid_resized = cv2.resize(grid_region,
                                      (OUTPUT_SIZE, OUTPUT_SIZE))

            # 颜色检测
            warped_hsv = cv2.cvtColor(grid_resized, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 43, 46])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([156, 43, 46])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(warped_hsv, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(warped_hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            lower_blue = np.array([100, 43, 46])
            upper_blue = np.array([124, 255, 255])
            blue_mask = cv2.inRange(warped_hsv, lower_blue, upper_blue)

            cells, step = split_grid_cells(grid_resized.shape)

            warped_grid_display = grid_resized.copy()
            for i in range(1, 3):
                cv2.line(warped_grid_display,
                         (i * step, 0), (i * step, OUTPUT_SIZE),
                         (0, 255, 0), 2)
                cv2.line(warped_grid_display,
                         (0, i * step), (OUTPUT_SIZE, i * step),
                         (0, 255, 0), 2)

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
                    cx_center = col * step + step // 2
                    cy_center = row * step + step // 2

                    # 在正视图上标记
                    cv2.putText(warped_grid_display, label,
                                (cx_center - 20, cy_center),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # 坐标转换：正视图 → 原始图像像素坐标
                    pt_warped = np.array(
                        [[[cx_center, cy_center]]], dtype=np.float32
                    )
                    pt_original = cv2.perspectiveTransform(pt_warped, invM)
                    orig_u, orig_v = pt_original[0][0]

                    cv2.circle(cv_image, (int(orig_u), int(orig_v)),
                               5, color, -1)
                    cv2.putText(cv_image, label,
                                (int(orig_u) - 15, int(orig_v) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # 像素 → 相机坐标系 3D 点
                    cam_x, cam_y, cam_z = pixel_to_camera_point(
                        orig_u, orig_v, FIXED_DEPTH
                    )

                    # 构建 PointStamped
                    now = self.get_clock().now().to_msg()
                    point_cam = PointStamped()
                    point_cam.header.frame_id = CAMERA_FRAME
                    point_cam.header.stamp = now
                    point_cam.point.x = cam_x
                    point_cam.point.y = cam_y
                    point_cam.point.z = cam_z

                    # TF 转换到目标坐标系
                    try:
                        can = self.tf_buffer.can_transform(
                            TARGET_FRAME,
                            CAMERA_FRAME,
                            rclpy.time.Time(),
                            timeout=Duration(seconds=0.1),
                        )
                        if not can:
                            self.get_logger().warn(
                                "TF 转换不可用: %s → %s"
                                % (CAMERA_FRAME, TARGET_FRAME)
                            )
                            continue

                        point_target = self.tf_buffer.transform(
                            point_cam,
                            TARGET_FRAME,
                            timeout=Duration(seconds=0.1),
                        )
                        target_x = point_target.point.x
                        target_y = point_target.point.y
                        target_z = point_target.point.z

                        self.get_logger().info(
                            "检测到 %s 在单元格 (%d, %d) | "
                            "相机坐标: (%.3f, %.3f, %.3f) | "
                            "%s 坐标: (%.3f, %.3f, %.3f)",
                            label, row, col,
                            cam_x, cam_y, cam_z,
                            TARGET_FRAME,
                            target_x, target_y, target_z,
                        )
                    except (tf2_ros.LookupException,
                            tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException) as e:
                        self.get_logger().warn("TF 转换失败: %s" % e)
                        self.get_logger().info(
                            "检测到 %s 在单元格 (%d, %d) | "
                            "相机坐标: (%.3f, %.3f, %.3f)",
                            label, row, col,
                            cam_x, cam_y, cam_z,
                        )
        else:
            self.get_logger().warn(
                "未检测到完整物体边框", throttle_duration_sec=5
            )

        cv2.imshow("RGB", cv_image)
        cv2.imshow("Edges", edges)
        cv2.imshow("Contour", contour_image)
        if warped_grid_display is not None:
            cv2.imshow("Warped Grid", warped_grid_display)
        else:
            blank = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), np.uint8)
            cv2.imshow("Warped Grid", blank)
        cv2.waitKey(5)


def main(args=None):
    rclpy.init(args=args)
    node = NineGridTFNode()
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
