#!/usr/bin/env python3
"""
ROS2 版本：HSV 颜色阈值检测 + 边缘检测节点
订阅 /usb_cam/image_raw，通过可调 HSV 阈值检测红/蓝色物体
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector")

        self.bridge = CvBridge()

        # 声明参数（带默认值）
        self.declare_parameter("h_low", 0)
        self.declare_parameter("h_high", 179)
        self.declare_parameter("s_low", 0)
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_low", 0)
        self.declare_parameter("v_high", 255)

        # 订阅图像话题（QoS 使用默认 SensorData 配置）
        self.sub = self.create_subscription(
            Image,
            "/usb_cam/image_raw",
            self.image_callback,
            1
        )

        # 创建 OpenCV 窗口和滑块
        cv2.namedWindow("Blue Threshold")
        cv2.createTrackbar('H_low', 'Blue Threshold', 0, 179, self.nothing)
        cv2.createTrackbar('H_high', 'Blue Threshold', 179, 179, self.nothing)
        cv2.createTrackbar('S_low', 'Blue Threshold', 0, 255, self.nothing)
        cv2.createTrackbar('S_high', 'Blue Threshold', 255, 255, self.nothing)
        cv2.createTrackbar('V_low', 'Blue Threshold', 0, 255, self.nothing)
        cv2.createTrackbar('V_high', 'Blue Threshold', 255, 255, self.nothing)

        cv2.namedWindow("RGB")
        cv2.namedWindow("HSV")
        cv2.namedWindow("Result")

        # 定时器：读取 trackbar 值并更新参数（30 Hz）
        self.timer = self.create_timer(1.0 / 30.0, self.trackbar_callback)

        self.get_logger().info("检测器节点已启动")

    def nothing(self, x):
        """Trackbar 空回调"""
        pass

    def trackbar_callback(self):
        """定时读取 trackbar 位置，更新参数"""
        h_low = cv2.getTrackbarPos("H_low", "Blue Threshold")
        h_high = cv2.getTrackbarPos("H_high", "Blue Threshold")
        s_low = cv2.getTrackbarPos("S_low", "Blue Threshold")
        s_high = cv2.getTrackbarPos("S_high", "Blue Threshold")
        v_low = cv2.getTrackbarPos("V_low", "Blue Threshold")
        v_high = cv2.getTrackbarPos("V_high", "Blue Threshold")

        self.set_parameters(
            rclpy.parameter.Parameter("h_low", value=h_low),
            rclpy.parameter.Parameter("h_high", value=h_high),
            rclpy.parameter.Parameter("s_low", value=s_low),
            rclpy.parameter.Parameter("s_high", value=s_high),
            rclpy.parameter.Parameter("v_low", value=v_low),
            rclpy.parameter.Parameter("v_high", value=v_high),
        )

    def image_callback(self, msg):
        """图像回调：HSV 阈值处理 + 边缘检测"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error("格式错误: %s" % e)
            return

        # 从参数读取当前阈值
        h_low = self.get_parameter("h_low").value
        h_high = self.get_parameter("h_high").value
        s_low = self.get_parameter("s_low").value
        s_high = self.get_parameter("s_high").value
        v_low = self.get_parameter("v_low").value
        v_high = self.get_parameter("v_high").value

        # 将 RGB 图片转换成 HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 在 HSV 空间做直方图均衡化
        h, s, v = cv2.split(hsv_image)
        v = cv2.equalizeHist(v)
        hsv_image = cv2.merge([h, s, v])

        # 设置红蓝阈值及掩码，红色双掩码
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        lower_blue = np.array([h_low, s_low, v_low])
        upper_blue = np.array([h_high, s_high, v_high])

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        r_b_mask = cv2.bitwise_or(red_mask, blue_mask)

        # 开操作（去除噪点）
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        r_b_mask = cv2.morphologyEx(r_b_mask, cv2.MORPH_OPEN, element)

        # 闭操作（连接连通域）
        r_b_mask = cv2.morphologyEx(r_b_mask, cv2.MORPH_CLOSE, element)

        # 边缘检测
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        filter_img = cv2.bilateralFilter(gray_image, 5, 15, 15)
        edges = cv2.Canny(filter_img, 100, 200)
        th_image = cv2.bitwise_or(edges, r_b_mask)

        # 显示结果
        cv2.imshow("RGB", cv_image)
        cv2.imshow("HSV", hsv_image)
        cv2.imshow("Result", th_image)
        cv2.waitKey(5)


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
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
