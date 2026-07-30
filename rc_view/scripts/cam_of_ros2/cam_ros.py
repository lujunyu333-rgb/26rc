#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 相机发布节点 - 将 UVC 和 RealSense 相机数据发布为 ROS2 Image 话题。
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import cv2
import message_filters
import threading
import queue
import copy
import numpy as np
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from rc_view.scripts.cam_of_ros2.allcamera import uvc_cam, rs_cam


class UVCameraPublisher(Node):
    """UVC USB 摄像头发布节点"""

    def __init__(self, device_path=0, name="uvc_cam", format_code=0,
                 width=640, height=480, fps=30, frame_id='camera_link'):
        super().__init__(f'uvc_publisher_{name}')

        self.cam = uvc_cam(
            device_path=device_path,
            name=name,
            funcation=format_code,
            width=width,
            height=height,
            fps=fps
        )

        if not self.cam.isOpened():
            self.get_logger().error(f"无法打开摄像头 {name}")
            raise RuntimeError(f"Camera open failed: {name}")

        self.pub_name = f"camera/color/image_raw/{name}"
        self.image_pub = self.create_publisher(Image, self.pub_name, 10)
        self.bridge = CvBridge()
        self.frame_id = frame_id
        self.cam_fps = self.cam.cam_fps

        # 使用 Timer 驱动发布，避免手动 spin 循环
        period = 1.0 / max(self.cam_fps, 1)
        self._timer = self.create_timer(period, self._publish_callback)

        self.get_logger().info(
            f"UVC 相机发布者启动 [{name}] → {self.pub_name} @ {self.cam_fps}Hz")

    def _publish_callback(self):
        """Timer 回调：读取一帧并发布"""
        ret, frame = self.cam.read_one_frame()
        if not ret:
            self.get_logger().warning(
                f"[{self.cam.cam_name}] 读取失败", throttle_duration_sec=5)
            return

        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 转换失败: {e}")
            return

        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)

    def shutdown(self):
        if self.cam is not None:
            self.cam.close_uvc_camera()
            self.get_logger().info("UVC 摄像头资源已释放")


class RealSensePublisher(Node):
    """RealSense D435i 发布节点"""

    def __init__(self, width=640, height=480, fps=30, periodic=True):
        super().__init__('realsense_publisher')

        # ROS2 参数
        self.declare_parameter('enable_filter', True)
        self.declare_parameter('color_frame_id', 'camera_color_optical_frame')
        self.declare_parameter('depth_frame_id', 'camera_depth_optical_frame')
        self.declare_parameter('depth_unit', 'mm')

        enable_filter = self.get_parameter('enable_filter').value
        self.color_frame_id = self.get_parameter('color_frame_id').value
        self.depth_frame_id = self.get_parameter('depth_frame_id').value
        self.depth_unit = self.get_parameter('depth_unit').value

        self.periodic = periodic
        self.cam = rs_cam(width=width, height=height, fps=fps)

        self.depth_scale = self.cam.depth_scale

        self.color_pub = self.create_publisher(Image, 'camera/color/image_raw', 60)
        self.depth_pub = self.create_publisher(Image, 'camera/depth/image_raw', 60)
        self.color_info_pub = self.create_publisher(CameraInfo, 'camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, 'camera/depth/camera_info', 10)
        self.depth_scale_pub = self.create_publisher(Float32, 'camera/depth/scale', 10)

        self.bridge = CvBridge()

        # 发布内参和深度缩放因子
        self.publish_camera_info()
        self.publish_depth_scale()

        # Timer 驱动
        period = 1.0 / max(fps, 1)
        self._pub_timer = self.create_timer(period, self._publish_callback)

        self.get_logger().info(
            f"RealSense 发布者启动 | color→{self.color_pub.topic} "
            f"depth→{self.depth_pub.topic} @ {fps}Hz")

    def _make_camera_info(self, frame_id):
        """创建一份新的 CameraInfo 消息 (独立副本)"""
        if self.cam.intrinsics is None:
            return None

        info = CameraInfo()
        info.height = self.cam.height
        info.width = self.cam.width
        info.distortion_model = "plumb_bob"

        # 内参矩阵 K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        info.k[0] = self.cam.intrinsics.fx
        info.k[2] = self.cam.intrinsics.ppx
        info.k[4] = self.cam.intrinsics.fy
        info.k[5] = self.cam.intrinsics.ppy
        info.k[8] = 1.0

        # 畸变系数
        info.d = self.cam.intrinsics.coeffs

        # 投影矩阵 P: [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0]
        info.p[0] = self.cam.intrinsics.fx
        info.p[2] = self.cam.intrinsics.ppx
        info.p[5] = self.cam.intrinsics.fy
        info.p[6] = self.cam.intrinsics.ppy
        info.p[10] = 1.0

        now = self.get_clock().now().to_msg()
        info.header.stamp = now
        info.header.frame_id = frame_id
        return info

    def publish_camera_info(self):
        """发布彩色和深度 CameraInfo（各独立副本）"""
        color_info = self._make_camera_info(self.color_frame_id)
        depth_info = self._make_camera_info(self.depth_frame_id)

        if color_info is None or depth_info is None:
            self.get_logger().warn("未能获取相机内参")
            return

        def _publish():
            now = self.get_clock().now().to_msg()
            color_info.header.stamp = now
            depth_info.header.stamp = now
            self.color_info_pub.publish(color_info)
            self.depth_info_pub.publish(depth_info)

        _publish()
        self.get_logger().info("相机内参已发布")

        if self.periodic and not hasattr(self, '_info_timer'):
            self._info_timer = self.create_timer(1.0, _publish)

    def publish_depth_scale(self):
        """发布深度缩放因子"""
        if self.depth_scale is None:
            self.get_logger().warn("深度缩放因子无效")
            return

        msg = Float32()
        msg.data = self.depth_scale

        def _publish():
            self.depth_scale_pub.publish(msg)

        _publish()
        self.get_logger().info(f"深度缩放因子: {self.depth_scale} m/count")

        if self.periodic and not hasattr(self, '_scale_timer'):
            self._scale_timer = self.create_timer(1.0, _publish)

    def _publish_callback(self):
        """Timer 回调：非阻塞获取最新帧并发布彩色+深度"""
        try:
            color_image, depth_image = self.cam.read_one_frame()

            if color_image is None or depth_image is None:
                self.get_logger().warning(
                    "获取 RealSense 帧失败 (后台线程尚未就绪)", throttle_duration_sec=5)
                return

            # 深度图单位转换
            if self.depth_unit == 'mm':
                depth_msg_image = (depth_image * self.depth_scale * 1000).astype(np.uint16)
                depth_encoding = '16UC1'
            else:
                depth_msg_image = (depth_image * self.depth_scale).astype(np.float32)
                depth_encoding = '32FC1'

            now = self.get_clock().now().to_msg()

            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = now
            color_msg.header.frame_id = self.color_frame_id

            depth_msg = self.bridge.cv2_to_imgmsg(depth_msg_image, encoding=depth_encoding)
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = self.depth_frame_id

            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)
        except Exception as e:
            self.get_logger().error(
                f"发布回调异常: {e}", throttle_duration_sec=5)

    def shutdown(self):
        if self.cam is not None:
            self.cam.close_rscam()
            self.get_logger().info("RealSense 相机资源已释放")


class AsyncImageReceiver(Node):
    """异步图像接收与显示节点"""

    def __init__(self, topics, queue_size=10, slop=0.1, display_fps=30):
        super().__init__('async_image_receiver')
        self.topics = topics
        self.bridge = CvBridge()
        self.image_queue = queue.Queue(maxsize=50)
        self.running = True
        self.display_fps = display_fps

        self.subs = [
            message_filters.Subscriber(self, Image, t, qos_profile=queue_size)
            for t in topics
        ]

        if len(self.subs) == 1:
            self.subs[0].registerCallback(self._single_cb)
        else:
            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.subs, queue_size, slop)
            self.sync.registerCallback(self._multi_cb)

        self.get_logger().info(f"异步接收器初始化 | topics={topics}")

    def _single_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        try:
            self.image_queue.put_nowait(([img], [msg]))
        except queue.Full:
            pass

    def _multi_cb(self, *msgs):
        imgs = [self.bridge.imgmsg_to_cv2(m, 'passthrough') for m in msgs]
        try:
            self.image_queue.put_nowait((imgs, msgs))
        except queue.Full:
            pass

    def display_loop(self):
        rate = self.create_rate(self.display_fps)
        while self.running and rclpy.ok():
            try:
                cv_images, _raw_msgs = self.image_queue.get(timeout=0.1)
            except queue.Empty:
                cv2.waitKey(1)
                rate.sleep()
                continue

            for i, img in enumerate(cv_images):
                win_name = f"Camera {i}"
                if img.dtype == np.uint16:
                    disp = cv2.normalize(
                        img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                elif img.dtype == np.float32:
                    disp = cv2.normalize(
                        img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    disp = img
                cv2.imshow(win_name, disp)

            cv2.waitKey(1)
            rate.sleep()

        cv2.destroyAllWindows()

    def start(self):
        self.display_thread = threading.Thread(target=self.display_loop)
        self.display_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'display_thread'):
            self.display_thread.join()
        self.get_logger().info("异步接收器已停止")


def main(args=None):
    """同时启动 UVC 和 RealSense 发布节点（使用 MultiThreadedExecutor）"""
    rclpy.init(args=args)

    try:
        uvc_node = UVCameraPublisher(device_path=0, name="uvc_cam")
    except RuntimeError:
        uvc_node = None

    try:
        rs_node = RealSensePublisher()
    except RuntimeError:
        rs_node = None

    executor = MultiThreadedExecutor(num_threads=2)
    if uvc_node is not None:
        executor.add_node(uvc_node)
    if rs_node is not None:
        executor.add_node(rs_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if uvc_node is not None:
            uvc_node.shutdown()
            uvc_node.destroy_node()
        if rs_node is not None:
            rs_node.shutdown()
            rs_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
