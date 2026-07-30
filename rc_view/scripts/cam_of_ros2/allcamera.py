#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import threading

class uvc_cam:
    def __init__(self, device_path=0, name="uvc_cam", funcation=0, width=640, height=480, fps=30):
        self.video_path = device_path
        self.cam_name = name
        self.cam_width = width
        self.cam_height = height
        self.cam_fps = fps
        self.funcation = funcation  # 0=MJPG, 其他=YUYV
        self.cap = self.open_uvc_camera()

    def open_uvc_camera(self):
        """使用V4L2设备路径打开摄像头"""
        cap = cv2.VideoCapture(self.video_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"Failed to open {self.cam_name} (device {self.video_path})")
            return None
   
        # 设置像素格式
        if self.funcation == 0:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        else:
            fourcc = cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V')
        if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            print(f"{self.cam_name}: 设置格式成功")
        else:
            print(f"{self.cam_name}: 设置格式失败，使用默认格式")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

        # 设置格式后立即验证
        if cap.get(cv2.CAP_PROP_FOURCC) != fourcc:
            print(f"Warning: {self.cam_name} 实际 FOURCC 与设置不符")
        # 检查实际尺寸
        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 0:
            print(f"Error: {self.cam_name} 宽度设置失败")
            cap.release()
            return None

        # 验证设置
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{self.cam_name}: 实际分辨率 {actual_width}x{actual_height} @ {actual_fps:.2f}fps")
        return cap

    def read_one_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame
    
    def get_video(self):    #仅适用于测试相机，不可直接使用于项目中
        try:
            while True:
                ret,mono=self.read_one_frame()
                if not ret:
                    print(f"Warning:{self.cam_name}相机读取失败")
                    break
                cv2.imshow(self.cam_name,mono)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                    break

        except Exception as e:
            print(f"{self.cam_name}出现错误：{e}")
        finally:
            cv2.destroyWindow(self.cam_name)
            print(f"{self.cam_name} 视频已关闭")

    def isOpened(self):
        """检查摄像头是否已打开且有效"""
        return self.cap is not None and self.cap.isOpened()

    def close_uvc_camera(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None            

class rs_cam:
    def __init__(self,width=640,height=480,fps=30):
        self.width=width
        self.height=height
        self.fps=fps

        self.pipeline=rs.pipeline()  
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, self.fps)
        #对齐
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        #率波
        self.decimation = rs.decimation_filter()
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"深度缩放因子: {self.depth_scale}")

        # 获取相机内参（需先获取一帧）
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        self.intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        print(f"Camera intrinsics: fx={self.intrinsics.fx}, fy={self.intrinsics.fy}, "
              f"ppx={self.intrinsics.ppx}, ppx={self.intrinsics.ppy}")

        # 后台帧捕获线程：避免 wait_for_frames 阻塞主线程
        self._frame_lock = threading.Lock()
        self._latest_color = None
        self._latest_depth = None
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        """后台线程：持续从硬件读取帧并存入缓冲区（非阻塞供主线程取用）"""
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                if not frames:
                    continue
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                # 从帧数据构建正确形状的 numpy 数组（深拷贝，独立于帧生命周期）
                color_data = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())

                color_img = np.array(color_data, copy=True).reshape(
                    (self.height, self.width, 3))
                # 深度流分辨率为 848×480（与彩色流不同）
                depth_h = depth_frame.get_height()
                depth_w = depth_frame.get_width()
                depth_img = np.array(depth_data, copy=True).reshape((depth_h, depth_w))

                with self._frame_lock:
                    self._latest_color = color_img
                    self._latest_depth = depth_img
            except Exception as e:
                print(f"rs_cam 后台捕获线程异常: {e}")
                time.sleep(0.01)

    def read_one_frame(self):
        """非阻塞获取最新帧（后台线程已做深拷贝，无需再拷贝）。
        返回 (color_ndarray, depth_ndarray) 或 (None, None)"""
        with self._frame_lock:
            if self._latest_color is None:
                return None, None
            return self._latest_color, self._latest_depth
    
    def get_video(self):     #仅适用于测试相机，不可直接使用于项目中
        try:
            while True:
                # 获取帧数据
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # 显示图像
                cv2.imshow("RealSense color", color_image)
                cv2.imshow("RealSense depth", depth_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):   # 按 'q' 退出
                    break
        except Exception as e:
            print(f"realsense_d435i 出现错误：{e}")
        finally:
            cv2.destroyWindow("RealSense color")
            cv2.destroyWindow("RealSense depth")
            print("realsense_d435i 视频已关闭")

    def close_rscam(self):
        """释放相机和窗口资源"""
        self._running = False
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        self.pipeline.stop()

#测试函数：两个单目相机，一个d435i，开启第三只眼
# (由于opencv同时启动两个相机进程，进程之间相互干扰有时可能无法打开2第二个，也有可能是usb口带宽的问题，两个相机以640*480 @30fps也有30—40GMB/s）
# 稳定性有待后辈提高
def test(width=640,height=480,fps=30):
    realsense_cam=rs_cam()
    cam_left = uvc_cam(0, name="left_cam",width=1024,height=768,fps=30)      # 根据实际情况修改设备索引或路径
    time.sleep(2)   # 等待左侧相机完全初始化
    cam_right = uvc_cam(2, name="right_cam",width=1024,height=768,fps=30)

    if not cam_left.isOpened():
        print("左侧相机打开失败，请检查设备索引")
        return
    if not cam_right.isOpened():
        print("右侧相机打开失败，请检查设备索引")
        return
    try:
        while True:
            color_image, depth_image = realsense_cam.read_one_frame()
            if color_image is None or depth_image is None:
                continue

            # 读取两个单目相机
            ret_left, mono_left = cam_left.read_one_frame()
            ret_right, mono_right = cam_right.read_one_frame()
            if not ret_left:
                print("Warning: 左侧单目相机读取失败")
                break
            if not ret_right:
                print("Warning: 右侧单目相机读取失败")
                break

            # 显示所有画面
            cv2.imshow("RealSense Color", color_image)
            cv2.imshow("Mono Camera Left", mono_left)
            cv2.imshow("Mono Camera Right", mono_right)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                break

    finally:
        realsense_cam.close_rscam()
        cam_left.close_uvc_camera()
        cam_right.close_uvc_camera()
        cv2.destroyAllWindows()

#测试区
if __name__ == "__main__":
    test()

