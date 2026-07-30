#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_downstream.py — 模拟下位机节点

功能：
  - 按 Enter → 发送 Int8=1 到 /camera/yolo/request，触发一次校准
  - 订阅 /camera/yolo/y_offset   查看 move_base_of_yolo 返回的偏移量
  - 订阅 /camera/yolo/grasp      查看抓取信号
  - 订阅 /camera/yolo/obstruction_ok  查看遮挡完成信号
  - 按 q → 退出

用法：
  ros2 run camera test_downstream
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, Float32, Bool
import threading
import sys
import select
import tty
import termios


class DownstreamSim(Node):
    """模拟下位机：键盘触发校准请求，打印收到的偏移量"""

    def __init__(self):
        super().__init__('downstream_simulator')

        # ── 发布：校准请求 ────────────────────────────
        self.pub_request = self.create_publisher(
            Int8, "/camera/yolo/request", 10)

        # ── 订阅：查看返回结果 ─────────────────────────
        self._last_offset = None
        self._last_grasp = None
        self._last_obstruction_ok = None

        self.sub_offset = self.create_subscription(
            Float32, "/camera/yolo/y_offset",
            self._offset_callback, 10)
        self.sub_grasp = self.create_subscription(
            Bool, "/camera/yolo/grasp",
            self._grasp_callback, 10)
        self.sub_obstruction = self.create_subscription(
            Bool, "/camera/yolo/obstruction_ok",
            self._obstruction_callback, 10)

        # ── 键盘输入线程 ───────────────────────────────
        self._running = True
        self._input_thread = threading.Thread(
            target=self._keyboard_loop, daemon=True, name="keyboard")
        self._input_thread.start()

        self.get_logger().info(
            "下位机模拟器就绪 | Enter=发送校准请求 | q=退出")

    def _offset_callback(self, msg: Float32):
        self._last_offset = msg.data
        self.get_logger().info(f"  ← 收到 y_offset = {msg.data:+.1f} px")

    def _grasp_callback(self, msg: Bool):
        if msg.data:
            self._last_grasp = msg.data
            self.get_logger().info("  🎯 收到抓取信号！")

    def _obstruction_callback(self, msg: Bool):
        if msg.data:
            self._last_obstruction_ok = msg.data
            self.get_logger().info("  ✓ 收到遮挡完成信号！")

    def send_request(self):
        """发送一次校准请求 Int8=1"""
        req = Int8()
        req.data = 1
        self.pub_request.publish(req)
        self.get_logger().info("→ 发送校准请求 (Int8=1)")

    def _keyboard_loop(self):
        """键盘监听线程：非阻塞读取单个按键"""
        # 保存终端设置
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self._running and rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\n' or ch == '\r':    # Enter
                        self.send_request()
                    elif ch == ' ':                  # 空格也触发
                        self.send_request()
                    elif ch == 'q' or ch == 'Q':    # 退出
                        self.get_logger().info("收到退出指令")
                        self._running = False
                        raise KeyboardInterrupt
                    elif ch == '\x03':               # Ctrl-C
                        self._running = False
                        raise KeyboardInterrupt
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def shutdown(self):
        self._running = False


def main(args=None):
    rclpy.init(args=args)
    node = DownstreamSim()

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
