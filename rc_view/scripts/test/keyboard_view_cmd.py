#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘模拟节点 — 键入任意数字发布到 /camera/view_cmd (UInt8)。

操作:
  0-9       → 输入数字（多位数自动拼接）
  Backspace → 删除最后一位
  Enter/Space → 发送当前数字
  g         → 快捷发送 view_cmd=1（抓取）
  r         → 发送 view_cmd=0（复位）
  q / Ctrl+C → 退出

用法：
  ros2 run rc_view keyboard_view_cmd
  ros2 run rc_view keyboard_view_cmd --ros-args -p topic:=/camera/view_cmd
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8


# ── 控制字符 ────────────────────────────────────────────
BACKSPACE = '\x7f'


def _get_key():
    """非阻塞读取单字符按键，无输入时返回 None"""
    dr, _, _ = select.select([sys.stdin], [], [], 0.05)
    if dr:
        return sys.stdin.read(1)
    return None


class KeyboardViewCmd(Node):
    """键盘 → /camera/view_cmd 桥接节点"""

    def __init__(self):
        super().__init__('keyboard_view_cmd')

        topic = self.declare_parameter('topic', '/camera/view_sig').value
        self._pub = self.create_publisher(UInt8, topic, 10)

        self._buf = ''          # 当前输入缓冲区

        self.get_logger().info(
            f'键盘模拟节点就绪 → 发布到 [{topic}]\n'
            f'  0-9       → 输入数字\n'
            f'  Backspace → 删除\n'
            f'  Enter/Space → 发送当前数字\n'
            f'  g         → 快捷发送 1\n'
            f'  r         → 发送 0\n'
            f'  q         → 退出')

        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        # 100 Hz 轮询键盘
        self._timer = self.create_timer(0.01, self._poll)

    def _send(self, value: int):
        """发布并打印"""
        value = max(0, min(255, value))   # 钳位到 UInt8 范围
        self._pub.publish(UInt8(data=value))
        self.get_logger().info(f'>>> view_cmd={value}')

    def _poll(self):
        key = _get_key()
        if key is None:
            return

        if '0' <= key <= '9':
            self._buf += key
            self.get_logger().info(f'输入: [{self._buf}]', throttle_duration_sec=0.3)

        elif key == BACKSPACE:
            if self._buf:
                self._buf = self._buf[:-1]
                self.get_logger().info(f'输入: [{self._buf}]' if self._buf else '输入: [空]',
                                       throttle_duration_sec=0.3)

        elif key in ('\r', ' '):          # Enter 或 Space → 发送
            if self._buf:
                self._send(int(self._buf))
                self._buf = ''
            else:
                self.get_logger().info('缓冲区为空，请先输入数字', throttle_duration_sec=0.5)

        elif key == 'g':                   # 快捷抓取
            self._send(1)
            self._buf = ''

        elif key == 'r':                   # 复位
            self._send(0)
            self._buf = ''

        elif key == 'q':
            self.get_logger().info('退出')
            raise KeyboardInterrupt

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardViewCmd()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
