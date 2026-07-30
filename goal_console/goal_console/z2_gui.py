#!/usr/bin/env python3
"""
z2_gui — 第二比赛 GUI

功能：
- BLUE/RED 阵营选择
- startup pose 状态显示
- 开始/取消任务
- 状态和目标显示
"""

import math
import signal
import tkinter as tk
import tkinter.ttk as ttk
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import ExternalShutdownException

from std_msgs.msg import UInt8, String


class Z2GUI(Node):
    def __init__(self):
        super().__init__('z2_gui')

        self._qos_transient = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # 发布
        self._alliance_pub = self.create_publisher(
            UInt8, '/competition/alliance', self._qos_transient)
        self._start_pub = self.create_publisher(
            UInt8, '/z2/mission/start', 10)
        self._cancel_pub = self.create_publisher(
            UInt8, '/z2/mission/cancel', 10)

        # 订阅
        self._startup_status_sub = self.create_subscription(
            UInt8, '/competition/startup_pose_status',
            self._startup_status_cb, self._qos_transient)
        self._startup_msg_sub = self.create_subscription(
            String, '/competition/startup_pose_message',
            self._startup_msg_cb, self._qos_transient)
        self._mission_status_sub = self.create_subscription(
            String, '/z2/mission/status', self._mission_status_cb, self._qos_transient)
        self._target_sub = self.create_subscription(
            String, '/z2/mission/target', self._target_cb, 10)

        # 状态
        self._alliance: Optional[int] = None
        self._startup_status: int = 0
        self._startup_msg: str = ''
        self._mission_status: str = 'WAIT_START'
        self._target: str = ''
        self._shutting_down = False

        # GUI
        self._root = tk.Tk()
        self._root.title("Z2 第二比赛")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

        main = ttk.Frame(self._root, padding=10)
        main.pack()

        # 阵营选择
        af = ttk.LabelFrame(main, text="阵营", padding=5)
        af.pack(fill=tk.X)
        self._alliance_var = tk.IntVar(value=0)
        self._rb_blue = tk.Radiobutton(af, text="蓝方 BLUE", variable=self._alliance_var,
                                       value=1, command=self._on_alliance)
        self._rb_blue.pack(side=tk.LEFT, padx=5)
        self._rb_red = tk.Radiobutton(af, text="红方 RED", variable=self._alliance_var,
                                      value=2, command=self._on_alliance)
        self._rb_red.pack(side=tk.LEFT, padx=5)

        # 启动区状态
        sf = ttk.LabelFrame(main, text="启动区定位", padding=5)
        sf.pack(fill=tk.X, pady=5)
        self._startup_label = tk.Label(sf, text="未选择阵营", fg="gray")
        self._startup_label.pack()

        # 控制按钮
        bf = ttk.Frame(main)
        bf.pack(pady=5)
        self._btn_start = tk.Button(bf, text="开始第二比赛", bg="lightblue",
                                    command=self._on_start, width=15)
        self._btn_start.pack(side=tk.LEFT, padx=3)
        self._btn_start.config(state=tk.DISABLED)
        self._btn_cancel = tk.Button(bf, text="取消任务", bg="lightcoral",
                                     command=self._on_cancel, width=12)
        self._btn_cancel.pack(side=tk.LEFT, padx=3)
        self._btn_cancel.config(state=tk.DISABLED)

        # 状态
        sf2 = ttk.LabelFrame(main, text="任务状态", padding=5)
        sf2.pack(fill=tk.X, pady=5)
        self._status_label = tk.Label(sf2, text="等待选择阵营", fg="gray")
        self._status_label.pack()

        # 目标
        tf = ttk.LabelFrame(main, text="当前目标", padding=5)
        tf.pack(fill=tk.X, pady=5)
        self._target_label = tk.Label(tf, text="—", fg="gray")
        self._target_label.pack()

        self._set_status_text("等待选择阵营")

        self._spin_period_ms = 50
        self._root.after(self._spin_period_ms, self._ros_spin_once)
        self.get_logger().info('[Z2] GUI started')

    # ================================================================
    #  ROS callbacks
    # ================================================================
    def _startup_status_cb(self, msg: UInt8):
        self._startup_status = msg.data
        self._update_startup_display()

    def _startup_msg_cb(self, msg: String):
        self._startup_msg = msg.data
        self._update_startup_display()

    def _mission_status_cb(self, msg: String):
        self._mission_status = msg.data
        self._set_status_text(msg.data)

    def _target_cb(self, msg: String):
        self._target = msg.data
        self._target_label.config(text=msg.data, fg="black")

    # ================================================================
    #  GUI callbacks
    # ================================================================
    def _on_alliance(self):
        val = self._alliance_var.get()
        if val in (1, 2):
            self._alliance = val
            self._alliance_pub.publish(UInt8(data=val))
            name = 'BLUE' if val == 1 else 'RED'
            self.get_logger().info(f'[Z2] GUI alliance={name}')
            self._update_startup_display()

    def _update_startup_display(self):
        if self._alliance is None:
            self._startup_label.config(text="未选择阵营", fg="gray")
            self._btn_start.config(state=tk.DISABLED)
            return
        if self._startup_status in (1, 2) and self._startup_status == self._alliance:
            color = "green"
            text = f"✓ 就绪 ({self._startup_msg})" if self._startup_msg else "✓ 就绪"
            self._btn_start.config(state=tk.NORMAL)
        elif self._startup_status == 0:
            color = "orange"
            text = "⏳ 定位中..."
            self._btn_start.config(state=tk.DISABLED)
        else:
            color = "red"
            text = f"✗ 失败 (status={self._startup_status})"
            self._btn_start.config(state=tk.DISABLED)
        self._startup_label.config(text=text, fg=color)

    def _on_start(self):
        self.get_logger().info(
            f'[Z2] GUI start button clicked: alliance={self._alliance} '
            f'startup_status={self._startup_status} '
            f'mission_status={self._mission_status}')
        self.get_logger().info('[Z2] GUI publishing /z2/mission/start data=1')
        self._start_pub.publish(UInt8(data=1))
        self.get_logger().info('[Z2] GUI /z2/mission/start published')
        self._btn_start.config(state=tk.DISABLED)
        self._btn_cancel.config(state=tk.NORMAL)
        self._rb_blue.config(state=tk.DISABLED)
        self._rb_red.config(state=tk.DISABLED)

    def _on_cancel(self):
        self.get_logger().info('[Z2] GUI cancel button clicked')
        self._cancel_pub.publish(UInt8(data=1))
        self._enable_controls()

    def _enable_controls(self):
        self._btn_start.config(state=tk.NORMAL)
        self._btn_cancel.config(state=tk.DISABLED)
        self._rb_blue.config(state=tk.NORMAL)
        self._rb_red.config(state=tk.NORMAL)
        self._alliance = None
        self._alliance_var.set(0)

    def _set_status_text(self, text: str):
        self._status_label.config(text=text)
        if 'FAIL' in text or 'CANCELL' in text:
            self._status_label.config(fg="red")
            self._enable_controls()
        elif 'MISSION_DONE' in text:
            self._status_label.config(fg="green")
            self._enable_controls()
        else:
            self._status_label.config(fg="blue")

    # ================================================================
    #  关闭
    # ================================================================
    def _handle_sigterm(self, signum, frame):
        self.get_logger().info('[Z2] SIGTERM received')
        self._root.after(0, self._shutdown)

    def _shutdown(self):
        self._shutting_down = True
        self._root.quit()

    def _on_close(self):
        self.get_logger().info('[Z2] window closed')
        self._shutdown()

    def _ros_spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.001)
        self._root.after(self._spin_period_ms, self._ros_spin_once)


def main(args=None):
    rclpy.init(args=args)
    node = Z2GUI()
    try:
        node._root.mainloop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
