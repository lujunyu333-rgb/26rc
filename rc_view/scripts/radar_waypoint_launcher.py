#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radar_waypoint_launcher.py — 基于雷达位置自动切换子节点

功能：
  1. 启动时创建子进程运行 start_rs.sh（RealSense 相机，常驻不杀）
  2. 订阅 /Odometry 话题获取雷达当前位置
  3. 三个目标点，以目标点为圆心 0.15m 为判定半径，单向顺序触发：
     · (0, 0)   → 杀旧 task 进程，启动 task1 (start_task1.sh)
     · (3, 3)   → 杀旧 task 进程，启动 task2 (task2.sh，传参切换颜色)
     · (6, 6)   → 杀旧 task 进程，启动 task3 (start_task3.sh)
  4. 父进程持续监听雷达消息，不阻塞

用法：
  python3 radar_waypoint_launcher.py

配置：
  修改下方 WAYPOINTS 列表中的坐标、脚本、参数即可。
"""

import subprocess
import signal
import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int8


# ══════════════════════════════════════════════════════════════
#  配置区 —— 按需修改
# ══════════════════════════════════════════════════════════════

# 脚本目录（相对于本文件）
SCRIPT_DIR = Path(__file__).resolve().parent
LSH_DIR = SCRIPT_DIR / "lsh"

# RealSense 守护脚本（始终运行）
RS_SCRIPT = str(LSH_DIR / "start_rs.sh")

# 三个目标点定义（单向顺序触发，触发后不再回头）
WAYPOINTS = [
    {
        "name": "task1",
        "x": 0.0,
        "y": 0.0,
        "radius": 0.15,               # 判定半径（米）
        "script": str(LSH_DIR / "start_task1.sh"),
        "args": [],                   # task1 无需额外参数
    },
    {
        "name": "task2",
        "x": 3.0,
        "y": 3.0,
        "radius": 0.15,
        "script": str(LSH_DIR / "task2.sh"),
        # task2 默认 blue，这里切换为 red（按需修改）
        "args": ["--ros-args", "-p", "target_color:=red"],
    },
    {
        "name": "task3",
        "x": 6.0,
        "y": 6.0,
        "radius": 0.15,
        "script": str(LSH_DIR / "start_task3.sh"),
        "args": [],
    },
]

# 里程计话题名
ODOM_TOPIC = "/Odometry"

# ══════════════════════════════════════════════════════════════
#  子进程管理
# ══════════════════════════════════════════════════════════════

def _kill_process_tree(proc: subprocess.Popen, stop_event: threading.Event = None):
    """安全终止子进程及其所有子进程（进程组）。

    通过 stop_event 支持外部中断：shutdown 时 set 事件即可快速退出等待，
    避免 proc.wait(timeout=5) 阻塞 shutdown 流程。
    """
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGINT)

        # 分片轮询等待（每次 0.5s），允许 stop_event 中断
        deadline = time.time() + 5.0
        while proc.poll() is None and time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                # shutdown 已触发 → 直接 SIGKILL，不墨迹
                os.killpg(pgid, signal.SIGKILL)
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
                return
            time.sleep(0.5)

        # 超时仍未退出 → SIGKILL
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                pass
    except (ProcessLookupError, OSError):
        pass  # 进程已退出


def _launch_script(script_path: str, args: list[str]) -> Optional[subprocess.Popen]:
    """启动一个脚本作为子进程（新进程组，便于整组终止）"""
    if not Path(script_path).is_file():
        print(f"[ERROR] 脚本不存在: {script_path}")
        return None

    cmd = ["bash", script_path] + args
    print(f"[LAUNCH] 启动子进程: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,   # 创建独立进程组
        )
        return proc
    except Exception as e:
        print(f"[ERROR] 启动失败: {e}")
        return None


def _start_stdout_reader(proc: subprocess.Popen, name: str):
    """后台线程读取子进程 stdout，实时打印到终端"""

    def _reader():
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    print(f"[{name}] {line}")
        except (ValueError, OSError):
            pass  # 管道关闭
        except Exception as e:
            print(f"[{name}] stdout 读取异常: {e}")

    t = threading.Thread(target=_reader, daemon=True, name=f"stdout-{name}")
    t.start()
    return t


# ══════════════════════════════════════════════════════════════
#  ROS2 节点：订阅 /Odometry
# ══════════════════════════════════════════════════════════════

class WaypointLauncherNode(Node):
    """订阅雷达里程计，到达目标点时触发子进程切换"""

    def __init__(self):
        super().__init__("radar_waypoint_launcher")

        # ── 状态 ────────────────────────────────
        # rs_proc:      RealSense 守护子进程（始终运行）
        # task_proc:    当前 task 子进程（到达新目标点时被替换）
        self.rs_proc: Optional[subprocess.Popen] = None
        self.task_proc: Optional[subprocess.Popen] = None

        # 已触发到第几个目标点（0-based index），-1 表示尚未触发任何点
        self._completed_index: int = -1

        # 上一次位置（用于日志节流）
        self._last_logged_waypoint = -1

        # 进程切换请求队列（回调 → 后台 worker）
        self._switch_queue: queue.Queue = queue.Queue()
        # 停止信号（shutdown 时设置，通知 worker / RS 健康检查退出）
        self._stop_event = threading.Event()
        # 进程操作锁
        self._proc_lock = threading.Lock()

        # ── RS 重启保护 ─────────────────────────
        self._rs_restart_count = 0
        self._rs_restart_window_start = 0.0
        self._RS_MAX_RESTARTS = 3          # 30s 内最多重启 3 次
        self._RS_RESTART_WINDOW_S = 30.0

        # ── 错误上报发布者 ─────────────────────
        # /camera/view_sig (Int8): 仅发生错误时发送一次，正常状态不发送
        #   11=RS崩溃  12=RS启动失败  13=RS重启超限
        #   21=Task脚本不存在  22=Task启动失败  23=Task异常退出
        self.pub_view_sig = self.create_publisher(Int8, "/camera/view_sig", 10)

        # 当前运行的 task 名称（用于错误日志），None 表示无 task 运行
        self._active_task_name: Optional[str] = None

        # ── 启动 RS 守护进程 ────────────────────
        self._start_rs()

        # ── 启动后台切换工作线程 ────────────────
        self._switch_thread = threading.Thread(
            target=self._switch_worker,
            daemon=True,
            name="switch-worker",
        )
        self._switch_thread.start()

        # ── RS 健康检查 Timer（每 3 秒） ────────
        self._rs_health_timer = self.create_timer(3.0, self._check_rs_health)

        # ── Task 健康检查 Timer（每 2 秒） ──────
        self._task_health_timer = self.create_timer(2.0, self._check_task_health)

        # ── 订阅雷达里程计 ──────────────────────
        self._odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self._odom_callback,
            30,
        )

        self.get_logger().info(
            f"waypoint_launcher 就绪 | RS守护已启动 | "
            f"监听话题: {ODOM_TOPIC} | "
            f"目标点: {len(WAYPOINTS)} 个 (单向顺序)"
        )
        for i, wp in enumerate(WAYPOINTS):
            self.get_logger().info(
                f"  [{i}] {wp['name']}: ({wp['x']}, {wp['y']}) r={wp['radius']}m → {wp['script']}"
            )

    # ── 错误上报 ──────────────────────────────────

    def _publish_error(self, code: int, detail: str = ""):
        """仅发生错误时发送一次 Int8 到 /camera/view_sig。"""
        msg = Int8(data=code)
        self.pub_view_sig.publish(msg)
        self.get_logger().error(
            f"📡 上报错误码 {code} → /camera/view_sig"
            + (f" | {detail}" if detail else ""))

    # ── RS 守护管理 ────────────────────────────────

    def _start_rs(self):
        """启动 RealSense 守护进程（始终运行）"""
        print(f"\n{'='*60}")
        print(f"[RS] 启动 RealSense 守护进程: {RS_SCRIPT}")
        print(f"{'='*60}")

        if not Path(RS_SCRIPT).is_file():
            self.get_logger().error(f"RS 脚本不存在: {RS_SCRIPT}")
            self._publish_error(21, f"RS脚本不存在: {RS_SCRIPT}")
            return

        self.rs_proc = _launch_script(RS_SCRIPT, [])
        if self.rs_proc:
            _start_stdout_reader(self.rs_proc, "RS")
            self.get_logger().info("RealSense 守护进程已启动")
        else:
            self.get_logger().error("RealSense 守护进程启动失败！")
            self._publish_error(12, "RS启动失败")

    # ── RS 健康检查 ──────────────────────────────

    def _check_rs_health(self):
        """Timer 回调：检查 RS 守护进程是否存活，崩溃则自动重启。

        保护：30 秒窗口内最多重启 3 次，防止无限重启风暴。
        """
        if self._stop_event.is_set():
            return

        if self.rs_proc is None or self.rs_proc.poll() is not None:
            # RS 已退出
            exit_code = self.rs_proc.poll() if self.rs_proc else "N/A"
            self.get_logger().error(
                f"❌ RS 守护进程已退出 (exit_code={exit_code})，尝试重启...")
            self._publish_error(11, f"RS崩溃 exit_code={exit_code}")

            # ── 重启频率保护 ──────────────────────
            now = time.time()
            if now - self._rs_restart_window_start > self._RS_RESTART_WINDOW_S:
                # 窗口过期，重置计数
                self._rs_restart_count = 0
                self._rs_restart_window_start = now

            self._rs_restart_count += 1
            if self._rs_restart_count > self._RS_MAX_RESTARTS:
                self.get_logger().error(
                    f"❌ RS 守护进程 {self._RS_RESTART_WINDOW_S}s 内已重启 "
                    f"{self._RS_MAX_RESTARTS} 次，停止自动重启！请手动检查硬件连接。")
                self._publish_error(13,
                    f"RS重启超限: {self._RS_RESTART_WINDOW_S}s内{self._rs_restart_count}次")
                return

            self.get_logger().warn(
                f"🔄 正在重启 RS 守护进程 "
                f"(第 {self._rs_restart_count}/{self._RS_MAX_RESTARTS} 次)...")
            self._start_rs()

    # ── Task 健康检查 ─────────────────────────────

    def _check_task_health(self):
        """Timer 回调：检查 task 子进程是否异常退出。

        只有在我们未主动杀它（task_proc 仍非 None）且它已退出时才上报。
        """
        if self._stop_event.is_set():
            return

        if self.task_proc is not None and self.task_proc.poll() is not None:
            exit_code = self.task_proc.poll()
            task_name = self._active_task_name or "unknown"
            self.get_logger().error(
                f"❌ Task 子进程异常退出 [{task_name}] "
                f"(pid={self.task_proc.pid}, exit_code={exit_code})")
            self._publish_error(23, f"Task异常退出: {task_name} exit_code={exit_code}")
            self.task_proc = None
            self._active_task_name = None

    # ── Task 子进程管理 ────────────────────────────

    def _switch_worker(self):
        """后台线程：阻塞等待切换请求，执行 kill 旧进程 + launch 新进程。

        通过 _stop_event 支持 shutdown 快速退出：set 后不再处理队列中的请求。
        """
        while not self._stop_event.is_set():
            try:
                wp_index = self._switch_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # 超时，回头检查 _stop_event

            if self._stop_event.is_set():
                break  # shutdown 了，丢弃队列中剩余请求

            self._switch_task(wp_index)

        # 退出前清空队列避免内存警告
        while True:
            try:
                self._switch_queue.get_nowait()
            except queue.Empty:
                break

    def _switch_task(self, wp_index: int):
        """切换到第 wp_index 号目标点对应的脚本"""
        with self._proc_lock:
            wp = WAYPOINTS[wp_index]

            # 1. 杀掉旧 task 子进程（可被 shutdown 中断）
            if self.task_proc is not None:
                self.get_logger().info(f"终止旧 task 子进程 (pid={self.task_proc.pid})")
                _kill_process_tree(self.task_proc, self._stop_event)
                self.task_proc = None
                self._active_task_name = None
                self.get_logger().info("旧 task 子进程已终止")

            # 2. 检查脚本是否存在
            if not Path(wp["script"]).is_file():
                self.get_logger().error(
                    f"Task 脚本不存在: {wp['script']} (目标点 [{wp_index}] {wp['name']})")
                self._publish_error(21, f"脚本不存在: {wp['name']} → {wp['script']}")
                return

            # 3. 启动新 task 子进程
            print(f"\n{'='*60}")
            print(f"[TASK] 到达目标点 [{wp_index}] {wp['name']} ({wp['x']}, {wp['y']})")
            print(f"       启动: {wp['script']} {' '.join(wp['args'])}")
            print(f"{'='*60}")

            new_proc = _launch_script(wp["script"], wp["args"])
            if new_proc:
                self.task_proc = new_proc
                self._active_task_name = wp["name"]
                _start_stdout_reader(new_proc, wp["name"])
                self.get_logger().info(f"[{wp['name']}] 子进程已启动 (pid={new_proc.pid})")
            else:
                self.get_logger().error(f"[{wp['name']}] Task 启动失败！")
                self._publish_error(22, f"Task启动失败: {wp['name']}")

    # ── 里程计回调 ────────────────────────────────

    def _odom_callback(self, msg: Odometry):
        """收到 /Odometry 消息，检查是否到达下一个目标点"""
        pos = msg.pose.pose.position
        x, y = pos.x, pos.y

        # 下一个待触发的目标点
        next_idx = self._completed_index + 1
        if next_idx >= len(WAYPOINTS):
            # 所有目标点都已触发，仅做低频日志
            if self._completed_index != self._last_logged_waypoint:
                self.get_logger().info("所有目标点已触发完毕，继续监听雷达位置...")
                self._last_logged_waypoint = self._completed_index
            return

        wp = WAYPOINTS[next_idx]
        dx = x - wp["x"]
        dy = y - wp["y"]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist <= wp["radius"]:
            # 到达目标点！
            self.get_logger().info(
                f"📍 到达目标点 [{next_idx}] {wp['name']} | "
                f"坐标=({x:.3f}, {y:.3f}) | 目标=({wp['x']}, {wp['y']}) | "
                f"距离={dist:.3f}m <= {wp['radius']}m"
            )
            self._switch_queue.put_nowait(next_idx)
            self._completed_index = next_idx
        else:
            # 低频日志：每隔一定距离打印一次当前位置
            if self._completed_index != self._last_logged_waypoint:
                self._last_logged_waypoint = self._completed_index
            # 每 30 条消息打印一次（大约 0.3-1s 一次，取决于里程计频率）
            if not hasattr(self, "_log_counter"):
                self._log_counter = 0
            self._log_counter += 1
            if self._log_counter % 30 == 0:
                next_wp = WAYPOINTS[next_idx]
                self.get_logger().info(
                    f"📍 当前位置=({x:.3f}, {y:.3f}) | "
                    f"下一个目标 [{next_idx}] {next_wp['name']}: "
                    f"({next_wp['x']}, {next_wp['y']}) | "
                    f"距离={dist:.3f}m (阈值={next_wp['radius']}m)",
                    throttle_duration_sec=3.0,
                )

    # ── 清理 ──────────────────────────────────────

    def shutdown(self):
        """节点退出时清理所有子进程。

        关闭顺序：
          1. stop_event.set()     → 通知 worker / RS 健康检查 "别再启动新进程了"
          2. 销毁 RS 健康 Timer    → 别再触发重启
          3. join worker 线程(5s)  → 等它完成当前 kill 或提前退出
          4. 获取 _proc_lock       → 此时 worker 已退出，安全
          5. 杀 task_proc + rs_proc → 最终清理
        """
        self.get_logger().info("正在清理子进程...")

        # 1. 通知所有后台组件停止
        self._stop_event.set()

        # 2. 销毁健康检查 Timer
        if hasattr(self, '_rs_health_timer') and self._rs_health_timer is not None:
            self.destroy_timer(self._rs_health_timer)
            self._rs_health_timer = None
        if hasattr(self, '_task_health_timer') and self._task_health_timer is not None:
            self.destroy_timer(self._task_health_timer)
            self._task_health_timer = None

        # 3. 等待 worker 线程退出
        if hasattr(self, '_switch_thread') and self._switch_thread.is_alive():
            self.get_logger().info("等待 switch worker 线程退出...")
            self._switch_thread.join(timeout=5.0)
            if self._switch_thread.is_alive():
                self.get_logger().warn("switch worker 线程未在 5s 内退出（可能卡在 kill 中）")

        # 4. 获取锁并最终清理
        with self._proc_lock:
            if self.task_proc is not None:
                self.get_logger().info(f"终止 task 子进程 (pid={self.task_proc.pid})")
                _kill_process_tree(self.task_proc, self._stop_event)
                self.task_proc = None
            if self.rs_proc is not None:
                self.get_logger().info(f"终止 RS 守护进程 (pid={self.rs_proc.pid})")
                _kill_process_tree(self.rs_proc, self._stop_event)
                self.rs_proc = None
        self.get_logger().info("所有子进程已清理")


# ══════════════════════════════════════════════════════════════
#  主入口
# ══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args)
    node = WaypointLauncherNode()

    # 注册 SIGINT / SIGTERM 处理，确保子进程被清理
    shutdown_flag = threading.Event()

    def _signal_handler(sig, frame):
        print(f"\n收到信号 {sig}，正在退出...")
        shutdown_flag.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # spin 直到收到退出信号
    try:
        while rclpy.ok() and not shutdown_flag.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        print("waypoint_launcher 已退出")


if __name__ == "__main__":
    main()
