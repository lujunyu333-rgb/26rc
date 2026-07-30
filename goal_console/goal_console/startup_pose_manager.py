#!/usr/bin/env python3
"""
startup_pose_manager — 根据阵营设置 map→odom 定位

20Hz 持续广播，从未停止。异步验证，不阻塞 TF 更新。
不发送任何 Nav2 goal。
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import ExternalShutdownException

import tf2_ros
from tf2_ros import TransformException
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import UInt8, String


# 默认启动区坐标（可通过参数覆盖）
DEFAULT_STARTUP_POSES = {
    1: {'x': -0.818, 'y': -3.784, 'yaw': 0.0},    # BLUE Z1
    2: {'x': -0.818, 'y': -6.634, 'yaw': 0.0},    # RED Z1
}

# 验证参数
VERIFY_INTERVAL_S = 0.1
VERIFY_MAX_TRIES = 30      # 30 × 0.1 = 3.0s
THROTTLE_PERIOD_S = 5.0    # 广播日志节流


class StartupPoseManager(Node):
    def __init__(self):
        super().__init__('startup_pose_manager')

        # ==================== TF ====================
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # map→odom 变换（20Hz 持续广播，永不停）
        self._m2o_x = 0.0
        self._m2o_y = 0.0
        self._m2o_qz = 0.0
        self._m2o_qw = 1.0
        self._broadcast_count = 0

        # 状态
        self._startup_status: int = 0   # 0=未设置, 1=BLUE, 2=RED, 10=失败
        self._selected_alliance: Optional[int] = None

        # 异步验证
        self._verify_pending = False
        self._verify_tries = 0
        self._verify_alliance: Optional[int] = None
        self._verify_desired: Optional[dict] = None

        # ==================== 定时器 ====================
        # 从参数加载启动区坐标，无参数时使用 Z1 默认值
        db = DEFAULT_STARTUP_POSES[1]
        dr = DEFAULT_STARTUP_POSES[2]
        self.declare_parameter('startup_blue', [db['x'], db['y'], db['yaw']])
        self.declare_parameter('startup_red', [dr['x'], dr['y'], dr['yaw']])

        def _parse_pose(param, default):
            try:
                vals = param.value
                if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                    return {'x': float(vals[0]), 'y': float(vals[1]), 'yaw': float(vals[2])}
            except Exception:
                pass
            return default

        sb = self.get_parameter('startup_blue')
        sr = self.get_parameter('startup_red')
        self._startup_poses = {
            1: _parse_pose(sb, DEFAULT_STARTUP_POSES[1]),
            2: _parse_pose(sr, DEFAULT_STARTUP_POSES[2]),
        }
        self.get_logger().info(f'[STARTUP_POSE] BLUE={self._startup_poses[1]}')
        self.get_logger().info(f'[STARTUP_POSE] RED={self._startup_poses[2]}')

        self._tf_timer = self.create_timer(0.05, self._publish_tf)           # 20Hz
        self._verify_timer = self.create_timer(
            VERIFY_INTERVAL_S, self._verify_timer_cb)

        # ==================== QoS ====================
        self._qos_ready = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ==================== 订阅 ====================
        self._alliance_sub = self.create_subscription(
            UInt8, '/competition/alliance',
            self._alliance_callback, self._qos_ready)
        self._set_sub = self.create_subscription(
            UInt8, '/competition/set_startup_pose',
            self._set_startup_pose_callback, 10)

        # ==================== 发布 ====================
        self._status_pub = self.create_publisher(
            UInt8, '/competition/startup_pose_status', self._qos_ready)
        self._msg_pub = self.create_publisher(
            String, '/competition/startup_pose_message', self._qos_ready)

        self.get_logger().info(
            '[STARTUP_POSE] broadcasting identity map→odom, awaiting alliance')

    # ================================================================
    #  20Hz TF 广播——永不停
    # ================================================================
    def _publish_tf(self):
        ts = TransformStamped()
        ts.header.stamp = self.get_clock().now().to_msg()
        ts.header.frame_id = 'map'
        ts.child_frame_id = 'odom'
        ts.transform.translation.x = self._m2o_x
        ts.transform.translation.y = self._m2o_y
        ts.transform.translation.z = 0.0
        ts.transform.rotation.z = self._m2o_qz
        ts.transform.rotation.w = self._m2o_qw
        self._tf_broadcaster.sendTransform(ts)

        # 节流日志
        self._broadcast_count += 1
        if self._broadcast_count % (20 * THROTTLE_PERIOD_S) == 0:
            m2o_yaw = 2.0 * math.atan2(self._m2o_qz, self._m2o_qw)
            self.get_logger().info(
                f'[STARTUP_POSE] broadcasting map→odom: '
                f'x={self._m2o_x:.3f} y={self._m2o_y:.3f} yaw={m2o_yaw:.3f} '
                f'status={self._startup_status}')

    # ================================================================
    #  阵营回调——非阻塞
    # ================================================================
    def _alliance_callback(self, msg: UInt8):
        if msg.data not in (1, 2):
            return
        name = 'BLUE' if msg.data == 1 else 'RED'
        self.get_logger().info(f'[STARTUP_POSE] alliance={name}')
        self._apply_startup_pose(msg.data)

    def _set_startup_pose_callback(self, msg: UInt8):
        if msg.data in (1, 2):
            name = 'BLUE' if msg.data == 1 else 'RED'
            self.get_logger().info(f'[STARTUP_POSE] explicit request: {name}')
            self._apply_startup_pose(msg.data)

    def _apply_startup_pose(self, alliance: int):
        pose = self._startup_poses.get(alliance)
        if not pose:
            self._set_status(10, 'unknown alliance')
            return

        name = 'BLUE' if alliance == 1 else 'RED'
        desired_x, desired_y, desired_yaw = pose['x'], pose['y'], pose['yaw']
        self.get_logger().info(
            f'[STARTUP_POSE] desired map→base_link='
            f'({desired_x:.3f}, {desired_y:.3f}, {math.degrees(desired_yaw):.1f}°)')

        # 过渡状态
        self._set_status(0, f'{name} startup pose being configured')

        # 读取 odom→base_link
        try:
            t = self._tf_buffer.lookup_transform(
                'odom', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            ob_x = t.transform.translation.x
            ob_y = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            ob_yaw = 2.0 * math.atan2(qz, qw)
        except (TransformException, Exception) as e:
            self.get_logger().error(
                f'[STARTUP_POSE] odom→base_link unavailable: {e}')
            self._set_status(10, f'odom→base_link TF error: {e}')
            return

        self.get_logger().info(
            f'[STARTUP_POSE] current odom→base_link: '
            f'x={ob_x:.4f} y={ob_y:.4f} yaw={math.degrees(ob_yaw):.3f}°')

        # 计算 map→odom
        cos_ob = math.cos(ob_yaw)
        sin_ob = math.sin(ob_yaw)
        self._m2o_x = desired_x - (cos_ob * ob_x - sin_ob * ob_y)
        self._m2o_y = desired_y - (sin_ob * ob_x + cos_ob * ob_y)
        m2o_yaw = desired_yaw - ob_yaw
        self._m2o_qz = math.sin(m2o_yaw / 2.0)
        self._m2o_qw = math.cos(m2o_yaw / 2.0)

        self.get_logger().info(
            f'[STARTUP_POSE] computed map→odom='
            f'({self._m2o_x:.4f}, {self._m2o_y:.4f}, {math.degrees(m2o_yaw):.3f}°)')

        # 启动异步验证
        self._verify_pending = True
        self._verify_tries = 0
        self._verify_alliance = alliance
        self._verify_desired = pose
        self._selected_alliance = alliance
        self._startup_status = 0  # 正在验证

    # ================================================================
    #  异步验证 timer（10Hz）
    # ================================================================
    def _verify_timer_cb(self):
        if not self._verify_pending:
            return
        if self._verify_alliance is None:
            return

        self._verify_tries += 1
        desired = self._verify_desired
        desired_x, desired_y, desired_yaw = desired['x'], desired['y'], desired['yaw']
        alliance = self._verify_alliance
        name = 'BLUE' if alliance == 1 else 'RED'

        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            mx = t.transform.translation.x
            my = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            myaw = 2.0 * math.atan2(qz, qw)
        except Exception:
            if self._verify_tries >= VERIFY_MAX_TRIES:
                self._verify_pending = False
                self._set_status(
                    10,
                    f'{name} verification timeout after '
                    f'{self._verify_tries * VERIFY_INTERVAL_S:.1f}s')
            return

        err_x = mx - desired_x
        err_y = my - desired_y
        err_yaw = myaw - desired_yaw
        pos_err = math.hypot(err_x, err_y)

        if pos_err <= 0.05 and abs(err_yaw) <= 0.05:
            self._verify_pending = False
            self._startup_status = alliance
            msg = (
                f'{name} startup pose ready: '
                f'actual=({mx:.4f},{my:.4f},{math.degrees(myaw):.3f}°) '
                f'pos_err={pos_err:.4f} yaw_err={math.degrees(err_yaw):.3f}°')
            self._set_status(alliance, msg)
            self.get_logger().info(f'[STARTUP_POSE] verification passed ({self._verify_tries} tries)')
            return

        if self._verify_tries >= VERIFY_MAX_TRIES:
            self._verify_pending = False
            self._set_status(
                10,
                f'{name} verification failed after {VERIFY_MAX_TRIES} tries: '
                f'pos_err={pos_err:.4f}m yaw_err={math.degrees(err_yaw):.3f}° '
                f'(threshold: 0.05m / 0.05rad)')
            return

        # 继续重试
        if self._verify_tries % 10 == 1:
            self.get_logger().info(
                f'[STARTUP_POSE] verifying try {self._verify_tries}/{VERIFY_MAX_TRIES}: '
                f'pos_err={pos_err:.4f} yaw_err={math.degrees(err_yaw):.3f}°')

    # ================================================================
    #  状态发布
    # ================================================================
    def _set_status(self, code: int, msg: str):
        self._status_pub.publish(UInt8(data=code))
        self._msg_pub.publish(String(data=msg))
        self.get_logger().info(f'[STARTUP_POSE] status={code}: {msg}')


def main(args=None):
    rclpy.init(args=args)
    node = StartupPoseManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # rclpy.shutdown() 在 spin 退出前销毁 context 会导致 RCLError，
        # 这是正常的退出路径，不是 bug。
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
