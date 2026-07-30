#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mission_manager.py

作用：
- 从 mission_routes.yaml 读取一条路线
- 一个点一个点发送给 Nav2 的 /navigate_to_pose
- 导航过程中读取 map -> base_link，判断当前位置和车头朝向
- 当前路线段满足 trigger 条件时，发布 /stair_action_cmd
- Nav2 不暂停，/cmd_vel 继续发；下位机收到动作命令后执行履带动作
"""

import math
import yaml

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float64

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import UInt8
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf_transformations import quaternion_from_euler, euler_from_quaternion


def normalize_angle(angle):
    """把角度限制到 [-pi, pi]，用于计算车头朝向误差。"""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager')

        # 路线文件、路线名、坐标系参数
        self.declare_parameter('route_file', '')
        self.declare_parameter('route_name', 'test_stair')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')

        self.route_file = self.get_parameter('route_file').value
        self.route_name = self.get_parameter('route_name').value
        self.global_frame = self.get_parameter('global_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value

        if self.route_file == '':
            raise RuntimeError('route_file parameter is empty')

        # 读取路线，每一段包含 nav_goal 和 trigger
        self.route = self.load_route(self.route_file, self.route_name)

        self.current_index = 0
        self.current_goal_active = False
        self.current_action_sent = False
        self.finished = False

        # 给 Nav2 发送单个目标点
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # 给下位机/串口桥准备的动作命令
        self.action_pub = self.create_publisher(UInt8, '/stair_action_cmd', 10)

        # 动作命令枚举映射 (与 protocol.yaml StairActionCmd 对应)
        self.ACTION_MAP = {
            'NONE': 0,
            'STOP': 0,
            'STAIR_UP_START': 1,
            'STAIR_UP_END': 2,
            'STAIR_DOWN_START': 3,
            'STAIR_DOWN_END': 4,
        }

        # 读取机器人当前位置：map -> base_link
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 主循环：发目标点 + 检查触发条件
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.get_logger().info(f'Mission manager started, route={self.route_name}')
        self.get_logger().info(f'Route file: {self.route_file}')
        self.get_logger().info(f'Total segments: {len(self.route)}')

    def publish_action(self, action_name, source, extra_info=None):
        """
        统一的动作发布函数。

        Args:
            action_name: 字符串动作名 (如 "STAIR_UP_START")
            source: "trigger" 或 "reached"
            extra_info: 可选额外日志字符串
        """
        if action_name is None:
            return
        if not isinstance(action_name, str):
            return
        action_name = action_name.strip()
        if action_name == '' or action_name == 'NONE':
            return

        action_code = self.ACTION_MAP.get(action_name, 0)
        msg = UInt8(data=action_code)
        self.action_pub.publish(msg)

        label = 'Trigger action sent' if source == 'trigger' else 'Reached action sent'
        log_msg = f'{label}: {action_name} (code={action_code})'
        if extra_info:
            log_msg += f', {extra_info}'
        self.get_logger().warn(log_msg)

    def load_route(self, path, route_name):
        """读取 mission_routes.yaml 里的指定路线。"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if 'routes' not in data:
            raise RuntimeError('routes key not found in route file')

        if route_name not in data['routes']:
            raise RuntimeError(f'route {route_name} not found')

        segments = data['routes'][route_name].get('segments', [])
        if len(segments) == 0:
            raise RuntimeError(f'route {route_name} has no segments')

        return segments

    def yaw_to_quat(self, yaw):
        """yaw 转四元数，Nav2 目标点需要 quaternion。"""
        return quaternion_from_euler(0.0, 0.0, yaw)

    def send_nav_goal(self, segment):
        """发送当前路线段的 Nav2 目标点。"""
        goal = segment['nav_goal']

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        # 2D 导航：只用 x/y/yaw，z 固定 0
        pose.pose.position.x = float(goal['x'])
        pose.pose.position.y = float(goal['y'])
        pose.pose.position.z = 0.0

        q = self.yaw_to_quat(float(goal.get('yaw', 0.0)))
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose

        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('/navigate_to_pose action server not available')
            return False

        self.get_logger().info(
            f'Send goal [{self.current_index + 1}/{len(self.route)}] '
            f'{segment["name"]}: x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}'
        )

        send_future = self.nav_client.send_goal_async(
            nav_goal,
            feedback_callback=self.nav_feedback_callback
        )
        send_future.add_done_callback(self.nav_goal_response_callback)

        self.current_goal_active = True
        self.current_action_sent = False
        return True

    def nav_goal_response_callback(self, future):
        """Nav2 是否接受目标点。"""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Nav goal rejected')
            self.current_goal_active = False
            self.finished = True
            return

        self.get_logger().info('Nav goal accepted')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_feedback_callback(self, feedback_msg):
        # 暂时不用反馈。后面需要剩余距离时可以在这里加。
        pass

    def nav_result_callback(self, future):
        """
        当前目标点完成后的回调。

        关键修改：
        - status == 4：目标成功，到下一个点
        - 其他状态：失败/取消，停止任务，不继续发后面的点

        常见 status：
        - 4 = SUCCEEDED
        - 5 = CANCELED
        - 6 = ABORTED
        """
        result = future.result()
        status = result.status

        self.get_logger().info(f'Nav result status={status}')

        if status == 4:
            segment = self.route[self.current_index]
            on_reached = segment.get('on_reached_action', None)
            self.publish_action(on_reached, 'reached')

            self.get_logger().info('Current segment reached, go to next segment')

            self.current_index += 1
            self.current_goal_active = False
            self.current_action_sent = False

            if self.current_index >= len(self.route):
                self.get_logger().info('Mission finished')
                self.finished = True

            return

        self.get_logger().error(
            f'Nav goal failed or canceled, status={status}. '
            f'Mission stopped at segment index={self.current_index}.'
        )

        self.current_goal_active = False
        self.finished = True

    def get_robot_pose_2d(self):
        """读取机器人在 map 坐标系下的 x/y/yaw。"""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(
                f'Cannot get TF {self.global_frame}->{self.robot_frame}: {e}',
                throttle_duration_sec=1.0
            )
            return None

        x = trans.transform.translation.x
        y = trans.transform.translation.y

        q = trans.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        return x, y, yaw

    def check_trigger(self, segment):
        """
        只检查当前 segment 的 trigger。
        触发条件：
        - trigger enabled
        - 当前段还没发过动作
        - 距离 trigger 点小于 radius
        - 车头朝向误差小于 yaw_tolerance
        """
        trigger = segment.get('trigger', None)

        if trigger is None:
            return

        if not trigger.get('enabled', False):
            return

        if self.current_action_sent:
            return

        pose = self.get_robot_pose_2d()
        if pose is None:
            return

        robot_x, robot_y, robot_yaw = pose

        tx = float(trigger['x'])
        ty = float(trigger['y'])
        tyaw = float(trigger.get('yaw', 0.0))
        radius = float(trigger.get('radius', 1.0))
        yaw_tol = float(trigger.get('yaw_tolerance', 0.52))
        action = str(trigger.get('action', 'NONE'))

        dist = math.hypot(robot_x - tx, robot_y - ty)
        yaw_error = abs(normalize_angle(robot_yaw - tyaw))

        self.get_logger().info(
            f'[{segment["name"]}] dist={dist:.2f}/{radius:.2f}, '
            f'yaw_error={yaw_error:.2f}/{yaw_tol:.2f}',
            throttle_duration_sec=1.0
        )

        if dist <= radius and yaw_error <= yaw_tol:
            self.publish_action(action, 'trigger',
                                f'dist={dist:.2f}, yaw_error={yaw_error:.2f}')
            self.current_action_sent = True

    def timer_callback(self):
        """
        主循环：
        - 如果当前没有目标，发送当前段 nav_goal
        - 如果当前目标正在执行，只检查当前段 trigger
        """
        if self.finished:
            return

        if self.current_index >= len(self.route):
            self.finished = True
            return

        segment = self.route[self.current_index]

        if not self.current_goal_active:
            self.send_nav_goal(segment)
            return

        self.check_trigger(segment)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
