#!/usr/bin/env python3
"""
z2_mission_manager — 第二比赛任务状态机

流程 (BLUE / RED)：
WAIT_START → GO_FIRST_GOAL → WAIT_FIRST_GOAL →
GO_SECOND_GOAL → WAIT_SECOND_GOAL →
SEND_STAIR_ACTION_4 → WAIT_STAIR_ACTION_DONE
"""

import math
import traceback
from enum import IntEnum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration as RclpyDuration
from rclpy.action import ActionClient

import tf2_ros
from tf2_ros import TransformException
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import UInt8, Int8, Float32, String


class State(IntEnum):
    WAIT_START = 0
    GO_FIRST_GOAL = 1
    WAIT_FIRST_GOAL = 2
    GO_SECOND_GOAL = 12
    WAIT_SECOND_GOAL = 13
    PAUSE_BEFORE_SECOND = 15
    REQUEST_NINE_GRID = 3
    WAIT_NINE_GRID_OFFSET = 4
    GO_DYNAMIC_GOAL = 5
    WAIT_DYNAMIC_GOAL = 6
    SEND_STAIR_ACTION_3 = 7
    SEND_STAIR_ACTION_4 = 14
    WAIT_STAIR_ACTION_DONE = 8
    FAILED = 9
    CANCELLED = 10
    MISSION_DONE = 11


STATE_NAMES = {v: k for k, v in State.__members__.items()}


class Z2MissionManager(Node):
    def __init__(self):
        super().__init__('z2_mission_manager')

        # ==================== TF ====================
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ==================== 参数 ====================
        self.declare_parameter('first_goal_blue', [0.0, 0.0, 0.0])
        self.declare_parameter('first_goal_red', [0.0, 0.0, 0.0])
        self.declare_parameter('second_goal_red', [0.0, 0.0, 0.0])
        self.declare_parameter('second_goal_blue', [0.0, 0.0, 0.0])
        self.declare_parameter('blue_dynamic_map_y_delta', -2.67)
        self.declare_parameter('red_dynamic_map_y_delta', 2.67)
        self.declare_parameter('nine_grid_request_value', 1)
        self.declare_parameter('stair_action_value', 3)

        self._first_goal_blue = list(self.get_parameter('first_goal_blue').value)
        self._first_goal_red = list(self.get_parameter('first_goal_red').value)
        self._second_goal_red = list(self.get_parameter('second_goal_red').value)
        self._second_goal_blue = list(self.get_parameter('second_goal_blue').value)
        self._blue_dy = self.get_parameter('blue_dynamic_map_y_delta').value
        self._red_dy = self.get_parameter('red_dynamic_map_y_delta').value
        self._nine_grid_val = self.get_parameter('nine_grid_request_value').value
        self._stair_val = self.get_parameter('stair_action_value').value

        # ==================== 状态 ====================
        self._state: State = State.WAIT_START
        self._alliance: Optional[int] = None
        self._startup_pose_status: int = 0
        self._goal_active = False
        self._goal_result_ready = False
        self._goal_succeeded = False
        self._goal_handle = None
        self._state_timestamp = self.get_clock().now()
        self._nine_grid_offset: Optional[float] = None
        self._stair_sent = False
        self._stair_action_4_sent = False
        self._execution_id: int = 0
        self._goal_sequence: int = 0
        self._pause_start = None

        # ==================== QoS ====================
        self._qos_transient = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ==================== 订阅 ====================
        self._alliance_sub = self.create_subscription(
            UInt8, '/competition/alliance', self._alliance_cb, self._qos_transient)
        self._startup_status_sub = self.create_subscription(
            UInt8, '/competition/startup_pose_status',
            self._startup_status_cb, self._qos_transient)
        self._start_sub = self.create_subscription(
            UInt8, '/z2/mission/start', self._start_cb, 10)
        self._cancel_sub = self.create_subscription(
            UInt8, '/z2/mission/cancel', self._cancel_cb, 10)
        self._offset_sub = self.create_subscription(
            Float32, '/camera/nine_grid/offset', self._offset_cb, 10)

        # ==================== 发布 ====================
        self._status_pub = self.create_publisher(
            String, '/z2/mission/status', self._qos_transient)
        self._target_pub = self.create_publisher(
            String, '/z2/mission/target', 10)
        self._nine_grid_pub = self.create_publisher(
            Int8, '/camera/nine_grid/request', 10)
        self._stair_pub = self.create_publisher(
            UInt8, '/stair_action_cmd', 10)

        # ==================== Action Client ====================
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self._send_goal_future = None
        self._goal_result_future = None

        # ==================== 定时器 ====================
        self._timer = self.create_timer(0.1, self._tick)

        self._publish_status(f'state={STATE_NAMES[self._state]}')
        self.get_logger().info('[Z2] mission_manager started, WAIT_START')

    # ================================================================
    #  Callbacks
    # ================================================================
    def _alliance_cb(self, msg: UInt8):
        if msg.data in (1, 2) and self._state in (State.WAIT_START,):
            self._alliance = msg.data
            name = 'BLUE' if msg.data == 1 else 'RED'
            self.get_logger().info(f'[Z2] alliance={name}')
            self._publish_status(f'alliance={name}')

    def _startup_status_cb(self, msg: UInt8):
        self._startup_pose_status = msg.data
        name = 'BLUE' if msg.data == 1 else 'RED' if msg.data == 2 else '?'
        self.get_logger().info(f'[Z2] startup status={msg.data} ({name})')

    def _start_cb(self, msg: UInt8):
        self.get_logger().info(
            f'[Z2] start message received: data={msg.data} state={STATE_NAMES[self._state]} '
            f'alliance={self._alliance} startup_status={self._startup_pose_status}')
        accepted_states = (State.WAIT_START, State.FAILED, State.CANCELLED,
                           State.WAIT_STAIR_ACTION_DONE, State.MISSION_DONE)
        if msg.data == 1 and self._state in accepted_states:
            self.get_logger().info('[Z2] mission start condition met, beginning')
            self._begin_mission()
        elif msg.data == 1:
            self.get_logger().warn(
                f'[Z2] start rejected: state={STATE_NAMES[self._state]} '
                f'(expected one of {[STATE_NAMES[s] for s in accepted_states]})')
        elif msg.data == 2:
            self.get_logger().warn('[Z2] start data=2 received, cancel trigger?')
        else:
            self.get_logger().warn(f'[Z2] unknown start data={msg.data}')

    def _cancel_cb(self, msg: UInt8):
        if msg.data == 1:
            self.get_logger().info('[Z2] mission cancelled')
            self._cancel_goal()
            self._state = State.CANCELLED
            self._publish_status(f'state=CANCELLED')
            self.get_logger().info('[Z2] mission cancelled')

    def _offset_cb(self, msg: Float32):
        if self._state != State.WAIT_NINE_GRID_OFFSET:
            return
        offset = float(msg.data)
        if not math.isfinite(offset):
            self.get_logger().error(f'[Z2] invalid nine_grid offset: {offset}')
            return
        self._nine_grid_offset = offset
        self.get_logger().info(f'[Z2] nine_grid offset={offset:.4f}')
        self._publish_status(f'offset={offset:.4f}')

    # ================================================================
    #  状态机 tick
    # ================================================================
    def _tick(self):
        try:
            if self._state == State.WAIT_START:
                pass  # 等待 /z2/mission/start
            elif self._state == State.GO_FIRST_GOAL:
                self._handle_go_first_goal()
            elif self._state == State.WAIT_FIRST_GOAL:
                self._handle_wait_first_goal()
            elif self._state == State.PAUSE_BEFORE_SECOND:
                self._handle_pause_before_second()
            elif self._state == State.GO_SECOND_GOAL:
                self._handle_go_second_goal()
            elif self._state == State.WAIT_SECOND_GOAL:
                self._handle_wait_second_goal()
            elif self._state == State.REQUEST_NINE_GRID:
                self._handle_request_nine_grid()
            elif self._state == State.WAIT_NINE_GRID_OFFSET:
                self._handle_wait_nine_grid_offset()
            elif self._state == State.GO_DYNAMIC_GOAL:
                self._handle_go_dynamic_goal()
            elif self._state == State.WAIT_DYNAMIC_GOAL:
                self._handle_wait_dynamic_goal()
            elif self._state == State.SEND_STAIR_ACTION_3:
                self._handle_send_stair_action_3()
            elif self._state == State.SEND_STAIR_ACTION_4:
                self._handle_send_stair_action_4()
            elif self._state == State.WAIT_STAIR_ACTION_DONE:
                pass  # 等待反馈协议
            elif self._state in (State.FAILED, State.CANCELLED, State.MISSION_DONE):
                pass
        except Exception as e:
            self.get_logger().error(
                f'[Z2] tick exception: {e}\n{traceback.format_exc()}')
            self._state = State.FAILED

    # ================================================================
    #  开始条件检查
    # ================================================================
    def _begin_mission(self):
        if self._alliance is None:
            self.get_logger().warn('[Z2] cannot start: no alliance')
            return
        if self._startup_pose_status not in (1, 2):
            self.get_logger().warn(f'[Z2] cannot start: startup status={self._startup_pose_status}')
            return
        if self._startup_pose_status != self._alliance:
            self.get_logger().warn('[Z2] cannot start: startup/alliance mismatch')
            return
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('[Z2] cannot start: navigate_to_pose unavailable')
            return

        # 清理可能残留的旧 goal
        self._cancel_goal()
        self._execution_id += 1
        self._goal_sequence = 0
        self._stair_action_4_sent = False
        self._stair_sent = False
        name = 'BLUE' if self._alliance == 1 else 'RED'
        goal = self._first_goal_blue if self._alliance == 1 else self._first_goal_red
        self.get_logger().info(
            f'[Z2] sending first goal: alliance={name} '
            f'x={goal[0]:.4f} y={goal[1]:.4f} yaw={goal[2]:.4f}')
        self._state = State.GO_FIRST_GOAL
        self._publish_status('GO_FIRST_GOAL')
        self._publish_target(goal)

    # ================================================================
    #  GO_FIRST_GOAL
    # ================================================================
    def _handle_go_first_goal(self):
        goal = self._first_goal_blue if self._alliance == 1 else self._first_goal_red
        self._send_nav_goal(goal[0], goal[1], goal[2])
        self._state = State.WAIT_FIRST_GOAL
        self._publish_status('WAIT_FIRST_GOAL')

    def _handle_wait_first_goal(self):
        if self._check_goal_result():
            name = 'BLUE' if self._alliance == 1 else 'RED'
            self.get_logger().info(f'[Z2] {name} first goal succeeded, pausing 5s')
            self._pause_start = self.get_clock().now()
            self._state = State.PAUSE_BEFORE_SECOND
            self._publish_status('PAUSE 5s')
        elif self._goal_result_ready and not self._goal_succeeded:
            self.get_logger().error('[Z2] first goal failed')
            self._state = State.FAILED
            self._publish_status('FAILED: first goal failed')

    # ================================================================
    #  PAUSE_BEFORE_SECOND（5秒停留）
    # ================================================================
    def _handle_pause_before_second(self):
        if self._pause_start is None:
            self._state = State.GO_SECOND_GOAL
            return
        elapsed = (self.get_clock().now() - self._pause_start).nanoseconds / 1e9
        if elapsed >= 5.0:
            self.get_logger().info('[Z2] pause 5s complete, proceeding to second goal')
            self._pause_start = None
            self._state = State.GO_SECOND_GOAL
            self._publish_status('GO_SECOND_GOAL')

    # ================================================================
    #  GO_SECOND_GOAL / WAIT_SECOND_GOAL (BLUE and RED)
    # ================================================================
    def _handle_go_second_goal(self):
        if self._alliance == 1:
            goal = self._second_goal_blue
            first = self._first_goal_blue
        else:
            goal = self._second_goal_red
            first = self._first_goal_red
        delta_y = goal[1] - first[1]
        name = 'BLUE' if self._alliance == 1 else 'RED'
        self.get_logger().info(
            f'[Z2] sending {name} second goal: '
            f'x={goal[0]:.4f} y={goal[1]:.4f} yaw={goal[2]:.4f} '
            f'delta_y={delta_y:+.4f}')
        self._send_nav_goal(goal[0], goal[1], goal[2])
        self._state = State.WAIT_SECOND_GOAL
        self._publish_status('WAIT_SECOND_GOAL')

    def _handle_wait_second_goal(self):
        if self._check_goal_result():
            name = 'BLUE' if self._alliance == 1 else 'RED'
            self.get_logger().info(f'[Z2] {name} second goal succeeded')
            self._state = State.SEND_STAIR_ACTION_4
            self._publish_status('SEND_STAIR_ACTION_4')
        elif self._goal_result_ready and not self._goal_succeeded:
            self.get_logger().error('[Z2] second goal failed')
            self._state = State.FAILED
            self._publish_status('FAILED: second goal failed')

    # ================================================================
    #  REQUEST_NINE_GRID
    # ================================================================
    def _handle_request_nine_grid(self):
        self._nine_grid_pub.publish(Int8(data=self._nine_grid_val))
        self.get_logger().info(
            f'[Z2] publishing /camera/nine_grid/request data={self._nine_grid_val}')
        self._nine_grid_offset = None
        self._state = State.WAIT_NINE_GRID_OFFSET
        self._publish_status('WAIT_NINE_GRID_OFFSET')

    def _handle_wait_nine_grid_offset(self):
        if self._nine_grid_offset is None:
            return
        # 使用收到 message 时的实时 TF
        pose = self._get_current_pose()
        if pose is None:
            self.get_logger().error('[Z2] TF unavailable for dynamic goal')
            self._state = State.FAILED
            self._publish_status('FAILED: TF unavailable')
            return
        cx, cy, cyaw = pose

        dy = self._blue_dy if self._alliance == 1 else self._red_dy
        target_x = cx + self._nine_grid_offset
        target_y = cy + dy
        target_yaw = -1.5708

        self.get_logger().info(
            f'[Z2] dynamic goal: current=({cx:.4f},{cy:.4f}) '
            f'offset={self._nine_grid_offset:.4f} dy={dy:.4f} '
            f'target=({target_x:.4f},{target_y:.4f}) yaw={target_yaw:.4f}')
        self._publish_target([target_x, target_y, target_yaw])

        self._state = State.GO_DYNAMIC_GOAL
        self._publish_status('GO_DYNAMIC_GOAL')

    # ================================================================
    #  GO_DYNAMIC_GOAL
    # ================================================================
    def _handle_go_dynamic_goal(self):
        pose = self._get_current_pose()
        if pose is None:
            return
        cx, cy, _ = pose
        dy = self._blue_dy if self._alliance == 1 else self._red_dy
        target_x = cx + (self._nine_grid_offset or 0.0)
        target_y = cy + dy
        self._send_nav_goal(target_x, target_y, -1.5708)
        self._state = State.WAIT_DYNAMIC_GOAL
        self._publish_status('WAIT_DYNAMIC_GOAL')

    def _handle_wait_dynamic_goal(self):
        if self._check_goal_result():
            self.get_logger().info('[Z2] dynamic goal succeeded')
            self._state = State.SEND_STAIR_ACTION_3
            self._publish_status('SEND_STAIR_ACTION_3')
        elif self._goal_result_ready and not self._goal_succeeded:
            self.get_logger().error('[Z2] dynamic goal failed')
            self._state = State.FAILED
            self._publish_status('FAILED: dynamic goal failed')

    # ================================================================
    #  SEND_STAIR_ACTION_3
    # ================================================================
    def _handle_send_stair_action_3(self):
        if self._stair_sent:
            return
        self._stair_pub.publish(UInt8(data=self._stair_val))
        self._stair_sent = True
        self.get_logger().info(
            f'[Z2] publishing /stair_action_cmd data={self._stair_val}')
        self._state = State.WAIT_STAIR_ACTION_DONE
        self._publish_status('WAIT_STAIR_ACTION_DONE')

    # ================================================================
    #  SEND_STAIR_ACTION_4 (RED/BLUE: 三区到位，通知下位机收腿)
    # ================================================================
    def _handle_send_stair_action_4(self):
        if self._stair_action_4_sent:
            self.get_logger().warn('[Z2] stair_action_cmd data=4 already sent; duplicate ignored')
            return
        self._stair_pub.publish(UInt8(data=4))
        self._stair_action_4_sent = True
        alliance_name = 'BLUE' if self._alliance == 1 else 'RED'
        self.get_logger().info(
            f'[Z2] publishing /stair_action_cmd data=4 '
            f'(alliance={alliance_name}, meaning=zone3 arrived, retract legs)')
        self._state = State.WAIT_STAIR_ACTION_DONE
        self._publish_status('WAIT_STAIR_ACTION_DONE')

    # ================================================================
    #  工具函数
    # ================================================================
    def _get_current_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=RclpyDuration(seconds=2.0))
            x = t.transform.translation.x
            y = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            yaw = 2.0 * math.atan2(qz, qw)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(f'[Z2] TF lookup failed: {e}')
            return None

    def _send_nav_goal(self, x: float, y: float, yaw: float):
        self._goal_sequence += 1
        capture_eid = self._execution_id
        capture_gseq = self._goal_sequence
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        cz = math.sin(yaw / 2.0)
        cw = math.cos(yaw / 2.0)
        goal.pose.pose.orientation.z = cz
        goal.pose.pose.orientation.w = cw
        self._goal_active = True
        self._goal_result_ready = False
        self._goal_succeeded = False
        self._goal_handle = None
        self._send_goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=self._goal_feedback_cb)
        self._send_goal_future.add_done_callback(
            lambda f: self._goal_response_cb(f, capture_eid, capture_gseq))
        self.get_logger().info(f'[Z2] NavigateToPose sent: x={x:.4f} y={y:.4f} yaw={yaw:.4f}')

    def _goal_response_cb(self, future, capture_eid: int, capture_gseq: int):
        if capture_eid != self._execution_id:
            self.get_logger().debug(
                f'[Z2] goal_response ignored: stale execution {capture_eid} != {self._execution_id}')
            return
        if capture_gseq != self._goal_sequence:
            self.get_logger().debug(
                f'[Z2] goal_response ignored: stale sequence {capture_gseq} != {self._goal_sequence}')
            return
        self._goal_handle = future.result()
        if not self._goal_handle or not self._goal_handle.accepted:
            self.get_logger().error('[Z2] NavigateToPose rejected')
            self._goal_active = False
            self._goal_result_ready = True
            self._goal_succeeded = False
            return
        self._goal_result_future = self._goal_handle.get_result_async()
        self._goal_result_future.add_done_callback(
            lambda f: self._goal_result_cb(f, capture_eid, capture_gseq))

    def _goal_result_cb(self, future, capture_eid: int, capture_gseq: int):
        if capture_eid != self._execution_id:
            self.get_logger().debug(
                f'[Z2] goal_result ignored: stale execution {capture_eid} != {self._execution_id}')
            return
        if capture_gseq != self._goal_sequence:
            self.get_logger().debug(
                f'[Z2] goal_result ignored: stale sequence {capture_gseq} != {self._goal_sequence}')
            return
        self._goal_result_ready = True
        self._goal_active = False
        try:
            result = future.result().result
            self._goal_succeeded = True
        except Exception:
            self._goal_succeeded = False

    def _goal_feedback_cb(self, msg):
        pass

    def _check_goal_result(self):
        """返回 True 表示 goal 成功"""
        return self._goal_result_ready and self._goal_succeeded

    def _cancel_goal(self):
        if self._goal_handle:
            self._goal_handle.cancel_goal_async()
        self._goal_active = False
        self._goal_result_ready = False

    def _time_in_state(self):
        return (self.get_clock().now() - self._state_timestamp).nanoseconds / 1e9

    def _publish_status(self, msg: str):
        self._status_pub.publish(String(data=msg))

    def _publish_target(self, goal):
        s = f'x={goal[0]:.4f} y={goal[1]:.4f} yaw={goal[2]:.4f}'
        self._target_pub.publish(String(data=s))


def main(args=None):
    rclpy.init(args=args)
    node = Z2MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
