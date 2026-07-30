#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
启动 YOLO 检测节点 + RealSense 发布节点 + 串口桥接。

YOLO 依赖 /camera/color/image_raw 和 /camera/depth/image_raw，
这些话题由 node_of_rs (RealSensePublisher) 提供，因此默认一并启动。
串口桥接节点（auto_serial_bridge）在包已安装时自动加载。

用法：
  ros2 launch camera yolo_launch.py
  ros2 launch camera yolo_launch.py start_rs:=false          # 已有 RS 实例时跳过
  ros2 launch camera yolo_launch.py confidence_threshold:=0.7
  ros2 launch camera yolo_launch.py visualize:=true           # 开启 cv2.imshow 可视化窗口
"""

import os
import sys
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """运行时组装节点列表（含串口桥接可选加载）"""

    start_rs = LaunchConfiguration('start_rs').perform(context).lower() in ('true', '1')
    show_viz = LaunchConfiguration('visualize').perform(context).lower() in ('true', '1')

    # ── RealSense 发布节点 ──────────────────────────
    rs_node = Node(
        package='camera',
        executable='node_of_rs',
        name='realsense_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_rs')),
        parameters=[{
            'enable_filter': True,
            'depth_unit': 'mm',
        }],
    )

    # ── YOLO 检测节点 ───────────────────────────────
    yolo_extra_args = ['--visualize'] if show_viz else []
    yolo_node = Node(
        package='camera',
        executable='yolo_detect',
        name='yolo_detection_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'min_box_area': LaunchConfiguration('min_box_area'),
            'target_class': LaunchConfiguration('target_class'),
            'select_mode': LaunchConfiguration('select_mode'),
        }],
        arguments=yolo_extra_args,
    )

    # ── 串口桥接节点（auto_serial_bridge 已安装时加载）──
    try:
        serial_pkg_share = get_package_share_directory('auto_serial_bridge')
        protocol_path = os.path.join(serial_pkg_share, 'config', 'protocol.yaml')
        with open(protocol_path, 'r', encoding='utf-8') as f:
            protocol_config = yaml.safe_load(f)
        serial_params = protocol_config['serial_controller']['ros__parameters']

        serial_node = Node(
            package='auto_serial_bridge',
            executable='serial_node',
            name='serial_controller',
            output='screen',
            emulate_tty=True,
            parameters=[serial_params],
            arguments=[
                '--ros-args',
                '--log-level', 'info',
                '--log-level', 'serial_controller:=debug',
            ],
        )
    except Exception:
        print('[yolo_launch] auto_serial_bridge 未安装，跳过串口桥接节点', file=sys.stderr)
        serial_node = None

    nodes = [rs_node, yolo_node]
    if serial_node is not None:
        nodes.append(serial_node)
    return nodes


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装节点"""

    return LaunchDescription([
        # ── 开关 ──────────────────────────────────────
        DeclareLaunchArgument(
            'start_rs', default_value='true',
            description='是否启动 RealSense 发布节点（已有实例时可设为 false）'),
        DeclareLaunchArgument(
            'visualize', default_value='false',
            description='开启 cv2.imshow 可视化窗口（需要 DISPLAY 环境变量）'),

        # ── YOLO 参数 ─────────────────────────────────
        DeclareLaunchArgument(
            'model_path',
            default_value='/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt',
            description='YOLO 模型文件路径'),
        DeclareLaunchArgument(
            'confidence_threshold', default_value='0.5',
            description='YOLO 置信度阈值'),
        DeclareLaunchArgument(
            'min_box_area', default_value='500',
            description='最小检测框面积'),
        DeclareLaunchArgument(
            'target_class', default_value='3',
            description='目标 COCO 类别 ID (3=car)'),
        DeclareLaunchArgument(
            'select_mode', default_value='highest_conf',
            description='候选框选择策略: highest_conf / leftmost'),

        OpaqueFunction(function=launch_setup),
    ])
