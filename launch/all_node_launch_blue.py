#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
一键启动所有视觉检测节点：
  - node_of_rs      RealSense D435i 彩色+深度发布
  - yolo            YOLOv8 目标检测 + 3D 坐标发布
  - qiyucolor1      USB 摄像头 0 黄+紫区域检测
  - qiyucolor2      USB 摄像头 2 黄色区域检测
  - res_kfs         RealSense 蓝/红区域检测（5s 延迟发送）
  - serial_node     串口桥接（auto_serial_bridge 已安装时自动加载）

用法：
  ros2 launch camera all_node_launch_blue.py
  ros2 launch camera all_node_launch_blue.py target_color:=red    # 改 res_kfs 检测颜色
  ros2 launch camera all_node_launch_blue.py start_yolo:=false    # 跳过 YOLO
  ros2 launch camera all_node_launch_blue.py start_rs:=false      # 跳过 RealSense

所有参数透传，可单独禁用任意节点。
"""

import os
import sys
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """运行时组装所有节点（含串口桥接可选加载）"""

    # ── 1. RealSense 发布节点 ─────────────────────────
    node_of_rs = Node(
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

    # ── 2. YOLO 检测节点 ──────────────────────────────
    yolo_node = Node(
        package='camera',
        executable='yolo_detect',
        name='yolo_detection_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('start_yolo')),
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'min_box_area': 500,
            'target_class': 3,
            'select_mode': 'highest_conf',
        }],
    )

    # ── 3. qiyucolor1 — USB 摄像头 0，黄+紫色检测 ────
    qiyucolor1_node = Node(
        package='camera',
        executable='qiyucolor1',
        name='qiyu1_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_qiyu1')),
    )

    # ── 4. qiyucolor2 — USB 摄像头 2，黄色检测 ────────
    qiyucolor2_node = Node(
        package='camera',
        executable='qiyucolor2',
        name='qiyu2_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_qiyu2')),
    )

    # ── 5. res_kfs — RealSense 蓝/红区域检测（5s 延迟发送）──
    res_kfs_node = Node(
        package='camera',
        executable='res_kfs',
        name='reg_kfs',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_res_kfs')),
        parameters=[{
            'target_color': LaunchConfiguration('target_color'),
        }],
    )

    # ── 6. 串口桥接节点（auto_serial_bridge 已安装时加载）──
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
        print('[all_node_launch_blue] auto_serial_bridge 未安装，跳过串口桥接节点', file=sys.stderr)
        serial_node = None

    nodes = [node_of_rs, yolo_node, qiyucolor1_node, qiyucolor2_node, res_kfs_node]
    if serial_node is not None:
        nodes.append(serial_node)
    return nodes


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装节点"""

    # ── 开关参数 ──────────────────────────────────────
    start_rs_arg = DeclareLaunchArgument(
        'start_rs', default_value='true',
        description='启动 RealSense 发布节点 (node_of_rs)')
    start_yolo_arg = DeclareLaunchArgument(
        'start_yolo', default_value='true',
        description='启动 YOLO 检测节点')
    start_qiyu1_arg = DeclareLaunchArgument(
        'start_qiyu1', default_value='true',
        description='启动 qiyucolor1 (USB 摄像头0, 黄+紫检测)')
    start_qiyu2_arg = DeclareLaunchArgument(
        'start_qiyu2', default_value='true',
        description='启动 qiyucolor2 (USB 摄像头2, 黄色检测)')
    start_res_kfs_arg = DeclareLaunchArgument(
        'start_res_kfs', default_value='true',
        description='启动 res_kfs (RealSense 蓝/红区域检测)')

    # ── res_kfs 参数 ──────────────────────────────────
    target_color_arg = DeclareLaunchArgument(
        'target_color', default_value='blue',
        description='res_kfs 检测颜色: blue / red / both')

    # ── YOLO 参数 ─────────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt',
        description='YOLO 模型文件路径')
    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.5',
        description='YOLO 置信度阈值')

    # ── 启动日志 ──────────────────────────────────────
    log_start = LogInfo(
        msg=[
            '══════════════════════════════════════',
            '\n  启动所有视觉检测节点 (Blue)',
            '\n  target_color = ', LaunchConfiguration('target_color'),
            '\n  start_rs     = ', LaunchConfiguration('start_rs'),
            '\n  start_yolo   = ', LaunchConfiguration('start_yolo'),
            '\n  start_qiyu1  = ', LaunchConfiguration('start_qiyu1'),
            '\n  start_qiyu2  = ', LaunchConfiguration('start_qiyu2'),
            '\n  start_res_kfs= ', LaunchConfiguration('start_res_kfs'),
            '\n══════════════════════════════════════',
        ]
    )

    return LaunchDescription([
        start_rs_arg,
        start_yolo_arg,
        start_qiyu1_arg,
        start_qiyu2_arg,
        start_res_kfs_arg,
        target_color_arg,
        model_path_arg,
        confidence_arg,
        log_start,
        OpaqueFunction(function=launch_setup),
    ])
