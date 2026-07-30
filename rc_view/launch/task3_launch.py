#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动 qiyucolor2（USB 摄像头黄色检测）+ YOLO（D435i 目标检测+3D坐标发布）

用法：
  ros2 launch rc_view task3_launch.py
  ros2 launch rc_view task3_launch.py start_qiyu2:=false
  ros2 launch rc_view task3_launch.py start_yolo:=false
  ros2 launch rc_view task3_launch.py model_path:=/path/to/best.pt confidence_threshold:=0.6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """运行时组装节点"""

    # ── qiyucolor2 — USB 摄像头 2，黄色区域检测 ────
    qiyucolor2_node = Node(
        package='camera',
        executable='qiyucolor2',
        name='qiyu2_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_qiyu2')),
    )

    # ── YOLO 检测节点 — D435i 彩色+深度 → 3D 坐标发布 ────
    yolo_node = Node(
        package='camera',
        executable='yolo_detect',
        name='yolo_detection_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('start_yolo')),
        parameters=[{
            'model_path':           LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'min_box_area':         LaunchConfiguration('min_box_area'),
            'target_class':         LaunchConfiguration('target_class'),
            'select_mode':          LaunchConfiguration('select_mode'),
            'frame_id':             LaunchConfiguration('frame_id'),
            'fallback_classes':     LaunchConfiguration('fallback_classes'),
        }],
    )

    return [qiyucolor2_node, yolo_node]


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装"""

    # ── 开关参数 ──────────────────────────────────────
    start_qiyu2_arg = DeclareLaunchArgument(
        'start_qiyu2', default_value='true',
        description='启动 qiyucolor2 (USB 摄像头2, 黄色检测)')

    start_yolo_arg = DeclareLaunchArgument(
        'start_yolo', default_value='true',
        description='启动 YOLO 检测节点 (D435i 目标检测+3D坐标)')

    # ── YOLO 模型参数 ─────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt',
        description='YOLO 模型文件路径')

    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.45',
        description='YOLO 置信度阈值')

    min_box_area_arg = DeclareLaunchArgument(
        'min_box_area', default_value='500',
        description='最小检测框面积 (像素)')

    # ── 目标选择 ──────────────────────────────────────
    target_class_arg = DeclareLaunchArgument(
        'target_class', default_value='3',
        description='主检测类别 ID (COCO: 3=car)')

    select_mode_arg = DeclareLaunchArgument(
        'select_mode', default_value='highest_conf',
        description='多目标选择策略: highest_conf / nearest / largest')

    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value='camera_color_optical_frame',
        description='相机光学 frame_id')

    fallback_classes_arg = DeclareLaunchArgument(
        'fallback_classes', default_value='[0, 1, 2]',
        description='回退检测类别列表 (COCO: 0=person, 1=bicycle, 2=car)')

    # ── 启动日志 ──────────────────────────────────────
    log_start = LogInfo(
        msg=[
            '══════════════════════════════════════',
            '\n  启动 qiyu2 + YOLO',
            '\n  start_qiyu2     = ', LaunchConfiguration('start_qiyu2'),
            '\n  start_yolo      = ', LaunchConfiguration('start_yolo'),
            '\n  model_path      = ', LaunchConfiguration('model_path'),
            '\n  confidence      = ', LaunchConfiguration('confidence_threshold'),
            '\n  target_class    = ', LaunchConfiguration('target_class'),
            '\n  select_mode     = ', LaunchConfiguration('select_mode'),
            '\n  fallback_classes= ', LaunchConfiguration('fallback_classes'),
            '\n══════════════════════════════════════',
        ]
    )

    return LaunchDescription([
        start_qiyu2_arg,
        start_yolo_arg,
        model_path_arg,
        confidence_arg,
        min_box_area_arg,
        target_class_arg,
        select_mode_arg,
        frame_id_arg,
        fallback_classes_arg,
        log_start,
        OpaqueFunction(function=launch_setup),
    ])
