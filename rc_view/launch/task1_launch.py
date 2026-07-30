#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动 qiyucolor1（USB 摄像头黄/紫检测）+ move_base_of_yolo（D435i+YOLO 视觉伺服）

用法：
  ros2 launch rc_view task1_launch.py
  ros2 launch rc_view task1_launch.py start_qiyu1:=false
  ros2 launch rc_view task1_launch.py start_move_base:=false
  ros2 launch rc_view task1_launch.py model_path:=/path/to/best.pt confidence_threshold:=0.6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """运行时组装节点"""

    # ── qiyucolor1 — USB 摄像头 0，黄+紫区域检测 ────
    qiyucolor1_node = Node(
        package='camera',
        executable='qiyucolor1',
        name='qiyu1_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_qiyu1')),
    )

    # ── move_base_of_yolo — D435i + YOLO 视觉伺伏 ────
    move_base_node = Node(
        package='camera',
        executable='move_base_of_yolo',
        name='move_base_of_yolo',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('start_move_base')),
        parameters=[{
            'model_path':           LaunchConfiguration('model_path'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'iou_threshold':        LaunchConfiguration('iou_threshold'),
            'min_box_area':         LaunchConfiguration('min_box_area'),
            'priority_class':       LaunchConfiguration('priority_class'),
            'dead_zone_px':         LaunchConfiguration('dead_zone_px'),
            'dark_threshold':       LaunchConfiguration('dark_threshold'),
            'obstruction_delay':    LaunchConfiguration('obstruction_delay'),
            'max_grasp_count':      LaunchConfiguration('max_grasp_count'),
            'frame_id':             LaunchConfiguration('frame_id'),
        }],
    )

    return [qiyucolor1_node, move_base_node]


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装"""

    # ── 开关参数 ──────────────────────────────────────
    start_qiyu1_arg = DeclareLaunchArgument(
        'start_qiyu1', default_value='true',
        description='启动 qiyucolor1 (USB 摄像头0, 黄+紫检测)')

    start_move_base_arg = DeclareLaunchArgument(
        'start_move_base', default_value='true',
        description='启动 move_base_of_yolo (D435i + YOLO 视觉伺服)')

    # ── YOLO 模型参数 ─────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/lyu/COD26/cod_-rm2026_-navigation/src/camera/best.pt',
        description='YOLO 模型文件路径')

    confidence_arg = DeclareLaunchArgument(
        'confidence_threshold', default_value='0.50',
        description='YOLO 置信度阈值')

    iou_arg = DeclareLaunchArgument(
        'iou_threshold', default_value='0.45',
        description='YOLO IoU 阈值')

    # ── 检测参数 ──────────────────────────────────────
    min_box_area_arg = DeclareLaunchArgument(
        'min_box_area', default_value='500',
        description='最小检测框面积 (像素)')

    priority_class_arg = DeclareLaunchArgument(
        'priority_class', default_value='3',
        description='优先跟踪的 YOLO 类别 ID')

    # ── 视觉伺服参数 ──────────────────────────────────
    dead_zone_arg = DeclareLaunchArgument(
        'dead_zone_px', default_value='15',
        description='死区像素范围 (±)，偏移量在此范围内视为已居中')

    dark_threshold_arg = DeclareLaunchArgument(
        'dark_threshold', default_value='30',
        description='遮挡判定：画面平均亮度低于此值视为变黑')

    obstruction_delay_arg = DeclareLaunchArgument(
        'obstruction_delay', default_value='1.5',
        description='遮挡后等待秒数再发送完成信号')

    max_grasp_arg = DeclareLaunchArgument(
        'max_grasp_count', default_value='2',
        description='最大抓取次数，超过后节点自动退出')

    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value='camera_color_optical_frame',
        description='相机光学 frame_id')

    # ── 启动日志 ──────────────────────────────────────
    log_start = LogInfo(
        msg=[
            '══════════════════════════════════════',
            '\n  启动 qiyu1 + move_base_of_yolo',
            '\n  start_qiyu1     = ', LaunchConfiguration('start_qiyu1'),
            '\n  start_move_base = ', LaunchConfiguration('start_move_base'),
            '\n  model_path      = ', LaunchConfiguration('model_path'),
            '\n  confidence      = ', LaunchConfiguration('confidence_threshold'),
            '\n  iou_threshold   = ', LaunchConfiguration('iou_threshold'),
            '\n  priority_class  = ', LaunchConfiguration('priority_class'),
            '\n  dead_zone_px    = ', LaunchConfiguration('dead_zone_px'),
            '\n  max_grasp_count = ', LaunchConfiguration('max_grasp_count'),
            '\n══════════════════════════════════════',
        ]
    )

    return LaunchDescription([
        start_qiyu1_arg,
        start_move_base_arg,
        model_path_arg,
        confidence_arg,
        iou_arg,
        min_box_area_arg,
        priority_class_arg,
        dead_zone_arg,
        dark_threshold_arg,
        obstruction_delay_arg,
        max_grasp_arg,
        frame_id_arg,
        log_start,
        OpaqueFunction(function=launch_setup),
    ])
