#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动 node_of_rs（RealSense D435i 彩色+深度发布）+ move_base_of_yolo（YOLO 视觉伺服）

用法：
  ros2 launch rc_view rs_movebase_launch.py
  ros2 launch rc_view rs_movebase_launch.py start_rs:=false         # 已有 RS 实例时跳过
  ros2 launch rc_view rs_movebase_launch.py start_move_base:=false  # 只跑 RS 驱动
  ros2 launch rc_view rs_movebase_launch.py model_path:=/path/to/best.pt confidence_threshold:=0.7
  ros2 launch rc_view rs_movebase_launch.py min_depth_m:=0.10 max_depth_m:=0.40

节点:
  node_of_rs          → /camera/color/image_raw, /camera/depth/image_raw, /camera/color/camera_info, /camera/depth/scale
  move_base_of_yolo   → /camera/yolo/y_offset, /camera/view_cmd, /camera/view_sig
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    """运行时组装节点"""

    # ── node_of_rs — RealSense D435i 彩色+深度发布 ────
    rs_node = Node(
        package='rc_view',
        executable='node_of_rs',
        name='realsense_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_rs')),
        parameters=[{
            'enable_filter': True,
            'depth_unit': 'mm',
        }],
    )

    # ── move_base_of_yolo — D435i + YOLO 视觉伺服 ────
    move_base_node = Node(
        package='rc_view',
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
            'dead_zone_m':          LaunchConfiguration('dead_zone_m'),
            'dark_threshold':       LaunchConfiguration('dark_threshold'),
            'obstruction_delay':    LaunchConfiguration('obstruction_delay'),
            'max_grasp_count':      LaunchConfiguration('max_grasp_count'),
            'frame_id':             LaunchConfiguration('frame_id'),
            'min_depth_m':          LaunchConfiguration('min_depth_m'),
            'max_depth_m':          LaunchConfiguration('max_depth_m'),
            'depth_filter_enabled': LaunchConfiguration('depth_filter_enabled'),
            'dark_roi_x1':          LaunchConfiguration('dark_roi_x1'),
            'dark_roi_x2':          LaunchConfiguration('dark_roi_x2'),
            'dark_roi_y1':          LaunchConfiguration('dark_roi_y1'),
            'dark_roi_y2':          LaunchConfiguration('dark_roi_y2'),
            'camera_offset_m':      LaunchConfiguration('camera_offset_m'),
        }],
    )

    return [rs_node, move_base_node]


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装"""

    # ── 开关参数 ──────────────────────────────────────
    start_rs_arg = DeclareLaunchArgument(
        'start_rs', default_value='true',
        description='启动 node_of_rs (RealSense D435i 发布节点)')

    start_move_base_arg = DeclareLaunchArgument(
        'start_move_base', default_value='true',
        description='启动 move_base_of_yolo (YOLO 视觉伺服)')

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

    # ── 检测过滤参数 ──────────────────────────────────
    min_box_area_arg = DeclareLaunchArgument(
        'min_box_area', default_value='500',
        description='最小检测框面积 (像素)')

    priority_class_arg = DeclareLaunchArgument(
        'priority_class', default_value='3',
        description='优先跟踪的 YOLO 类别 ID')

    # ── 视觉伺服参数 ──────────────────────────────────
    dead_zone_arg = DeclareLaunchArgument(
        'dead_zone_m', default_value='0.07',
        description='死区范围 (米)，偏移量在此范围内视为已居中')

    dark_threshold_arg = DeclareLaunchArgument(
        'dark_threshold', default_value='30',
        description='遮挡判定：画面平均亮度低于此值视为变黑')

    obstruction_delay_arg = DeclareLaunchArgument(
        'obstruction_delay', default_value='0.5',
        description='遮挡后等待秒数再发送 obstruction 信号')

    max_grasp_arg = DeclareLaunchArgument(
        'max_grasp_count', default_value='2',
        description='最大抓取次数，超过后节点自动退出')

    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value='camera_color_optical_frame',
        description='相机光学 frame_id')

    # ── 深度误报过滤参数 ──────────────────────────────
    min_depth_arg = DeclareLaunchArgument(
        'min_depth_m', default_value='0.15',
        description='深度过滤下限 (米)，低于此值视为误报')

    max_depth_arg = DeclareLaunchArgument(
        'max_depth_m', default_value='0.35',
        description='深度过滤上限 (米)，高于此值视为误报')

    depth_filter_arg = DeclareLaunchArgument(
        'depth_filter_enabled', default_value='true',
        description='是否启用深度误报过滤')

    # ── 遮挡检测 ROI ──────────────────────────────────
    dark_roi_x1_arg = DeclareLaunchArgument(
        'dark_roi_x1', default_value='0.667',
        description='遮挡 ROI 左边界（距图像左侧比例，默认 1-1/3≈0.667）')

    dark_roi_x2_arg = DeclareLaunchArgument(
        'dark_roi_x2', default_value='0.800',
        description='遮挡 ROI 右边界（距图像左侧比例，默认 1-1/5=0.800）')

    dark_roi_y1_arg = DeclareLaunchArgument(
        'dark_roi_y1', default_value='0.500',
        description='遮挡 ROI 上边界（距图像顶部比例，默认 1/2=0.500）')

    dark_roi_y2_arg = DeclareLaunchArgument(
        'dark_roi_y2', default_value='1.000',
        description='遮挡 ROI 下边界（距图像顶部比例，默认 1.000=底部）')

    # ── 相机安装偏置 ──────────────────────────────────
    camera_offset_arg = DeclareLaunchArgument(
        'camera_offset_m', default_value='0.01',
        description='RGB镜头在机械臂中心左侧的偏移量（米，正值=左）')

    # ── 启动日志 ──────────────────────────────────────
    log_start = LogInfo(
        msg=[
            '══════════════════════════════════════',
            '\n  启动 node_of_rs + move_base_of_yolo',
            '\n  start_rs             = ', LaunchConfiguration('start_rs'),
            '\n  start_move_base      = ', LaunchConfiguration('start_move_base'),
            '\n  model_path           = ', LaunchConfiguration('model_path'),
            '\n  confidence_threshold = ', LaunchConfiguration('confidence_threshold'),
            '\n  iou_threshold        = ', LaunchConfiguration('iou_threshold'),
            '\n  priority_class       = ', LaunchConfiguration('priority_class'),
            '\n  dead_zone_m          = ', LaunchConfiguration('dead_zone_m'),
            '\n  depth_filter_enabled = ', LaunchConfiguration('depth_filter_enabled'),
            '\n  min_depth_m          = ', LaunchConfiguration('min_depth_m'),
            '\n  max_depth_m          = ', LaunchConfiguration('max_depth_m'),
            '\n  max_grasp_count      = ', LaunchConfiguration('max_grasp_count'),
            '\n══════════════════════════════════════',
        ]
    )

    return LaunchDescription([
        start_rs_arg,
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
        min_depth_arg,
        max_depth_arg,
        depth_filter_arg,
        dark_roi_x1_arg,
        dark_roi_x2_arg,
        dark_roi_y1_arg,
        dark_roi_y2_arg,
        camera_offset_arg,
        log_start,
        OpaqueFunction(function=launch_setup),
    ])
