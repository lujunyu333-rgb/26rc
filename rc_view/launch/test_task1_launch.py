#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动 node_of_rs（RealSense D435i 彩色+深度发布）+
     move_base_of_yolo（YOLO 视觉伺服）+
     qiyuQR（USB摄像头 QR码检测）+
     res_kfs（RealSense 颜色区域检测）

用法：
  ros2 launch rc_view test_task1_launch.py
  ros2 launch rc_view test_task1_launch.py start_rs:=false             # 已有 RS 实例时跳过
  ros2 launch rc_view test_task1_launch.py start_move_base:=false      # 跳过视觉伺服
  ros2 launch rc_view test_task1_launch.py start_qiyuqr:=false         # 跳过QR检测
  ros2 launch rc_view test_task1_launch.py start_res_kfs:=false        # 跳过区域检测
  ros2 launch rc_view test_task1_launch.py target_color:=red           # 改检测颜色
  ros2 launch rc_view test_task1_launch.py model_path:=/path/to/best.pt confidence_threshold:=0.7

节点:
  node_of_rs          → /camera/color/image_raw, /camera/depth/image_raw, /camera/color/camera_info, /camera/depth/scale
  move_base_of_yolo   → /camera/yolo/y_offset, /camera/view_cmd, /camera/view_sig
  qiyuQR              → /camera/view_cmd (QR码检测 flag: 仅发送 2/3, 其他内容过滤)
  res_kfs             → 蓝/红区域检测结果 (0.3s 延迟发送)
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
        executable='move_base',
        name='move_base',
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

    # ── qiyuQR — USB摄像头 QR码检测 ────────────────────
    qiyuqr_node = Node(
        package='rc_view',
        executable='qrqiyu',
        name='qiyuQR_cam',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('start_qiyuqr')),
        parameters=[{
            'detection_window_s': LaunchConfiguration('qr_detection_window_s'),
            'roi_x_ratio':        LaunchConfiguration('qr_roi_x_ratio'),
            'roi_y_ratio':        LaunchConfiguration('qr_roi_y_ratio'),
            'roi_w_ratio':        LaunchConfiguration('qr_roi_w_ratio'),
            'roi_h_ratio':        LaunchConfiguration('qr_roi_h_ratio'),
            'gamma':              LaunchConfiguration('qr_gamma'),
            'blur_ksize':         LaunchConfiguration('qr_blur_ksize'),
            'thresh_block_size':  LaunchConfiguration('qr_thresh_block_size'),
            'thresh_C':           LaunchConfiguration('qr_thresh_C'),
            'display_scale':      LaunchConfiguration('qr_display_scale'),
            'enable_display':     LaunchConfiguration('qr_enable_display'),
            'device_path':        LaunchConfiguration('qr_device_path'),
        }],
    )

    # ── res_kfs — RealSense 颜色区域检测（0.3s 延迟发送）──
    res_kfs_node = Node(
        package='rc_view',
        executable='res_kfs',
        name='reg_kfs',
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_res_kfs')),
        parameters=[{
            'target_color': LaunchConfiguration('target_color'),
        }],
    )

    return [rs_node, move_base_node, qiyuqr_node, res_kfs_node]


def generate_launch_description():
    """声明参数，通过 OpaqueFunction 延迟组装"""

    # ── 开关参数 ──────────────────────────────────────
    start_rs_arg = DeclareLaunchArgument(
        'start_rs', default_value='true',
        description='启动 node_of_rs (RealSense D435i 发布节点)')

    start_move_base_arg = DeclareLaunchArgument(
        'start_move_base', default_value='true',
        description='启动 move_base_of_yolo (YOLO 视觉伺服)')

    start_qiyuqr_arg = DeclareLaunchArgument(
        'start_qiyuqr', default_value='true',
        description='启动 qiyuQR (USB摄像头 QR码检测)')

    start_res_kfs_arg = DeclareLaunchArgument(
        'start_res_kfs', default_value='true',
        description='启动 res_kfs (RealSense 颜色区域检测)')

    # ── res_kfs 参数 ──────────────────────────────────
    target_color_arg = DeclareLaunchArgument(
        'target_color', default_value='both',
        description='res_kfs 检测颜色: blue / red / both')

    # ── YOLO 模型参数 ─────────────────────────────────
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/lyu/COD26/cod_-rm2026_-navigation/src/rc_view/rc_view/best.pt',
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
        'dead_zone_m', default_value='0.04',
        description='死区范围 (米)，偏移量在此范围内视为已居中')

    dark_threshold_arg = DeclareLaunchArgument(
        'dark_threshold', default_value='66',
        description='遮挡判定：画面平均亮度低于此值视为变黑')

    obstruction_delay_arg = DeclareLaunchArgument(
        'obstruction_delay', default_value='0.8',
        description='遮挡后等待秒数再发送 obstruction 信号')

    max_grasp_arg = DeclareLaunchArgument(
        'max_grasp_count', default_value='6',
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
        description='遮挡 ROI 左边界（距图像左侧比例）')

    dark_roi_x2_arg = DeclareLaunchArgument(
        'dark_roi_x2', default_value='0.800',
        description='遮挡 ROI 右边界（距图像左侧比例）')

    dark_roi_y1_arg = DeclareLaunchArgument(
        'dark_roi_y1', default_value='0.500',
        description='遮挡 ROI 上边界（距图像顶部比例）')

    dark_roi_y2_arg = DeclareLaunchArgument(
        'dark_roi_y2', default_value='1.000',
        description='遮挡 ROI 下边界（距图像顶部比例）')

    # ── 相机安装偏置 ──────────────────────────────────
    camera_offset_arg = DeclareLaunchArgument(
        'camera_offset_m', default_value='0.01',
        description='RGB镜头在机械臂中心左侧的偏移量（米，正值=左）')

    # ── qiyuQR 参数 ───────────────────────────────────
    qr_detection_window_arg = DeclareLaunchArgument(
        'qr_detection_window_s', default_value='180.0',
        description='QR检测窗口时长 (秒)')

    qr_roi_x_arg = DeclareLaunchArgument(
        'qr_roi_x_ratio', default_value='0',
        description='QR ROI 左边界（占全图比例）')

    qr_roi_y_arg = DeclareLaunchArgument(
        'qr_roi_y_ratio', default_value='0',
        description='QR ROI 上边界（占全图比例）')

    qr_roi_w_arg = DeclareLaunchArgument(
        'qr_roi_w_ratio', default_value='1',
        description='QR ROI 宽度（占全图比例）')

    qr_roi_h_arg = DeclareLaunchArgument(
        'qr_roi_h_ratio', default_value='1',
        description='QR ROI 高度（占全图比例）')

    qr_gamma_arg = DeclareLaunchArgument(
        'qr_gamma', default_value='2.0',
        description='QR Gamma校正 (大于1降低曝光)')

    qr_blur_arg = DeclareLaunchArgument(
        'qr_blur_ksize', default_value='5',
        description='QR 高斯模糊核大小 (奇数)')

    qr_thresh_block_arg = DeclareLaunchArgument(
        'qr_thresh_block_size', default_value='11',
        description='QR 自适应阈值邻域大小 (奇数)')

    qr_thresh_c_arg = DeclareLaunchArgument(
        'qr_thresh_C', default_value='2',
        description='QR 自适应阈值常数')

    qr_enable_display_arg = DeclareLaunchArgument(
        'qr_enable_display', default_value='true',
        description='QR 是否开启可视化窗口 (true=显示, false=无头模式)')

    qr_display_scale_arg = DeclareLaunchArgument(
        'qr_display_scale', default_value='0.3',
        description='QR imshow 缩放比例')

    qr_device_path_arg = DeclareLaunchArgument(
        'qr_device_path', default_value='6',
        description='QR USB摄像头设备索引 (/dev/videoX)')

    # ── 启动日志 ──────────────────────────────────────
    log_start = LogInfo(
        msg=[
            '══════════════════════════════════════',
            '\n  启动 node_of_rs + move_base_of_yolo + qiyuQR + res_kfs',
            '\n  start_rs             = ', LaunchConfiguration('start_rs'),
            '\n  start_move_base      = ', LaunchConfiguration('start_move_base'),
            '\n  start_qiyuqr         = ', LaunchConfiguration('start_qiyuqr'),
            '\n  start_res_kfs        = ', LaunchConfiguration('start_res_kfs'),
            '\n  target_color         = ', LaunchConfiguration('target_color'),
            '\n  model_path           = ', LaunchConfiguration('model_path'),
            '\n  confidence_threshold = ', LaunchConfiguration('confidence_threshold'),
            '\n  iou_threshold        = ', LaunchConfiguration('iou_threshold'),
            '\n  priority_class       = ', LaunchConfiguration('priority_class'),
            '\n  dead_zone_m          = ', LaunchConfiguration('dead_zone_m'),
            '\n  depth_filter_enabled = ', LaunchConfiguration('depth_filter_enabled'),
            '\n  min_depth_m          = ', LaunchConfiguration('min_depth_m'),
            '\n  max_depth_m          = ', LaunchConfiguration('max_depth_m'),
            '\n  max_grasp_count      = ', LaunchConfiguration('max_grasp_count'),
            '\n  dark_threshold       = ', LaunchConfiguration('dark_threshold'),
            '\n  obstruction_delay    = ', LaunchConfiguration('obstruction_delay'),
            '\n  camera_offset_m      = ', LaunchConfiguration('camera_offset_m'),
            '\n  qr_detection_window_s= ', LaunchConfiguration('qr_detection_window_s'),
            '\n  qr_device_path       = ', LaunchConfiguration('qr_device_path'),
            '\n══════════════════════════════════════',
        ]
    )

    return LaunchDescription([
        # 开关
        start_rs_arg,
        start_move_base_arg,
        start_qiyuqr_arg,
        start_res_kfs_arg,
        # res_kfs
        target_color_arg,
        # YOLO
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
        # qiyuQR
        qr_detection_window_arg,
        qr_roi_x_arg,
        qr_roi_y_arg,
        qr_roi_w_arg,
        qr_roi_h_arg,
        qr_gamma_arg,
        qr_blur_arg,
        qr_thresh_block_arg,
        qr_thresh_c_arg,
        qr_enable_display_arg,
        qr_display_scale_arg,
        qr_device_path_arg,
        # 日志
        log_start,
        OpaqueFunction(function=launch_setup),
    ])
