"""goal_gui 启动文件

启动桌面 GUI 点位导航工具。

用法:
    ros2 launch goal_console goal_gui.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 源码目录下的点位文件（不会被 colcon build 覆盖）
    src_points_yaml = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'src', 'goal_console', 'config', 'goal_points.yaml')
    src_routes_yaml = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'src', 'goal_console', 'config', 'routes.yaml')
    src_map_yaml = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'src', 'cod_bringup', 'maps', '2026rc.yaml')

    declare_points_yaml = DeclareLaunchArgument(
        'points_yaml',
        default_value=os.path.abspath(src_points_yaml),
        description='预存目标点 YAML 文件路径'
    )
    declare_routes_yaml = DeclareLaunchArgument(
        'routes_yaml',
        default_value=os.path.abspath(src_routes_yaml),
        description='预设路线 YAML 文件路径'
    )
    declare_map_yaml = DeclareLaunchArgument(
        'map_yaml',
        default_value=os.path.abspath(src_map_yaml),
        description='地图 YAML 文件路径'
    )
    declare_map_frame = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='地图坐标系 frame_id'
    )
    declare_base_frame = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='机器人基座 frame_id'
    )
    declare_action_name = DeclareLaunchArgument(
        'action_name',
        default_value='/navigate_to_pose',
        description='Nav2 NavigateToPose action 名称'
    )

    goal_gui_node = Node(
        package='goal_console',
        executable='goal_gui_node',
        name='goal_gui_node',
        output='screen',
        parameters=[{
            'points_yaml': LaunchConfiguration('points_yaml'),
            'routes_yaml': LaunchConfiguration('routes_yaml'),
            'map_frame': LaunchConfiguration('map_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'action_name': LaunchConfiguration('action_name'),
        }],
    )

    return LaunchDescription([
        declare_points_yaml,
        declare_routes_yaml,
        declare_map_yaml,
        declare_map_frame,
        declare_base_frame,
        declare_action_name,
        goal_gui_node,
    ])
