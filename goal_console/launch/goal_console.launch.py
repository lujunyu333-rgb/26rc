"""goal_console 启动文件

启动终端交互式Nav2目标点导航控制台。

用法:
    ros2 launch goal_console goal_console.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('goal_console')

    # 声明启动参数
    declare_points_yaml = DeclareLaunchArgument(
        'points_yaml',
        default_value=os.path.join(pkg_share, 'config', 'goal_points.yaml'),
        description='预存目标点 YAML 文件路径'
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

    points_yaml = LaunchConfiguration('points_yaml')
    map_frame = LaunchConfiguration('map_frame')
    base_frame = LaunchConfiguration('base_frame')
    action_name = LaunchConfiguration('action_name')

    # 控制台节点（独占终端）
    goal_console_node = Node(
        package='goal_console',
        executable='goal_console_node',
        name='goal_console_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'points_yaml': points_yaml,
            'map_frame': map_frame,
            'base_frame': base_frame,
            'action_name': action_name,
        }],
    )

    return LaunchDescription([
        declare_points_yaml,
        declare_map_frame,
        declare_base_frame,
        declare_action_name,
        goal_console_node,
    ])
