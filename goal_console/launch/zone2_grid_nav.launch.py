"""二区格子导航 GUI 启动文件

用法:
    ros2 launch goal_console zone2_grid_nav.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='goal_console',
            executable='zone2_grid_nav_gui',
            name='zone2_grid_nav_gui',
            output='screen',
        ),
    ])
