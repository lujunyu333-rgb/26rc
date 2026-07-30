"""route2_only 启动文件

跳过一区，直接执行二区路线 route_2。

用法:
    ros2 launch goal_console route2_only.launch.py
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("goal_console"))
    default_config = str(pkg_share / "config" / "mission_manager.yaml")

    declare_config_yaml = DeclareLaunchArgument(
        'config_yaml',
        default_value=default_config,
        description='mission_manager 配置文件路径'
    )

    mission_manager_node = Node(
        package='goal_console',
        executable='mission_manager_node',
        name='mission_manager',
        output='screen',
        parameters=[
            LaunchConfiguration('config_yaml'),
            {'selected_route_id': 2},
            {'route_only_mode': True},
            {'bench_test_mode': False},
            {'dry_run_nav': False},
        ],
    )

    return LaunchDescription([
        declare_config_yaml,
        mission_manager_node,
    ])
