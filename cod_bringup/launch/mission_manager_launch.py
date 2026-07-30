import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bringup_dir = get_package_share_directory('cod_bringup')

    route_file = LaunchConfiguration('route_file')
    route_name = LaunchConfiguration('route_name')

    declare_route_file = DeclareLaunchArgument(
        'route_file',
        default_value=os.path.join(bringup_dir, 'wps', 'mission_routes.yaml'),
        description='Mission route yaml file'
    )

    declare_route_name = DeclareLaunchArgument(
        'route_name',
        default_value='test_stair',
        description='Route name in mission_routes.yaml'
    )

    mission_manager = Node(
        package='cod_bringup',
        executable='mission_manager.py',
        name='mission_manager',
        output='screen',
        parameters=[
            {
                'route_file': route_file,
                'route_name': route_name,
                'global_frame': 'map',
                'robot_frame': 'base_link',
            }
        ]
    )

    return LaunchDescription([
        declare_route_file,
        declare_route_name,
        mission_manager,
    ])
