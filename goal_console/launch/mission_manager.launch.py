"""mission_manager 启动文件

启动比赛任务状态机节点。

用法:
    ros2 launch goal_console mission_manager.launch.py                    # 默认路线 1
    ros2 launch goal_console mission_manager.launch.py selected_route_id:=2
    ros2 launch goal_console mission_manager.launch.py selected_route_id:=3
    ros2 launch goal_console mission_manager.launch.py selected_route_id:=4

    工作台联调:
    ros2 launch goal_console mission_manager.launch.py selected_route_id:=1 bench_test_mode:=true dry_run_nav:=true
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

    declare_selected_route_id = DeclareLaunchArgument(
        'selected_route_id',
        default_value='2',
        description='选定路线 ID (1/2/3/4)'
    )

    declare_bench_test_mode = DeclareLaunchArgument(
        'bench_test_mode',
        default_value='false',
        description='工作台测试模式：true=跳过 zone1_start，直接进入视觉偏移等待'
    )

    declare_dry_run_nav = DeclareLaunchArgument(
        'dry_run_nav',
        default_value='false',
        description='dry run：true=只打印目标点，不发送 NavigateToPose'
    )

    mission_manager_node = Node(
        package='goal_console',
        executable='mission_manager_node',
        name='mission_manager',
        output='screen',
        parameters=[
            LaunchConfiguration('config_yaml'),
            {'selected_route_id': LaunchConfiguration('selected_route_id')},
            {'bench_test_mode': LaunchConfiguration('bench_test_mode')},
            {'dry_run_nav': LaunchConfiguration('dry_run_nav')},
        ],
    )

    return LaunchDescription([
        declare_config_yaml,
        declare_selected_route_id,
        declare_bench_test_mode,
        declare_dry_run_nav,
        mission_manager_node,
    ])
