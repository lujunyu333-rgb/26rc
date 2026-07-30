# auto_save_map.launch.py
import os
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess

def generate_launch_description():
    ld = LaunchDescription()

    workspace = "/home/lyu/COD26/cod_-rm2026_-navigation"
    save_dir = os.path.join(workspace, "src/cod_bringup/maps/auto_save")

    def create_save_command(suffix: str) -> list:
        return [
            'bash', '-c',
            f'source /opt/ros/humble/setup.bash && '
            f'source {workspace}/install/setup.bash && '
            f'mkdir -p {save_dir} && '
            f'ros2 run nav2_map_server map_saver_cli -f {save_dir}/auto_map_{suffix}'
        ]

    intervals = [60, 120, 180]

    for t in intervals:
        ld.add_action(
            TimerAction(
                period=float(t),
                actions=[
                    ExecuteProcess(
                        cmd=create_save_command("$(date +%H%M%S)"),
                        output='screen'
                    )
                ]
            )
        )

    return ld