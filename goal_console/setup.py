import os
from setuptools import find_packages, setup

package_name = 'goal_console'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            ['launch/goal_console.launch.py',
             'launch/goal_gui.launch.py',
             'launch/mission_manager.launch.py',
             'launch/route2_only.launch.py',
             'launch/zone2_grid_nav.launch.py']),
        (os.path.join('share', package_name, 'config'),
            ['config/goal_points.yaml',
             'config/routes.yaml',
             'config/mission_manager.yaml',
             'config/z2_mission.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lyu',
    maintainer_email='lyu@todo.todo',
    description='终端交互式Nav2目标点导航控制台',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'goal_console_node = goal_console.goal_console_node:main',
            'goal_gui_node = goal_console.goal_gui_node:main',
            'mission_manager_node = goal_console.mission_manager_node:main',
            'zone2_grid_nav_gui = goal_console.zone2_grid_nav_gui:main',
            'startup_pose_manager = goal_console.startup_pose_manager:main',
            'z2_gui = goal_console.z2_gui:main',
            'z2_mission_manager = goal_console.z2_mission_manager:main',
        ],
    },
)
