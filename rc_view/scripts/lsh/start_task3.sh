#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# Task3: qiyucolor2（USB 摄像头黄色检测）+ YOLO（D435i 目标检测+3D坐标）
#
# 用法：
#   ./start_task3.sh
#   ./start_task3.sh start_qiyu2:=false                      # 只跑 YOLO
#   ./start_task3.sh start_yolo:=false                       # 只跑颜色检测
#   ./start_task3.sh confidence_threshold:=0.7 target_class:=0
#
# 所有参数透传给 ros2 launch rc_view task3_launch.py
# ─────────────────────────────────────────────────────────

set -e

# ── 环境 ──────────────────────────────────────────────
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26/cod_-rm2026_-navigation"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching task3_launch.py $*"
ros2 launch rc_view task3_launch.py "$@"
