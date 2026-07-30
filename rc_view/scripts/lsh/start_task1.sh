#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# Task1: qiyucolor1（USB 摄像头黄/紫检测）+ move_base_of_yolo（视觉伺服）
#
# 用法：
#   ./start_task1.sh
#   ./start_task1.sh start_qiyu1:=false                      # 只跑视觉伺服
#   ./start_task1.sh start_move_base:=false                  # 只跑颜色检测
#   ./start_task1.sh confidence_threshold:=0.7 dead_zone_px:=20
#
# 所有参数透传给 ros2 launch rc_view task1_launch.py
# ─────────────────────────────────────────────────────────

set -e

# ── 环境 ──────────────────────────────────────────────
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26/cod_-rm2026_-navigation"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching task1_launch.py $*"
ros2 launch rc_view task1_launch.py "$@"
