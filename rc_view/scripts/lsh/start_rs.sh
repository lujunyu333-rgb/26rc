#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# RealSense D435i 彩色+深度发布节点
# ─────────────────────────────────────────────────────────

set -e

ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26/cod_-rm2026_-navigation"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching node_of_rs"
ros2 run camera node_of_rs "$@"
