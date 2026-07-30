#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# RealSense 蓝/红区域检测节点
#
# 用法：
#   ./start_kfs.sh                                        # 默认 blue+red
#   ./start_kfs.sh --ros-args -p target_color:=red        # 只检测红色
#   ./start_kfs.sh --ros-args -p target_color:=blue       # 只检测蓝色
# ─────────────────────────────────────────────────────────

set -e

ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26/cod_-rm2026_-navigation"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

# 默认只检测蓝色，传参可覆盖
if [ $# -eq 0 ]; then
    set -- --ros-args -p target_color:=blue
fi

echo ">>> launching res_kfs $*"
ros2 run camera res_kfs "$@"
