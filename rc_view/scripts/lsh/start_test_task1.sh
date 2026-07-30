#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# test_task1: node_of_rs + move_base_of_yolo + qiyuQR + res_kfs
#
# 四个节点全部默认启动，可通过参数单独关闭：
#   ./start_test_task1.sh
#   ./start_test_task1.sh start_rs:=false              # 跳过 RealSense 发布节点
#   ./start_test_task1.sh start_move_base:=false       # 跳过 YOLO 视觉伺服
#   ./start_test_task1.sh start_qiyuqr:=false          # 跳过 QR 码检测
#   ./start_test_task1.sh start_res_kfs:=false         # 跳过颜色区域检测
#   ./start_test_task1.sh target_color:=red            # 改检测颜色 (blue/red/both)
#   ./start_test_task1.sh model_path:=/path/to/best.pt confidence_threshold:=0.7
#
# 所有参数透传给 ros2 launch rc_view test_task1_launch.py
# ─────────────────────────────────────────────────────────

set -e

# ── 环境 ──────────────────────────────────────────────
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching test_task1_launch.py $*"
ros2 launch rc_view test_task1_launch.py "$@"
