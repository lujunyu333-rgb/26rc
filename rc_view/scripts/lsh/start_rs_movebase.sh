#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# RS + MoveBase: node_of_rs（RealSense D435i）+ move_base_of_yolo（YOLO 视觉伺服）
#
# 用法：
#   ./start_rs_movebase.sh
#   ./start_rs_movebase.sh start_rs:=false                      # 跳过 RS 驱动（已有实例时）
#   ./start_rs_movebase.sh start_move_base:=false               # 只跑 RS 驱动
#   ./start_rs_movebase.sh confidence_threshold:=0.7 dead_zone_m:=0.05
#   ./start_rs_movebase.sh min_depth_m:=0.10 max_depth_m:=0.40  # 调整深度过滤范围
#
# 所有参数透传给 ros2 launch rc_view rs_movebase_launch.py
# ─────────────────────────────────────────────────────────

set -e

# ── 环境 ──────────────────────────────────────────────
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26/cod_-rm2026_-navigation"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching rs_movebase_launch.py $*"
ros2 launch rc_view rs_movebase_launch.py "$@"
