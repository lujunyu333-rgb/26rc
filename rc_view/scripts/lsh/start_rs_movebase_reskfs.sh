#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
# RS + MoveBase + ResKFS:
#   node_of_rs（RealSense D435i）+ move_base_of_yolo（YOLO 视觉伺服）+ res_kfs（颜色区域检测）
#
# 用法：
#   ./start_rs_movebase_reskfs.sh
#   ./start_rs_movebase_reskfs.sh start_rs:=false                      # 跳过 RS 驱动
#   ./start_rs_movebase_reskfs.sh start_move_base:=false               # 跳过视觉伺服
#   ./start_rs_movebase_reskfs.sh start_res_kfs:=false                 # 跳过区域检测
#   ./start_rs_movebase_reskfs.sh target_color:=red                    # 改检测颜色
#   ./start_rs_movebase_reskfs.sh confidence_threshold:=0.7 dead_zone_m:=0.05
#   ./start_rs_movebase_reskfs.sh min_depth_m:=0.10 max_depth_m:=0.40  # 调整深度过滤
#
# 所有参数透传给 ros2 launch rc_view rs_movebase_reskfs_launch.py
# ─────────────────────────────────────────────────────────

set -e

# ── 环境 ──────────────────────────────────────────────
ROS2_DISTRO="${ROS2_DISTRO:-humble}"
WS_ROOT="/home/lyu/COD26"

echo ">>> sourcing /opt/ros/${ROS2_DISTRO}/setup.bash"
source "/opt/ros/${ROS2_DISTRO}/setup.bash"

echo ">>> sourcing ${WS_ROOT}/install/setup.bash"
source "${WS_ROOT}/install/setup.bash"

echo ">>> launching rs_movebase_reskfs_launch.py $*"
ros2 launch rc_view rs_movebase_reskfs_launch.py "$@"
