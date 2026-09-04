#!/usr/bin/env bash
# Practice leader-to-follower teleoperation without recording or cameras.

set -euo pipefail
cd "$(dirname "$0")/.."

uv run --no-sync lerobot-teleoperate \
  --robot.discover_packages_path=lerobot_robot_trossen \
  --robot.type=widowxai_follower_robot \
  --robot.ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.loop_rate=20 \
  --robot.min_time_to_move_multiplier=1.5 \
  --robot.max_relative_target=0.07 \
  --teleop.type=widowxai_leader_teleop \
  --teleop.ip_address=192.168.1.2 \
  --teleop.id=leader \
  --display_data=false
