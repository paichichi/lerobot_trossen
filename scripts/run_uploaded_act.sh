#!/usr/bin/env bash
set -euo pipefail

model="${1:-ours_rn50}"
mode="${2:-download}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$model" in
  ours_rn50)
    policy_repo=Chipaipai/act-lite-ours-rn50-carrot-100
    policy_revision=424a8456d2e43f3730e5d684b122ed151728ec23
    ;;
  rn18)
    policy_repo=Chipaipai/act-official-rn18-carrot-100
    policy_revision=8f3cf3b8358d46928bc12271027787cc1f7b0499
    ;;
  *)
    echo "usage: $0 {ours_rn50|rn18} [--execute]" >&2
    exit 2
    ;;
esac

if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 {ours_rn50|rn18} [--execute]" >&2
  exit 2
fi

uv sync --extra act
policy_path="$repo_root/checkpoints/$model"
uv run --no-sync hf download "$policy_repo" \
  --revision "$policy_revision" \
  --local-dir "$policy_path"

if [[ "$model" == ours_rn50 ]]; then
  export BACKBONE_SOURCE_ROOT="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"
  export BACKBONE_CHECKPOINT="$repo_root/assets/tcc-policy-assets/backbones/ours_rn50/checkpoint_040000.pt"
  uv run --no-sync hf download Chipaipai/tcc-core-real-robot-policies \
    backbones/ours_rn50/checkpoint_040000.pt \
    --revision 466bc1e4be7d1899da7281a5d8a30add04bf7e3c \
    --local-dir="$repo_root/assets/tcc-policy-assets"
  if [[ "$mode" == --execute && ! -f "$BACKBONE_SOURCE_ROOT/xirl/models.py" ]]; then
    echo "TCC source missing: $BACKBONE_SOURCE_ROOT/xirl/models.py" >&2
    exit 1
  fi
fi

uv run --no-sync python -m json.tool "$policy_path/first_frame_report.json"
if [[ "$mode" != --execute ]]; then
  echo "Downloaded only. The robot was not connected or moved."
  exit 0
fi

echo "Starting physical evaluation: $model. Keep the E-stop ready."
uv run --no-sync lerobot-record \
  --robot.discover_packages_path=lerobot_robot_trossen \
  --robot.type=widowxai_follower_robot \
  --robot.ip_address=192.168.1.4 \
  --robot.id=follower \
  --robot.loop_rate=20 \
  --robot.min_time_to_move_multiplier=2.0 \
  --robot.max_relative_target='{"joint_0": 0.07, "joint_1": 0.07, "joint_2": 0.07, "joint_3": 0.07, "joint_4": 0.07, "joint_5": 0.07, "left_carriage_joint": 0.003}' \
  --robot.cameras='{cam_main: {type: intelrealsense, serial_number_or_name: "838212073584", width: 640, height: 480, fps: 30}, cam_wrist: {type: intelrealsense, serial_number_or_name: "409122274608", width: 640, height: 480, fps: 30}}' \
  --dataset.repo_id="Chipaipai/act-${model}-carrot-eval" \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Pick up the carrot and place it in the pan" \
  --dataset.push_to_hub=false \
  --display_data=true \
  --policy.discover_packages_path=lerobot_policy_backbone_act \
  --policy.path="$policy_path"
