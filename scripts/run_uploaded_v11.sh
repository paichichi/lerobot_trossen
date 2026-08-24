#!/usr/bin/env bash
set -euo pipefail

mode="${1:-download}"
if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

tcc_real_robot_root="${TCC_REAL_ROBOT_SOURCE_ROOT:-/home/robotarm/tcc-core-real-robot}"
backbone_source_root="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"
policy_revision=466bc1e4be7d1899da7281a5d8a30add04bf7e3c
policy_file=policies_v11_basic_chunked_mlp/ours_rn50/checkpoint_040000.pt
backbone_file=backbones/ours_rn50/checkpoint_040000.pt
asset_root="$repo_root/assets/tcc-policy-assets"
raw_policy="$asset_root/$policy_file"
backbone_checkpoint="$asset_root/$backbone_file"
policy_path="$repo_root/checkpoints/v11_ours_rn50"

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$repo_root/output/v11_ours_rn50_${run_stamp}"
mkdir -p "$run_dir"

finish_run() {
  exit_code=$?
  trap - EXIT
  {
    printf 'exit_code=%s\n' "$exit_code"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$run_dir/status.txt"
  exit "$exit_code"
}
trap finish_run EXIT

exec > >(tee -a "$run_dir/console.log") 2>&1
echo "All run information will be saved to: $run_dir"

uv sync --extra act
uv run --no-sync hf download Chipaipai/tcc-core-real-robot-policies \
  "$policy_file" "$backbone_file" \
  --revision "$policy_revision" \
  --local-dir "$asset_root"

export TCC_REAL_ROBOT_SOURCE_ROOT="$tcc_real_robot_root"
export BACKBONE_SOURCE_ROOT="$backbone_source_root"
export BACKBONE_CHECKPOINT="$backbone_checkpoint"

if [[ ! -f "$tcc_real_robot_root/src/tcc_real_robot/policy.py" ]]; then
  echo "TCC real-robot source missing: $tcc_real_robot_root" >&2
  exit 1
fi
if [[ ! -f "$backbone_source_root/xirl/models.py" ]]; then
  echo "TCC backbone source missing: $backbone_source_root" >&2
  exit 1
fi

uv run --no-sync python -m lerobot_policy_v11.export_v11 \
  --checkpoint "$raw_policy" \
  --output-dir "$policy_path" \
  --backbone-checkpoint "$backbone_checkpoint" \
  --backbone-source-root "$backbone_source_root" \
  --tcc-real-robot-source-root "$tcc_real_robot_root"

{
  printf 'model=v11_ours_rn50\n'
  printf 'mode=%s\n' "$mode"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'policy_revision=%s\n' "$policy_revision"
  printf 'raw_policy=%s\n' "$raw_policy"
  printf 'policy_path=%s\n' "$policy_path"
} > "$run_dir/run_metadata.txt"
sha256sum "$raw_policy" "$backbone_checkpoint" "$policy_path/model.safetensors" \
  > "$run_dir/weights.sha256"

if [[ "$mode" != --execute ]]; then
  echo "Converted only. The robot was not connected or moved."
  exit 0
fi

echo "Starting official LeRobot rollout for V11. Keep the E-stop ready."
record_command=(
  uv run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
  --robot.min_time_to_move_multiplier=2.0
  --robot.max_relative_target='{"joint_0": 0.07, "joint_1": 0.07, "joint_2": 0.07, "joint_3": 0.07, "joint_4": 0.07, "joint_5": 0.07, "left_carriage_joint": 0.003}'
  --robot.cameras='{cam_main: {type: intelrealsense, serial_number_or_name: "838212073584", width: 640, height: 480, fps: 30}, cam_wrist: {type: intelrealsense, serial_number_or_name: "409122274608", width: 640, height: 480, fps: 30}}'
  --strategy.type=episodic
  --fps=20
  --task="Pick up the carrot and place it in the pan"
  --return_to_initial_position=true
  --dataset.repo_id=Chipaipai/rollout_v11-ours-rn50-carrot-eval
  --dataset.root="$run_dir/dataset"
  --dataset.num_episodes=1
  --dataset.episode_time_s=30
  --dataset.reset_time_s=10
  --dataset.single_task="Pick up the carrot and place it in the pan"
  --dataset.push_to_hub=false
  --display_data=false
  --policy.path="$policy_path"
)
printf '%q ' "${record_command[@]}" > "$run_dir/resolved_command.txt"
printf '\n' >> "$run_dir/resolved_command.txt"
"${record_command[@]}"
