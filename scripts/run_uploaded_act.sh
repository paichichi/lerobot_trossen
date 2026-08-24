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

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$repo_root/output/${model}_${run_stamp}"
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

{
  printf 'model=%s\n' "$model"
  printf 'mode=%s\n' "$mode"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'policy_repo=%s\n' "$policy_repo"
  printf 'policy_revision=%s\n' "$policy_revision"
  printf 'output_dir=%s\n' "$run_dir"
  printf 'invocation='
  printf '%q ' "$0" "$@"
  printf '\n'
} > "$run_dir/run_metadata.txt"

exec > >(tee -a "$run_dir/console.log") 2>&1
echo "All run information will be saved to: $run_dir"

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

cp "$policy_path/first_frame_report.json" "$run_dir/first_frame_report.json"
sha256sum "$policy_path/model.safetensors" > "$run_dir/weights.sha256"
if [[ "$model" == ours_rn50 ]]; then
  sha256sum "$BACKBONE_CHECKPOINT" >> "$run_dir/weights.sha256"
fi

uv run --no-sync python -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({"decision": d["decision"], "summary": d["summary"], "gates": d["gates"]}, indent=2))' \
  "$policy_path/first_frame_report.json"
if [[ "$mode" != --execute ]]; then
  echo "Downloaded only. The robot was not connected or moved."
  exit 0
fi

echo "Starting base-mode physical evaluation: $model. Keep the E-stop ready."
echo "This ACT checkpoint has no task-completion output; press Ctrl+C after success."
record_command=(
  uv run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
  --robot.min_time_to_move_multiplier=2.0
  --robot.startup_home_positions='[-0.00019073777366429567, 1.046196699142456, 0.5239566564559937, 0.6292439103126526, -0.000572213320992887, 0.00019073777366429567, 0.0]'
  --robot.startup_home_goal_time_s=10.0
  --robot.startup_home_settle_time_s=1.0
  --robot.startup_home_max_arm_error_rad=0.03
  --robot.startup_home_max_gripper_error_m=0.002
  --robot.controller_connect_attempts=3
  --robot.controller_connect_retry_delay_s=2.0
  --robot.fold_on_disconnect=true
  --robot.fold_staging_goal_time_s=3.0
  --robot.folded_positions='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
  --robot.fold_goal_time_s=3.0
  --robot.fold_max_arm_error_rad=0.12
  --robot.fold_max_gripper_error_m=0.003
  --robot.arm_max_velocity_rad_s=0.35
  --robot.arm_max_acceleration_rad_s2=1.75
  --robot.gripper_max_velocity_m_s=0.02
  --robot.gripper_deadband_m=0.001
  --robot.postprocess_max_dt_multiplier=2.0
  --robot.max_relative_target='{"joint_0": 0.07, "joint_1": 0.07, "joint_2": 0.07, "joint_3": 0.07, "joint_4": 0.07, "joint_5": 0.07, "left_carriage_joint": 0.003}'
  --robot.cameras='{cam_main: {type: intelrealsense, serial_number_or_name: "838212073584", width: 640, height: 480, fps: 30}, cam_wrist: {type: intelrealsense, serial_number_or_name: "409122274608", width: 640, height: 480, fps: 30}}'
  --strategy.type=base
  --fps=20
  --duration=0
  --task="Pick up the carrot and place it in the pan"
  --return_to_initial_position=true
  --display_data=false
  --policy.path="$policy_path"
)
printf '%q ' "${record_command[@]}" > "$run_dir/resolved_command.txt"
printf '\n' >> "$run_dir/resolved_command.txt"
"${record_command[@]}"
