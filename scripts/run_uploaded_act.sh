#!/usr/bin/env bash
set -euo pipefail

model="${1:-ours_rn50}"
mode="${2:-download}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$model" in
  ours_rn50)
    policy_repo=Chipaipai/act-ours-rn50-end-to-end-carrot-100
    policy_revision=5f0fd733e9098ba2e4c7143d44ada99087e0ae7d
    policy_dir_name=ours_rn50_end_to_end
    ;;
  rn18)
    policy_repo=Chipaipai/act-official-rn18-carrot-100
    policy_revision=8f3cf3b8358d46928bc12271027787cc1f7b0499
    policy_dir_name=rn18
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
policy_path="$repo_root/checkpoints/$policy_dir_name"
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

echo "Starting physical evaluation: $model. Keep the E-stop ready."
rollout_command=(
  uv run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
  --robot.min_time_to_move_multiplier=2.0
  --robot.max_relative_target=0.07
  --robot.cameras='{cam_main: {type: intelrealsense, serial_number_or_name: "838212073584", width: 640, height: 480, fps: 30}, cam_wrist: {type: intelrealsense, serial_number_or_name: "409122274608", width: 640, height: 480, fps: 30}}'
  --strategy.type=base
  --fps=20
  --task="Pick up the carrot and place it in the pan"
  --return_to_initial_position=true
  --display_data=false
  --policy.path="$policy_path"
)
printf '%q ' "${rollout_command[@]}" > "$run_dir/resolved_command.txt"
printf '\n' >> "$run_dir/resolved_command.txt"
"${rollout_command[@]}"
