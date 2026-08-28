#!/usr/bin/env bash
set -euo pipefail

model="${1:-rn50_newcam_8k}"
mode="${2:-download}"
n_action_steps_override="${3:-}"
c920_path="${4:-${TROSSEN_C920_PATH:-/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920-video-index0}}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
max_relative_target=0.07
policy_prefix=""
task_description="Pick up the carrot and place it in the pot"
usage_models="rn18_newcam_2k|rn18_newcam_4k|rn18_newcam_6k|rn18_newcam_8k|rn50_newcam_2k|rn50_newcam_4k|rn50_newcam_6k|rn50_newcam_8k"

case "$model" in
  rn18_newcam_2k|rn18_newcam_4k|rn18_newcam_6k|rn18_newcam_8k)
    policy_repo=Chipaipai/act-official-rn18-carrot-to-pot-40-train40-8k
    policy_revision=fb23c5b528828d745a8b7bf466bd9bd4465efd08
    policy_dir_name=act_official_rn18_carrot_to_pot_40
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_prefix="checkpoints/$(printf '%06d' "$policy_step")"
    ;;
  rn50_newcam_2k|rn50_newcam_4k|rn50_newcam_6k|rn50_newcam_8k)
    policy_repo=Chipaipai/act-official-rn50-carrot-to-pot-40-train40-8k
    policy_revision=fd369e1ef0b8ea5b5cc539c34353c7ae750f117e
    policy_dir_name=act_official_rn50_carrot_to_pot_40
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_prefix="checkpoints/$(printf '%06d' "$policy_step")"
    ;;
  *)
    echo "usage: $0 {$usage_models} [--execute|download] [n_action_steps] [c920_path]" >&2
    exit 2
    ;;
esac

if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 {$usage_models} [--execute|download] [n_action_steps] [c920_path]" >&2
  exit 2
fi
if [[ -n "$n_action_steps_override" ]]; then
  if [[ ! "$n_action_steps_override" =~ ^[0-9]+$ ]] || (( n_action_steps_override < 1 || n_action_steps_override > 40 )); then
    echo "n_action_steps must be an integer from 1 to the trained chunk_size of 40." >&2
    exit 2
  fi
fi
if [[ ! "$c920_path" =~ ^/dev/v4l/by-id/[A-Za-z0-9._-]+$ ]]; then
  echo "c920_path must be a stable /dev/v4l/by-id device path: $c920_path" >&2
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
  printf 'policy_prefix=%s\n' "${policy_prefix:-repository_root}"
  printf 'n_action_steps_override=%s\n' "${n_action_steps_override:-checkpoint_default}"
  printf 'c920_path=%s\n' "$c920_path"
  printf 'camera_mode=c920_only\n'
  printf 'output_dir=%s\n' "$run_dir"
  printf 'invocation='
  printf '%q ' "$0" "$@"
  printf '\n'
} > "$run_dir/run_metadata.txt"

exec > >(tee -a "$run_dir/console.log") 2>&1
echo "All run information will be saved to: $run_dir"

uv sync --extra act
policy_download_root="$repo_root/checkpoints/$policy_dir_name"
if [[ -n "$policy_prefix" ]]; then
  uv run --no-sync hf download "$policy_repo" \
    --revision "$policy_revision" \
    --include "$policy_prefix/*" \
    --local-dir "$policy_download_root"
  policy_path="$policy_download_root/$policy_prefix"
else
  policy_path="$policy_download_root"
  uv run --no-sync hf download "$policy_repo" \
    --revision "$policy_revision" \
    --local-dir "$policy_path"
fi

sha256sum "$policy_path/model.safetensors" > "$run_dir/weights.sha256"

if [[ -f "$policy_path/first_frame_report.json" ]]; then
  cp "$policy_path/first_frame_report.json" "$run_dir/first_frame_report.json"
  uv run --no-sync python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({"decision": d["decision"], "summary": d["summary"], "gates": d["gates"]}, indent=2))' \
    "$policy_path/first_frame_report.json"
else
  uv run --no-sync python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({k:d.get(k) for k in ("type", "input_features", "output_features", "chunk_size", "n_action_steps")}, indent=2))' \
    "$policy_path/config.json"
fi
if [[ -n "$n_action_steps_override" ]]; then
  echo "Runtime policy override: n_action_steps=$n_action_steps_override"
fi
if [[ "$mode" != --execute ]]; then
  echo "Downloaded only. The robot was not connected or moved."
  exit 0
fi

echo "Starting physical evaluation: $model. Keep the E-stop ready."
if [[ ! -e "$c920_path" ]]; then
  echo "C920 device is unavailable: $c920_path" >&2
  exit 1
fi
uv run --no-sync python scripts/lock_c920_focus.py
robot_cameras="{cam_main: {type: opencv, index_or_path: \"$c920_path\", width: 640, height: 480, fps: 20, fourcc: MJPG}}"
rollout_command=(
  uv run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
  --robot.min_time_to_move_multiplier=2.0
  --robot.max_relative_target="$max_relative_target"
  --robot.cameras="$robot_cameras"
  --strategy.type=base
  --fps=20
  --task="$task_description"
  --return_to_initial_position=true
  --display_data=false
  --policy.path="$policy_path"
)
if [[ -n "$n_action_steps_override" ]]; then
  rollout_command+=(--policy.n_action_steps="$n_action_steps_override")
fi
printf '%q ' "${rollout_command[@]}" > "$run_dir/resolved_command.txt"
printf '\n' >> "$run_dir/resolved_command.txt"
"${rollout_command[@]}"
