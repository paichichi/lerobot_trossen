#!/usr/bin/env bash
set -euo pipefail

model="${1:-ours_rn50}"
mode="${2:-download}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
max_relative_target=0.07

case "$model" in
  ours_rn50)
    policy_repo=Chipaipai/act-ours-rn50-end-to-end-carrot-100
    policy_revision=5f0fd733e9098ba2e4c7143d44ada99087e0ae7d
    policy_dir_name=ours_rn50_end_to_end
    ;;
  rn50_full)
    policy_repo=Chipaipai/act-rn50-full-carrot-100
    policy_revision=f599bba7e80f22a513f98a8579dae6cbf44ca627
    policy_dir_name=act_rn50_full_6k
    # Stateless spike limiting only: preserve the ACT target sequence and
    # avoid the lag introduced by velocity/acceleration filters.
    max_relative_target=0.06
    ;;
  rn50_full_36k)
    policy_repo=Chipaipai/act-rn50-full-carrot-100
    policy_revision=e9fb5faf5dcd5ea75112e736997dd98bdba9833a
    policy_dir_name=act_rn50_full_36k
    max_relative_target=0.06
    ;;
  rn50_rms_5k)
    policy_repo=Chipaipai/act-rn50-rms-ln-v1-carrot-100-5k
    policy_revision=7b2b73c574435301f37f87c443123549743c68bd
    policy_dir_name=act_rn50_rms_ln_v1_5k
    ;;
  rn50_rms_8k)
    policy_repo=Chipaipai/act-rn50-rms-ln-v1-carrot-100-8k
    policy_revision=a892129e578f3e86c51b1654644cde201898802a
    policy_dir_name=act_rn50_rms_ln_v1_8k
    ;;
  rn50_train100_8k)
    policy_repo=Chipaipai/act-rn50-rms-ln-v1-carrot-100-train100-8k
    policy_revision=b614602e3fdf004450a4072a17b0ade3653e32f8
    policy_dir_name=act_rn50_rms_ln_v1_train100_8k
    ;;
  rn50_visual_goal_8k)
    policy_repo=Chipaipai/act-rn50-visual-goal-v1-carrot-100-train100-8k
    policy_revision=0b12a4cbc6a36061cd73b54849213e57a8fd3943
    policy_dir_name=act_rn50_visual_goal_v1_train100_8k
    ;;
  rn18)
    policy_repo=Chipaipai/act-official-rn18-carrot-100
    policy_revision=8f3cf3b8358d46928bc12271027787cc1f7b0499
    policy_dir_name=rn18
    ;;
  *)
    echo "usage: $0 {rn50_visual_goal_8k|rn50_train100_8k|rn50_rms_5k|rn50_rms_8k|rn18|rn50_full|rn50_full_36k|ours_rn50} [--execute]" >&2
    exit 2
    ;;
esac

if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 {rn50_visual_goal_8k|rn50_train100_8k|rn50_rms_5k|rn50_rms_8k|rn18|rn50_full|rn50_full_36k|ours_rn50} [--execute]" >&2
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

if [[ "$model" == ours_rn50 || "$model" == rn50_full || "$model" == rn50_full_36k || "$model" == rn50_rms_5k || "$model" == rn50_rms_8k || "$model" == rn50_train100_8k || "$model" == rn50_visual_goal_8k ]]; then
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

sha256sum "$policy_path/model.safetensors" > "$run_dir/weights.sha256"
if [[ "$model" == ours_rn50 || "$model" == rn50_full || "$model" == rn50_full_36k || "$model" == rn50_rms_5k || "$model" == rn50_rms_8k || "$model" == rn50_train100_8k || "$model" == rn50_visual_goal_8k ]]; then
  sha256sum "$BACKBONE_CHECKPOINT" >> "$run_dir/weights.sha256"
fi

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
  --robot.max_relative_target="$max_relative_target"
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
