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
policy_path_override=""
backbone_checkpoint_override=""
backbone_source_root_override=""
task_description="Pick up the carrot and place it in the pot"
usage_models="rn18_newcam_2k|rn18_newcam_4k|rn18_newcam_6k|rn18_newcam_8k|rn50_newcam_2k|rn50_newcam_4k|rn50_newcam_6k|rn50_newcam_8k|rn50_newcam_9k|rn50_newcam_10k|rn50_spatial_8k|rn50_color_2k|rn50_color_4k|rn50_color_6k|rn50_color_8k|rn50_color3_4k|rn50_color3_5k|rn50_color3_6k|rn50_color3_7k|rn50_color3_8k|vit_ours_2k|vit_ours_4k|vit_ours_6k|vit_ours_8k|ours_vit_new_2k|ours_vit_new_4k|ours_vit_new_6k|ours_vit_new_8k|ours_rn50_new_2k|ours_rn50_new_4k|ours_rn50_new_6k|ours_rn50_new_8k|d4r_imagenet_2k|d4r_imagenet_4k|d4r_imagenet_6k|d4r_imagenet_8k|hrp_imagenet_2k|hrp_imagenet_4k|hrp_imagenet_6k|hrp_imagenet_8k|vit_compact_1k|vit_compact_3k|vit_last1_250|vit_last1_500|vit_last1_750|vit_last1_1k|vit_ln_1k|vit_ln_2k|vit_ln_3k|vit_ln_4k|vit_ln_5k|vit_ln_6k|vit_ln_7k|vit_ln_8k|vit_later_1k|vit_later_2k|vit_later_3k|vit_later_4k|vit_later_5k|vit_later_6k|vit_later_7k|vit_later_8k|vit_e2e_2k|vit_e2e_4k|vit_e2e_6k|vit_e2e_8k|rn50_layerwise_2k|rn50_layerwise_4k|rn50_layerwise_6k|rn50_layerwise_8k|rn50_late_2k|rn50_late_4k|rn50_late_6k|rn50_late_8k"

case "$model" in
  rn18_newcam_2k|rn18_newcam_4k|rn18_newcam_6k|rn18_newcam_8k)
    policy_repo=Chipaipai/act-official-rn18-carrot-to-pot-40-train40-8k
    policy_revision=fb23c5b528828d745a8b7bf466bd9bd4465efd08
    policy_dir_name=act_official_rn18_carrot_to_pot_40
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_prefix="checkpoints/$(printf '%06d' "$policy_step")"
    ;;
  rn50_newcam_2k|rn50_newcam_4k|rn50_newcam_6k|rn50_newcam_8k|rn50_newcam_9k|rn50_newcam_10k)
    policy_repo=Chipaipai/act-official-rn50-carrot-to-pot-40-train40-8k
    policy_revision=fd369e1ef0b8ea5b5cc539c34353c7ae750f117e
    policy_dir_name=act_official_rn50_carrot_to_pot_40
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_prefix="checkpoints/$(printf '%06d' "$policy_step")"
    ;;
  rn50_spatial_8k)
    policy_repo=local/act-official-rn50-carrot-to-pot-40-spatial-lr1e6-8k
    policy_revision=local
    policy_dir_name=act_official_rn50_carrot_to_pot_40_spatial_lr1e6
    policy_path_override="$repo_root/outputs/train/act_official_rn50_carrot_to_pot_40_spatial_lr1e6_train40_8000steps/checkpoints/008000/pretrained_model"
    ;;
  rn50_color_2k|rn50_color_4k|rn50_color_6k|rn50_color_8k)
    policy_repo=local/act-official-rn50-carrot-to-pot-40-color-lr1e6-8k
    policy_revision=local
    policy_dir_name=act_official_rn50_carrot_to_pot_40_color_lr1e6
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_official_rn50_carrot_to_pot_40_color_lr1e6_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    ;;
  rn50_color3_4k|rn50_color3_5k|rn50_color3_6k|rn50_color3_7k|rn50_color3_8k)
    policy_repo=local/act-official-rn50-carrot-to-pot-40-color3-lr1e6-8k
    policy_revision=local
    policy_dir_name=act_official_rn50_carrot_to_pot_40_color3_lr1e6
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_official_rn50_carrot_to_pot_40_color3_lr1e6_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    ;;
  vit_ours_2k|vit_ours_4k|vit_ours_6k|vit_ours_8k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-frozen-8k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_frozen
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_frozen_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  ours_vit_new_2k|ours_vit_new_4k|ours_vit_new_6k|ours_vit_new_8k)
    policy_repo=local/act-ours-vit-new-frozen-carrot-to-pot-40-color3-8k
    policy_revision=local
    policy_dir_name=act_ours_vit_new_frozen_carrot_to_pot_40_color3
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_frozen_carrot_to_pot_40_color3_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  ours_rn50_new_2k|ours_rn50_new_4k|ours_rn50_new_6k|ours_rn50_new_8k)
    policy_repo=local/act-ours-rn50-new-frozen-carrot-to-pot-40-color3-8k
    policy_revision=local
    policy_dir_name=act_ours_rn50_new_frozen_carrot_to_pot_40_color3
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_rn50_frozen_carrot_to_pot_40_color3_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_rn50/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  d4r_imagenet_2k|d4r_imagenet_4k|d4r_imagenet_6k|d4r_imagenet_8k)
    policy_repo=local/act-d4r-imagenet-frozen-carrot-to-pot-40-color3-8k
    policy_revision=local
    policy_dir_name=act_d4r_imagenet_frozen_carrot_to_pot_40_color3
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_d4r_imagenet_frozen_carrot_to_pot_40_color3_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/d4r_imagenet/D4R_IN_1M.pth"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  hrp_imagenet_2k|hrp_imagenet_4k|hrp_imagenet_6k|hrp_imagenet_8k)
    policy_repo=local/act-hrp-imagenet-frozen-carrot-to-pot-40-color3-8k
    policy_revision=local
    policy_dir_name=act_hrp_imagenet_frozen_carrot_to_pot_40_color3
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_hrp_imagenet_frozen_carrot_to_pot_40_color3_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/hrp_imagenet/HRP_IN.pth"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  vit_compact_1k|vit_compact_3k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-compact-val20-4k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_compact_val20
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_compact_val20_4000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  vit_last1_250|vit_last1_500|vit_last1_750|vit_last1_1k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-last1-lr1e7-1k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_last1_lr1e7
    policy_step="${model##*_}"
    if [[ "$policy_step" == *k ]]; then
      policy_step="${policy_step%k}000"
    fi
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_last1_lr1e7_from_frozen8k_1000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  vit_ln_1k|vit_ln_2k|vit_ln_3k|vit_ln_4k|vit_ln_5k|vit_ln_6k|vit_ln_7k|vit_ln_8k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-layernorm-only-lr1e6-8k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_layernorm_only_lr1e6
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_layernorm_only_lr1e6_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  vit_later_1k|vit_later_2k|vit_later_3k|vit_later_4k|vit_later_5k|vit_later_6k|vit_later_7k|vit_later_8k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-later1-lr1e6-8k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_later1_lr1e6
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_later1_lr1e6_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  vit_e2e_2k|vit_e2e_4k|vit_e2e_6k|vit_e2e_8k)
    policy_repo=local/act-ours-vit-carrot-to-pot-40-color3-e2e-lr1e7-8k
    policy_revision=local
    policy_dir_name=act_ours_vit_carrot_to_pot_40_color3_e2e_lr1e7
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_e2e_lr1e7_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    backbone_checkpoint_override="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    backbone_source_root_override=/home/robotarm/TCC-core
    ;;
  rn50_layerwise_2k|rn50_layerwise_4k|rn50_layerwise_6k|rn50_layerwise_8k)
    policy_repo=local/act-official-rn50-carrot-to-pot-40-color-layerwise-8k
    policy_revision=local
    policy_dir_name=act_official_rn50_carrot_to_pot_40_color_layerwise
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_official_rn50_carrot_to_pot_40_color_layerwise_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    ;;
  rn50_late_2k|rn50_late_4k|rn50_late_6k|rn50_late_8k)
    policy_repo=local/act-official-rn50-carrot-to-pot-40-color-late-8k
    policy_revision=local
    policy_dir_name=act_official_rn50_carrot_to_pot_40_color_late
    policy_step="${model##*_}"
    policy_step="${policy_step%k}000"
    policy_path_override="$repo_root/outputs/train/act_official_rn50_carrot_to_pot_40_color_late_train40_8000steps/checkpoints/$(printf '%06d' "$policy_step")/pretrained_model"
    ;;
  *)
    echo "usage: $0 {$usage_models} [--execute|download] [n_action_steps] [c920_path]" >&2
    exit 2
    ;;
esac

if [[ -n "$backbone_checkpoint_override" ]]; then
  if [[ ! -f "$backbone_checkpoint_override" ]]; then
    echo "Backbone checkpoint is unavailable: $backbone_checkpoint_override" >&2
    exit 1
  fi
  if [[ ! -f "$backbone_source_root_override/xirl/models.py" ]]; then
    echo "Backbone source is unavailable: $backbone_source_root_override" >&2
    exit 1
  fi
  export BACKBONE_CHECKPOINT="$backbone_checkpoint_override"
  export BACKBONE_SOURCE_ROOT="$backbone_source_root_override"
fi

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

release_stale_c920_rollout() {
  local holder_pid holder_cmd holder_pgid current_pgid attempt
  local -a holder_pids=() holder_pgids=()

  command -v fuser >/dev/null 2>&1 || {
    echo "fuser is required to verify exclusive C920 access." >&2
    return 1
  }
  while read -r holder_pid; do
    [[ -n "$holder_pid" ]] && holder_pids+=("$holder_pid")
  done < <(fuser "$c920_device_node" 2>/dev/null | tr ' ' '\n')
  (( ${#holder_pids[@]} > 0 )) || return 0

  current_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  for holder_pid in "${holder_pids[@]}"; do
    holder_cmd="$(tr '\0' ' ' < "/proc/$holder_pid/cmdline" 2>/dev/null || true)"
    if [[ "$holder_cmd" != *"$repo_root/.venv/bin/lerobot-rollout"* ]]; then
      echo "C920 is held by a non-rollout process; refusing to stop it automatically." >&2
      fuser -v "$c920_device_node" >&2 || true
      return 1
    fi
    holder_pgid="$(ps -o pgid= -p "$holder_pid" | tr -d ' ')"
    if [[ ! "$holder_pgid" =~ ^[0-9]+$ || "$holder_pgid" == "$current_pgid" ]]; then
      echo "Could not safely isolate stale rollout PID $holder_pid (PGID $holder_pgid)." >&2
      return 1
    fi
    if [[ " ${holder_pgids[*]} " != *" $holder_pgid "* ]]; then
      holder_pgids+=("$holder_pgid")
    fi
  done

  echo "C920 is held by stale rollout PID(s): ${holder_pids[*]}"
  echo "Disconnecting stale rollout process group(s): ${holder_pgids[*]}"
  for holder_pgid in "${holder_pgids[@]}"; do
    # Stale rollouts are commonly job-control stopped. SIGKILL avoids briefly
    # resuming a queued robot action before the old process disconnects.
    kill -KILL -- "-$holder_pgid" 2>/dev/null || true
  done
  for ((attempt = 0; attempt < 30; attempt++)); do
    if ! fuser "$c920_device_node" >/dev/null 2>&1; then
      echo "C920 released; continuing with reset and reconnect."
      return 0
    fi
    sleep 0.1
  done

  echo "C920 did not release after stopping the stale rollout." >&2
  fuser -v "$c920_device_node" >&2 || true
  return 1
}

release_stale_act_rollouts() {
  local pid pgid command current_pgid
  local -a stale_pgids=()

  current_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  while read -r pid pgid command; do
    [[ -n "${pid:-}" && -n "${pgid:-}" ]] || continue
    [[ "$pgid" != "$current_pgid" ]] || continue
    if [[ "$command" == *"$repo_root/.venv/bin/lerobot-rollout"* || \
          "$command" == *"scripts/run_uploaded_act.sh"* ]]; then
      if [[ " ${stale_pgids[*]} " != *" $pgid "* ]]; then
        stale_pgids+=("$pgid")
      fi
    fi
  done < <(ps -u "$(id -u)" -o pid=,pgid=,args=)

  (( ${#stale_pgids[@]} > 0 )) || return 0
  echo "Disconnecting stale ACT rollout process group(s): ${stale_pgids[*]}"
  for pgid in "${stale_pgids[@]}"; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
}

hardware_reset_c920() {
  local attempt reset_device_node reset_output

  if [[ ! -x /usr/bin/usbreset ]]; then
    echo "C920 hardware reset requires /usr/bin/usbreset." >&2
    return 1
  fi

  echo "Hardware-resetting C920 USB device 046d:08e5..."
  reset_output="$(/usr/bin/usbreset 046d:08e5 2>&1 || true)"
  printf '%s\n' "$reset_output"
  if [[ "$reset_output" != *"Resetting HD Pro Webcam C920 ... ok"* ]]; then
    echo "C920 hardware reset permission is not installed." >&2
    echo "Install scripts/99-c920-usbreset.rules once; rollout will not request a password." >&2
    return 1
  fi

  # A USB reset can change /dev/videoN. Only the stable by-id path is reused.
  for ((attempt = 0; attempt < 100; attempt++)); do
    if [[ -e "$c920_path" ]]; then
      reset_device_node="$(readlink -f "$c920_path")"
      if [[ "$reset_device_node" =~ ^/dev/video[0-9]+$ && -e "$reset_device_node" ]]; then
        c920_device_node="$reset_device_node"
        echo "C920 re-enumerated at $c920_device_node."
        return 0
      fi
    fi
    sleep 0.1
  done

  echo "C920 did not re-enumerate after USB hardware reset." >&2
  return 1
}

finish_run() {
  exit_code=$?
  trap - EXIT
  {
    printf 'exit_code=%s\n' "$exit_code"
    printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$run_dir/status.txt"
  exit "$exit_code"
}

run_dir=""
if [[ "$mode" == --execute ]]; then
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  run_dir="$repo_root/output/${model}_${run_stamp}"
  mkdir -p "$run_dir"
  trap finish_run EXIT

  {
    printf 'model=%s\n' "$model"
    printf 'mode=%s\n' "$mode"
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
    printf 'policy_repo=%s\n' "$policy_repo"
    printf 'policy_revision=%s\n' "$policy_revision"
    printf 'policy_prefix=%s\n' "${policy_prefix:-repository_root}"
    printf 'policy_path_override=%s\n' "${policy_path_override:-none}"
    printf 'n_action_steps=%s\n' "${n_action_steps_override:-checkpoint_default}"
    printf 'c920_path=%s\n' "$c920_path"
    printf 'camera_mode=c920_only\n'
    printf 'camera_reset=usb_hardware_046d:08e5\n'
    printf 'output_dir=%s\n' "$run_dir"
    printf 'invocation='
    printf '%q ' "$0" "$@"
    printf '\n'
  } > "$run_dir/run_metadata.txt"

  exec > >(tee -a "$run_dir/console.log") 2>&1
  echo "All run information will be saved to: $run_dir"
fi

uv sync --extra act
if [[ -n "$policy_path_override" ]]; then
  policy_path="$policy_path_override"
  if [[ ! -f "$policy_path/model.safetensors" ]]; then
    echo "Local checkpoint is missing: $policy_path/model.safetensors" >&2
    exit 1
  fi
  echo "Using local checkpoint: $policy_path"
else
  policy_download_root="$repo_root/checkpoints/$policy_dir_name"
  if [[ -n "$policy_prefix" ]]; then
  policy_path="$policy_download_root/$policy_prefix"
  if [[ -f "$policy_path/model.safetensors" ]]; then
    echo "Using local checkpoint: $policy_path"
  else
    uv run --no-sync hf download "$policy_repo" \
      --revision "$policy_revision" \
      --include "$policy_prefix/*" \
      --local-dir "$policy_download_root"
  fi
  else
    policy_path="$policy_download_root"
    uv run --no-sync hf download "$policy_repo" \
      --revision "$policy_revision" \
      --local-dir "$policy_path"
  fi
fi

if [[ "$mode" == --execute ]]; then
  sha256sum "$policy_path/model.safetensors" > "$run_dir/weights.sha256"
else
  sha256sum "$policy_path/model.safetensors"
fi

if [[ -f "$policy_path/first_frame_report.json" ]]; then
  if [[ "$mode" == --execute ]]; then
    cp "$policy_path/first_frame_report.json" "$run_dir/first_frame_report.json"
  fi
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
c920_device_node="$(readlink -f "$c920_path")"
if [[ ! "$c920_device_node" =~ ^/dev/video[0-9]+$ || ! -e "$c920_device_node" ]]; then
  echo "C920 path did not resolve to an available /dev/videoN node: $c920_path -> $c920_device_node" >&2
  exit 1
fi
release_stale_act_rollouts
release_stale_c920_rollout
hardware_reset_c920
uv run --no-sync python scripts/lock_c920_focus.py --device "$c920_path"
robot_cameras="{cam_main: {type: opencv, index_or_path: \"$c920_path\", width: 640, height: 480, fps: 20, fourcc: MJPG}}"
rollout_command=(
  uv run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
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
