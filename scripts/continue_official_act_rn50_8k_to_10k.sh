#!/usr/bin/env bash
# Exactly resume the official RN50 ACT run at step 8k and continue to 10k.
# The Hub checkpoint includes model, optimizer, RNG, and training-step state.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_repo="${ACT_SOURCE_REPO:-Chipaipai/act-official-rn50-carrot-to-pot-40-train40-8k}"
dataset_root="${ACT_DATASET_ROOT:-$repo_root/datasets/carrot_to_pot_40}"
target_steps="${ACT_TARGET_STEPS:-10000}"
save_freq="${ACT_SAVE_FREQ:-1000}"
run_name="act_official_rn50_carrot_to_pot_40_resume_8k_to_10k"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"
local_policy_root="${ACT_LOCAL_POLICY_ROOT:-$repo_root/checkpoints/act_official_rn50_carrot_to_pot_40}"

if [[ ! -f "$dataset_root/meta/info.json" ]]; then
  echo "Missing local carrot dataset: $dataset_root" >&2
  exit 1
fi
if [[ "$target_steps" != 10000 || "$save_freq" != 1000 ]]; then
  echo "This comparison run requires ACT_TARGET_STEPS=10000 and ACT_SAVE_FREQ=1000." >&2
  exit 2
fi
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing continuation run: $output_dir" >&2
  exit 1
fi
if [[ -e "$local_policy_root/checkpoints/009000" || -e "$local_policy_root/checkpoints/010000" ]]; then
  echo "Refusing to overwrite existing logical 9k/10k checkpoints." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -d "$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$(dirname "$output_dir")" "$HF_DATASETS_CACHE"
set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" .venv/bin/lerobot-train \
  --config_path="$source_repo" \
  --resume=true \
  --dataset.root="$dataset_root" \
  --output_dir="$output_dir" \
  --steps="$target_steps" \
  --save_freq="$save_freq" \
  --log_freq=100 \
  --policy.push_to_hub=false \
  --save_checkpoint_to_hub=false \
  --job_name="$run_name" \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

checkpoint_9k="$output_dir/checkpoints/009000/pretrained_model"
checkpoint_10k="$output_dir/checkpoints/010000/pretrained_model"
if [[ ! -f "$checkpoint_9k/model.safetensors" || ! -f "$checkpoint_10k/model.safetensors" ]]; then
  echo "Continuation finished without both expected checkpoints." >&2
  exit 1
fi

mkdir -p "$local_policy_root/checkpoints"
ln -s "$checkpoint_9k" "$local_policy_root/checkpoints/009000"
ln -s "$checkpoint_10k" "$local_policy_root/checkpoints/010000"

sha256sum \
  "$checkpoint_9k/model.safetensors" \
  "$checkpoint_10k/model.safetensors"
printf 'RN50 exact continuation complete.\n9k: %s\n10k: %s\n' \
  "$local_policy_root/checkpoints/009000" \
  "$local_policy_root/checkpoints/010000"
