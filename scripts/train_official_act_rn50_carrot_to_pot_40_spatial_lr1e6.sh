#!/usr/bin/env bash
# Train official LeRobot ACT RN50 with spatial-only augmentation and a 1e-6 backbone LR.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

steps="${ACT_STEPS:-8000}"
batch_size="${ACT_BATCH_SIZE:-16}"
save_freq="${ACT_SAVE_FREQ:-2000}"
video_backend="${ACT_VIDEO_BACKEND:-pyav}"
dataset_root="${ACT_DATASET_ROOT:-$repo_root/datasets/carrot_to_pot_40}"
source_config="$repo_root/checkpoints/act_official_rn50_carrot_to_pot_40/checkpoints/008000/train_config.json"
spatial_config="$repo_root/configs/act_rn50_spatial_image_transforms.json"
run_name="act_official_rn50_carrot_to_pot_40_spatial_lr1e6_train40_${steps}steps"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ ! -d "$dataset_root/meta" || ! -d "$dataset_root/videos" ]]; then
  echo "ACT dataset is incomplete: $dataset_root" >&2
  exit 1
fi
if [[ ! -f "$source_config" ]]; then
  echo "Original RN50 training config is missing: $source_config" >&2
  exit 1
fi
if [[ ! -f "$spatial_config" ]]; then
  echo "Spatial transform config is missing: $spatial_config" >&2
  exit 1
fi
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing ACT run: $output_dir" >&2
  exit 1
fi

# Reuse the exact original 40-episode order so augmentation and backbone LR are
# the only optimization changes relative to the official RN50 baseline.
episode_order="$(.venv/bin/python - "$source_config" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    episodes = json.load(f)["dataset"]["episodes"]
if len(episodes) != 40 or sorted(episodes) != list(range(40)):
    raise SystemExit(f"Unexpected source episode order: {episodes}")
print(json.dumps(episodes, separators=(",", ":")))
PY
)"
image_transforms="$(<"$spatial_config")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$repo_root/.cache/huggingface/datasets}"
if [[ -d "$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$(dirname "$output_dir")"
set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" .venv/bin/lerobot-train \
  --dataset.repo_id=UoA-Trossen-Arm/carrot_to_pot_40 \
  --dataset.root="$dataset_root" \
  --dataset.episodes="$episode_order" \
  --dataset.eval_split=0.0 \
  --dataset.return_uint8=true \
  --dataset.video_backend="$video_backend" \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=1 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$image_transforms" \
  --policy.type=act \
  --policy.repo_id=Chipaipai/act-official-rn50-carrot-to-pot-40-spatial-lr1e6-8k \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=40 \
  --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet50 \
  --policy.pretrained_backbone_weights=ResNet50_Weights.IMAGENET1K_V1 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-6 \
  --policy.optimizer_weight_decay=1e-4 \
  --batch_size="$batch_size" \
  --num_workers=8 \
  --prefetch_factor=4 \
  --steps="$steps" \
  --eval_steps=0 \
  --save_freq="$save_freq" \
  --log_freq=100 \
  --output_dir="$output_dir" \
  --job_name="$run_name" \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

printf 'Training complete. Final checkpoint: %s\n' \
  "$output_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
