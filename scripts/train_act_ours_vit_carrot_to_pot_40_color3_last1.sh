#!/usr/bin/env bash
# Adapt a frozen ViT ACT checkpoint by tuning only ViT block 11 + final LayerNorm.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

steps="${ACT_STEPS:-1000}"
batch_size="${ACT_BATCH_SIZE:-8}"
save_freq="${ACT_SAVE_FREQ:-250}"
allow_nonstandard="${ACT_ALLOW_NONSTANDARD:-false}"
video_backend="${ACT_VIDEO_BACKEND:-pyav}"
dataset_root="${ACT_DATASET_ROOT:-$repo_root/datasets/carrot_to_pot_40}"
frozen_policy="${ACT_FROZEN_POLICY:-$repo_root/outputs/train/act_ours_vit_carrot_to_pot_40_color3_frozen_train40_8000steps/checkpoints/008000/pretrained_model}"
source_config="$frozen_policy/train_config.json"
color_config="$repo_root/configs/act_rn50_full_image_transforms.json"
backbone_source_root="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"
backbone_checkpoint="${BACKBONE_CHECKPOINT:-$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt}"
run_name="act_ours_vit_carrot_to_pot_40_color3_last1_lr1e7_from_frozen8k_${steps}steps"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ "$allow_nonstandard" != true && ( "$steps" != 1000 || "$save_freq" != 250 ) ]]; then
  echo "This controlled comparison requires ACT_STEPS=1000 and ACT_SAVE_FREQ=250." >&2
  exit 2
fi
if [[ ! -d "$dataset_root/meta" || ! -d "$dataset_root/videos" ]]; then
  echo "ACT dataset is incomplete: $dataset_root" >&2
  exit 1
fi
if [[ ! -f "$frozen_policy/model.safetensors" || ! -f "$source_config" ]]; then
  echo "Frozen ViT 8k policy is incomplete: $frozen_policy" >&2
  exit 1
fi
if [[ ! -f "$color_config" ]]; then
  echo "Color transform config is missing: $color_config" >&2
  exit 1
fi
if [[ ! -f "$backbone_source_root/xirl/models.py" ]]; then
  echo "TCC backbone source is missing: $backbone_source_root" >&2
  exit 1
fi
if [[ ! -f "$backbone_checkpoint" ]]; then
  echo "ViT backbone checkpoint is missing: $backbone_checkpoint" >&2
  exit 1
fi
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing ACT run: $output_dir" >&2
  exit 1
fi

# Reuse the exact episode order and photometric augmentation of the frozen run.
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
image_transforms="$(<"$color_config")"

export BACKBONE_SOURCE_ROOT="$backbone_source_root"
export BACKBONE_CHECKPOINT="$backbone_checkpoint"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$repo_root/.cache/huggingface/datasets}"
export PYTHONPATH="$repo_root/packages/lerobot_policy_backbone_act/src${PYTHONPATH:+:$PYTHONPATH}"
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
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$image_transforms" \
  --policy.path="$frozen_policy" \
  --policy.repo_id=Chipaipai/act-ours-vit-carrot-to-pot-40-color3-last1-lr1e7-1k \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.backbone_checkpoint="$backbone_checkpoint" \
  --policy.backbone_source_root="$backbone_source_root" \
  --policy.freeze_vision_backbone=false \
  --policy.vit_trainable_last_blocks=1 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-7 \
  --policy.optimizer_weight_decay=1e-4 \
  --batch_size="$batch_size" \
  --num_workers=8 \
  --prefetch_factor=4 \
  --steps="$steps" \
  --eval_steps=0 \
  --save_freq="$save_freq" \
  --log_freq=50 \
  --output_dir="$output_dir" \
  --job_name="$run_name" \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

if [[ "$steps" == 1000 && "$save_freq" == 250 ]]; then
  for step in 250 500 750 1000; do
    checkpoint="$output_dir/checkpoints/$(printf '%06d' "$step")/pretrained_model/model.safetensors"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Training finished without expected checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
fi

printf 'Selective ViT tuning complete. Final checkpoint: %s\n' \
  "$output_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
