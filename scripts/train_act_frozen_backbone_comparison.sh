#!/usr/bin/env bash
# Train a fresh official ACT with one strictly frozen visual backbone.

set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 {ours_vit|ours_rn50|d4r_imagenet|hrp_imagenet}" >&2
  exit 2
fi
variant="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

steps="${ACT_STEPS:-8000}"
batch_size="${ACT_BATCH_SIZE:-8}"
save_freq="${ACT_SAVE_FREQ:-2000}"
allow_nonstandard="${ACT_ALLOW_NONSTANDARD:-false}"
video_backend="${ACT_VIDEO_BACKEND:-pyav}"
dataset_root="${ACT_DATASET_ROOT:-$repo_root/datasets/carrot_to_pot_40}"
source_config="$repo_root/checkpoints/act_official_rn50_carrot_to_pot_40/checkpoints/008000/train_config.json"
color_config="$repo_root/configs/act_rn50_full_image_transforms.json"
backbone_source_root="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"

case "$variant" in
  ours_vit)
    backbone_family=ours_vit
    backbone_checkpoint="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    expected_sha256=1326a13fffae82c6e4057ff6c3d98ae2066dea9daa651eb32fa75691f3091955
    ;;
  ours_rn50)
    backbone_family=ours_rn50
    backbone_checkpoint="$repo_root/assets/tcc-policy-assets/backbones/ours_rn50/checkpoint_040000.pt"
    expected_sha256=708926a8e5759c2b9cf4308996532f0b691d7ce646fb42b27a6bdb16db1b142e
    ;;
  d4r_imagenet)
    backbone_family=pretrained_vit
    backbone_checkpoint="$repo_root/assets/tcc-policy-assets/backbones/d4r_imagenet/D4R_IN_1M.pth"
    expected_sha256=194b920f81d419c2d51f9ff133b712c0e17a8d54101ff854fcb40deacca40a80
    ;;
  hrp_imagenet)
    backbone_family=pretrained_vit
    backbone_checkpoint="$repo_root/assets/tcc-policy-assets/backbones/hrp_imagenet/HRP_IN.pth"
    expected_sha256=6be28e1343dc66b5ca8441ecf42b16fbf829692caab4107969fb3d70eae5a874
    ;;
  *)
    echo "usage: $0 {ours_vit|ours_rn50|d4r_imagenet|hrp_imagenet}" >&2
    exit 2
    ;;
esac

run_name="act_${variant}_frozen_carrot_to_pot_40_color3_train40_${steps}steps"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ "$allow_nonstandard" != true && ( "$steps" != 8000 || "$save_freq" != 2000 ) ]]; then
  echo "This controlled comparison requires ACT_STEPS=8000 and ACT_SAVE_FREQ=2000." >&2
  exit 2
fi
if [[ ! -d "$dataset_root/meta" || ! -d "$dataset_root/videos" ]]; then
  echo "ACT dataset is incomplete: $dataset_root" >&2
  exit 1
fi
for required_file in "$source_config" "$color_config" "$backbone_checkpoint"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required comparison input is missing: $required_file" >&2
    exit 1
  fi
done
if [[ ! -f "$backbone_source_root/xirl/models.py" ]]; then
  echo "TCC backbone source is missing: $backbone_source_root" >&2
  exit 1
fi
actual_sha256="$(sha256sum "$backbone_checkpoint" | cut -d' ' -f1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Backbone hash mismatch for $variant: expected $expected_sha256, got $actual_sha256" >&2
  exit 1
fi
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing ACT run: $output_dir" >&2
  exit 1
fi

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
export ACCELERATE_MIXED_PRECISION=bf16
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
  --policy.discover_packages_path=lerobot_policy_backbone_act \
  --policy.type=backbone_act \
  --policy.repo_id="Chipaipai/act-${variant}-frozen-carrot-to-pot-40-color3-8k" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.backbone_checkpoint="$backbone_checkpoint" \
  --policy.backbone_source_root="$backbone_source_root" \
  --policy.backbone_family="$backbone_family" \
  --policy.freeze_vision_backbone=true \
  --policy.backbone_image_size=224 \
  --policy.chunk_size=40 \
  --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet50 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.n_vae_encoder_layers=4 \
  --policy.dropout=0.1 \
  --policy.optimizer_lr=1e-4 \
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

if [[ "$steps" == 8000 && "$save_freq" == 2000 ]]; then
  for step in 2000 4000 6000 8000; do
    checkpoint="$output_dir/checkpoints/$(printf '%06d' "$step")/pretrained_model/model.safetensors"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Training finished without expected checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
fi

printf '%s frozen-backbone ACT training complete: %s\n' \
  "$variant" "$output_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
