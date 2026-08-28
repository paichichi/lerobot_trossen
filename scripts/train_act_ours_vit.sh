#!/usr/bin/env bash
# Train official LeRobot ACT on Trossen data with the TCC ours_vit patch tokens.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset_root="${ACT_DATASET_ROOT:-/home/paichichi/data/act-lite-review/carrot_100_v3}"
backbone_source_root="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"
steps="${ACT_STEPS:-8000}"
batch_size="${ACT_BATCH_SIZE:-8}"
eval_steps="${ACT_EVAL_STEPS:-0}"
save_freq="${ACT_SAVE_FREQ:-1000}"
eval_split="${ACT_EVAL_SPLIT:-0.0}"
run_name="act_ours_vit_carrot_100_${steps}steps"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ ! -d "$dataset_root/meta" || ! -d "$dataset_root/videos" ]]; then
  echo "ACT dataset is incomplete: $dataset_root" >&2
  exit 1
fi
if [[ ! -f "$backbone_source_root/xirl/models.py" ]]; then
  echo "TCC backbone source is missing: $backbone_source_root" >&2
  exit 1
fi
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing ACT run: $output_dir" >&2
  exit 1
fi

if [[ -n "${BACKBONE_CHECKPOINT:-}" ]]; then
  backbone_checkpoint="$BACKBONE_CHECKPOINT"
else
  backbone_checkpoint="$(.venv/bin/python -c \
    "from huggingface_hub import hf_hub_download; print(hf_hub_download('Chipaipai/tcc-core-real-robot-policies', 'backbones/ours_vit/checkpoint_040000.pt'))")"
fi
if [[ ! -f "$backbone_checkpoint" ]]; then
  echo "Backbone checkpoint is missing: $backbone_checkpoint" >&2
  exit 1
fi

episode_order="$(.venv/bin/python scripts/make_act_episode_order.py \
  --repo-id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --root="$dataset_root" \
  --eval-split="$eval_split" \
  --seed=1000 | tail -n 1)"
image_transforms="$(<"$repo_root/configs/act_rn50_full_image_transforms.json")"

export BACKBONE_SOURCE_ROOT="$backbone_source_root"
export BACKBONE_CHECKPOINT="$backbone_checkpoint"
export ACCELERATE_MIXED_PRECISION=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$repo_root/packages/lerobot_policy_backbone_act/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -d "$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$(dirname "$output_dir")"
set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" .venv/bin/lerobot-train \
  --dataset.repo_id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --dataset.root="$dataset_root" \
  --dataset.episodes="$episode_order" \
  --dataset.eval_split="$eval_split" \
  --dataset.return_uint8=true \
  --dataset.video_backend=torchcodec \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$image_transforms" \
  --policy.discover_packages_path=lerobot_policy_backbone_act \
  --policy.type=backbone_act \
  --policy.repo_id=Chipaipai/act-ours-vit-carrot-100 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.backbone_checkpoint="$backbone_checkpoint" \
  --policy.backbone_source_root="$backbone_source_root" \
  --policy.backbone_family=ours_vit \
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
  --eval_steps="$eval_steps" \
  --save_freq="$save_freq" \
  --log_freq=100 \
  --output_dir="$output_dir" \
  --job_name="$run_name" \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

printf 'Training complete. Final checkpoint: %s\n' \
  "$output_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
