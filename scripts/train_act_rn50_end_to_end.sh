#!/usr/bin/env bash
# Official LeRobot ACT with the upstream ours RN50 fine-tuned end to end.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset_root="${ACT_DATASET_ROOT:-/home/paichichi/data/act-lite-review/carrot_100_v3}"
backbone_source_root="${BACKBONE_SOURCE_ROOT:-/home/paichichi/projects/TCC-core}"
backbone_checkpoint="${BACKBONE_CHECKPOINT:-$repo_root/assets/tcc-policy-assets/backbones/ours_rn50/checkpoint_040000.pt}"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/act_ours_rn50_end_to_end_carrot_100_50k}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ ! -d "$dataset_root/meta" || ! -d "$dataset_root/videos" ]]; then
  echo "ACT dataset is incomplete: $dataset_root" >&2
  exit 1
fi
if [[ ! -f "$backbone_checkpoint" ]]; then
  echo "Backbone checkpoint is missing: $backbone_checkpoint" >&2
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

episode_order="$(.venv/bin/python scripts/make_act_episode_order.py \
  --repo-id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --root="$dataset_root" \
  --seed=1000 | tail -n 1)"

export BACKBONE_SOURCE_ROOT="$backbone_source_root"
export BACKBONE_CHECKPOINT="$backbone_checkpoint"
export ACCELERATE_MIXED_PRECISION=bf16
if [[ -d "$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$(dirname "$output_dir")"
set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" .venv/bin/lerobot-train \
  --dataset.repo_id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --dataset.root="$dataset_root" \
  --dataset.episodes="$episode_order" \
  --dataset.eval_split=0.2 \
  --dataset.return_uint8=true \
  --dataset.video_backend=torchcodec \
  --policy.discover_packages_path=lerobot_policy_backbone_act \
  --policy.type=backbone_act \
  --policy.repo_id=Chipaipai/act-ours-rn50-end-to-end-carrot-100 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.backbone_checkpoint="$backbone_checkpoint" \
  --policy.backbone_source_root="$backbone_source_root" \
  --policy.backbone_family=ours_rn50 \
  --policy.freeze_vision_backbone=false \
  --policy.chunk_size=40 \
  --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet50 \
  --policy.dim_model=256 \
  --policy.n_heads=4 \
  --policy.dim_feedforward=1024 \
  --policy.n_encoder_layers=2 \
  --policy.n_decoder_layers=1 \
  --policy.n_vae_encoder_layers=2 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --steps=50000 \
  --eval_steps=2000 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir="$output_dir" \
  --job_name=act_ours_rn50_end_to_end_carrot_100_50k \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

.venv/bin/python scripts/select_best_act_checkpoint.py \
  "$output_dir" \
  --log-path="$train_log"
