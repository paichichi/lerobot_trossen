#!/usr/bin/env bash
# Train one frozen-ViT ACT policy.

set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 {open_lid|push_pot|press_button} {ours_vit|hrp_imagenet|d4r_imagenet}" >&2
  exit 2
fi

task="$1"
backbone="$2"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$task" in
  open_lid)
    dataset_name=open_the_pot_by_lifting_the_lid_40
    dataset_root="$repo_root/datasets/open_the_pot_by_lifting_the_lid/dataset"
    ;;
  push_pot)
    dataset_name=push_the_pot_into_the_marked_area_40
    dataset_root="$repo_root/datasets/push_the_pot_into_the_marked_area/dataset"
    ;;
  press_button)
    dataset_name=press_the_button_40
    dataset_root="$repo_root/datasets/press_the_button/dataset"
    ;;
  *) echo "unknown task: $task" >&2; exit 2 ;;
esac

case "$backbone" in
  ours_vit)
    backbone_family=ours_vit
    backbone_file="$repo_root/assets/tcc-policy-assets/backbones/ours_vit/checkpoint_040000.pt"
    expected_hash=1326a13fffae82c6e4057ff6c3d98ae2066dea9daa651eb32fa75691f3091955
    ;;
  hrp_imagenet)
    backbone_family=pretrained_vit
    backbone_file="$repo_root/assets/tcc-policy-assets/backbones/hrp_imagenet/HRP_IN.pth"
    expected_hash=6be28e1343dc66b5ca8441ecf42b16fbf829692caab4107969fb3d70eae5a874
    ;;
  d4r_imagenet)
    backbone_family=pretrained_vit
    backbone_file="$repo_root/assets/tcc-policy-assets/backbones/d4r_imagenet/D4R_IN_1M.pth"
    expected_hash=194b920f81d419c2d51f9ff133b712c0e17a8d54101ff854fcb40deacca40a80
    ;;
  *) echo "unknown backbone: $backbone" >&2; exit 2 ;;
esac

steps=8000
run_name="act_${backbone}_frozen_${task}_color3_train40_8000steps"
output_dir="$repo_root/outputs/train/$run_name"
train_log="$output_dir.train.log"
transforms="$repo_root/configs/image_transforms.json"
backbone_source="${BACKBONE_SOURCE_ROOT:-/home/robotarm/TCC-core}"

for required in "$dataset_root/meta/info.json" "$backbone_file" "$transforms" "$backbone_source/xirl/models.py"; do
  test -f "$required" || { echo "missing required file: $required" >&2; exit 1; }
done
actual_hash="$(sha256sum "$backbone_file" | cut -d' ' -f1)"
test "$actual_hash" = "$expected_hash" || { echo "backbone hash mismatch" >&2; exit 1; }
if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "output already exists: $output_dir" >&2
  exit 1
fi

export BACKBONE_SOURCE_ROOT="$backbone_source"
export BACKBONE_CHECKPOINT="$backbone_file"
export ACCELERATE_MIXED_PRECISION=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export PYTHONPATH="$repo_root/packages/lerobot_policy_backbone_act/src${PYTHONPATH:+:$PYTHONPATH}"
image_transforms="$(<"$transforms")"
episodes='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]'

mkdir -p "$(dirname "$output_dir")"
.venv/bin/lerobot-train \
  --dataset.repo_id="local/$dataset_name" \
  --dataset.root="$dataset_root" \
  --dataset.episodes="$episodes" \
  --dataset.eval_split=0.0 \
  --dataset.return_uint8=true \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=false \
  --dataset.image_transforms.tfs="$image_transforms" \
  --policy.discover_packages_path=lerobot_policy_backbone_act \
  --policy.type=backbone_act \
  --policy.repo_id="local/act-${backbone}-frozen-${task}-color3-8k" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.backbone_checkpoint="$backbone_file" \
  --policy.backbone_source_root="$backbone_source" \
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
  --batch_size=8 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --steps="$steps" \
  --eval_steps=0 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir="$output_dir" \
  --job_name="$run_name" \
  --wandb.enable=false 2>&1 | tee "$train_log"
