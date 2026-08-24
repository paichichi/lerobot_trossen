#!/usr/bin/env bash
# Queue the official ACT ResNet18 control after an existing GPU run finishes.

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

wait_pid="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 20
done

ours_output="$repo_root/outputs/train/act_lite_ours_rn50_carrot_100_50k"
ours_log="$repo_root/outputs/train/act_lite_ours_rn50_carrot_100_50k.train.log"
if [[ ! -d "$ours_output/checkpoints/050000" ]]; then
  echo "ours_rn50 did not produce its final checkpoint" >&2
  exit 1
fi
"$repo_root/.venv/bin/python" scripts/select_best_act_checkpoint.py \
  "$ours_output" \
  --log-path="$ours_log"

export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ACCELERATE_MIXED_PRECISION=bf16

dataset_root="/home/paichichi/data/act-lite-review/carrot_100_v3"
episode_order="$("$repo_root/.venv/bin/python" scripts/make_act_episode_order.py \
  --repo-id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --root="$dataset_root" \
  --seed=1000 | tail -n 1)"
output_dir="$repo_root/outputs/train/act_official_rn18_carrot_100_50k"
train_log="$repo_root/outputs/train/act_official_rn18_carrot_100_50k.train.log"

if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output: $output_dir" >&2
  exit 1
fi

set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" "$repo_root/.venv/bin/lerobot-train" \
  --dataset.repo_id=UoA-Trossen-Arm/pick_and_place_carrot_100 \
  --dataset.root="$dataset_root" \
  --dataset.episodes="$episode_order" \
  --dataset.eval_split=0.2 \
  --dataset.return_uint8=true \
  --dataset.video_backend=torchcodec \
  --policy.type=act \
  --policy.repo_id=Chipaipai/act-official-rn18-carrot-100 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=40 \
  --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet18 \
  --batch_size=16 \
  --num_workers=8 \
  --prefetch_factor=4 \
  --steps=50000 \
  --eval_steps=2000 \
  --save_freq=2000 \
  --log_freq=100 \
  --output_dir="$output_dir" \
  --job_name=act_official_rn18_carrot_100_50k \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

"$repo_root/.venv/bin/python" scripts/select_best_act_checkpoint.py \
  "$output_dir" \
  --log-path="$train_log"
