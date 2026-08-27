#!/usr/bin/env bash
# Train the official LeRobot ACT ResNet-18 policy on all 40 oblique-view demos.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

steps="${ACT_STEPS:-8000}"
batch_size="${ACT_BATCH_SIZE:-16}"
save_freq="${ACT_SAVE_FREQ:-2000}"
video_backend="${ACT_VIDEO_BACKEND:-pyav}"
run_name="act_official_rn18_carrot_to_pot_40_train40_${steps}steps"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing ACT run: $output_dir" >&2
  exit 1
fi

dataset_root="$(.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("UoA-Trossen-Arm/carrot_to_pot_40", repo_type="dataset"))
PY
)"

episode_order="$(.venv/bin/python scripts/make_act_episode_order.py \
  --repo-id=UoA-Trossen-Arm/carrot_to_pot_40 \
  --root="$dataset_root" \
  --eval-split=0.0 \
  --seed=1000 | tail -n 1)"

export ACCELERATE_MIXED_PRECISION=bf16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
  --dataset.image_transforms.enable=false \
  --policy.type=act \
  --policy.repo_id=Chipaipai/act-official-rn18-carrot-to-pot-40-train40-8k \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=40 \
  --policy.n_action_steps=10 \
  --policy.vision_backbone=resnet18 \
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
