#!/usr/bin/env bash
# Resume the complete new-camera ACT 8K checkpoint from Hugging Face.

set -euo pipefail

backbone="${1:-}"
total_steps="${2:-10000}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

case "$backbone" in
  rn18)
    policy_repo=Chipaipai/act-official-rn18-carrot-to-pot-40-train40-8k
    ;;
  rn50)
    policy_repo=Chipaipai/act-official-rn50-carrot-to-pot-40-train40-8k
    ;;
  *)
    echo "usage: $0 {rn18|rn50} [total_steps]" >&2
    exit 2
    ;;
esac

if [[ ! "$total_steps" =~ ^[0-9]+$ ]] || (( total_steps <= 8000 )); then
  echo "total_steps must be an integer greater than the saved step 8000." >&2
  exit 2
fi

run_name="act_official_${backbone}_carrot_to_pot_40_train40_${total_steps}steps_resume_from_8k"
output_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/$run_name}"
train_log="${ACT_TRAIN_LOG:-$output_dir.train.log}"

if [[ -e "$output_dir" || -e "$train_log" ]]; then
  echo "Refusing to overwrite an existing resumed run: $output_dir" >&2
  exit 1
fi

uv sync --extra act
dataset_root="$(.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("UoA-Trossen-Arm/carrot_to_pot_40", repo_type="dataset"))
PY
)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -d "$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="$repo_root/.local/ffmpeg6/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$(dirname "$output_dir")"
set -o pipefail
/usr/bin/time -f "WALL=%e RSS_KB=%M" .venv/bin/lerobot-train \
  --config_path="$policy_repo" \
  --resume=true \
  --steps="$total_steps" \
  --output_dir="$output_dir" \
  --dataset.root="$dataset_root" \
  --policy.device=cuda \
  --save_freq=2000 \
  --log_freq=100 \
  --wandb.enable=false \
  2>&1 | tee "$train_log"

printf 'Resume complete. Final checkpoint: %s\n' \
  "$output_dir/checkpoints/$(printf '%06d' "$total_steps")/pretrained_model"
