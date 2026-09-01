#!/usr/bin/env bash
# Train the 3 tasks x 3 backbones sequentially; skip completed 8K policies.

set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

for task in open_lid push_pot press_button; do
  for backbone in ours_vit hrp_imagenet d4r_imagenet; do
    model="outputs/train/act_${backbone}_frozen_${task}_color3_train40_8000steps/checkpoints/008000/pretrained_model/model.safetensors"
    if [[ -f "$model" ]]; then
      echo "SKIP completed: $task $backbone"
    else
      bash scripts/train_policy.sh "$task" "$backbone"
    fi
  done
done
