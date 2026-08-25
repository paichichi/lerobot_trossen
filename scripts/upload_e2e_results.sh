#!/usr/bin/env bash
# Upload validation-selected V11 and ACT end-to-end RN50 artifacts.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="${E2E_WORKSPACE_ROOT:-/home/paichichi/work/v11_act_e2e_20260825}"
v11_dir="${V11_OUTPUT_DIR:-$workspace_root/outputs/v11_end_to_end_rn50_100k}"
act_dir="${ACT_OUTPUT_DIR:-$repo_root/outputs/train/act_ours_rn50_end_to_end_carrot_100_50k}"
python_bin="$repo_root/.venv/bin/python"
hf_bin="$repo_root/.venv/bin/hf"
result_file="${E2E_UPLOAD_RESULT:-$workspace_root/hf_upload_result.txt}"

v11_checkpoint="$v11_dir/checkpoint_100000.pt"
v11_metrics="$v11_dir/metrics.json"
act_best_file="$act_dir/best_checkpoint.txt"
test -f "$v11_checkpoint"
test -f "$v11_metrics"
test -f "$act_best_file"
act_best="$(< "$act_best_file")"
test -d "$act_best"

export BACKBONE_SOURCE_ROOT="${BACKBONE_SOURCE_ROOT:-/home/paichichi/projects/TCC-core}"
export BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-$repo_root/assets/tcc-policy-assets/backbones/ours_rn50/checkpoint_040000.pt}"

"$python_bin" -c '
import sys
from pathlib import Path
import torch

source = Path(sys.argv[1])
output = Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
torch.save(
    {
        "backbone_model": checkpoint["backbone_model"],
        "backbone_metadata": checkpoint.get("backbone_metadata"),
        "policy_step": checkpoint.get("step"),
    },
    output,
)
' "$v11_checkpoint" "$v11_dir/backbone_finetuned_best.pt"

"$python_bin" scripts/eval_act_first_frame.py \
  "$act_best" \
  --device=cuda \
  --output="$act_best/first_frame_report.json" \
  > "$act_dir/first_frame_eval.log"

"$hf_bin" upload Chipaipai/tcc-core-real-robot-policies \
  "$v11_checkpoint" \
  policies_v11_end_to_end_rn50/ours_rn50/checkpoint_100000.pt \
  --commit-message "Upload V11 end-to-end RN50 policy"
"$hf_bin" upload Chipaipai/tcc-core-real-robot-policies \
  "$v11_metrics" \
  policies_v11_end_to_end_rn50/ours_rn50/metrics.json \
  --commit-message "Upload V11 end-to-end metrics"
"$hf_bin" upload Chipaipai/tcc-core-real-robot-policies \
  "$v11_dir/backbone_finetuned_best.pt" \
  policies_v11_end_to_end_rn50/ours_rn50/backbone_finetuned_best.pt \
  --commit-message "Upload V11 fine-tuned RN50"
"$hf_bin" upload Chipaipai/act-ours-rn50-end-to-end-carrot-100 \
  "$act_best" . \
  --no-private \
  --commit-message "Upload validation-selected end-to-end ACT RN50"

v11_revision="$("$python_bin" -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("Chipaipai/tcc-core-real-robot-policies").sha)')"
act_revision="$("$python_bin" -c \
  'from huggingface_hub import HfApi; print(HfApi().repo_info("Chipaipai/act-ours-rn50-end-to-end-carrot-100").sha)')"
{
  printf 'v11_revision=%s\n' "$v11_revision"
  printf 'act_revision=%s\n' "$act_revision"
  printf 'v11_checkpoint=%s\n' "$v11_checkpoint"
  printf 'act_checkpoint=%s\n' "$act_best"
} > "$result_file"
cat "$result_file"
