#!/usr/bin/env bash
# Compatibility entry point. New runs use ACT-RN50-full.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_act_rn50_full.sh" "$@"
