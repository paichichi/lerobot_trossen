#!/usr/bin/env bash
set -euo pipefail

mode="${1:-download}"
if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${TCC_REAL_ROBOT_SOURCE_ROOT:-}" ]]; then
  tcc_real_robot_root="$TCC_REAL_ROBOT_SOURCE_ROOT"
elif [[ -d "$repo_root/../tcc-core-real-robot" ]]; then
  tcc_real_robot_root="$(cd "$repo_root/../tcc-core-real-robot" && pwd)"
else
  tcc_real_robot_root="${HOME}/projects/tcc-core-real-robot"
fi

if [[ -n "${BACKBONE_SOURCE_ROOT:-}" ]]; then
  backbone_source_root="$BACKBONE_SOURCE_ROOT"
elif [[ -d "$repo_root/../TCC-core" ]]; then
  backbone_source_root="$(cd "$repo_root/../TCC-core" && pwd)"
else
  backbone_source_root="${HOME}/projects/TCC-core"
fi
policy_revision=56690ddea1023ebe840c2d0dd1a07cfad67377b0
policy_file=policies_v11_end_to_end_rn50/ours_rn50/checkpoint_100000.pt
backbone_file=backbones/ours_rn50/checkpoint_040000.pt
asset_root="$repo_root/assets/tcc-policy-assets"
raw_policy="$asset_root/$policy_file"
backbone_checkpoint="$asset_root/$backbone_file"
policy_path="$repo_root/checkpoints/v11_end_to_end_ours_rn50"

text_log="$repo_root/output/v11_latest.txt"
mkdir -p "$repo_root/output"

finish_run() {
  exit_code=$?
  trap - EXIT
  printf 'Run finished with exit code %s at %s\n' \
    "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$exit_code"
}
trap finish_run EXIT

exec > >(tee "$text_log") 2>&1
echo "V11 output log: $text_log"

echo "[1/5] Checking the locked official LeRobot environment"
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "${HOME}/.local/bin/uv" ]]; then
  uv_bin="${HOME}/.local/bin/uv"
else
  echo "uv was not found on PATH or at ${HOME}/.local/bin/uv" >&2
  exit 1
fi
echo "Using uv: $uv_bin"

# LeRobot's Linux core-scripts dependency includes evdev, which is built from
# source for CPython 3.12. Keep its compiler and Python headers in a small,
# user-owned toolchain when the robot OS does not provide build-essential.
if [[ "$(uname -s)" == Linux ]] && ! command -v cc >/dev/null 2>&1; then
  toolchain_root="${V11_BUILD_TOOLCHAIN_ROOT:-${HOME}/.local/share/v11-build-toolchain}"
  toolchain_cc="$toolchain_root/bin/x86_64-conda-linux-gnu-cc"
  if [[ ! -x "$toolchain_cc" || ! -f "$toolchain_root/include/python3.12/Python.h" ]]; then
    if command -v conda >/dev/null 2>&1; then
      conda_bin="$(command -v conda)"
    elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
      conda_bin="${HOME}/miniconda3/bin/conda"
    else
      echo "A C compiler and Python 3.12 headers are required to build evdev." >&2
      echo "Install build-essential/python3.12-dev or set V11_BUILD_TOOLCHAIN_ROOT." >&2
      exit 1
    fi
    echo "Preparing the user-owned V11 build toolchain: $toolchain_root"
    "$conda_bin" create -y -p "$toolchain_root" --override-channels \
      -c conda-forge gcc_linux-64 python=3.12
  fi
  export CC="$toolchain_cc"
  export CXX="$toolchain_root/bin/x86_64-conda-linux-gnu-c++"
  export CFLAGS="${CFLAGS:+$CFLAGS }-I$toolchain_root/include/python3.12"
  echo "Using user-owned C compiler: $CC"
fi
"$uv_bin" sync --locked --extra act

echo "[2/5] Downloading the pinned V11 policy and backbone"
"$uv_bin" run --no-sync hf download Chipaipai/tcc-core-real-robot-policies \
  "$policy_file" "$backbone_file" \
  --revision "$policy_revision" \
  --local-dir "$asset_root"

export TCC_REAL_ROBOT_SOURCE_ROOT="$tcc_real_robot_root"
export BACKBONE_SOURCE_ROOT="$backbone_source_root"
export BACKBONE_CHECKPOINT="$backbone_checkpoint"

if [[ ! -f "$tcc_real_robot_root/src/tcc_real_robot/policy.py" ]]; then
  echo "TCC real-robot source missing: $tcc_real_robot_root" >&2
  exit 1
fi
if [[ ! -f "$backbone_source_root/xirl/models.py" ]]; then
  echo "TCC backbone source missing: $backbone_source_root" >&2
  exit 1
fi

echo "[3/5] Converting V11 to the LeRobot policy contract"
"$uv_bin" run --no-sync python -m lerobot_policy_v11.export_v11 \
  --checkpoint "$raw_policy" \
  --output-dir "$policy_path" \
  --backbone-checkpoint "$backbone_checkpoint" \
  --backbone-source-root "$backbone_source_root" \
  --tcc-real-robot-source-root "$tcc_real_robot_root"

if [[ "$mode" != --execute ]]; then
  echo "[4/5] Export self-check passed"
  echo "Converted and validated only. The robot was not connected or moved."
  exit 0
fi

echo "[4/5] Export self-check passed"
echo "[5/5] Starting hardware rollout"
"$uv_bin" run --no-sync python -c \
  'import sys, torch; available = torch.cuda.is_available(); print("CUDA device:", torch.cuda.get_device_name(0) if available else "unavailable"); sys.exit(0 if available else 1)'
echo "Starting official LeRobot base rollout for V11. Keep the E-stop ready."
echo "This policy has no task-completion output; press Ctrl+C after success."
record_command=(
  "$uv_bin" run --no-sync lerobot-rollout
  --robot.discover_packages_path=lerobot_robot_trossen
  --robot.type=widowxai_follower_robot
  --robot.ip_address=192.168.1.4
  --robot.id=follower
  --robot.loop_rate=20
  --robot.min_time_to_move_multiplier=2.0
  --robot.startup_home_positions='[-0.00019073777366429567, 1.046196699142456, 0.5239566564559937, 0.6292439103126526, -0.000572213320992887, 0.00019073777366429567, 0.0]'
  --robot.startup_home_goal_time_s=10.0
  --robot.startup_home_settle_time_s=1.0
  --robot.startup_home_max_arm_error_rad=0.03
  --robot.startup_home_max_gripper_error_m=0.002
  --robot.controller_connect_attempts=3
  --robot.controller_connect_retry_delay_s=2.0
  --robot.fold_on_disconnect=true
  --robot.fold_staging_goal_time_s=3.0
  --robot.folded_positions='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
  --robot.fold_goal_time_s=3.0
  --robot.fold_max_arm_error_rad=0.12
  --robot.fold_max_gripper_error_m=0.003
  --robot.max_relative_target='{"joint_0": 0.07, "joint_1": 0.07, "joint_2": 0.07, "joint_3": 0.07, "joint_4": 0.07, "joint_5": 0.07, "left_carriage_joint": 0.003}'
  --robot.cameras='{cam_main: {type: intelrealsense, serial_number_or_name: "838212073584", width: 640, height: 480, fps: 30}}'
  --strategy.type=base
  --device=cuda
  --fps=20
  --duration=0
  --task="Pick up the carrot and place it in the pan"
  --return_to_initial_position=true
  --display_data=false
  --policy.path="$policy_path"
)
"${record_command[@]}"
