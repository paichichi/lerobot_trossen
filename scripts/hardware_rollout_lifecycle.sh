#!/usr/bin/env bash

# Single-owner lifecycle for real-arm rollout launchers. This uses an atomic
# directory lock rather than flock: no lock descriptor can leak into uv/Python,
# and an empty lock left by a reboot can be verified and refreshed safely.

HARDWARE_ROLLOUT_PID=""
HARDWARE_ROLLOUT_PID_FILE=""
HARDWARE_ROLLOUT_LOCK_DIR=""
HARDWARE_ROLLOUT_OWNER_FILE=""

hardware_process_cwd() {
  readlink -f "/proc/$1/cwd" 2>/dev/null || true
}

hardware_process_command() {
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || true
}

hardware_rollout_pid_is_managed() {
  local pid="$1"
  local repo_root="$2"
  local process_command

  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ "$(hardware_process_cwd "$pid")" == "$repo_root" ]] || return 1
  process_command="$(hardware_process_command "$pid")"
  [[ "$process_command" == *lerobot-rollout* \
    || "$process_command" == *lerobot_rollout* ]]
}

hardware_launcher_pid_is_managed() {
  local pid="$1"
  local repo_root="$2"
  local process_command

  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ "$(hardware_process_cwd "$pid")" == "$repo_root" ]] || return 1
  process_command="$(hardware_process_command "$pid")"
  [[ "$process_command" == *scripts/run_uploaded_act.sh* \
    || "$process_command" == *scripts/run_uploaded_v11.sh* ]]
}

hardware_rollout_group_has_managed_process() {
  local process_group="$1"
  local repo_root="$2"
  local member_pid

  [[ "$process_group" =~ ^[0-9]+$ ]] || return 1
  while read -r member_pid; do
    if hardware_rollout_pid_is_managed "$member_pid" "$repo_root"; then
      return 0
    fi
  done < <(pgrep -g "$process_group" 2>/dev/null || true)
  return 1
}

hardware_rollout_interrupt_and_wait() {
  local process_group="$1"
  local label="$2"
  local repo_root="$3"
  local attempt

  echo "Stopping $label rollout group $process_group with SIGINT for driver cleanup..."
  kill -INT -- "-$process_group" 2>/dev/null \
    || kill -INT "$process_group" 2>/dev/null \
    || true
  for attempt in {1..300}; do
    if ! hardware_rollout_group_has_managed_process "$process_group" "$repo_root"; then
      wait "$process_group" 2>/dev/null || true
      echo "$label rollout completed controller cleanup."
      return 0
    fi
    sleep 0.1
  done
  echo "$label rollout did not clean up after 30 seconds." >&2
  echo "It was not force-killed because SIGKILL would skip driver cleanup." >&2
  return 1
}

hardware_legacy_lock_holders() {
  local repo_root="$1"
  local legacy_lock="$2"
  local fd_path holder_pid target

  for fd_path in /proc/[0-9]*/fd/*; do
    [[ -e "$fd_path" ]] || continue
    target="$(readlink "$fd_path" 2>/dev/null || true)"
    [[ "$target" == "$legacy_lock" ]] || continue
    holder_pid="${fd_path#/proc/}"
    holder_pid="${holder_pid%%/*}"
    [[ "$holder_pid" != "$$" ]] || continue
    [[ "$(hardware_process_cwd "$holder_pid")" == "$repo_root" ]] || continue
    printf '%s\n' "$holder_pid"
  done | sort -n -u
}

hardware_refresh_legacy_flock() {
  local repo_root="$1"
  local legacy_lock="$repo_root/output/.hardware_rollout.lock"
  local holder_pid process_command process_group
  local -a holders=()
  local attempt

  [[ -e "$legacy_lock" ]] || return 0
  mapfile -t holders < <(hardware_legacy_lock_holders "$repo_root" "$legacy_lock")
  if ((${#holders[@]})); then
    echo "Migrating legacy hardware lock held by: ${holders[*]}"
    for holder_pid in "${holders[@]}"; do
      process_command="$(hardware_process_command "$holder_pid")"
      if [[ "$process_command" == *lerobot-rollout* \
        || "$process_command" == *lerobot_rollout* ]]; then
        process_group="$(ps -o pgid= -p "$holder_pid" 2>/dev/null | tr -d ' ' || true)"
        [[ "$process_group" =~ ^[0-9]+$ ]] || process_group="$holder_pid"
        kill -INT -- "-$process_group" 2>/dev/null \
          || kill -INT "$holder_pid" 2>/dev/null \
          || true
      else
        # Shell/tee/helper holders do not control the arm. TERM only closes
        # their inherited descriptor; launchers trap TERM and clean children.
        kill -TERM "$holder_pid" 2>/dev/null || true
      fi
    done
    # Give normal TERM/SIGINT cleanup five seconds. Old non-driver shell/tee
    # helpers can ignore TERM while waiting on an already-dead child; because
    # they do not control the arm, force-closing only those verified helpers is
    # safe and releases their inherited descriptor.
    for attempt in {1..50}; do
      mapfile -t holders < <(hardware_legacy_lock_holders "$repo_root" "$legacy_lock")
      ((${#holders[@]} == 0)) && break
      sleep 0.1
    done
    if ((${#holders[@]})); then
      for holder_pid in "${holders[@]}"; do
        process_command="$(hardware_process_command "$holder_pid")"
        if [[ "$process_command" != *lerobot-rollout* \
          && "$process_command" != *lerobot_rollout* ]]; then
          echo "Force-closing stale non-driver lock helper $holder_pid"
          kill -KILL "$holder_pid" 2>/dev/null || true
        fi
      done
    fi
    # Actual rollout processes still receive up to 30 seconds to run their
    # driver teardown. They are never SIGKILLed by this migration.
    for attempt in {1..300}; do
      mapfile -t holders < <(hardware_legacy_lock_holders "$repo_root" "$legacy_lock")
      ((${#holders[@]} == 0)) && break
      sleep 0.1
    done
    if ((${#holders[@]})); then
      echo "Legacy lock holders remain: ${holders[*]}; refusing a second rollout." >&2
      return 1
    fi
  fi
  rm -f "$legacy_lock"
}

hardware_release_stale_directory_lock() {
  local repo_root="$1"
  local owner_pid=""
  local rollout_pid=""

  [[ -d "$HARDWARE_ROLLOUT_LOCK_DIR" ]] || return 0
  [[ -f "$HARDWARE_ROLLOUT_OWNER_FILE" ]] \
    && read -r owner_pid < "$HARDWARE_ROLLOUT_OWNER_FILE" || true
  [[ -f "$HARDWARE_ROLLOUT_PID_FILE" ]] \
    && read -r rollout_pid < "$HARDWARE_ROLLOUT_PID_FILE" || true

  if hardware_rollout_group_has_managed_process "$rollout_pid" "$repo_root"; then
    hardware_rollout_interrupt_and_wait "$rollout_pid" "previous" "$repo_root" || return 1
  fi
  if hardware_launcher_pid_is_managed "$owner_pid" "$repo_root"; then
    echo "Stopping previous launcher $owner_pid so it can finish teardown..."
    kill -TERM "$owner_pid" 2>/dev/null || true
    for _ in {1..350}; do
      kill -0 "$owner_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$owner_pid" 2>/dev/null; then
      echo "Previous launcher $owner_pid did not exit; refusing a second rollout." >&2
      return 1
    fi
  fi

  # No verified launcher or rollout remains, so this directory is stale rather
  # than an active hardware owner. Fixed filenames make the refresh bounded.
  rm -f "$HARDWARE_ROLLOUT_OWNER_FILE" "$HARDWARE_ROLLOUT_PID_FILE"
  rmdir "$HARDWARE_ROLLOUT_LOCK_DIR" 2>/dev/null || {
    echo "Hardware lock directory contains unexpected files; refusing to remove it." >&2
    return 1
  }
  echo "Refreshed stale hardware ownership."
}

hardware_rollout_prepare() {
  local repo_root="$1"
  local attempt

  command -v setsid >/dev/null 2>&1 || {
    echo "The required 'setsid' command is unavailable (install util-linux)." >&2
    return 1
  }
  command -v pgrep >/dev/null 2>&1 || {
    echo "The required 'pgrep' command is unavailable (install procps)." >&2
    return 1
  }

  mkdir -p "$repo_root/output"
  HARDWARE_ROLLOUT_LOCK_DIR="$repo_root/output/.hardware_rollout.lock.d"
  HARDWARE_ROLLOUT_OWNER_FILE="$HARDWARE_ROLLOUT_LOCK_DIR/launcher.pid"
  HARDWARE_ROLLOUT_PID_FILE="$HARDWARE_ROLLOUT_LOCK_DIR/rollout.pid"

  hardware_refresh_legacy_flock "$repo_root"
  for attempt in {1..3}; do
    if mkdir "$HARDWARE_ROLLOUT_LOCK_DIR" 2>/dev/null; then
      printf '%s\n' "$$" > "$HARDWARE_ROLLOUT_OWNER_FILE"
      return 0
    fi
    # Give a concurrent launcher time to publish its owner PID before deciding
    # whether the directory is stale.
    sleep 0.2
    hardware_release_stale_directory_lock "$repo_root" || return 1
  done
  echo "Could not acquire exclusive hardware ownership." >&2
  return 1
}

hardware_rollout_run() {
  local exit_code

  setsid "$@" &
  HARDWARE_ROLLOUT_PID=$!
  printf '%s\n' "$HARDWARE_ROLLOUT_PID" > "$HARDWARE_ROLLOUT_PID_FILE"
  echo "Managed hardware rollout group: $HARDWARE_ROLLOUT_PID"

  set +e
  wait "$HARDWARE_ROLLOUT_PID"
  exit_code=$?
  set -e

  rm -f "$HARDWARE_ROLLOUT_PID_FILE"
  HARDWARE_ROLLOUT_PID=""
  return "$exit_code"
}

hardware_rollout_cleanup() {
  local repo_root="$1"
  local owner_pid=""

  if [[ -n "$HARDWARE_ROLLOUT_PID" ]] \
    && hardware_rollout_group_has_managed_process "$HARDWARE_ROLLOUT_PID" "$repo_root"; then
    hardware_rollout_interrupt_and_wait \
      "$HARDWARE_ROLLOUT_PID" "current" "$repo_root" || return 1
  fi

  if [[ -f "$HARDWARE_ROLLOUT_OWNER_FILE" ]]; then
    read -r owner_pid < "$HARDWARE_ROLLOUT_OWNER_FILE" || true
  fi
  if [[ "$owner_pid" == "$$" ]]; then
    rm -f "$HARDWARE_ROLLOUT_PID_FILE" "$HARDWARE_ROLLOUT_OWNER_FILE"
    rmdir "$HARDWARE_ROLLOUT_LOCK_DIR" 2>/dev/null || true
  fi
  HARDWARE_ROLLOUT_PID=""
}
