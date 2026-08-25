#!/usr/bin/env bash

# Shared single-owner lifecycle for real-arm rollout launchers. The caller is
# expected to use `set -euo pipefail`, call hardware_rollout_prepare before
# launch, hardware_rollout_run for the command, and hardware_rollout_cleanup
# from its EXIT trap.

HARDWARE_ROLLOUT_PID=""
HARDWARE_ROLLOUT_PID_FILE=""
HARDWARE_ROLLOUT_LOCK_FD=9

hardware_rollout_pid_is_ours() {
  local pid="$1"
  local repo_root="$2"
  local process_cwd process_command

  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  process_command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$process_cwd" == "$repo_root" ]] || return 1
  [[ "$process_command" == *lerobot-rollout* \
    || "$process_command" == *lerobot_rollout* ]]
}

hardware_rollout_find_managed_lock_holders() {
  local repo_root="$1"
  local lock_file="$2"
  local fd_path holder_pid process_cwd process_command target

  for fd_path in /proc/[0-9]*/fd/*; do
    [[ -e "$fd_path" ]] || continue
    target="$(readlink "$fd_path" 2>/dev/null || true)"
    [[ "$target" == "$lock_file" ]] || continue
    holder_pid="${fd_path#/proc/}"
    holder_pid="${holder_pid%%/*}"
    [[ "$holder_pid" != "$$" ]] || continue
    process_cwd="$(readlink -f "/proc/$holder_pid/cwd" 2>/dev/null || true)"
    process_command="$(tr '\0' ' ' < "/proc/$holder_pid/cmdline" 2>/dev/null || true)"
    [[ "$process_cwd" == "$repo_root" ]] || continue
    if [[ "$process_command" == *lerobot-rollout* \
      || "$process_command" == *lerobot_rollout* \
      || "$process_command" == *scripts/run_uploaded_act.sh* \
      || "$process_command" == *scripts/run_uploaded_v11.sh* ]]; then
      printf '%s\n' "$holder_pid"
    fi
  done | sort -n -u
}

hardware_rollout_interrupt_and_wait() {
  local pid="$1"
  local label="$2"
  local attempt process_state

  echo "Stopping $label rollout process $pid with SIGINT so the driver can clean up..."
  # hardware_rollout_run starts a new session whose process-group ID equals
  # its PID. Signal the whole group so both `uv` and Python receive Ctrl+C.
  kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  for attempt in {1..300}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$label rollout process $pid exited cleanly."
      return 0
    fi
    process_state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)"
    if [[ "$process_state" == Z ]]; then
      # Reap it when it is our child. A previous launcher's parent will reap
      # its own child before releasing the single-owner lock.
      wait "$pid" 2>/dev/null || true
      echo "$label rollout process $pid completed controller cleanup."
      return 0
    fi
    sleep 0.1
  done
  echo "Rollout process $pid did not exit after 30 seconds." >&2
  echo "It was not force-killed because SIGKILL would skip controller cleanup." >&2
  return 1
}

hardware_rollout_prepare() {
  local repo_root="$1"
  local holder_pid previous_pid=""
  local -a holder_pids=()
  local lock_file="$repo_root/output/.hardware_rollout.lock"

  command -v flock >/dev/null 2>&1 || {
    echo "The required 'flock' command is unavailable (install util-linux)." >&2
    return 1
  }
  command -v setsid >/dev/null 2>&1 || {
    echo "The required 'setsid' command is unavailable (install util-linux)." >&2
    return 1
  }

  mkdir -p "$repo_root/output"
  HARDWARE_ROLLOUT_PID_FILE="$repo_root/output/.active_hardware_rollout.pid"
  exec 9>"$lock_file"

  if ! flock -n "$HARDWARE_ROLLOUT_LOCK_FD"; then
    if [[ -f "$HARDWARE_ROLLOUT_PID_FILE" ]]; then
      read -r previous_pid < "$HARDWARE_ROLLOUT_PID_FILE" || true
    fi
    if hardware_rollout_pid_is_ours "$previous_pid" "$repo_root"; then
      hardware_rollout_interrupt_and_wait "$previous_pid" "previous" || return 1
    else
      # The recorded leader may already be gone while one of its descendants
      # still has the inherited lock open. Resolve the actual kernel lock
      # holders instead of trusting an empty or stale PID file.
      mapfile -t holder_pids < <(
        hardware_rollout_find_managed_lock_holders "$repo_root" "$lock_file" 9>&-
      )
      if ((${#holder_pids[@]} == 0)); then
        echo "The hardware lock is held by an unknown process; refusing an unsafe refresh." >&2
        return 1
      fi
      echo "Refreshing stale hardware ownership held by: ${holder_pids[*]}"
      for holder_pid in "${holder_pids[@]}"; do
        kill -INT "$holder_pid" 2>/dev/null || true
      done
    fi
    echo "Waiting for the previous launcher to finish hardware teardown..."
    flock -w 35 "$HARDWARE_ROLLOUT_LOCK_FD" || {
      echo "The previous launcher still owns the robot; refusing a second rollout." >&2
      return 1
    }
  fi

  # Also handle a managed rollout whose launcher disappeared and therefore no
  # longer owns the lock, but whose rollout process and PID file remain.
  previous_pid=""
  if [[ -f "$HARDWARE_ROLLOUT_PID_FILE" ]]; then
    read -r previous_pid < "$HARDWARE_ROLLOUT_PID_FILE" || true
  fi
  if hardware_rollout_pid_is_ours "$previous_pid" "$repo_root"; then
    hardware_rollout_interrupt_and_wait "$previous_pid" "orphaned" || return 1
  fi

  # A PID file can remain after a host crash. It is safe to discard only after
  # obtaining the exclusive lock and confirming there is no managed owner.
  rm -f "$HARDWARE_ROLLOUT_PID_FILE"
}

hardware_rollout_run() {
  local exit_code

  # Keep the exclusive lock in the launcher only. If `uv` or Python inherited
  # this descriptor, a dead launcher could leave an empty PID file while an
  # orphaned descendant continued holding the kernel lock indefinitely.
  (
    exec 9>&-
    exec setsid "$@"
  ) &
  HARDWARE_ROLLOUT_PID=$!
  printf '%s\n' "$HARDWARE_ROLLOUT_PID" > "$HARDWARE_ROLLOUT_PID_FILE"
  echo "Managed hardware rollout PID: $HARDWARE_ROLLOUT_PID"

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
  local recorded_pid=""

  if [[ -n "$HARDWARE_ROLLOUT_PID" ]] \
    && hardware_rollout_pid_is_ours "$HARDWARE_ROLLOUT_PID" "$repo_root"; then
    if ! hardware_rollout_interrupt_and_wait "$HARDWARE_ROLLOUT_PID" "current"; then
      # Preserve the PID file so the next launch refuses to compete with a
      # controller owner that could not complete its cleanup.
      return 1
    fi
  fi

  if [[ -n "$HARDWARE_ROLLOUT_PID_FILE" && -f "$HARDWARE_ROLLOUT_PID_FILE" ]]; then
    read -r recorded_pid < "$HARDWARE_ROLLOUT_PID_FILE" || true
    if [[ -z "$recorded_pid" || "$recorded_pid" == "$HARDWARE_ROLLOUT_PID" ]] \
      || ! hardware_rollout_pid_is_ours "$recorded_pid" "$repo_root"; then
      rm -f "$HARDWARE_ROLLOUT_PID_FILE"
    fi
  fi
  HARDWARE_ROLLOUT_PID=""
}
