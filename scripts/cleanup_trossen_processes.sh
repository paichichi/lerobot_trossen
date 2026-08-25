#!/usr/bin/env bash
set -euo pipefail

# Stop only rollout/driver processes owned by the current user. The cleanup
# script and its parent shell are deliberately excluded from every scan.
collect_pids() {
  local pid command

  while read -r pid command; do
    [[ -n "${pid:-}" ]] || continue
    [[ "$pid" != "$$" && "$pid" != "$PPID" && "$pid" != "${BASHPID:-$$}" ]] || continue

    case "$command" in
      *cleanup_trossen_processes.sh*) continue ;;
      *lerobot-rollout*|*scripts/run_uploaded_act.sh*|*scripts/run_uploaded_v11.sh*|*/lerobot_trossen/.venv/bin/python*|*lerobot_robot_trossen*|*widowxai_follower*|*trossen_arm*)
        printf '%s\n' "$pid"
        ;;
    esac
  done < <(ps -u "$(id -u)" -o pid= -o args=)
}

show_targets() {
  local pid
  for pid in "$@"; do
    ps -p "$pid" -o pid=,etime=,args= 2>/dev/null || true
  done
}

wait_for_exit() {
  local attempts="$1"
  shift
  local pid alive

  while (( attempts > 0 )); do
    alive=0
    for pid in "$@"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && return 0
    sleep 0.5
    ((attempts -= 1))
  done
  return 1
}

refresh_targets() {
  local pid
  target_pids=()
  while read -r pid; do
    [[ -n "$pid" ]] && target_pids+=("$pid")
  done < <(collect_pids | sort -nu)
}

target_pids=()
refresh_targets
if (( ${#target_pids[@]} == 0 )); then
  echo "No active LeRobot/Trossen rollout processes found."
  exit 0
fi

echo "Stopping these LeRobot/Trossen processes:"
show_targets "${target_pids[@]}"

# SIGINT gives lerobot-rollout its normal Ctrl+C path so it can return to the
# initial position and disconnect. Escalation is used only for stuck processes.
kill -INT "${target_pids[@]}" 2>/dev/null || true
if ! wait_for_exit 30 "${target_pids[@]}"; then
  refresh_targets
  if (( ${#target_pids[@]} > 0 )); then
    echo "Some processes ignored SIGINT; sending SIGTERM:"
    show_targets "${target_pids[@]}"
    kill -TERM "${target_pids[@]}" 2>/dev/null || true
    wait_for_exit 6 "${target_pids[@]}" || true
  fi
fi

refresh_targets
if (( ${#target_pids[@]} > 0 )); then
  echo "Force-closing remaining stuck processes:"
  show_targets "${target_pids[@]}"
  kill -KILL "${target_pids[@]}" 2>/dev/null || true
  sleep 1
fi

refresh_targets
if (( ${#target_pids[@]} > 0 )); then
  echo "Unable to stop the following processes:" >&2
  show_targets "${target_pids[@]}" >&2
  exit 1
fi

echo "All LeRobot/Trossen rollout processes have stopped."
