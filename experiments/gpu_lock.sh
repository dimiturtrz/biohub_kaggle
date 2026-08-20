#!/usr/bin/env bash
# Cross-project GPU mutex. One physical card, one lock directory, honored by every repo on this box.
#
# Protocol agreed with the rsna_knee_kaggle agent over the inbox/ mailbox (2026-08-20). Origin is their
# experiments/lib.sh (hold_card / drop_own_card / release_card); ported here rather than sourced across
# repos so each project owns its copy in its own lane. Shared surface is the lock path alone.
#
# Neutral path under the shared projects root so it serialises ACROSS projects — a project-local lock
# only serialises its own chains and lets a second project's arm split the card.
#
# Four scars baked into these lines, each from a measured failure on their side. Port them or re-earn them:
#   1. mkdir IS the lock — atomic on NTFS. A test-then-write on a lockfile interleaves; two arms once
#      both started and the card paged over PCIe at 32-37 img/s vs ~520 to itself: 15x slower, SILENT
#      (nvidia-smi read 99% util at a third of the power budget). This is the failure the lock exists for.
#   2. A dead holder is CLEARED, not waited on (kill -0). A timeout-killed arm leaves the lock behind;
#      without this the next job parks on a corpse for its whole wall-clock budget.
#   3. Release ONLY if still the holder. A bare `rm -rf $LOCK` at exit wipes a lock a DIFFERENT chain
#      took in between — the serialiser itself producing scar 1.
#   4. Release PER ARM, not per chain. Taking the lock once for a multi-arm chain deadlocks at arm two:
#      mkdir fails, the holder is this very shell, kill -0 says alive, and it sleeps forever.
set -u

GPU_LOCK="${GPU_LOCK:-D:/personal_projects/.gpu.lock}"
GPU_LOCK_PROJECT="biohub_kaggle"
GPU_LOCK_WAIT_HEARTBEAT_MINUTES="${GPU_LOCK_WAIT_HEARTBEAT_MINUTES:-5}"

_gpu_holder_pid() { awk '{print $1}' "$GPU_LOCK/holder" 2>/dev/null; }

# Release only when this shell is still the recorded holder — covers a killed arm, not just a clean exit.
drop_own_card() { [ "$(_gpu_holder_pid)" = "$$" ] && rm -rf "$GPU_LOCK"; }

hold_card() {  # $1 = arm name, recorded in the holder line for the blocked side to read
  local arm="${1:-unnamed}" blocked=0 holder
  until mkdir "$GPU_LOCK" 2>/dev/null; do
    holder="$(_gpu_holder_pid)"
    if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
      echo "gpu-lock: clearing lock held by dead process $holder"
      rm -rf "$GPU_LOCK"
      continue
    fi
    [ $((blocked % GPU_LOCK_WAIT_HEARTBEAT_MINUTES)) -eq 0 ] &&
      echo "gpu-lock: parked ${blocked}m behind $(cat "$GPU_LOCK/holder" 2>/dev/null || echo unknown)"
    blocked=$((blocked + 1))
    sleep 60
  done
  echo "$$ $GPU_LOCK_PROJECT $arm $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$GPU_LOCK/holder"
  trap drop_own_card EXIT
  [ "$blocked" -gt 0 ] && echo "gpu-lock: took the card after ${blocked}m parked"
  return 0  # never leak the heartbeat test's exit status — `hold_card && run_arm` must see success
}

release_card() { drop_own_card; trap - EXIT; }

# Direct use: `bash experiments/gpu_lock.sh <arm-name> -- <command...>` runs the command under the lock.
# Sourced use: `source experiments/gpu_lock.sh` then call hold_card/release_card around an arm.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  arm="${1:?usage: gpu_lock.sh <arm-name> -- <command...>}"
  shift
  [ "${1:-}" = "--" ] && shift
  hold_card "$arm"
  "$@"
  status=$?
  release_card
  exit "$status"
fi
