#!/usr/bin/env bash
# Poll flow + division kernels; submit each to the competition the moment it reaches COMPLETE. Exit when both resolved.
set -u
K=".venv/Scripts/kaggle.exe"
COMP="biohub-cell-tracking-during-development"
LOG="kaggle/submit_ab.log"
declare -A DONE=()
KERNELS=("celltrack-ourflow-global:1" "celltrack-ourdiv-recover:1")

log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

log "=== submit_ab start ==="
for i in $(seq 1 60); do
  all_done=1
  for entry in "${KERNELS[@]}"; do
    slug="${entry%:*}"; ver="${entry#*:}"
    [ -n "${DONE[$slug]:-}" ] && continue
    st=$("$K" kernels status "dimiturnt/$slug" 2>&1 | tail -1)
    case "$st" in
      *COMPLETE*)
        out=$("$K" competitions submit "$COMP" -k "dimiturnt/$slug" -v "$ver" -f submission.csv -m "auto $slug v$ver" 2>&1)
        log "SUBMIT $slug v$ver :: $(echo "$out"|tail -1)"; DONE[$slug]=1 ;;
      *ERROR*|*CANCEL*) log "$slug FAILED :: $st"; DONE[$slug]=1 ;;
      *) all_done=0 ;;
    esac
  done
  [ $all_done -eq 1 ] && { log "=== all resolved ==="; break; }
  sleep 60
done
log "=== submit_ab end ==="
