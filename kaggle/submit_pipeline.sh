#!/usr/bin/env bash
# Fill Kaggle submission slots before deadline. 2 concurrent GPU sessions max.
# Poll active kernels; on COMPLETE -> submit output -> push next slug from kaggle/queue.txt. Stop at MAX_SUBMITS.
set -u
K=".venv/Scripts/kaggle.exe"
COMP="biohub-cell-tracking-during-development"
LOG="kaggle/submit_pipeline.log"
QF="kaggle/queue.txt"     # one slug per line, remaining kernels to push (edit live to reprioritise)
MAX_SUBMITS=5

# already pushed+running: "slug:version"
declare -a ACTIVE=("celltrack-frontier-thr097:2" "celltrack-dualseed-assign-affinity-stlf6:30")
declare -A DONE=()
declare -A SUBMITTED=()
SUBMIT_COUNT=0

log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }

pop_queue(){ # prints+removes first non-empty line of QF
  local line=""
  while IFS= read -r l; do [ -n "$l" ] && { line="$l"; break; }; done < "$QF"
  [ -z "$line" ] && return 1
  grep -vxF "$line" "$QF" > "$QF.tmp" 2>/dev/null; mv "$QF.tmp" "$QF"
  echo "$line"
}

push_next(){
  [ -s "$QF" ] || return 1
  local slug; slug=$(pop_queue) || return 1
  local out; out=$("$K" kernels push -p "kaggle/kernels/$slug" 2>&1)
  local ver; ver=$(echo "$out" | grep -oE 'version [0-9]+' | grep -oE '[0-9]+' | head -1)
  if [ -n "$ver" ]; then ACTIVE+=("$slug:$ver"); log "PUSHED $slug v$ver";
  else log "PUSH-FAIL $slug :: $(echo "$out"|tail -1)"; printf '%s\n' "$slug" >> "$QF"; return 1; fi
}

submit(){
  local slug="$1" ver="$2"
  local out; out=$("$K" competitions submit "$COMP" -k "dimiturnt/$slug" -v "$ver" -f submission.csv -m "auto $slug v$ver" 2>&1)
  log "SUBMIT $slug v$ver :: $(echo "$out"|tail -1)"; SUBMITTED[$slug]=1; SUBMIT_COUNT=$((SUBMIT_COUNT+1))
}

log "=== pipeline start (target $MAX_SUBMITS) ==="
for i in $(seq 1 180); do
  [ $SUBMIT_COUNT -ge $MAX_SUBMITS ] && { log "DONE $SUBMIT_COUNT submits"; break; }
  for entry in "${ACTIVE[@]}"; do
    slug="${entry%:*}"; ver="${entry#*:}"
    [ -n "${DONE[$slug]:-}" ] && continue
    st=$("$K" kernels status "dimiturnt/$slug" 2>&1 | tail -1)
    case "$st" in
      *COMPLETE*) log "$slug COMPLETE"; submit "$slug" "$ver"; DONE[$slug]=1; push_next || true ;;
      *ERROR*|*CANCEL*) log "$slug FAILED :: $st"; DONE[$slug]=1; push_next || true ;;
      *) : ;;
    esac
  done
  running=0; for e in "${ACTIVE[@]}"; do s="${e%:*}"; [ -z "${DONE[$s]:-}" ] && running=$((running+1)); done
  [ $running -lt 2 ] && push_next || true
  sleep 60
done
log "=== end: $SUBMIT_COUNT submits fired ==="
