#!/usr/bin/env bash
# Fire the ONE allowed submission for the geometric-division kernel the moment two conditions hold:
#   (1) the kernel run is COMPLETE (submission.csv exists on Kaggle), and
#   (2) the daily submission quota has reset (0 UTC) — detected by the submit call SUCCEEDING rather than
#       being rejected for "maximum submissions". No wall-clock math: retry the submit until it takes.
# Exits on the first accepted submission, or after the poll budget is spent.
set -u
K=".venv/Scripts/kaggle.exe"
COMP="biohub-cell-tracking-during-development"
SLUG="dimiturnt/celltrack-dualseed-geometric-div"
VER=1
LOG="kaggle/submit_geometric.log"

log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

log "=== submit_geometric start (kernel=$SLUG v$VER) ==="
for i in $(seq 1 240); do   # 240 * 90s = 6h poll budget
  st=$("$K" kernels status "$SLUG" 2>&1 | tail -1)
  case "$st" in
    *COMPLETE*)
      out=$("$K" competitions submit "$COMP" -k "$SLUG" -v "$VER" -f submission.csv \
            -m "geometric division candidacy vs 0.900 base (candidacy=geometric, thr0.97, symmetry) — first arm to move div_jac" 2>&1)
      last=$(echo "$out" | tail -1)
      log "SUBMIT-ATTEMPT :: $last"
      # A clean accept prints the CLI's "N submissions remaining today." — the ONLY reliable success token; an
      # over-quota reject is a 4xx / 'maximum submissions' error and never carries it. (The earlier
      # "successfully submitted" pattern matched no real message and re-fired every 90s, burning the quota.)
      if echo "$last" | grep -qiE "submissions remaining|successfully submitted|Your submission"; then
        log "=== SUBMITTED — one shot fired ==="; exit 0
      fi
      log "not accepted yet (quota not reset?) — retrying"
      ;;
    *ERROR*|*CANCEL*) log "KERNEL FAILED :: $st — aborting, needs a re-push"; exit 2 ;;
    *) log "kernel status: $st (waiting)" ;;
  esac
  sleep 90
done
log "=== poll budget spent without an accepted submission ==="
exit 1
