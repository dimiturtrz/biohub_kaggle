#!/usr/bin/env bash
# Recall-recovery via the CORRECT axis: the flow linker's disappearance_cost, not detection threshold.
# thr-sweep (0.90/0.93/0.95) proved threshold inert — node recall pinned at 0.8938 regardless. The recall
# loss is the linker dropping true edges under HOCT's sharper affinity, not a detection-count problem.
# Raising disappearance_cost (shipped 3.0) makes ending a track costlier -> linker continues tracks over
# weak edges -> recovers dropped true edges. HOCT ranks the true successor #1 (top1 0.224), so confusors
# stay low even as the cost rises: a window should exist where recall climbs toward standard's 0.9512 while
# top1 holds >> standard's 0.012. A config with faithful >= standard-finer 0.6877 AND top1 >> 0.012 is the
# first B-driven pipeline-win candidate (paired with the oracle-justified finer-A axis = the submission thesis).
set -euo pipefail
cd "$(dirname "$0")"
LOG=logs/hoct_disc_sweep_upper.log
: > "$LOG"
for d in 3 6 10 20 40; do
  echo "=== HOCT disc=$d (thr 0.97) ===" | tee -a "$LOG"
  uv run python -m celltrack.eval.joint_eval \
    --checkpoint hoct_real_converge.pt \
    --threshold 0.97 \
    --set linker.disappearance_cost=$d >> "$LOG" 2>&1
done
echo "=== disc sweep done ===" | tee -a "$LOG"
