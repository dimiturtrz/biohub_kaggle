#!/usr/bin/env bash
# Score the synth A/B arms on the 20-movie held-out split: cache each arm's predict_video once (GPU), then sweep
# the linking grid on CPU, so the arms are compared at their own best settings, not at one shared default.
#   bash tools/eval_synth_ab.sh <grid.json> [arm...]
set -euo pipefail
cd "$(dirname "$0")/.."
GRID=$(realpath "$1"); shift
ARMS=("${@:-real_from_synth real_scratch}")
source tools/data_root.sh
REPO=$PWD/external/frontier_ds/biohub-tracking-support-pack-50ep-v1/repo
REAL=$DATA_RAW/biohub_cell_tracking/train
SPLITS=$DATA_PROCESSED/real_holdout20_splits.json
CACHE_ROOT=$DATA_PROCESSED/frontier_cache
export USER=local PYTHONWARNINGS=ignore

for arm in ${ARMS[@]}; do
  .venv/Scripts/python tools/frontier_ilp_sweep.py --frontier-repo "$REPO" --data-dir "$REAL" cache --splits "$SPLITS" \
    --weights "$REPO/weights/$arm/split_0/edge_predictor_best.pth" --out "$CACHE_ROOT/$arm" > "logs/frontier/cache_$arm.log" 2>&1
done
for arm in ${ARMS[@]}; do
  OMP_NUM_THREADS=2 .venv/Scripts/python tools/frontier_ilp_sweep.py --frontier-repo "$REPO" --data-dir "$REAL" sweep \
    --cache "$CACHE_ROOT/$arm" --grid "$GRID" --workers "${WORKERS:-4}" > "logs/frontier/sweep_$arm.log" 2>&1
done
