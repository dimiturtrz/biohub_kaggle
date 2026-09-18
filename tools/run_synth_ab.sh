#!/usr/bin/env bash
# Synth-pretrain A/B on the 20-movie real held-out split: R1 = warm from synth pretrain, R0 = scratch.
# Waits for the pretrain checkpoint to stop changing (pretrain process gone), then runs both arms in sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=external/frontier_ds/biohub-tracking-support-pack-50ep-v1/repo
REAL=D:/data/volumetric/microscopy/raw/biohub_cell_tracking/train
SPLITS=D:/data/volumetric/microscopy/processed/real_holdout20_splits.json
PRE=$REPO/weights/${PRE_METHOD:-synth_pre20}/split_0/edge_predictor_best.pth
EPOCHS=${EPOCHS:-40}
export USER=local PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore

while powershell -NoProfile -c "if (Get-CimInstance Win32_Process | ? { \$_.CommandLine -like '*method ${PRE_METHOD:-synth_pre20}*' -and \$_.Name -like 'python*' }) { exit 0 } else { exit 1 }"; do
  sleep 60
done

run_arm() {
  local method=$1; shift
  .venv/Scripts/python tools/frontier_train.py --frontier-repo "$REPO" "$@" --method "$method" \
    --data-dir "$REAL" --splits "$SPLITS" --split 0 --epochs "$EPOCHS" --batch-size 8 --single-gpu --num-workers 8 \
    > "logs/frontier/$method.log" 2>&1
}

run_arm real_from_synth --init-full "$PRE"
run_arm real_scratch
