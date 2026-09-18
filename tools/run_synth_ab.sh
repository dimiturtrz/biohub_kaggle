#!/usr/bin/env bash
# Synth-pretrain A/B on the 20-movie real held-out split, strictly sequential (one GPU job at a time):
# pretrain on synthetic sequences, then R1 = real fine-tune warm from that pretrain, then R0 = real from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=external/frontier_ds/biohub-tracking-support-pack-50ep-v1/repo
REAL=D:/data/volumetric/microscopy/raw/biohub_cell_tracking/train
SPLITS=D:/data/volumetric/microscopy/processed/real_holdout20_splits.json
SYNTH=D:/data/volumetric/microscopy/processed/synth_trainer_600
PRE_METHOD=${PRE_METHOD:-synth_pre20}
PRE=$REPO/weights/$PRE_METHOD/split_0/edge_predictor_best.pth
EPOCHS=${EPOCHS:-40}
export USER=local PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONWARNINGS=ignore

run_arm() {
  local method=$1 data=$2 splits=$3 epochs=$4; shift 4
  .venv/Scripts/python tools/frontier_train.py --frontier-repo "$REPO" "$@" --method "$method" \
    --data-dir "$data" --splits "$splits" --split 0 --epochs "$epochs" --batch-size 8 --single-gpu --num-workers 8 \
    > "logs/frontier/$method.log" 2>&1
}

[[ -n "${SKIP_PRETRAIN:-}" ]] || run_arm "$PRE_METHOD" "$SYNTH" "$SYNTH/splits.json" "${PRE_EPOCHS:-20}"
run_arm real_from_synth "$REAL" "$SPLITS" "$EPOCHS" --init-full "$PRE"
run_arm real_scratch "$REAL" "$SPLITS" "$EPOCHS"
