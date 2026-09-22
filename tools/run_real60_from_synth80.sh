#!/usr/bin/env bash
# The 60-epoch real fine-tune warm from the 80-epoch synth pretrain, on the 6-movie held-out split.
#
# Its only question is whether 80 synth pre-epochs behave differently from 20 at a LONG real schedule: at a
# 10-epoch schedule synth pretrain LOST (real_from_synth 0.9063 vs real_scratch 0.9127). It is an INSTRUMENT,
# not a submission candidate -- the local proxy blinds above ~0.90 and detector-side arms are 0-for-5 on the LB.
#
# Only the UNet crosses over (--init-unet reads the unet. subtree of the full pretrain checkpoint in memory, so
# no second file is carved on disk). Snapshots land every epoch and --resume picks the run up at the next one,
# so the arm is preemptible: kill it for the GPU, re-run this script to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=external/frontier_ds/biohub-tracking-support-pack-50ep-v1/repo
METHOD=${METHOD:-synth_real60_bf16}
PRE=$REPO/weights/${PRE_METHOD:-synth_pre80}/split_0/edge_predictor_best.pth
LOG=logs/frontier/$METHOD.log
export USER=local PYTHONWARNINGS=ignore
# This arm runs within ~600MiB of the card's 32.6GB, so it is sensitive to its own footprint rather than to the
# allocator: an earlier read blamed PYTORCH_CUDA_ALLOC_CONF, and the matched control refuted that (the default
# allocator was slower still, 3.15 s/batch against expandable_segments' 2.3). Set ALLOC to opt back in.
[[ -z "${ALLOC:-}" ]] || export PYTORCH_CUDA_ALLOC_CONF="$ALLOC"

mkdir -p "$(dirname "$LOG")"
# shellcheck disable=SC2086 # FLAGS carries whitespace-separated passthrough flags, e.g. FLAGS=--no-bf16
.venv/Scripts/python tools/frontier_train.py --frontier-repo "$REPO" --init-unet "$PRE" --resume ${FLAGS:-} \
  --method "$METHOD" --data-dir D:/data/volumetric/microscopy/raw/biohub_cell_tracking/train \
  --splits D:/data/volumetric/microscopy/processed/real_holdout6_splits.json \
  --split 0 --epochs "${EPOCHS:-60}" --batch-size 8 --single-gpu --num-workers 8 >> "$LOG" 2>&1
