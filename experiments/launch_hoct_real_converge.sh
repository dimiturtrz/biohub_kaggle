#!/usr/bin/env bash
# Converge the HOCT edge head on REAL (1,2,2) GT pairs, warm detector frozen-shared from the finer co-adapt.
#
# Single-variable head-class comparison (kneemri 0905z): finer122_coadapt_long IS the converged STANDARD head
# under the SAME (1,2,2) detector; converging HOCT under --detector-from that same checkpoint makes head class
# the only difference. A converged lift in the scale-free top1 ruler (now wired) is conclusive; a flat is
# confounded (the detector co-adapted to the standard/pack head — licensed follow-up is the mirrored arm).
#
# Grid audit (2026-08-25): the velocity-inversion confusor is UNREACHABLE at (1,2,2) — both synth paths render
# fixed 64^3 = (1,4,4)-grid (SceneCorpus raises; gpu-scene stack-mismatches the 128^2 finer window), and at
# (1,4,4) the two confusor cells collide in one voxel so B is unlearnable there. So this arm trains on REAL GT
# pairs only; the HOCT line-to-line bias is what attacks the confusor. Inversion-at-(1,2,2) is filed separately.
set -euo pipefail
cd "$(dirname "$0")/.."  # the arm runs from the repo root, wherever it is launched from
LOG="logs/hoct_real_converge.log"
uv run python -m celltrack.training.joint_detector \
  --head hoct --norm group --downsample 1 2 2 \
  --detector-from finer122_coadapt_long.pt --slack-links \
  --epochs 3 --evals-per-epoch 3 --patience-epochs 1 \
  --batch-size 4 --weights hoct_real_converge.pt --device cuda \
  > "$LOG" 2>&1
