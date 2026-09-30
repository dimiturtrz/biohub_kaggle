#!/usr/bin/env bash
# Directional relative-PE warm arm — the one live independent mechanism (attacks association DISCRIMINATION,
# orthogonal to the exhausted recall axis). Warm from pilkwang (calibrated base -> BaseCalibration.gate passes).
#
# GO/NO-GO IS PROXY-INDEPENDENT (the LB proxy inverts/blinds >0.90, one submission left):
#   after train, run:  python -m celltrack.eval.dense_diagnosis --checkpoint logs/directional_pe_warm_v1.pt
#   champion baseline (same instrument): mean P_true 0.134  vs  P_chosen 0.686 ; inverted_fraction high.
#   SUBMIT only if P_true materially lifts toward/over P_chosen AND inverted_fraction drops. Else: from-scratch
#   disambiguator (zero-init warm may null ambiguously) or no-submit. A flat P_true = mechanism OR no confusor
#   gradient in the data; --detected-videos exposes the confusor negatives so the directional term gets signal.
set -euo pipefail
cd "$(dirname "$0")/.."  # the arm runs from the repo root, wherever it is launched from

# LoRA freezes the pilkwang base -> floor ~= champion 0.900 (no threshold/recall drift like dw0's -0.028);
# the zero-init directional bias is an installed (kept-trainable) param, so it + the low-rank adapters learn
# the confusor discrimination on the --detected-videos crowded candidates (where the confusor negatives live).
uv run python -m celltrack.training.joint_detector \
  --warm-start \
  --lora --lora-rank 16 \
  --relative-position \
  --relative-position-directional \
  --detected-videos 40 --det-weight 0 \
  --lr 1e-4 \
  --selection-objective confusor_margin \
  --weights directional_pe_warm_v1.pt

# --selection-objective confusor_margin: the LB proxy REGRESSED as the confusor margin improved last run
# (0.8485 init > 0.844 final), so save-best kept the ZERO-INIT ckpt and discarded the trained directional
# weight. Selecting on (mean_p_true - mean_p_chosen) keeps the ckpt where the mechanism actually fired.
