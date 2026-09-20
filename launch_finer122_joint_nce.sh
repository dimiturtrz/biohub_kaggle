#!/usr/bin/env bash
# THE correctly-specified never-run cell: finer122 detector, JOINT (trunk trainable), contrastive ON,
# trained on the DETECTED corpus.
#
# Why this and not launch_hoct_finer122_nce.sh (2026-09-20, failed 0.6239 vs a 0.6877 gate):
#   That arm froze the detector. Under a frozen trunk the contrastive term is INERT — at the default
#   FEATURES site it builds no projection and holds ZERO parameters (contrastive_term.py:59-63), and
#   info_nce.py contrasts backbone features directly, so its gradient reaches nothing trainable.
#   nce sat 1.5399 -> 1.5496 across 72,096 steps. avl8 was frozen too, so "nce ON vs OFF" was never the
#   live variable in ANY head-only run. The real 0.79-vs-0.62 split is the DETECTOR: both arms init a
#   FRESH UNTRAINED head and already read 0.7875 (pilkwang) vs 0.6164 (finer122) before a gradient step.
#   See interpretations/celltrack/2026-09-20_head_only_contrastive_is_structurally_inert.md.
#
#   A log sweep for "detector frozen" finds only six frozen runs ever; everything else is JOINT, where the
#   term IS live — and that is where the 0.61 -> 0.87 nce lift was measured. finer122_coadapt_long is joint
#   but nce-DARK, and sits at 0.6150: the same plateau signature nce fixes in the joint family. Nobody has
#   run joint + finer122 + nce ON. That is this arm.
#
# --downsample 1 2 2 IS REQUIRED and is not implied by --init-weights. from_checkpoint rebuilds the MODEL
# from the checkpoint's own config (layers, norm), but the decode GRID comes from the run's flags. Omitting
# it built a (1,4,4) pipeline around the (1,2,2) checkpoint and BaseCalibration aborted at window 0:
#   node ratio +1.03 outside [-0.50, +0.50]
# The band is grid-scaled — 0.5 * (16 / dy*dx), so (1,2,2) allows +2.00 and +1.03 passes. The abort was the
# grid mismatch, not the base.
#
# The inverse holds for the HEAD, and passing --relative-position here is a second bug, not a fix: the head
# config IS restored by from_checkpoint, so re-passing the flags wraps an already-wrapped head and dies with
#   AttributeError: 'RelativePositionEdgeTransformer' object has no attribute 'blocks'
# Rule for a chained run: pass what describes the PIPELINE (decode grid, corpus, loss weights); pass nothing
# that describes the MODEL, which the checkpoint already carries.
#
# Three deliberate choices:
#   --init-weights  : whole model, TRAINABLE (unlike --detector-from, which freezes). Freezing costs ~0.05
#                     (avl8 0.8024 frozen vs 0.8475/0.8567/0.8630 joint) — more than any head choice buys.
#   --detected-videos : the failed arm trained on GT pairs at in-gate mean 1.00 / rival 2.6%. An association
#                     objective needs something to choose between; the detected corpus carries ~53% contested.
#   --batch-size 2  : the 30% GPU budget is ~9.5GB. The FROZEN arm used 9.66GB at batch 4; unfreezing adds
#                     detector grads + optimizer state, so halve it. Watch the first eval window — the failed
#                     arm ballooned to 16.9GB after eval and expandable_segments is unsupported on Windows.
#
# Gate: proxy must clear ~0.70 to show the objective moved anything at all (plateau is 0.6150); ~0.85 puts
# finer122 in the joint regime and REOPENS the finer-decode leg. Below 0.70 = the substrate is the bound,
# not the objective, and the finer122 axis closes for good. NOT a submission arm either way: 0.947 is banked.
set -euo pipefail
LOG="logs/finer122_joint_nce.log"
uv run python -m celltrack.training.joint_detector \
  --init-weights finer122_coadapt_long.pt \
  --downsample 1 2 2 \
  --detected-videos 20 \
  --contrastive-weight 1.5 --hard-negative-weight 0.015 --temperature 0.07 \
  --lr 5e-5 \
  --epochs 8 --evals-per-epoch 2 --patience-epochs 3 \
  --batch-size 2 --weights finer122_joint_nce.pt --device cuda \
  > "$LOG" 2>&1
