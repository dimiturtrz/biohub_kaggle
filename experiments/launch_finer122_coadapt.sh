#!/usr/bin/env bash
# finer122 co-adapt — FROM-SCRATCH train at the (1,2,2) decode grid. The highest wall-clock/gain lever open.
#
# WHY from-scratch (not warm, not graft): the (1,4,4) decode fuses two crowded cells into ONE basin
# (detection-merge bottleneck A, ~part of the 0.900->0.950 dense-regime gap). A frozen (1,4,4)-trained head
# decoded at (1,2,2) is OFF-GRID -> merge_referee scored it CONFOUNDED (+60.9% peak inflation, fabricated
# fragments). So the finer grid must be TRAINED, everything co-adapting at (1,2,2): detector + edge head.
#
# GO/NO-GO IS GT-FREE + PROXY-BLIND (finer-decode is LB-visible / proxy-blind; the LB proxy can't see it).
#   TWO steps, both card-free once the checkpoint exists (see bd biohub_kaggle-h3zi):
#   1. DUMP finer122's fine (1,2,2) logits over the dense confusor movie under weights-key
#      finer122_coadapt_v3_logits (from_joint mounts a PERSISTENT ResponseCache; run_scored persists the volume):
#        uv run python -m celltrack.eval.dump_joint_logits \
#            --checkpoint finer122_coadapt_v3.pt --movie 44b6_e57ff5c6
#      The coarse (1,4,4) baseline is already cached as pilkwang_seed1_logits. A (1,2,2)-locked joint checkpoint
#      CANNOT self-supply the coarse band (from_joint asserts grid==checkpoint; off-grid decode = garbage) ->
#      the referee compares finer122's fine vs the pilkwang coarse BASELINE.
#   2. run the physics referee (two-key, coarse=baseline / fine=candidate):
#        uv run python -m celltrack.analysis.merge_referee \
#            --coarse-key pilkwang_seed1_logits --fine-key finer122_coadapt_v3_logits \
#            --video-key 44b6_e57ff5c6.zarr --floor 0.5
#   PASS iff matched-1.6um peaks AGREE across grids (no inflation) AND the single band GROWS / merged-mass
#   drains -> the finer grid un-merges REAL cells. CONFOUNDED (inflation) or NO-signal -> do not ship.
#   Only a PASS + a recall axis the champion structurally lacks earns the one held submission.
set -euo pipefail
cd "$(dirname "$0")/.."  # the arm runs from the repo root, wherever it is launched from

# NO --warm-start / NO --lora: full co-adapt so the detector learns to place TWO centres where (1,4,4)
# placed one. det-weight left at default (detection MUST train -- it is the grid that merges).
# SCHEDULE (EXPLICIT -- the parser default is --steps 1500 = 0.08 epoch, uselessly short for a from-scratch
# detector; the v3-1500 run proved it: init proxy 0.0, 0 nodes, never left init). --epochs 8 = 144192 steps
# ~1.6h train at ~25 it/s; memory says most loss falls early, ~10 ep reaches 0.847. --evals-per-epoch 3 (NOT
# the default eval@500 = 288 evals x ~40s = +3h of pure eval overhead) -> 24 evals, ~16min. --patience-epochs
# 3 so proxy-based early-stop (faithful below 0.89 during the from-scratch climb) doesn't trip on scatter.
# MUST be launched DETACHED (run_in_background / nohup) -- the v2 attempt died at init from a foreground
# parent-exit, not OOM, not a code fault. ~1.6h at ~25 it/s; needs the card mostly to itself (finer grid
# ~4x y/x activation memory, will not co-reside with a ~16GB neighbour).
#
# DIRECTIONAL RIDER (added 2026-08-25, free): --relative-position --relative-position-directional installs
# the per-head offset-direction bias. This full from-scratch co-adapt is the ONLY regime where that bias
# gets a CONSISTENT gradient (audit: under warm+LoRA the frozen base can't co-adapt to USE it, so it random-
# walked to noise 5.6e-3, zero flips). Bias is init-zero = a no-op at start, grows only if it lowers loss ->
# negligible risk to detection training. Selector stays default PROXY (detection recovers from scratch;
# confusor_margin would ignore detection quality -- wrong for a detection-merge run). One run now tests BOTH
# levers: gate detection-merge with merge_referee (below); check the directional axis POST-HOC on the same
# ckpt via the confusor instrument (inv_frac / P_chosen margin).
#
# RIDER SAFETY (RESOLVED-CORRECT 2026-08-25, was "check at first eval"): the directional bias is PHYSICALLY
# CORRECT at (1,2,2), NOT blunted. Coords arrive in RAW-voxel units (hoct_edge_transformer.py:211 "node coords
# arrive in raw-grid voxels"; voxel_um=(1.625,0.40625,0.40625)=raw voxel size, downsample-INVARIANT -- the
# detector upsamples centroids to raw before emitting). offset_um = coord_delta * spacing (relative_position_
# bias.py:98), spacing = config.spacing = raw ome-zarr voxel (pair_split.py:97, also invariant). So
# separation_um is physical um at ANY downsample: the radial basis tiles [0,gate_um=10um] identically at
# (1,2,2) and (1,4,4) -> no 2x mis-scale, no basis saturation. The rider gets its full directional signal.
uv run python -m celltrack.training.joint_detector \
  --downsample 1 2 2 \
  --relative-position \
  --relative-position-directional \
  --lr 1e-4 \
  --batch-size 4 \
  --epochs 2 \
  --evals-per-epoch 3 \
  --patience-epochs 1 \
  --weights finer122_coadapt_v3.pt \
  2>&1 | tee logs/finer122_coadapt_v3.log
