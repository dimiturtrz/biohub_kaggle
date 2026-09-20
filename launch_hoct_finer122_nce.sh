#!/usr/bin/env bash
# THE never-run cell: finer122 detector + HOCT edge head + CONTRASTIVE (nce) ON.
#
# Finding (2026-08-28): the entire finer122 head family was trained with contrastive OFF.
#   - launch_hoct_real_converge.sh (HOCT) omits --contrastive-weight / --hard-negative-weight
#   - finer122_coadapt_long (the "standard" A/B control) likewise: nce 0.0000 hn 0.0000 for 144k steps
#   - LossCfg defaults both weights to 0.0 (joint_config.py:288,298), so an omitted flag = dark term
#   Both plateau proxy ~0.61 with inv 0.97-0.99 (P_true 0.07 vs chosen 0.39) = the degenerate
#   inverted-affinity signature of an untrained association objective. avl8 (SAME joint arch, nce ON)
#   reaches proxy 0.79 / node R 0.92 / inv 0.54. So every finer122 head verdict (coadapt 0.779,
#   HOCT HEAD-BOUND @ referee 05:16z) rode a crippled objective — NOT a refutation of the head axis.
#
# This arm closes the untested cell: the converged finer (1,2,2) detector (frozen, from coadapt_long)
# + the HOCT line-to-line head, trained with the contrastive term the champion recipe carries and these
# runs lacked. If proxy clears ~0.87 (avl8 regime on the finer detector) the finer122 axis is alive.
#
# Weight PIN (recovered from avl8 loss decomposition, NOT derived):
#   avl8 loss line prints RAW unweighted terms; total = weighted sum. Solve two lines:
#     13160.2369 = edge 18.6874 + det 12.8443 + w_nce*2.4670 + w_hn*875000.3253  (avl8, hn pre-fix)
#        35.2359 = edge 18.6874 + det 12.8453 + w_nce*2.4652 + w_hn*0.3681         (avl8hnfix)
#   Both fit to 1e-2 at w_nce=1.5, w_hn=0.015 (edge/det weight 1.0). So avl8 ACTUAL:
#   --contrastive-weight 1.5  : recovered exact (hn line residual 0.005, pre-fix line 0.007).
#   --hard-negative-weight 0.015 : recovered exact, AND == memory knife-edge (0.57 collapses). Do not raise.
#
# CARD-GATED: needs ~28GB, will not co-tenant with knee's raddino arm (to ~0900z). Launch after card frees.
#
# NORM FIX 2026-09-20: this script said `--norm group`, which would have silently voided the arm.
# The checkpoint's own config is norm='batch' with 10 BatchNorm running_mean buffers
# (downsample [1,2,2], head 'pack'); a GroupNorm backbone has no home for those buffers, so the
# warm detector would have been discarded and the run would have produced a second invalid verdict
# for the same reason the nce=0 run did. Also added --freeze-backbone-norm: freezing the detector's
# WEIGHTS does not freeze its BN running stats, which model.train() keeps updating from small
# batches (the documented BN-pollution trap). Verify in the log that the warm load reports 76
# detector tensors before trusting any number this produces.
set -euo pipefail
LOG="logs/hoct_finer122_nce.log"
uv run python -m celltrack.training.joint_detector \
  --head hoct --norm batch --downsample 1 2 2 \
  --detector-from finer122_coadapt_long.pt --slack-links --freeze-backbone-norm \
  --contrastive-weight 1.5 --hard-negative-weight 0.015 --temperature 0.07 \
  --epochs 12 --evals-per-epoch 3 --patience-epochs 3 \
  --batch-size 4 --weights hoct_finer122_nce.pt --device cuda \
  > "$LOG" 2>&1
