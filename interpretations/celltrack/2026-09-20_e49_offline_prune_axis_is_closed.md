# E49 — the offline prune axis is closed: no cache-available feature set reaches the operating point

E48 killed two scores one at a time, and inferred from them that real-vs-FP status is not recoverable
from anything downstream of the detector. Two scores do not establish that. E49 gives the claim its best
shot instead of its weakest: **every** feature the cache and the volumes hold, fed jointly to a
gradient-boosted model, scored on **held-out movies** so nothing leaks across the split.
Instrument: `kaggle/prune_feature_ceiling.py`.

```
train 368975 rows / test 354158 rows on held-out movies   real 5557
JOINT model: AUC(real > FP) 0.6622
    collateral 0.02 -> FP-recall 0.0846     (E47 requires 0.75 at 0.02)
    collateral 0.05 -> FP-recall 0.1422
    collateral 0.10 -> FP-recall 0.2207
  alone  intensity     AUC 0.5515
  alone  best_out      AUC 0.6366
  alone  best_in       AUC 0.6226
  alone  margin        AUC 0.6170
  alone  degree        AUC 0.4278
  alone  crowding      AUC 0.4815
  alone  support_prev  AUC 0.4837
  alone  support_next  AUC 0.4838
```

**The joint model buys +0.026 AUC over its best single feature and lands 9× short of the operating
point.** Eight features, a model free to combine them nonlinearly, and FP-recall at the required 2 %
collateral goes 0.0578 → 0.0846 against the 0.75 that E47 says the axis needs to pay. That is not a
tuning gap; it is a different order of magnitude, and it is measured on movies the model never saw.

## The nulls are the informative part

Four of the eight features sit at or below chance: degree 0.4278, crowding 0.4815, temporal support
0.4837 / 0.4838. Temporal support is the one that matters, because it was the strongest remaining
candidate — "a spurious detection should have no continuation" is exactly the intuition `kiw1` runs on.
It carries nothing, and E46 says why: the confusor is a **near-static lookalike sitting where the cell
was**, so it has the same close neighbour in t−1 and t+1 that a real cell has. The pairwise-unresolvable
finding and the temporal-support null are the same fact seen twice.

The affinity family (best_out 0.6366, best_in 0.6226, margin 0.6170) carries essentially all of the
available signal, and they are near-duplicates of one another — which is why eight features barely beat
one.

## What this closes, and what it does not

**Closed:** pruning our tunet candidate set with information available *after* detection. Threshold,
affinity filter, crowding rule, temporal-consistency filter, and any learned combination of them — all
sit far below the operating point E47 derived. No further offline prune variant deserves wall-clock, and
any new proposal should be required to state its FP-recall at 2 % collateral before it is built.

**Not closed:** bd `g89y`. Hierarchy selection does not score *the detections we already have* — it
chooses among **nested contour hypotheses that this candidate set never contained**, so it is outside
this measurement's reach rather than refuted by it. E49 sharpens its precondition into the sharpest form
yet: the hierarchy must supply information that is *not a function of the existing detections' position,
intensity, crowding, temporal support, or affinity field*, because all of those are now measured and
jointly insufficient. That is a strong requirement, and it should be tested before the build.

**Also not closed:** the detector's own confidence score, still absent from the cache (`coords` is
`[t,z,y,x]`); center-voxel intensity is a proxy for it. Worth one cheap read on the next GPU pass.

Converges with `celltrack-true-bottleneck-is-dense-regime-model-gap` and
`celltrack-selection-is-binding-not-detection` from a third direction: the lever is the detector, not
anything reading its output.
