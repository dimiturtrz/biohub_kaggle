# E48 — the diagonal, measured: neither score we already have can reach E47's operating point

E47 priced the FP-prune axis and found the binding constraint is collateral: a pruner needs FP-recall
≈ 0.75 at ≤ 2 % collateral to earn +0.035, and at 5 % collateral even a *perfect* FP pruner nets only
+0.021. It then *argued* that a threshold fails because its decision variable is intensity-ordered and so
buys FP-recall and collateral along the same axis. That was an argument. This measures it.

Instrument: `kaggle/prune_score_quality.py`, CPU-only, on the same caches. Two rankings — the only two
the cache and the volumes already provide — scored on separability and, decisively, on the FP-recall they
deliver **at** the required collateral (the cut is placed on the real detections' own quantile, so
collateral is exact by construction).

```
detections n=723133   real 12482   FP 710651
center-voxel intensity: AUC(real > FP) 0.5221
    collateral 0.02 -> FP-recall 0.0222     (E47 requires 0.75 at 0.02)
    collateral 0.05 -> FP-recall 0.0476
    collateral 0.10 -> FP-recall 0.1011
best incident affinity: AUC(real > FP) 0.6380
    collateral 0.02 -> FP-recall 0.0578
    collateral 0.05 -> FP-recall 0.1168
    collateral 0.10 -> FP-recall 0.1964
```

## The diagonal is not a metaphor

For intensity, **FP-recall equals collateral to three decimals at every cut** (0.0222 at 0.02, 0.0476 at
0.05, 0.1011 at 0.10). That is the q = r diagonal exactly, and it is what an AUC of 0.5221 means in
operating terms: the score is a coin flip between a real cell and a false one. Every threshold move
trades one for the other at par, and E47's grid says every point on that diagonal is a loss. The
LB-refuted recovery stack (≤ 0.900) sat there by construction — this upgrades E47's retro-explanation
from mechanism-shaped argument to measurement.

Affinity is genuinely better (AUC 0.6380) and still nowhere near: **0.0578 against the 0.75 required, a
13× shortfall**. This was pre-registered as a failure before it ran, and for a stated reason — the
E46 lookalike's signature is *high* affinity to the source, since it sits where the cell was. A score
that ranks detections by how much the tracker wants to link them cannot demote the one detection the
tracker most wants to link.

## What it means, and what it does not

The strong reading: on our tunet caches, a detection's real-vs-FP status is **not recoverable from
anything downstream of the detector**. That is the same conclusion as
`celltrack-true-bottleneck-is-dense-regime-model-gap` and `celltrack-selection-is-binding-not-detection`,
reached from a new direction — cite the convergence, do not claim it as new.

For bd `g89y` this sharpens the precondition E47 named rather than closing it. Hierarchy selection is
still the one candidate whose decision variable is not forced onto the intensity axis, but E48 raises the
bar it must clear: it has to bring information that neither raw intensity nor learned edge affinity
carries, and the number to beat is not "better than chance" but **FP-recall 0.75 at 2 % collateral**.
Any prune proposal should now be asked for that one number first; it is cheap to estimate and it has
killed two scores in an afternoon.

## Caveats

- **The detector's own confidence score is not measured here.** The cache carries `coords` (`[t,z,y,x]`)
  and edge affinity only — no per-detection score column. The recovery stack thresholded the *detector
  score*; center-voxel intensity is a proxy for it, not it. So this refutes the intensity axis directly
  and the score axis by analogy. Getting the real score means a detector pass (GPU) — worth one cheap
  run whenever the GPU is next in use, not worth waking it for.
- **Our tunet substrate**, FP field 0.972 of all detections. On 0947's sparser field the absolute
  FP-recall numbers move; the diagonal argument does not.
- Center-voxel value, not an integrated or peak intensity over the cell body. A better intensity feature
  would move 0.5221 somewhat — it will not move it to the 0.9-plus that the operating point implies.
