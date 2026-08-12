# The dense mislinks are a wall — closed from seven directions

*2026-08-12 — analysis only. Tools: `motion_statistics`, `drift_field`, `appearance_identity`,
`trajectory_prediction`. NO score change: everything here is diagnostic and every lever it tested is refuted.*

## The bottom line first

This session added no score. The public best is **0.899** (flow + fusion + symmetry-division) before and after.
The drift field, motion statistics, Kalman/trajectory criterion, and full-resolution appearance were all
**diagnostic**, and all of them **refuted** the lever they tested. What the session bought is a *negative*: the
38 dense mislinks — the association failure that carries essentially all of the dense movie's lost score — are
**unfixable from the data**, proven from seven independent directions. That stops the campaign chasing ~0.02 of
permanently unrecoverable score.

## The ledger — every cue tested on the 38 dense mislinks

| cue | result |
|---|---|
| distance (per-frame) | inverted — decoy nearer by +6.44 µm median |
| learned affinity | inverted — P_chosen 0.686 vs P_true 0.134, 100% |
| bidirectional agreement | shipped, does not rescue them |
| global one-parent flow | +0.0007 (flat) |
| full-resolution raw appearance | below chance (in-gate top-1 0.115 vs 0.260) |
| tissue drift field | 0/37 rescued |
| **oracle trajectory (Kalman)** | **0/35 rescued** |

The true successor is the anomalous choice on *every physical axis in the data*. That is the definition of a
wall — not "we could not find a trick" but "the information is not there".

## Why the Kalman criterion — the last one — fails, at its ceiling

The natural fix for a fast mover is to replace distance with deviation from the *predicted* position: advance
the source by its velocity, then score candidates by distance from that prediction. Measured with the source's
**ground-truth** velocity (an oracle — the best any filter could do):

```
                        |velocity|   align cos(v, true step)   rescued
mislinks  raw            0.91 µm       +0.00                    0/35
mislinks  drift-removed  1.58 µm       −0.37                    1/35
correct   (control)      1.68 µm       +0.35                    —
```

The mislinked cells are **not persistently-moving cells**. Two failures at once:
1. The incoming velocity is small (~1 µm), not the 3.3 µm of the outgoing step — the "fast" step is mostly
   shared tissue drift, which moves every candidate alike and cannot disambiguate.
2. The velocity is **uncorrelated** (cos +0.00) or **reversed** (−0.37 drift-removed) with where the cell
   actually went — against +0.35 persistence on correct edges. The past does not point at the successor.

So "cost = deviation from predicted" cannot help: the prediction is a random ~1 µm nudge in the wrong
direction, not a lunge toward the true successor. No filter recovers a trajectory that is not there.

## What that −0.37 also says: some are probably annotation errors

The reversal is the "jump out and back" signature. At the mislinked edges the annotated successor sits *against*
the cell's own motion, while distance, affinity and appearance all prefer the alternative. When four independent
signals agree the annotated choice is the odd one out, the parsimonious reading for some fraction is that the
label is wrong. This does not need to be resolved to act on it — annotation error and irreducible ambiguity have
the same consequence: not a lever.

## How much the wall costs

```
dense movie   shipped jaccard 0.903 -> oracle (all mislinks fixed) 0.989   = 0.086 on dense
per mislink   ~0.0023 dense jaccard
4-movie score dense is 1 of 4, others near-perfect on association       ~= 0.021 pooled
```

So the mislinks cost **~0.02 of the pooled score, permanently**. It is the floor, not headroom. For scale, that
is about the whole gap to the reproducible frontier (~0.915) — but the frontier is walled on these too, so its
lead comes from other axes, not from fixing mislinks.

## Where score can still come from (none of it is here)

The transferable lever this whole campaign has is **decision-rule changes** (greedy→global +0.005, flow+fusion
+0.004, symmetry-division +0.004). Recall-side and cost-tuning tweaks anti-transfer. The remaining documented
same-weights path is the frontier bundle (bead 73g): the ILP-global division-aware linker plus the output
repairs (enforce-next-frame, single-parent, prune-isolated) that took the public clean baseline to ~0.908 on
the *same pilkwang weights we mount*. That is the next real submission candidate. The dense-association lever,
by contrast, is now closed.

## What was kept

Four diagnostic tools, all default-nothing and reusable: the motion-structure/Fürth fit, the drift-field
estimator (validated, +0.43 explained variance out-of-sample, over-correction failure mapped), the
full-resolution appearance gate, and the oracle trajectory criterion. None ship; all are the instruments that
will re-answer these questions if the assets change (a finer detector, a different resolution, denser labels).
