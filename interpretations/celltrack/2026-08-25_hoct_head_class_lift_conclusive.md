# HOCT edge head fixes bottleneck B — conclusive instrument LIFT, recall-cost caveat

2026-08-25. First conclusive win of a head class on the confusor axis (bottleneck B), measured on the
scale-free top1 ruler under a single-variable design.

## Design (single variable = head class)

`launch_hoct_real_converge.sh`: HOCT edge head trained on REAL (1,2,2) GT pairs, detector warmed FROZEN
from `finer122_coadapt_long.pt` (the converged STANDARD-head detector). `--detector-from` freezes the
detector + backbone norm, fresh-inits and trains only the head → head class is the ONLY difference.
Frozen detector is the standard/pack head's home ground (it co-adapted to that head), so a HOCT win here
is understated, not flattered (kneemri 0905z). Pre-registered LIFT bar (0952z): top1 ≥ +0.05 absolute
over the standard head's re-eval on the identical ruler; P_true ≥ 0.20 corroborating.

## Result — same ruler (TEST four, shipped flow linker, thr 0.970, frozen detector)

| head     | faithful | node R | mislinks | inverted | P_true | top1  |
|----------|----------|--------|----------|----------|--------|-------|
| STANDARD | 0.6877   | 0.9512 | 83       | 0.988    | 0.050  | 0.012 |
| HOCT     | 0.6432   | 0.8910 | 84       | 0.583    | 0.202  | 0.214 |

- **top1 0.012 → 0.214 (+0.202)** — clears the +0.05 bar 4×. The affinity ranks the confusor's true
  successor FIRST 18× more often.
- **inverted 0.988 → 0.583** — mislinks stop being systematically inverted; P_true 0.050 → 0.202.
- Head class CARRIES the confusor. This is bottleneck B moving for the first time on a non-inverting
  instrument, on the incumbent detector's own ground → conclusive.

## The caveat that keeps it from being a submission (yet)

faithful DROPPED 0.6877 → 0.6432 and node recall 0.9512 → 0.8910. Same frozen detector, so the recall
loss is the LINKER's: HOCT's sharper affinity makes the flow linker drop marginal-but-true edges it kept
under the standard head's softer scores. The linker's threshold/disappearance cost are calibrated for the
STANDARD head's affinity distribution, not HOCT's. Same tension made the training arm early-stop — as
node recall grew (0.872 → 0.909 over epoch 0.67 → 1.0) the instrument REGRESSED (top1 0.265 → 0.134);
best checkpoint is the early, lower-recall epoch.

So: B is solved on the instrument; the recall cost erases it at the shipped operating point.

## Live lever (mechanism-sound, no retrain)

Convert the instrument lift into a pipeline win by recalibrating the linker to HOCT's affinity —
lower detection threshold to restore recall, and/or blend HOCT's edge head with the packs' (ensemble) to
keep recall while adding confusor ranking. Sweep running (`logs/hoct_recall_sweep.log`). A config that
lifts faithful past the standard 0.6877 WHILE holding top1 >> 0.012 is a pipeline win driven by the B fix —
and B is exactly the LB-visible dense-regime axis (proxy-blind at the top). That would be the first
mechanism worth pricing against the held submission.
