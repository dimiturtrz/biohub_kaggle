# GT-free blob/peak merge referee — INCONCLUSIVE (finer-field spikiness confounds static readouts)

2026-08-24. Built `celltrack/analysis/merge_referee.py` to answer, GT-free: does the (1,2,2) finer
decode un-merge the crowded cells that (1,4,4) fuses? GT can't referee this — annotation enforces
>4µm inter-cell spacing, so the merge regime holds zero labelled pairs (dng6 premise).

## Data
Same weights (`pilkwang_seed1`), same dense movie (`44b6_e57ff5c6`), two cached decode grids already on
disk (no card, no inference): `logit.ds1x4x4` (64³) and `logit.ds1x2x2` (64×128×128, 2× finer in y,x).

## Three readouts, all confounded

**Basin volume (connected components > prob floor).** Finer grid: +60% basins, merged-mass frac
0.837→0.799. Looks like un-merge — but band-resolving kills it: the +60% is **fragmentation-dominated**
(+9001 sub-0.5·V0 fragment basins carrying +2.6% mass; the real single-cell band gains only +572).
Basin volume is not scale-invariant across decode grids — the finer field sharpens (median physical V
272→108), producing tighter cores + broken halos, which reads as "more basins" independent of any merge.

**Peak count (local maxima after NMS — the detector's own count).** At a *matched* 1.6µm NMS the finer
grid still yields **+47–61% more peaks**, and this gap does NOT converge under thresholding (floor
0.5→0.99). The finer decode is intrinsically spikier at every physical scale; the sub-1.6µm "un-merge"
gain (+2000–4400 peaks) is swamped by it.

**Spatial locality (are extra fine peaks doublets near coarse peaks, or scattered?).** Only 34–35% of
fine peaks fall within 1.6µm of a coarse peak. A clean un-merge would be ~100% (split doublets hug their
parent blob). BUT 34% vs ~1.5% chance (241 coarse peaks/frame, 1.6µm ball in 64×256×256) means the fine
peaks ARE correlated with coarse detections — real structure, not pure noise, just not clean enough to
arbitrate.

## Verdict
The GT-free static referee is **inconclusive**, not a clean GO/NO-GO. Real un-merge signal is present but
buried under the finer decode's field spikiness, and no static readout (volume, count, locality)
separates them at the dense-movie scale. This is the faulty-TEST branch: the ruler can't see the axis.
**LB remains the only arbiter of the finer-decode lever**, consistent with proxy inversion above 0.90.

Caveat that keeps the lever alive: this tested the finer *decode* of frozen pilkwang weights (an oracle),
NOT the *trained* `joint_finer122_v1.pt` (a model trained at 1,2,2, +0.18 node R). A trained model places
mass differently than a finer decode of coarse-trained weights, so this does not refute finer122 itself —
it refutes the cheap offline referee that was meant to gate it. dng6 (referee) → closed inconclusive.

Tool kept: `merge_referee.py` (per kneemri's "keep the constant in a tool, not a doc" — this is now a
re-runnable instrument, not a number in prose).
