# Operating-point sweep: under-detection collapses the score, and recall isn't the score

*2026-07-23 — `biohub_kaggle-sct`, learned detector + NN linker on validation fold 0, two videos per
acquisition.*

## The curve

End-to-end competition score (adjusted edge Jaccard + division term) as the detector's node budget varies
as a fraction of each video's estimated true cell count:

| N_pred / N_total | 44b6 | 6bba |
|---|---|---|
| 0.50 | 0.243 | 0.300 |
| 0.75 | **0.455** | 0.468 |
| 1.00 | 0.454 | **0.479** |
| 1.25 | 0.454 | 0.467 |
| 1.50 | 0.454 | 0.455 |

## What it settles

**Under-detection does not pay.** Halving the node budget collapses the score on both acquisitions
(0.243, 0.300) — the node-count bonus from predicting fewer nodes is buried by the edges those missing
nodes take with them as false negatives. This is the empirical confirmation of the arithmetic in
`2026-07-23_metric-shape.md`: the earlier PLAN reading that an under-counting detector wins the ≤×1.1
bonus was backwards. The bonus is real but small and always dominated by the recall it costs.

**The operating point is ≈ 1.0, with a per-acquisition shape.** Sparse 44b6 is flat from 0.75 upward
(0.454–0.455) — once enough of its few annotated cells are covered, extra predictions in its vast
unannotated tissue are free, so overshooting costs almost nothing. Dense 6bba peaks cleanly at 1.0 and
declines above it (1.5 → 0.455), because with 9 % annotated the node-count penalty actually bites. So the
safe single choice is **N_pred/N_total = 1.0**; a per-acquisition policy would push 44b6 slightly lower
(~0.75) for a marginal gain. Detecting *to* the estimate, not below it, is the rule.

## The question the sweep surfaced

The learned detector's end-to-end score here (~0.45–0.48) sits **below** the classical floor's ~0.56 from
the bracket — even though the learned detector's *recall* beat the classical one (`3ws`: 0.766 vs 0.656 on
44b6, 0.742 vs 0.664 on 6bba). Higher per-frame recall produced a *lower* tracking score. That is a
question, not a verdict (per the project's own rule on physically-sound methods that score worse).

The most likely mechanism: the NN linker rewards temporal *consistency*, not per-frame recall. A detector
that finds more cells but localises each one a little differently frame to frame gives the linker worse
edges — the extra detections are real cells, but their centres wander, so the one-to-one assignment picks
wrong successors. The classical LoG, for all it misses, places what it finds at a stable intensity
maximum. Two caveats keep this tentative: the bracket floor and this sweep are different video samples
(not a matched A/B), and the detector saw only 2000 untuned steps.

## What to do with it

- **`sct` is met:** the curve, the operating point (1.0), and the finding that the bonus does not survive
  under-detection are recorded.
- **The next lever is detection *consistency*, not just recall** — or a linker that tolerates jitter.
  Concretely: a matched learned-vs-classical end-to-end comparison at the 1.0 operating point (same
  videos), then either more/steadier detector training or the learned linker (`cdo`) that can absorb
  noisy centres. The recall win is real; converting it to a score win is the open work.
- **Do not chase the node-count bonus.** Operate at the estimate; spend the effort on detection quality
  and linking, where the sweep says the score actually lives.
