# What the metric actually rewards

*2026-07-23 — from building the local evaluator (`biohub_kaggle-92j`) against the organizers'
`tracking_cellmot` (pinned `075fc5f5`).*

## What was built

`core/metrics/` scores a predicted track graph the way the leaderboard does: per-timepoint optimal node
matching at 7 µm in physical space (`matching.py`), edge TP/FP/FN against a sparse annotation
(`edges.py`), the local-window division rules (`divisions.py`), and the two aggregations (`score.py`).

**Agreement:** exact — all six counts, on all 158 frozen cases, 88 of which contain divisions.
Cases use integer voxel coordinates at the real anisotropic spacing (1.625 / 0.40625 / 0.40625 µm), and
include 60 deliberately crowded ones where nodes sit within the cutoff of several rivals, plus 80 with
dividing lineages and forks displaced a timepoint either way. Fixtures are committed
(`tests/assets/metric_*.json`), so the check runs without the organizers' package.

Getting there required copying two things that are not in the prose spec and would have diverged
silently:

- the assignment maximises `Σ 1/(1+d)`, **not** minimises `Σ d`. The two disagree exactly where cells
  are crowded, which is where the score is decided.
- the solver chain is full-bipartite → pad-empty-rows-and-columns → dense assignment, and *which*
  fallback fires changes the matching when no assignment covers every node.

## The finding: recall in unannotated tissue is free

A predicted edge is only counted when one endpoint matches a GT node the annotation continues through
(`metrics.py:193-198`, `pred_valid = out_valid | in_valid`). Endpoints that match nothing make the edge
invisible — not a false positive, simply ignored. The ground truth annotates **6.1 % of cells on
average**, so the overwhelming majority of predicted links are unpriced.

Verified rather than reasoned: the `free_edges_in_unannotated_tissue` case adds a whole spurious
two-node track to a perfect prediction and the oracle returns `fp = 0`.

The only price for predicting too much is the node-count term, `max(0, J·(1 − 0.1·(N_pred−N_total)/N_total))`.

## This inverts the operating point I had written into PLAN

The earlier reading was that, because the ratio is signed and clamped only from below, under-predicting
multiplies the Jaccard by up to 1.1 and the incentive is a high-precision detector run below the true
cell count. The bonus is real, but the trade is not:

| direction | node-count term | edge term |
|---|---|---|
| overshoot by `f` | −`0.1·f` | recall gained at full value; extra links unpriced |
| undershoot by `f` | +`0.1·f` | −≈`f`, since every unfound node's links become FNs |

Undershooting pays `0.1·f` to lose `f`. **Maximal recall is right; the operating point sits at or a
little above `N_total`**, and how far above is set by how fast recall still climbs — worth 0.1 per unit
of ratio. That is the sweep in `biohub_kaggle-sct`, and it is now cheap to run.

Two caveats on the table, both arguing for measuring rather than trusting it:

- `f` is a fraction of *estimated total* cells, but the edge loss falls on *annotated* ones. It is only
  ≈`f` if detections are lost uniformly across annotated and unannotated cells; a detector that fails
  preferentially on dim or crowded cells breaks that, in either direction.
- the adjusted Jaccard is weight-averaged per video by `TP+FP+FN`, so the arithmetic is dominated by the
  densely annotated `6bba` acquisition (9.0 % annotated) rather than `44b6` (0.99 %) — and those are
  also the videos where `N_total` is smallest, so the ratio moves fastest there.

## What this means for the model

Detection should be tuned for recall, not precision, and the threshold chosen by the node-count sweep
rather than by a detection F1. Linking should propose generously: a wrong link between two unannotated
cells is free, so the linker's real cost is wrong links *at* annotated cells and missed true ones.

None of this is a licence to spam nodes — every predicted node counts in `N_pred` whether or not its
links are ever priced.
