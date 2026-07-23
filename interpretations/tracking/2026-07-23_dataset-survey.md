# What the training data actually is (2026-07-23)

Survey of all 199 annotated training videos: `python -m celltrack.explore --split train --limit 199`.
Reader code in `core/data/`; metric read from the organizers' `src/tracking_cellmot/metrics.py`
(cloned at the v1 competition release).

## The ground truth is a thin sample of lineages, not a labelled frame

| | all 199 | `44b6` (71) | `6bba` (128) |
|---|---|---|---|
| annotated nodes / video (mean) | 670 | 284 | 884 |
| estimated real cells / video (mean) | 23 700 | 36 887 | 16 454 |
| annotated fraction (mean) | 6.1 % | 0.99 % | 9.0 % |
| divisions / video (mean) | 0.76 | 0.37 | 0.98 |

Totals: 133 318 annotated nodes, **151 divisions** across the whole training split.

Per-video the annotated fraction spans 0.13 % to 20 %. The sparsest videos carry a *single* tracked
cell — ~50 nodes, one per timepoint, no branch. So the supervision is a set of hand-followed lineages
through a dense field of 16–37 k cells, not a segmented frame with a few holes.

**Consequence for training.** A per-voxel detection loss over these labels is ~94 % wrong by
construction: almost every unlabelled bright blob is a real cell. The organizers' baseline sidesteps
this by backpropagating only on annotated edges. Any detector trained here needs the same treatment —
unlabelled positives must be masked out, not scored as background.

## The two prefixes are different acquisitions

`44b6` has ~2.2× the cell count of `6bba` at ~1/9 the annotation density. That is not sampling noise
across 71 vs 128 videos; it is a different developmental stage or imaging setting. **CV must be
grouped by prefix**, and the over-detection penalty (below) has a different operating point in each.

## Motion is the same size as the matching tolerance

Per-link displacement, micrometres per timepoint, over all 133 k annotated links:

| median | p90 | p99 | max |
|---|---|---|---|
| 1.82 | 4.14 | 8.38 | 60.8 |

**2.1 % of true links move further than the 7 µm node-matching cutoff.** Two things follow. A
nearest-neighbour linker with a 7 µm radius has a hard ceiling around 0.98 edge Jaccard before any
model quality enters. And the linking radius is *not* the matching cutoff — it has to cover the p99
tail (~8.5 µm) and up, so the linker sees several candidates per cell and must actually discriminate.

## The node-count penalty rewards under-prediction

From `metrics.py`:

```
total_node_ratio = (N_pred − N_total) / N_total          # signed, NOT clamped
adj_edge_jaccard = max(0, edge_jaccard · (1 − 0.1 · total_node_ratio))
score            = adj_edge_jaccard + 0.1 · division_jaccard
```

`N_total` is the GEFF `estimated_number_of_nodes` — the estimate of *all* real cells, ~23 700 per
video, against ~670 annotated ones. The clamp is only at zero from below; there is no cap at 1.0.
Predicting **fewer** nodes than `N_total` makes the multiplier exceed 1 and *inflates* the Jaccard, up
to ×1.1 as `N_pred → 0`.

This inverts the naive strategy. Detecting every cell in the field (`N_pred ≈ N_total`) earns
multiplier ≈ 1.0. Detecting only a well-linked subset earns a bonus on top of whatever Jaccard that
subset achieves. Since edge TP only requires that the *annotated* lineages be matched and correctly
linked, and unmatched predicted nodes generate no edge FP, the incentive is a **high-precision
detector deliberately run below the true cell count** — the opposite of maximising recall.

Two caveats before this becomes a plan. Under-detecting risks dropping the annotated cells themselves
(a missed GT node costs a full edge FN, which is far more expensive than the ≤10 % bonus). And the
aggregation weights each video's adjusted Jaccard by its own `TP+FP+FN`, so the bonus is worth most on
the videos with the most annotated links — which are the `6bba` ones, exactly where `N_total` is
smallest and the ratio moves fastest.

## Open, unmeasured

- Where the operating point actually sits: sweep `N_pred / N_total` against realised score once a
  detector exists. The argument above is arithmetic on the formula, not an experiment.
- Divisions: 151 in 199 videos is thin for training a division head, and the term is worth ≤ 0.1 of
  the score. Confirm the test split even contains divisions — `summarise()` drops the term entirely
  when none are present anywhere.
