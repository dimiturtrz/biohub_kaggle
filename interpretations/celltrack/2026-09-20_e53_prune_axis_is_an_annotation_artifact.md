# E53 — the prune axis was never a tracking axis: unannotated detections are metric-invisible except through the node-count bonus

**Verdict: read from the official scorer's own source. A predicted edge is TP or FP only if an endpoint
matched a GT node; an edge between two unannotated detections is NEITHER, so it does not enter the
jaccard at all. The only channel by which a spurious detection touches the score is the node-count term
`adjusted = jaccard * (1 - 0.1 * (t_pred - t_true) / t_true)`. E47's +0.1139 "oracle prune" gain is
therefore mostly that multiplier, not better tracking — and the label a pruner would have to learn is
"did a human annotate this cell", which on two of four movies is a thin slab in y.**

## The scorer, primary source

`external/frontier_ds/biohub-temporal-unet3d-seed314159-v1/repo/src/biohub_tracking/metrics.py:55-114`
and the champion's faithful reimplementation
`kaggle/kernels/celltrack-ilp-faithful-0923/champion_submission.py:3584-3620`:

    is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
    is_fp = (mt is not None and mt in gt_incoming_source) or (ms is not None and bool(gt_outgoing.get(ms)))

`ms`/`mt` are the GT nodes the predicted endpoints matched. Both `None` → the edge is counted **nowhere**.
Upstream this is the same rule, spelled `pred_valid = out_valid | in_valid` with `fill_null(False)`.

So an FP detection is free unless it sits within the match radius of an annotated cell. What is *not*
free is the node count:

    adjusted_jaccard = max(0, jac * (1 - a * (t_pred - t_true) / t_true)),   a = 0.1

`t_true` is the dataset's own **estimated true node count**, read from the geff
(`read_estimated_true_node_count`) — not the annotated count. The term is **signed**: `t_pred < t_true`
multiplies the jaccard **up**.

## Where the champion already sits

| movie | t_pred | t_true | multiplier |
|---|---|---|---|
| 44b6_0113de3b | 25 637 | 25 755 | 1.0005 |
| 44b6_0b24845f | 20 729 | 32 795 | **1.0368** |
| 6bba_05b6850b | 6 152 | 6 362 | 1.0033 |
| 6bba_05db0fb1 | 70 290 | 69 800 | 0.9993 |

The champion is at or below the estimated true count on three of four movies. There is no over-detection
penalty left to recover.

## What this re-reads in E47-E49

- **E47's oracle gain (+0.1139) is largely the multiplier, not tracking.** Deleting every unmatched
  detection leaves the jaccard untouched (those edges were invisible) and lifts the multiplier by
  `0.1 × 0.98 ≈ 0.098`. The number was right; its *mechanism* was mis-attributed to recovered edges.
- **E48/E49 asked for an impossible label.** Their positive class was "matched to an annotated GT node";
  98.4 % of rows were negatives (5 557 real of 354 158). But an unmatched detection is usually a **real
  cell nobody labelled**, so the feature being sought is not "is this a false positive" — it is "did an
  annotator pick this cell". AUC 0.6622 against that target is not a statement about detection quality.
- **E52's SHORT5 arithmetic now closes numerically too.** Removing 11 343 nodes (9.2 %) buys at most
  `0.1 × 0.092 ≈ +0.009` — under the 0.01-0.02 noise floor — while its 1.18 % collateral costs ~2× that in
  destroyed true edges. Net negative before any feature quality enters.

## Is the label even learnable

Spatial spread of matched-GT detections vs all detections (µm, per axis):

| movie | GT std | all std |
|---|---|---|
| 44b6_0113de3b | 30.3 / **5.7** / 16.6 | 30.2 / 30.7 / 25.7 |
| 44b6_0b24845f | 19.1 / **8.4** / 15.8 | 33.9 / 24.9 / 27.8 |
| 6bba_05b6850b | 36.3 / 30.2 / 31.0 | 32.9 / 27.5 / 30.3 |
| 6bba_05db0fb1 | 19.0 / 23.1 / 27.3 | 26.7 / 27.1 / 29.7 |

On the two sparsest movies the annotation is a **thin slab in y**, ~5× narrower than the detections — so
"which cells are annotated" is partly predictable from position alone. On the two denser movies it is not.

**We are not building that.** Predicting the annotator's slab in order to delete real cells for the
node-count bonus improves no tracking; it games the sparse-GT structure of the metric. It is recorded
here because it explains the numbers, and because any future prune proposal has to state which of the two
channels it claims — recovered edges (real) or the node-count multiplier (not).

## Standing consequence

**The offline prune axis is closed, and for a better reason than E48/E49 gave.** It was never measuring
detection quality. Future work on false positives must be priced on edges that touch annotated cells —
the mislink/confusor problem (E45/E46) — which is where the remaining real headroom lives.
