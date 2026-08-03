# Kaggle Biohub Cell Tracking: Metric Scoring & Test Split Resolution

**Date**: 2026-08-03
**Status**: partial
**Supersedes**: none

## TL;DR

Metric confirmed: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`, where `adjusted_jaccard = max(0, jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))` and `T_true` is per-movie total node estimate [S2]. Division Jaccard exploit (synthetic hub node outside image) was patched post-discovery and all submissions re-scored [S1]. **Test set overlap with train remains UNCONFIRMED** — no public source found confirming whether the 4 test movies appear in train data with sparse annotations [OPEN].

## Question

1. How exactly is the public/private test split constructed — same movies with different annotations, or different movies/timepoints?
2. What are the exact metric edge-cases, scoring code paths, and node-count ratio definition?
3. Do the 4 test movies (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1) appear in training data?
4. What is `estimated_node_count` and how is it defined per-movie?
5. Is there discussion of node-count bonus exploitation or leaderboard shakeup risk?

## Findings

### Metric Formulas (Confirmed)

- **Edge Jaccard**: `edge_jaccard = TP / (TP + FP + FN)` [S2]
- **Adjusted Edge Jaccard**: `adjusted_jaccard = max(0, jaccard * (1 - a * (T_pred - T_true) / T_true))` where `a = 0.1` [S2]
- **Division Jaccard**: `division_jaccard = TP / (TP + FP + FN)` [S2]
- **Final Score**: `score = adjusted_edge_jaccard + w * division_jaccard` where `w = 0.1` [S2]

### Node Count Penalty (T_true / estimated_node_count)

- `T_true` is described as "a provided coarse estimate of the total number of true nodes (including those the ground truth doesn't annotate)" [S2], making it a **per-movie TOTAL count** estimate, not a per-timeframe measure.
- The adjusted Jaccard formula penalizes over-prediction: if `T_pred > T_true`, the Jaccard is multiplied by a factor `< 1`, scaling the penalty by `0.1 * (T_pred - T_true) / T_true`.
- This creates an asymmetric bonus structure: under-predicting reduces false positives (higher Jaccard) but risks missing nodes; over-predicting lowers the Jaccard multiplicatively. [S2]
- No forum discussion found confirming whether participants actively exploit or cap predictions near `T_true` [OPEN].

### Division Jaccard Exploit & Patch

- **Pre-patch behavior**: Division matching relied on weakly-connected-component structure in the predicted graph, not strict parent→two-daughter fork validation against ground truth [S1].
- **The exploit**: Participants added synthetic graph structure far outside the image volume — a single "hub" node at coordinates like (−10000, −10000, −10000) with edges to the root of every (or largest N) track components, plus a handful of synthetic fork chains at those same far-away coordinates. This merged the whole prediction into one weakly-connected component and artificially inflated division_jaccard from ~0 to ~1.0, adding close to the full +0.1 bonus to the final score, despite zero biological correspondence [S1].
- **Patch**: After discovery, a patch was implemented requiring divisions to be genuine strongly-connected parent→two-daughter structures matched within a 7 µm distance radius. All submissions were re-scored; submissions not actively exploiting the vulnerability were unaffected [S1].
- **Status**: Patch publicly available in the GitHub repository; the exploit is now closed [S1, S2].

### Test Set Composition & Public/Private Split

**Confirmed facts**:
- Train directory contains `{name}.zarr` images + paired `{name}.geff` ground-truth track graphs [S2].
- Test directory contains `{name}.zarr` images only (no ground truth for submission) [S2].
- Data format: OME-Zarr (T, Z, Y, X) with scale (1.625, 0.40625, 0.40625) µm/pixel; tracks as GEFF sparse spatial graphs with nodes (t, z, y, x) and temporal edges [S2].
- Ground truth annotations are sparse — only a subset of cells annotated per video [S2].

**Unconfirmed assumptions**:
- Your working hypothesis: "the 4 test movies (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1) are ALSO in the train set with sparse annotations, and public/private splits their ANNOTATIONS (not different movies)" — **NOT VERIFIED** from any primary source. [OPEN]
- Public LB = 29% of test, private = 71% — **NOT VERIFIED** from competition docs, forum, or host announcement found. [OPEN]
- **No forum or host statement found** explicitly stating whether test movies appear in training data or how the 29/71 split is constructed. [OPEN]

### Data Structure & Node Estimates

- Each dataset is a `{name}.zarr` + `{name}.geff` pair. Divisions encoded as one source node at t with two target edges at t+1 [S2].
- `estimated_node_count` (T_true) is **per-movie**, not per-timeframe. No details found on how it is computed (actual count, ground-truth count, or statistical estimate) [OPEN].

### Node-Count Bonus Exploitation

- The adjusted Jaccard structure means over-predicting nodes incurs a multiplicative penalty. No public forum discussion or exploit report found confirming teams actively cap predictions near T_true or discuss leaderboard shakeup risk from this bonus [OPEN].

## Open Questions

1. **Test set origin**: Are the 4 test movies present in the training directory with sparse annotations, or are they held-out unseen data?
2. **Public/private split mechanism**: How is the 29%/71% split actually constructed? Different movies, or same movies with annotation subset?
3. **T_true computation**: How is `estimated_node_count` calculated per movie? Actual GT count, field-specific average, or participant estimate?
4. **Leaderboard shakeup risk**: Has the private test set shown significant score divergence from public, and if so, what caused it?
5. **Node-count bonus exploitation**: Do top teams deliberately cap predictions near T_true to leverage the penalty asymmetry? Is this documented in solutions?
6. **Division matching edge-cases**: What is the exact "7 µm distance radius" matching criterion post-patch? Is it volumetric distance, temporal window, or both?

## Sources

- [S1] Kaggle discussion + earlier search results: Division Jaccard exploit discovery — synthetic hub node outside image coordinates at (−10000, −10000, −10000) with edges merging components and synthetic fork chains inflating score from ~0 to ~1.0. Patch re-scored all submissions. URL: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/727154
- [S2] GitHub metrics.md: Exact formulas for adjusted_jaccard, division_jaccard, combined score. T_true definition as "coarse estimate of total true nodes including unannotated." Data format (OME-Zarr, GEFF, sparse annotations). URL: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- [S3] GitHub repository README (raw fetch): Data structure (paired .zarr + .geff), train/test split (ground truth in train, none in test), no mention of test-train overlap. URL: https://github.com/royerlab/kaggle-cell-tracking-competition
- [S4] Image.sc Forum announcement: Competition launched June 29, 2026; largest publicly available cell tracking dataset (CC0), zebrafish embryo videos, sparse annotations. URL: https://forum.image.sc/t/biohub-cell-tracking-during-development-kaggle-competition/121671
