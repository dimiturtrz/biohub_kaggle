# celltrack — PLAN (source of truth)

This is the canonical plan. Read it at session start before proposing work. Substance lives here;
ordered execution status lives in `ROADMAP.md`.

## Goal

Track individual cells through 3D+time light-sheet microscopy of developing zebrafish embryos:
detect cell centers per timepoint and link them across time, including divisions (one parent → two
daughters). Target is the Kaggle competition
[Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(launched 2026-06-29, final submission 2026-09-29). Data is CC0 from the Royer group (Ultrack,
*Nature Methods* 2025).

## Headline metrics

The competition score IS the headline metric — no separate proxy. Micro-averaged across videos:

```
score = adjusted_edge_jaccard + 0.1 · division_jaccard
```

- **node matching** — optimal bipartite assignment on centroid distance, cutoff **7 µm**. Anisotropic
  in voxels: 7 / 1.625 = 4.31 vox in z, 7 / 0.40625 = 17.23 vox in y and x.
- **edge TP** — both endpoints matched to GT nodes joined by a GT edge. FP when a predicted edge hits
  a GT node connected to a *different* partner. Unmatched predicted nodes produce **no** FP.
- `edge_jaccard = TP / (TP + FP + FN)`
- `adjusted_edge_jaccard = max(0, jaccard · (1 − 0.1 · (T_pred − T_true) / T_true))` — over-detection
  penalty against the *estimated* true node count (annotated + unannotated), shipped per video in the
  GEFF attrs as `extra.estimated_number_of_nodes`.
- **division TP** — a predicted fork (node with ≥2 outgoing edges) inside a local window around a GT
  division, requiring parent-anchor match, two distinct daughters to distinct predicted branches,
  directed local topology, no cross-component conflict, unmerged branches.
- `division_jaccard = TP / (TP + FP + FN)` over bipartite-paired forks/divisions.

Metric source: [royerlab/kaggle-cell-tracking-competition `metrics.md`](https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md).

| metric | baseline | target | current |
|---|---|---|---|
| competition score | _(tbd — run the official U-Net+transformer baseline)_ | ≥ 0.929 (LB top, 2026-07-23) | — |
| adjusted edge jaccard | — | — | — |
| division jaccard | — | — | — |

**Three consequences the metric forces, all non-obvious** (verified against the organizers'
`src/tracking_cellmot/metrics.py`, not just the prose spec):

1. *The node-count ratio is signed and uncapped above.* `total_node_ratio = (N_pred − N_total)/N_total`
   with only a `max(0, …)` clamp, so predicting **fewer** nodes than the estimated true count pushes
   the multiplier above 1 and inflates the Jaccard — up to ×1.1. Detecting every cell earns ×1.0.
   The incentive is a high-precision detector run deliberately below the true cell count, not maximal
   recall. Caveat: a missed GT node is a full edge FN, which outweighs the ≤10 % bonus.
2. *Motion is the size of the matching tolerance.* 2.1 % of true links move more than 7 µm per
   timepoint (median 1.82, p99 8.38). A 7 µm-radius nearest-neighbour linker is capped near 0.98
   before model quality enters, and the linker's search radius must exceed the matcher's cutoff.
3. *Divisions are worth 0.1 max and are the thinnest signal.* 151 divisions in 199 training videos.
   Edge Jaccard dominates; division work is only worth doing once edge linking saturates.

Measured in [`interpretations/tracking/2026-07-23_dataset-survey.md`](../interpretations/tracking/2026-07-23_dataset-survey.md).

## Data

Local root: `paths.yaml` → `data:`; competition unzips under `<data>/raw/biohub_cell_tracking/`.

- **Images** — OME-Zarr v3, `(T, Z, Y, X) = (100, 64, 256, 256)`, `uint16`, blosc/zstd, chunked one
  timepoint per chunk. Voxel spacing 1.625 µm (z), 0.40625 µm (y, x) — **4× anisotropic**.
- **Tracks** — GEFF 1.1 (tracksdata), directed. Node props `t, z, y, x` as `int64` **voxel indices**;
  the µm scale lives in the GEFF axes metadata, so every metric-space computation must scale first.
- **Split** — 199 train videos (`.zarr` + `.geff` each), 4 test videos in the public download.
  Prefixes `44b6` (144) and `6bba` (258 across train+test) mark two source embryos/acquisitions.
- **Sparse GT** — hand-followed lineages, not labelled frames: 670 annotated nodes per video on
  average against ~23 700 real cells (6.1 % mean, 0.13–20 % range; the sparsest videos hold a single
  tracked cell). A correct prediction *will* contain unannotated cells, so unlabelled positives must
  be masked out of the loss rather than scored as background.
- **The prefixes are different acquisitions** — `44b6` averages 36.9 k cells at 0.99 % annotated,
  `6bba` 16.5 k at 9.0 %. CV must be grouped by prefix.
- Download is 87.6 GB / 24 886 files.

**Submission** — CSV: `id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`, `row_type ∈
{node, edge}`, unused fields `-1`. Coordinates in voxel indices, matching the GEFF convention.

## Approach

_(placeholder — to be argued from public evidence → our-data statistics → method.)_ Starting point is
the organizers' published baseline: 3D U-Net with temporal attention producing a per-voxel detection
map (cell centers via local-maximum suppression), plus a cross-attention transformer scoring node
pairs between consecutive frames, trained only on GT-annotated edges. Reproduce it first to get an
honest baseline number, then decide where the headroom is (detection recall vs. linking vs. gap
closing) from measured error decomposition, not from guesswork.

## Open questions

- Is this a code competition (notebook rerun, runtime cap) or a file-submission competition? The
  public download ships only 4 test videos, which suggests a hidden rerun test set — **unverified**;
  the API endpoints reachable with the current OAuth scope do not expose the rules.
- Prize structure — unverified.
- Where does `N_pred / N_total` actually maximise the score? The under-prediction bonus is arithmetic
  on the formula so far, not an experiment — sweep it once a detector exists.
- Does the test split contain divisions at all? `summarise()` drops the division term entirely when
  none are present anywhere, which would make the score pure adjusted edge Jaccard.
- Inter-cell spacing at a timepoint is still unmeasured — the GT is too sparse to give it (often one
  annotated cell per frame), so it needs a detector's output or the image itself. It sets the NMS
  radius and how often a detection can match the wrong GT cell.
