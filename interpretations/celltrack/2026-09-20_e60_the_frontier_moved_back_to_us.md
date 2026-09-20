# E60 — six top-voted kernels audited: five are our champion with worse constants, one is a closed axis

2026-09-20 · CPU only, no GPU, no submission · `research/frontier_kernels/` (six new dirs)

## Why look at all

The Einstein criterion says check what was tried before pulling anything new. The public list had six
top-voted kernels we had never harvested — 263, 97, 82, 78, 58 and 30 votes. Votes are popularity, so the
first question for each is not "what does it do" but **"what does it set that our banked 0.947 does not"**.
Every kernel in the 0947 lineage is driven by `BIOHUB_*` environment constants, so that question is a diff,
not a read: extract each kernel's env assignments and `comm` them against the champion's.

## The diff

| kernel (author, votes) | sets that 0947 does not | lacks that 0947 has |
|---|---|---|
| `biohub-harmonic-fusion` (flexonafft, 263) | `DET_THRESHOLD=0.965`, `PPSWEEP_SELECT_MARGIN=0.001` | `DET_THRESHOLD=0.960`, `MOTION_RELINK_TIGHT_UM=5.5` |
| `biohub-lineage-forge-precision-tracking` (flexonafft, 97) | as above | as above |
| `biohub-lf-dctta` (sjlee101, 82) | as above | as above |
| `biohub-harmonic-fusion-v3` (raunakdey07, 78) | as above | as above |
| `biohub-0-95` (anvithpothula, 58) | — (shares **no** `BIOHUB_*` constant) | the entire lineage |
| `biohub-lf-hoctveto-div-b` (sjlee101, 30) | **`HOCT_VETO=2`**, `SECONDARY_EDGE_FEATURE_TTA_WEIGHT=1.0`, `DEEPCENTER_SAFE_DIV_THRESHOLD=0.20` | `MOTION_RELINK_TIGHT_UM=5.5`, and the two above |

**Four of the six are our banked champion with two constants moved the wrong way and the motion relink
missing.** `biohub-harmonic-fusion`'s own header says so: `BIOHUB_SCORE_AXIS = 'public 0.939 base +
holdout-selected post-process configuration'`. Its 263 votes buy a **0.939**, against the 0.947 we hold.

## The one thing that looked new, and wasn't

`BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT=0.15` with `BIOHUB_BIDIRECTIONAL_FUSION_MODE=harmonic_probability` reads
like a training-free lever — a second `predict_edges(tgt, src)` pass, harmonic-meaned in probability space.
It is **already set in our banked 0947**, identically, and we independently built and shipped the same
mechanism a month earlier: `TrackerConfig.shipped()` carries `bidirectional_edges=True`, implemented at
`celltrack/edges/blended_edge_scoring.py:191-194` (softmax forward, softmax reverse transposed, `fuse`).
Our own record already prices it both ways — −0.0027 under the per-frame assignment linker, **+0.004 on
the leaderboard under flow** (`celltrack/operating_point.py:121`). Nothing to harvest; the axis is ours.

## `biohub-0-95` is the metric hack again, under a new name

`augment_dataset` (`code.py:849-897`) appends a node at `t=-1000, z=y=x=-10000` — outside every volume, so
never matchable — links the roots of the `MAX_COMPONENTS = 1400` largest components to it, and chains
`FORKS = 5` fabricated off-image divisions. That is the kirneo family, already recorded in the conclusion
tree (line 701, and 1391-1396 under codezzzsleep's `095-owned-validation`, same 1400/5 constants). It farms
the component-count term; our own measurement says the fork half earns **exactly zero** division Jaccard.
**Not portable and not submitted** — it games the scorer rather than tracking anything, and a host fix
voids it on private. Its "0.95" is not a tracking number, which is the useful part: **public scores above
our 0.947 are not all evidence of a better tracker.**

## `HOCT_VETO` is the only genuinely new element, and it is a closed axis

`biohub-lf-hoctveto-div-b` carries ~550 lines absent from every other kernel: after the champion's own
post-processing has produced the final graph, it runs **HOCT general_v0** (royerlab, arXiv 2607.11754) over
that same node set — a 3 µm sphere per final node, intensity features from the raw frames — and **drops
every edge HOCT does not also propose**. Mode 2 vetoes division edges too. Nodes are never changed.

The author's own offline number, on 20 videos: **+0.0040 [+0.0006, +0.0058]**, with scorer-visible false
divisions 55 → 29; the edge *union* loses 0.004.

It is pre-refuted on three independent legs of our own measurement, before any run:

1. **The effect size is under our floor.** +0.0040 against a 0.01–0.02 noise floor, and a 1.5 % bar. Its own
   confidence interval reaches 0.0006.
2. **It is an edge deletion, and there is nothing to delete.** E54 measured **wrong-association = 0** on all
   four public-GT movies: no GT edge is linked to the wrong partner, so a removed edge is either a GT edge
   (a loss) or an edge between unannotated nodes, which E53 showed is **metric-invisible**. E47 priced the
   whole prune axis at **+0.021 even if perfect**, with ~2× collateral; E58 found the champion already holds
   51 of 79 in-scope broken edges and **0 of 102 decoys isolated**, losing 1.85 per 1 gained.
3. **Its stated gain is division-side, and our division axis is LB-flat.** Swept on the real leaderboard
   flat; the global division-ILP scored −0.055.

Leg 2 is the general form and worth keeping: **an edge-deleting proposal must first answer whether the
edges it deletes can ever have been GT edges.** That is the same question E55 retired gap repair with.

## Net

Six kernels, ~1.4 MB of notebook, **zero shippable elements**. The frontier's top-voted public work is our
banked champion with worse constants; the one number above it is an exploit; the one new mechanism deletes
edges on an axis four of our own experiments closed. This does not raise the score, and it does something
almost as useful late in a competition: it says the public list holds nothing we have not already priced,
so the remaining wall-clock belongs to the one lever E59 left open — the **centre-offset head**, which
cannot be priced offline and has to be trained or dropped.
