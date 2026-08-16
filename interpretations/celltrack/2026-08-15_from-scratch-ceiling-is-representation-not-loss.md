# The from-scratch ceiling is representation, not loss

*2026-08-15 — resolves the two items `2026-08-10_joint-training-and-the-honest-split.md` left open
(from-scratch untested, `det_weight` never swept).*

## The number

The 3-stage from-scratch curriculum (synth → real → real+test finetune) tops out at **proxy 0.7121**
on the dense val-four. The same ruler reads **0.8474** for the pilkwang warm weights with zero training.
The 0.135 gap is not noise and not budget: stage-2 plateaued by step ~5500, not step 40000.

The honest-split doc guessed the detection term might be "dominating and dragging the association head
with it." This window tested that guess directly, three ways, and it is wrong.

## What was tested — and refuted

**Association is not starved by loss weight.** Two arms shifted the objective toward association:

- *edgeboost* (`--det-weight 0.3 --hard-negative-weight 0.5 --balanced-links`): proxy flat at 0.6944,
  early-stopped at step 4000. A wash. (And the InfoNCE line it logged was cosmetic — `contrastive_weight`
  defaulted to 0, so the term contributed nothing to the total. The arm only tested det-down + hard-neg
  margin.)
- *contrastive* (`--contrastive-weight 0.1 --contrastive-site projection`): actively **harmful** — proxy
  declined monotonically 0.674 → 0.645 while mislinks rose monotonically 67 → 129, node recall unharmed.
  Same failure mode `contrastive_term.py` already recorded for the FEATURES site (node recall 0.966 →
  0.867): contrastive gradient into the shared trunk corrupts the features detection and linking both read.
  The projection head only softened it. The damage persisted even as the term converged to 0.13 — it is
  the *representation shift*, not the live loss magnitude, that hurts.

**The recall deficit is not a threshold artifact.** A no-train sweep of `--eval-threshold` on the stage-3
checkpoint (the eval default 0.97 is the LB-measured best for *warm* weights, `operating_point.py:31`):

| threshold | 0.97 | 0.90 | 0.80 | 0.70 | 0.60 |
|-----------|------|------|------|------|------|
| node R    | 0.904 | 0.907 | 0.905 | 0.904 | 0.904 |
| proxy     | 0.7121 | 0.7094 | 0.7057 | 0.7050 | 0.7033 |
| mislinks  | — | 105 | 108 | 105 | 106 |

Node recall is **flat to three decimals** from 0.97 down to 0.60. The ~6% of cells the from-scratch
detector misses relative to warm (0.966) produce **no detection response at any threshold** — they are
absent from the map, not suppressed by it. Lowering the threshold buys zero recall and costs a little
precision. The shipped 0.97 is correct for from-scratch too.

## What it means

The from-scratch ceiling is set by two threshold-invariant, loss-invariant floors:

1. **Detector recall saturates at ~0.90 node R** — a genuine capacity/pretraining-data gap, not calibration.
2. **~105 mislinks** — the same association wall `2026-08-12_dense-mislinks-are-a-wall.md` found, here
   unmoved by every association-loss lever available.

Both are properties of *what the weights learned*, and warm-init simply learned more of it (pilkwang's
broader pretraining). No knob exposed on the trainer — detection weight, contrastive term, hard negatives,
balanced links, eval threshold — moves either floor materially. This is why from-scratch cannot reach warm:
it is a representation gap, and the levers we reached for shape the *objective*, not the *representation*.

## The one lever left — now characterized

Raising from-scratch means raising detector recall past 0.90, and that is a data problem, not a loss
problem. So `celltrack/analysis/missed_cells.py` (`python -m celltrack.analysis.missed_cells`) asks *which*
GT cells the from-scratch detector misses, reusing the **shipped tracker assembly** so its missed set is
exactly the one the measured node-recall counts (no divergent hand-rolled detector = no harness fork).

Pooled over the dense val-four: **2045 GT nodes, 193 missed (recall 0.9056)** — matching the 0.904 ruler.
Per-axis, missed-median vs caught-median and the concentration ratio in the extreme fifth:

| axis | caught | missed | ratio | verdict |
|------|--------|--------|-------|---------|
| intensity (dim tail) | 0.515 | 0.354 | **1.68** | **SYSTEMATIC** — dim, 3/4 big movies (1.65/1.86/1.46) |
| xy-edge (µm to border) | 21.5 | 10.6 | **1.73** | systematic pooled, movie-mixed (4.91/1.96/3.65/1.01) |
| z-edge (vox to face) | 18.0 | 16.0 | 0.65 | diffuse |
| crowding (rivals) | 0.000 | 0.000 | — | **unmeasurable** (see below) |
| dividing (mitotic) | 0.006 | 0.005 | 0.88 | unmeasurable (GT division base-rate ~0.006) |

Intensity is sampled as the **local peak over a small anisotropic box**, not the centre voxel — a cell whose
centroid sits on the dim centre of a bright ring is not a dim cell. The named centre-sampling artefact was
tested and refuted: swapping centre-voxel for local-max moved the pooled ratio only 1.74 → 1.68, both
SYSTEMATIC. The dim skew is a property of the cells, not the sampler.

**The finding: from-scratch misses a systematic DIM class.** The cells it drops are the low-intensity tail
(dimmest fifth over-missed 1.68×, direction-consistent across the three big-n movies; the 4th is n=5 noise).
XY-frame-border is a weaker, movie-mixed second axis. This is a coverable class with a plausible mechanism —
**targeted low-SNR / dim-cell synth** — not a diffuse capacity gap. That is why `biohub_kaggle-ie55` is worth
building rather than a dead end.

**Two axes are unmeasurable on this input, not features of the model — denominator failures, flagged
honestly.** Crowding reads all-zero for *every* node because the ruler counts same-timepoint rivals among the
**sparse GT subset**, where a 10 µm gate genuinely holds ~0 neighbours; the linker's real rivals are
*detections*, not GT. (`_crowding`'s arithmetic is correct — `voxels × spacing` — so this is not a code bug to
fix; measuring it would need detection-space crowding.) Dividing is likewise starved by a ~0.006 GT
division base-rate. Neither says anything about the missed class; both are the wrong denominator.

Everything short of the dim-synth lever — including reverting to the warm baseline as the end state — is
either a wash or a retreat.

## Standing decision

The one submission stays HELD. From-scratch at 0.7121 is 0.135 below warm on the shared ruler and would
waste the slot; the warm pipeline (proxy 0.847 / LB 0.892–0.899) remains the deliverable until a from-scratch
run demonstrably clears it.
