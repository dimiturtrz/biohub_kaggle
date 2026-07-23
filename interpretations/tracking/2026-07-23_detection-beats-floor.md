# The learned detector beats the classical floor on recall

*2026-07-23 — `biohub_kaggle-3ws`, on validation fold 0, two videos per acquisition.*

## Result

At a matched node count (both detectors asked for each video's estimated cell count), the trained 3D
U-Net finds more of the annotated cells than the classical Laplacian-of-Gaussian, on **both**
acquisitions:

| acquisition | classical recall | learned recall | gain |
|---|---|---|---|
| 44b6 | 0.656 | **0.766** | +0.110 |
| 6bba | 0.664 | **0.742** | +0.078 |

Recall is the fraction of ground-truth nodes with a predicted centre inside the competition's 7 µm
matching radius, measured with the same `DistanceMatcher` the edge metric uses. Both detectors end in the
identical anisotropic non-max suppression (`PeakExtractor`), so the only thing that differs is the
response volume — the LoG for the floor, the network's heatmap for the learned one. The comparison
therefore isolates *where the detector looks*, not how it picks peaks or how many it keeps.

## What produced it

- **Architecture:** anisotropy-aware 3D U-Net (`ANISOTROPIC_STRIDES = (1,2,2),(1,2,2),(2,2,2)`) — y/x
  pool twice before z, so the receptive field is physically isotropic on the 4× grid at native
  resolution (the recorded decision; resampling was rejected — see the model docstring).
- **Supervision:** the masked scheme from the harness — Gaussian heatmap at annotated centres, dark
  tissue as negatives, bright-but-unlabelled excluded so un-annotated cells are never taught as
  background.
- **Training:** 40 videos, 2000 steps, masked MSE 0.040 → 0.006. That is *all* — no tuning, no schedule.

## How much to trust it

This is a single run on two videos per prefix, so treat the exact numbers as indicative, not hardened.
Two things make the direction credible rather than noise: both acquisitions move the same way, and they
are genuinely different (44b6 is 1 % annotated, 6bba 9 %), so a +0.08–0.11 gain on both is not a
one-acquisition fluke. Before leaning on the magnitude, this wants more held-out videos and a couple of
seeds — cheap follow-ups now that the pipeline exists.

## What it means for the epic

- **3ws is met:** the learned detector beats the classical floor's recall at a matched node count on the
  held-out prefix split, which is exactly the DONE-WHEN.
- **The 0.44 bracket gap has a mover.** The bracket (`dll`) put nearly the whole score gap in detection
  and ~0.998 already reachable in linking. A detector that lifts recall ~8–11 points over the classical
  floor with no tuning is the first real step into that gap. How much of the *score* it buys depends on
  the operating point (the node-count sweep, `sct`) and on the linker — next.
- **Recall isn't the finish.** Recall at matched count ignores precision and the node-count penalty; the
  score also cares about how many nodes are predicted. The sweep and an end-to-end score on the learned
  detector are the honest next measurements.
