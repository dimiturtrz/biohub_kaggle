# E50 — does a contour HIERARCHY hold what our candidate set does not? (the `g89y` precondition)

**Verdict: NULL under an intensity proxy, and the proxy is confounded — this does NOT refute `g89y`.**
It does retire the cheap route to it, and it overturns a runtime claim I made six hours earlier.

## Why this was run

E49 closed the offline prune axis: eight features of our existing detections reach AUC 0.6622 jointly
and FP-recall 0.0846 at E47's 2 % collateral, against the 0.75 required. `g89y` survived that only on a
scope argument — Ultrack's segment-SELECTION ILP does not score the detections we hold, it scores
**nested contour hypotheses the candidate set never contained** (Nature Methods 2025: candidates from
multiple algorithms, temporal consistency used to *select* among them). Two numbers decide whether the
GPU build earns anyone's wall-clock:

- **CONTAINMENT** — fraction of GT cells with a hierarchy segment inside the cell-separation ruler.
  Our detector reaches 0.824 @ 2.87 µm (E36). Beating that means the hierarchy offers *new candidates*.
- **SEPARABILITY** — whether a hierarchy-**intrinsic** score (contour `frontier`, ultrametric `height`,
  `area`) hits E47's operating point. These are not functions of E49's eight features: they describe how
  a segment survives across merge thresholds, which a fixed detection cannot express.

Pre-registered in the instrument's docstring before the run: the contour map is a raw-intensity proxy,
not a trained contour head, so **a PASS licenses the build while a NULL only says "not with this map"**.

## What happened first: a magic constant produced a fake result

The first full run pooled 81300 segments over 4 dense movies and reported containment **0.6047** @2.87 µm
— comfortably below the detector, which reads as a clean kill. Disaggregating by movie killed that read:

| movie | mean containment | s/frame |
|---|---|---|
| 6bba_09961292 | 0.728 | 5.6 |
| 6bba_bb9f20c3 | 0.742 | 5.5 |
| 6bba_784a78c9 | 0.789 | 5.7 |
| **6bba_57b7cc1e** | **0.102** | **17.6** |

One movie collapses and runs 3× slower. Cause, measured rather than guessed:

| movie | raw median | GT-voxel norm. intensity | q98 cut | **GT above cut** |
|---|---|---|---|---|
| 6bba_09961292 | 114 | 0.4145 | 0.3814 | 0.722 |
| 6bba_57b7cc1e | 1115 | 0.3602 | 0.4762 | **0.000** |

`6bba_57b7cc1e` has a bright hazy background (raw min 432, median 1115) and fills z=3–60 vs 3–24. The
fixed `FOREGROUND_QUANTILE = 0.98` therefore cuts **above every GT cell**: containment is zero *by
construction*, and the pooled 0.6047 was measuring my threshold, not the hierarchy. A magic number with
no derivation behind it — the exact failure CLAUDE.md names. Its frames also contribute ~1200 all-FP
segments each, so the pooled separability AUCs were contaminated too.

## The threshold has no valid-and-affordable setting

Otsu, derived from the data rather than legislated, admits the cells (GT-above-cut 1.000 / 1.000 / 0.833)
— but inflates foreground from 2 % of voxels to **21–48 %**, and one frame then failed to finish in
**>3 minutes** against 5.5 s. So:

- the cheap regime **misses cells** on some movies,
- the cell-containing regime costs ~20–40× and is unaffordable,
- and no intensity quantile fixes `6bba_57b7cc1e`, because there the background haze is simply brighter
  than the cells. Intensity is not a contour cue on that movie at any threshold.

`proxy_validity()` now refuses such a movie up front (GT-above-cut < 0.5) instead of averaging it in.

## The measurement, on the movies where the proxy is valid

`6bba_57b7cc1e` excluded by the guard, on its own arithmetic (`proxy sees 0.000 of GT (< 0.50)`), not by
my noticing an outlier. Remaining: **57651 segments, 1151 GT cells, 3 movies, 60 frames.** All four
movies are DeepCenter-**held-out** (see the manifest check below), so this is not a memorised-detector
number.

| | value | bar |
|---|---|---|
| CONTAINMENT @ 2.87 µm | **0.7498** | detector 0.824 (E36) |
| CONTAINMENT @ 7.00 µm | 0.8801 | official ruler |
| `frontier` AUC | 0.4421 | — |
| `height` AUC | 0.5694 | — |
| `area` AUC | **0.6145** | E49 joint 0.6622 |
| best FP-recall @ 2 % collateral | **0.0203** (`intensity`); `area` 0.0077 | **0.75** (E47) |

**Both gates fail, and neither is close.** Containment is *below* the detector, so on these movies the
intensity hierarchy offers no candidates we don't already have. Separability is 37–97× short of E47's
operating point and lands under E49's own 8-feature joint ceiling (0.0846) — a hierarchy-intrinsic score
is not buying anything a detection-side feature already failed to buy. `height` is the sharpest null:
FP-recall **0.0000** at every collateral target, i.e. the ultrametric depth at which a segment dies
carries no information at all about whether it is a real cell, under this contour map.

## What this overturns

**My own claim from NEWS 2026-09-20 03:30Z — "`g89y`'s runtime gate is CLOSED IN ITS FAVOUR" — was
wrong.** That arithmetic (4.5 s/frame → ~4800 s over 40 movies × 100 frames, against ~27000 s of kernel
headroom from `twrp`) priced the *cheap* foreground, the one that misses cells. Hierarchy cost scales
with foreground volume; at a threshold that actually contains the GT it is >3 min/frame, which overruns
the headroom ~4×. Kernel budget is an argument against `g89y` again, until the trained contour head is
shown to be both tight and cell-containing. That is now the gate, and it is a GPU question.

## What this does NOT say

Not a refutation of `g89y`. The real build uses a **trained contour head**, which is exactly the
component this proxy stands in for and exactly the component that could be simultaneously sparse and
cell-aligned — the property intensity lacks. Recorded per the pre-registration: NULL-leaning,
confounded, re-testable only on GPU (currently parked by instruction).

What it does buy, for ~40 min of CPU: nobody spends a day building the intensity-proxy route, the
runtime gate is corrected before it misled a build decision, and `g89y`'s remaining claim is now stated
in one falsifiable line — *the trained contour head must contain GT above 0.824 @2.87 µm at a foreground
sparse enough to run in the kernel budget.*
