# Appearance identity at full resolution — the mislinks are genuinely ambiguous

*2026-08-12 — analysis only, no training, no submission. Tool: `celltrack.analysis.appearance_identity`.*

## The question

The dense movie's association failure is 38 mislinks where the head prefers a wrong near neighbour to the true
far successor. Three prior measurements said the *learned* features cannot separate those cells — raw feature
cosine prefers the wrong one 77% of the time, a probe reads chance held out, two independent inits fail
identically — but **all three were on the `(1,4,4)`-downsampled grid**, where a nucleus is 2–3 voxels. The one
untried structural argument was resolution: the full-resolution image is 16× finer in plane. This asks whether
the identity information is there in the raw voxels.

## Cell size, measured

FWHM at annotated centres (n=4, interior cells; order-of-magnitude):

| | FWHM | full-res voxels | downsampled (1,4,4) |
|---|---|---|---|
| in-plane (y/x) | 6.5 µm | 16 | **4** |
| z | 8.1 µm | ~5 | ~5 (z not downsampled) |

**Downsampled 4×4 in plane is the whole problem** — a blob that size has no shape, orientation, or texture,
only position and rough size. Full resolution has 16×16, genuinely enough to describe the cell. So resolution
was worth checking.

## Result

Raw normalised cross-correlation of 5×17×17 voxel boxes (full resolution), no model, no training. The honest
metric is the **in-gate ranking** — rank the true successor among *all* candidates in the 10 µm gate, with the
correct null `chance = mean(1/candidates)`:

| population | top-1 | chance | read |
|---|---|---|---|
| correct edges | **0.796** | 0.284 | 2.8× chance — appearance is a real identity cue in general |
| mislinks | **0.115** | 0.260 | below chance — points at the wrong cell on the failures |

**Appearance is a real identity cue in general** (correct edges, 2.8× chance) — cells *do* look different
enough, usually. **But on the mislinks it is below chance**, ranking the truth 3rd of ~4. At full resolution,
the wrong cell genuinely looks more like the source than the true successor does.

Chain, all on the dense movie:
```
downsampled learned features        0.229   (prior)
full-res raw appearance, 2-way      0.267   below its (>0.5) null
full-res, miscentring-corrected     0.400   still below
full-res in-gate ranking            0.115   below the mean-1/N chance of 0.260
```

## Two confounds closed

**Rotation is negligible.** On correct edges, allowing the successor box to rotate in-plane to best-fit the
source improves correlation by a median **+0.002** (identity 0.809 → best-rotated 0.819). Cells hold their
orientation frame to frame, so there is no hidden shape/orientation structure a fixed-orientation correlation
was discarding — and the shape-change-under-rotation worry does not bite here.

**The 0.5 null was wrong for the two-way metric** and is corrected. `prefers_true` is a 2-alternative forced
choice, but the alternatives are not exchangeable: raw-box correlation tracks spatial overlap and the mislink
rival is nearer by construction, so the nearer cell wins under the no-identity null and the honest baseline is
*above* 0.5. That makes the mislink 0.267 worse than it looked, not better. The in-gate ranking's `mean(1/N)`
baseline is the trustworthy one and the conclusions rest on it.

## Consequences

**The identity model is refuted as a lever on the failures.** It helps where distance already wins (~99.8% on
correct edges, so appearance's 80% adds nothing) and fails where distance fails (the mislinks, below chance).
Replacing the 0.19-worth edge transformer with pairwise cosine is a non-starter; adding appearance as an extra
cost term has no signal to add on the tail.

**The annotation-error hypothesis is elevated.** On the 38 mislinks, three independent physical signals —
distance, learned features, and now raw full-resolution appearance — all rank the rival above the true
successor, unanimously. When geometry, features, and pixels agree against the label, "the label is wrong" is
the parsimonious reading for some fraction. If k of 38 are annotation errors, the dense association ceiling is
below the computed 0.086 and part of the wall is unreachable by construction. The cheap next test is the
annotated turn cosine at the mislinked edges: a genuine fast move continues in a direction, a swapped label
jumps out and back (cos ≈ −1), and that signature already appeared in the motion run's largest-step band.

**What is NOT refuted:** a *learned* full-res representation could still weight informative voxels where raw
correlation weights all equally. But the prior dropped hard — raw full-res appearance is below chance exactly
where we need it, and the rotation result argues there is little rich structure to learn. The honest statement
is "enough pixels to describe the cell, not enough distinctness between the cells": 1280 sharp samples of two
similar nuclear-fluorescence blobs still gives two similar blobs.

## Denominator, for context

The distance-only baseline measured the same day: the edge transformer is worth **+0.19** of the score (0.746
distance-only vs 0.938 shipped, on the four test movies). So the association head is not a marginal component —
it carries a fifth of the score — and the 38 mislinks are the residual tail it gets wrong, now shown ambiguous
on every measurable axis including native-resolution appearance.
