# The peak read-out was over-merging dense cells — a reference-window fix worth +0.0117

**Date:** 2026-08-04 · **Bottom line:** our non-maximum-suppression window was **5³ where the reference is
3³**, silently merging crowded nuclei two voxels apart. Matching the reference recovers them: proxy
**0.9227 → 0.9344**, driven entirely by *raw* edge Jaccard on the dense movie (0.849 → 0.892) — a genuine
recall gain, not a node-count-bonus artefact. It partly reframes the "dense is signal-limited" capstone:
some of the missing dense cells were an **NMS-merge artefact**, not a detector ceiling.

## What was wrong

The learned detector's centres are the suppressed local maxima of a per-voxel response. The suppression is a
strided-1 max-pool: a voxel survives where it equals the pooled maximum over an anisotropy-aware neighbourhood.
The neighbourhood *size* is the bug. We built it as

```
radius = ceil(anisotropic_radius(pool_kernel_um / 2))   # ceil(2.5µm / 1.625µm) = 2  ->  5³ kernel, ±2 voxels
```

while the reference read-out (`predict_unet_transformer.py`) is

```
k = round(pool_kernel_um / spacing)                     # round(5µm / 1.625µm) = 3   ->  3³ kernel, ±1 voxel
```

At the `(1,4,4)` downsample the grid is isotropic 1.625 µm, so our ±2 window suppresses any second peak within
**3.25 µm** — well inside a ~5 µm cell diameter. Two touching nuclei collapse to one centre; the lost node
takes its in- and out-edges with it as false negatives. Exactly the crowding regime that caps the dense movie.

## The fix, and why it is safe

`radius = rint(2 · scale_um / spacing) // 2`, kernel `2·radius+1`. Rounding the *diameter* then halving gives
the reference's effective radius (1, a 3³ kernel) on the detector path, and the `2r+1` form keeps the kernel
**odd** so stride-1 symmetric padding is size-preserving — an even kernel grows the pooled volume by a voxel
and mismatches. On integer spacing ratios the formula is identical to the old one, so the classical detectors
and every peak unit test are unchanged; only the non-integer downsampled detector ratio moves.

## It is recall, not bonus-farming

The metric's node-count adjustment can flatter a prediction that merely trims toward the estimated count, so
the gain had to be checked against *raw* Jaccard (bonus-independent). Per-movie A/B at threshold 0.99:

| movie | raw Jaccard before → after | note |
|-------|----------------------------|------|
| 44b6_0113de3b | 0.960 → 0.960 | flat (sparse, uncrowded) |
| 44b6_0b24845f | 0.960 → 1.000 | un-merged, now perfect |
| 6bba_05b6850b | 0.994 → 0.970 | small regression — false splits on the easy movie, low sample weight |
| **6bba_05db0fb1 (dense)** | **0.849 → 0.892** | **+0.043 raw — the weighted-dominant win** |

The dense movie carries by far the most annotation weight, so its +0.043 raw Jaccard drives the pooled
+0.0117. And threshold 0.98 and 0.99 now score *identically* — a node-count-bonus gain would be
threshold-sensitive (more nodes at 0.98); its absence confirms the gain is structural recall. Recall is the
one lever the leaderboard has consistently rewarded (0.99 > 0.995), so this has a high transfer prior, unlike
the per-frame cost-tunes and bonus-farms that overshot on the proxy and did not transfer.

## A companion fix that did nothing (and that is fine)

The audit also found we ran the equality-NMS on the **sigmoid** volume, where the reference pools the
**logits** (thresholding on the sigmoid). We moved suppression to the blended logit volume — reference-faithful
and it removes a latent footgun: at fp16 (the old cache bug, 0.909 → 0.816) sigmoid saturates distinct logits
to identical values and manufactures false plateaus. But at fp32 it is **inert**: our peak logits are ~5–10,
whose sigmoids (0.99–0.9999) are all distinct in fp32, so no plateau forms and the score is unchanged
(0.9227 → 0.9227). Kept for correctness; the real over-merge was the window, not the value space.

## Where this leaves the climb

Submitted as a clean A/B against the 0.887 baseline (same config, no fix) at the known-good threshold 0.99. If
it transfers, it is the first *structural* gain over 0.887 since the global linker. It also says the dense
ceiling is a little higher than the capstone claimed: not every missing dense cell is a no-response cell — some
were merged by our own read-out. The remaining gap is still detector no-response cells (irreducible with these
weights), but the merge artefact was a real, fixable slice of it.
