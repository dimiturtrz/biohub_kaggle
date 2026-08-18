# Motion persistence is tissue drift, not per-cell direction — the linker, not a wider window

*2026-08-18 · CPU-only, `python -m celltrack.analysis.motion_statistics`, 8 annotated movies*

## Question

The joint backbone's temporal attention is order-blind (permutation-invariant, no positional
encoding). A proposed architecture change (bd **f5w**, `--temporal-position` flag) widens the temporal
window past the current 3-frame pair (t-1, t, t+1) and adds a frame-position embedding, so the model
can integrate displacement over several steps.

That only pays if displacement **accumulates** — if a cell holds a direction, averaging over frames
beats per-step noise (ballistic). If it tumbles, averaging converges to zero and the wider window buys
nothing (diffusive). The decider is the per-cell velocity correlation time τ (Furth fit) — but measured
in the **co-moving frame**, because lab-frame persistence can belong to the tissue, not the cell.

## Numbers

Per movie: MSD exponent · lab-frame τ · **drift-free τ** · drift share of mean step.

| movie | chains | MSD exp | τ (lab) | τ (drift-free) | drift share |
|---|---|---|---|---|---|
| 6bba_05b6850b | 21 | 1.25 (diff) | 2.86 | **1.55** | 61% |
| 6bba_05db0fb1 (dense) | 46 | 1.59 | 7.80 | **2.44** | 65% |
| 6bba_969618f6 | 33 | 1.83 | 12.03 | **1.24** | 88% |
| 6bba_fc83837d | 28 | 1.65 | 8.45 | **2.02** | 71% |
| 44b6_341df25f | 16 | 1.63 | 4.88 | **5.35** | 83% |
| 44b6_e57ff5c6 | 19 | 1.44 (diff) | 2.22 | **3.94** | 92% |
| 44b6_0113de3b | 2 | 1.79 | 11.76 | nan | nan |
| 44b6_0b24845f | 2 | 1.93 | degenerate | nan | nan |

Denominator honesty: the two 44b6 2-chain movies are too thin — drift uneval'd (nan), τ degenerate.
The load-bearing evidence is the **6 well-sampled movies** (16–46 chains).

## Read

1. **Lab-frame persistence is mostly tissue drift.** τ collapses from 7–12 frames (lab) to ~1.2–2.4
   frames (drift-free) in the three high-drift movies (969618f6 12.03→1.24, fc83837d 8.45→2.02,
   05db0fb1 7.80→2.44). The long persistence a wide window would exploit is **common-mode drift**, not
   the cell's own direction.

2. **Per-cell direction decorrelates fast.** Drift-free τ ≈ 1.2–5.3 frames, mostly ~2. The existing
   3-frame pair span already covers ~1 correlation time; widening the window integrates mostly noise
   for the individual-motion signal. Two movies are outright diffusive (exp 1.25, 1.44).

3. **Drift is 61–92% of the mean step** — the dominant, persistent, exploitable motion signal. It
   shifts every candidate in a frame equally, so it's estimable from the current frame's mean
   displacement (hundreds of samples) — a **linker** term, not something the backbone needs a wide
   temporal window to see.

4. **Neighbour coupling ≈ shuffled** (observed +0.44 vs shuffled +0.41 on fc83837d; +0.36 vs +0.32 on
   05db0fb1). The local "neighbours predict my motion" signal is barely above the whole-frame drift
   that survives permutation → a local spatial-coupling model buys little beyond the global drift term.

## Conclusion

- **f5w (wider temporal window + positional encoding) is LOW-VALUE for individual cell motion.** The
  persistent long-τ signal is tissue drift, which disambiguates nothing at the per-cell head (it moves
  all candidates equally); the per-cell direction it *would* help with decorrelates in ~2 frames,
  already inside the current span. Deprioritize — not refuted (dense movie 05db0fb1 still has
  drift-free τ 2.44 > 1-frame, and f5w might help the linker see drift), but not the well-motivated
  lever it looked like.
- **mpwq (subtract the tissue-drift field in MotionHungarian) is the better-motivated confusor lever.**
  It targets the 61–92% common-mode directly, at the stage where LB gains have actually come from (the
  linker — [[celltrack-selection-is-binding-not-detection]]). Code already landed (`use_drift`,
  `_estimate_drift`, off-by-default); needs one GPU affinity pass to validate.

Matches bd **duw4**'s thesis: every LB gain altered *how the linker decides*; every recall/backbone-side
tweak died.
