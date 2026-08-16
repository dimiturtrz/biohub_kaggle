# The from-scratch ceiling is detection generalization, not data or synth

**Date:** 2026-08-15
**Question the user posed:** "is that our best historical ceiling? why can't we hit higher?" — and the
follow-up: *attribute* why theoretically-better (dim-tail) synth data does not lift the from-scratch model,
instead of monkey-fixing knob values.

## The decisive metric-free test

`celltrack/analysis/differential_missed.py` runs BOTH detectors through the shipped tracker on the dense
validation four and intersects their per-GT-node missed masks. No fragile intensity histogram — it asks a
binary, physical question: does the warm (pilkwang-initialised) detector CATCH the cells the from-scratch one
misses, from the same pixels?

| set | count | share | meaning |
|---|---|---|---|
| both caught | 1839 | 0.899 | — |
| both missed | 15 | 0.007 | shared physical floor |
| **from-scratch missed & warm caught** | **178** | **0.087** | **recoverable** |
| warm-only missed | 13 | 0.006 | — |

from-scratch recall **0.906**, warm recall **0.986**. Of the 193 from-scratch misses, warm recovers
**178 (92.2%)**. Only 15 cells (0.7%) are a floor no detector clears.

Local contrast (peak − background-shell median), the SNR-flavoured axis raw intensity misses:

| set | median contrast |
|---|---|
| both-caught | 89.0 |
| recoverable | 81.5 |
| both-missed | 75.5 |

The recoverable cells are **normal-contrast, ordinary cells** — barely dimmer than the caught ones, far above
the both-missed floor. They are not a dim tail, not low-SNR, not a synthesisable class.

## Attribution

The ~8% recall gap is **not** a data problem:
- **Not data availability** — the train GT carries the same dim brightness tail as val (p5/p95 2.02× vs 2.51×;
  `intensity_calibration.py`). The from-scratch detector had dim cells in its real supervision and still misses
  val's.
- **Not a dim/low-SNR class** — the missed cells are normal-contrast (median 81.5 vs 89.0). Dim-tail synth
  attacks a class that isn't the bottleneck.
- **Not synth quality** — a better synthetic dim tail cannot help a gap that isn't about dim cells.

It **is** a detection representation/generalization gap: the pixels are sufficient (pilkwang's backbone reads
them correctly), so the from-scratch weights simply have not learned the representation. Every from-scratch run
floors its train detection loss at ~0.025 regardless of step budget (4k → 40k reach the same val nodeR ~0.92),
while val recall stalls — the low-train-loss / capped-val-recall signature of a generalization deficit, not
undertraining. **None of the from-scratch runs used augmentation.**

## Consequence

- **The synth dim-tail line is dead** (refuted twice: data availability, and the missed cells aren't dim).
  Stop tuning `intensity_log_std` / `noise_std`.
- The lever that could plausibly deliver the ~5% nodeR is **generalization of the from-scratch detection
  representation** — augmentation (flip + brightness/offset jitter), which multiplies effective diversity over
  the 12 train movies for free. Test in flight: `det_scratch_aug.pt` (from-scratch detection pretrain, flip +
  brightness 0.3 + offset 0.1, cosine). Climbs past the 0.93 cap → generalization confirmed, honest from-scratch
  lever found; also caps → a real capacity limit, next lever is architecture/backbone size.
- Warm-start's edge is **external published pilkwang weights** — legitimate as a reference ceiling, but not a
  from-scratch answer.

## Reproduce

```
uv run python -m celltrack.analysis.differential_missed \
  --from-scratch joint_pt_finetune.pt --warm U_nosym.pt
```
