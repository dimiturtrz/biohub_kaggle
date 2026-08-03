# The dense ceiling, quantified — why we sit at ~0.92 and what it would take to move (capstone)

A night of levers (lna edge-retrain, 884 complementary detector, AUC reframe, edge-fate decomposition)
converges on one quantified conclusion: the dense movie `6bba_05db0fb1` is at its **achievable ceiling** with
the current assets, and the residual loss is largely irreducible crowd ambiguity — not a fixable signal,
detector, or cost-knob deficit.

## The full chain, each link measured

| stage | number | headroom |
|-------|--------|----------|
| detector recall (dense) | 0.988 | ~0 (saturated; 884 adds none) |
| edge-signal crowd-AUC (dense) | **0.9860** | ~0 (near-ceiling; no fine-tune beats it) |
| linker GT-edge correctness (dense) | **91.4%** | Hungarian is per-frame optimal |

### Edge-fate decomposition (1183 dense GT edges, bonus=40)

- **correct 91.4%** (1081)
- **mislinked 4.4%** (52) ← the core residual lever
- **skipped 2.2%** (26) — cost threshold `distance − bonus·P < 0` rejected a true edge
- **endpoint-missing 2.0%** (24) — detector, near-irreducible

The long-standing "0.11 headroom to the 0.98 linkability ceiling" framing was misleading: the *achievable*
headroom is ~6.6% (mislink + skip), and mislinks are the bulk.

### Why the 52 mislinks happen

- **98%** — the signal P ranks the wrong target *above* the true one. These mislinks are exactly the 1.4%
  tail of the 0.986 AUC. The signal is not merely un-improvable (lna proved that); on these specific cases it
  is actively wrong.
- **87%** — the chosen wrong target is physically *closer* than the true successor.
- 40% are 1-to-1 conflicts (true target stolen by a competing source); 60% are free cost-errors.

The picture: a fast-moving cell's true successor lands far, while a slower/nearby cell sits closer, and
**both distance and the learned appearance signal prefer the near wrong one**. This is the crowd-ambiguity
floor — the true and false continuations are not separable from position + this UNet's features at the
(1,4,4) resolution.

## What was refuted, with mechanism (three independent axes)

1. **Signal retrain (lna)** — fine-tuning the edge transformer (synth / real / curriculum, every bonus)
   degrades or ties; AUC baseline 0.986 has no headroom to retrain into. Dead.
   (`2026-08-04_edge-retrain-refuted.md`)
2. **Per-frame cost knobs (e9b)** — velocity, bonus, blend, threshold, division, NMS-prune, affinity
   renorm: all flat or worse. Velocity specifically *hurts* dense (damped prediction lands on wrong
   neighbours — the same fast-mover-in-crowd failure the mislink diagnosis shows).
3. **Complementary detector (884)** — cellpose-SAM: no recall gain (recall already saturated), and its
   centroids score 0.37 through the pipeline (features misalign with the UNet the affinity reads). Dead.

## What remains (honest EV)

- **Lever 2, track-level global-over-time optimization** — the only untested axis, targeting the 40% of
  mislinks that are 1-to-1 conflicts. But 60% are free cost-errors the signal gets wrong, and 98% have the
  signal preferring the closer wrong target, so a better *solver* (not signal) caps out well below the full
  4.4%. And its cheap approximation (velocity) is already refuted. Low-to-moderate EV, real build.
- Everything else is refuted or saturated.

## Verdict for the campaign

We are at the achievable ceiling (~0.92 public predicted) with the pilkwang detector + edge signal + our
linker. The pack (0.930–0.947) needs assets we do not have: a stronger/differently-trained detector+edge
model, or more resolution/data to separate crowded fast-movers — consistent with the ceh pack-decode ("no
public method published; the pack is ahead via undisclosed means"). The remaining in-hand gains are
noise-level (bonus 20→40 = +0.0003) or the node-count bonus (a0u, +0.006–0.02, already pending on the LB).
The honest near-term result is the pending submissions, not a new dense lever.
