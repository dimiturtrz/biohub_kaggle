# Edge-transformer retrain refuted — the dense loss is signal-limited (lna)

aet ranked "retrain the edge signal" as lever 1, the only measured route into the 0.930+ pack (the dense
movie `6bba_05db0fb1` has ~0.11 edge-jaccard headroom to the 0.98 linkability ceiling, all crowding-mislinks).
66g named the technique: hard-negative mining + parental softmax. This is the result of testing it.

## Setup

Fine-tune **only** the pilkwang `SimpleNodeTransformer` edge head (UNet frozen — the user's call, and it keeps
train/inference features identical), warm-started from pilkwang, with the reference focal-BCE-over-sources loss
(`compute_loss`: softmax dim=0 = parental constraint already built in). Evaluated single-seed affinity into
`AssignmentLinker(gate=10, bonus)` + gap + stlf6 on the 4-movie proxy. Baseline (pilkwang, no fine-tune) =
pooled **0.9273** at its optimal bonus=40 (dense **0.8667**).

Three substrates, each swept over bonus (the linker cost is `distance − bonus·P`, so a shifted P-sharpness
needs a re-tuned bonus — swept to rule out bonus-mismatch as the cause):

| config | best bonus | pooled | dense |
|--------|-----------|--------|-------|
| **baseline (no fine-tune)** | 40 | **0.9273** | **0.8667** |
| real-only (from pilkwang, 600 steps) | 80 | 0.9212 | 0.8582 |
| synthetic-only (600 steps) | — | 0.9105 | 0.8448 |
| curriculum synth→real | 80 | 0.9058 | 0.8409 |

## Findings

1. **Every fine-tune loses, monotonically with dose.** Even 20 synthetic steps hurt; 600 hurt more. The best
   any variant reaches at its *own* optimal bonus is real-only 0.9212 — still −0.006 below baseline, and the
   dense movie never improves.
2. **Not a bug.** Ruled out: GT training coords match detection inference coords (both native-voxel, y/x to
   ~252); training loss *does* drop (real 0.0067→0.0036, so the head is learning the objective); UNet frozen
   so features are identical train-vs-inference; the loss mirrors the reference exactly.
3. **Not bonus-mismatch.** Fine-tuning flattens the P distribution (optimal bonus shifts 40→80 = P got less
   sharp, needs more weight), but re-tuning the bonus recovers only part of the loss — never baseline.
4. **Not the synthetic domain gap alone.** Real-only fine-tuning (same distribution pilkwang trained on) also
   loses. The synthetic substrate is separately dead (monotonic dose-damage, worst on the fast-mover sparse
   movie 44b6 = motion-prior mismatch), but the real control shows the problem is deeper than the gap.

## Verdict

The pilkwang edge transformer sits at a **strong optimum**; any fine-tune — synthetic, real, or curriculum, at
any bonus — degrades dense association. The dense mislinks are **not fixable by retraining P** with this
detector+linker. Two independent axes now agree the dense loss is **signal-limited**: e9b (every per-frame
cost-function knob flat) and this (every retrain of the signal itself worse). The crowded nuclei are genuinely
ambiguous from position + this UNet's features at the (1,4,4)-downsampled resolution — the true successor is
not separable from its near-neighbours by the available signal.

## Consequence for the campaign

- **lna closed.** The edge-signal lever, ranked P0 by aet, does not crack dense. aet's ranked-levers doc
  lever 1 is now refuted — update: the only *measured* route into the pack is exhausted.
- **du3 (synthetic) training-feed use closed** with it. The dataset's recall-vs-density diagnostic use
  survives but is low value now.
- **What remains** (both untested, both lower-confidence than lna was): lever 2 track-level global
  optimization (a different *mechanism* — solver uses multi-frame trajectory context, not a better P; note
  e9b showed 1-frame velocity *hurts*, but full multi-frame is untested), and lever 3 a complementary
  detector (884) — but recall is saturated (0.988) and thinning the crowd already collapsed (NMS-prune),
  so its mechanism for helping dense is weak.
- **Realistic near-term ceiling stays the pending submissions** (~0.92 predicted). Bonus=40 (vs shipped 20)
  is a noise-level nudge, not worth a slot alone.
