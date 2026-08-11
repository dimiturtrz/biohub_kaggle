# The "short optimum" was partly our training loop

**Date:** 2026-08-12
**Status:** one seed per arm; the mechanics result is large enough to act on, the balance result is still running.

## What the campaign believed

Every warm-start arm peaked within a few hundred steps and then degraded, across many months and many
methods. That was recorded as a property of the problem — "warm-start fine-tuning on 1-2% dense labels has a
short optimum" — and used to justify 250-step arms as the right horizon.

It was challenged on the obvious ground: *a score should not collapse with more training if the training is
adequate.* Auditing the loop rather than defending the curve found two defects, both in our own setup.

## Defect 1 — the batch was one

`JointTrainer._step` did `zero_grad -> backward -> step` per pair. The loop's comment said

> counts are ragged, so there is no batching across pairs

which is true of stacking pairs into a batch **tensor**, and had been silently treated as ruling out
averaging their **gradients**. It does not. Every joint arm this campaign followed one pair's gradient per
update, with no averaging anywhere — so "degradation past the peak" and "a random walk at batch size one"
were never separated.

`OptimCfg.accumulate_pairs` now averages N pairs into one update, dividing by N so the effective rate does
not scale with the batch, and clipping the mean the optimiser actually follows. `1` reproduces the old step
exactly.

## Defect 2 — nothing anchored the shared trunk

The association arms ran `--det-weight 0`. Both heads read ONE backbone, so with detection weighted zero
nothing held the trunk as a good detector while association gradients reshaped it. The logs show it plainly:
node recall drifted in runs that train no detection at all.

## What changed when both were fixed

Same arm — detected-pair corpus, hard-negative margin, unfrozen backbone — held out on the untouched test
four through the shipped pipeline:

| arm | faithful | mislinks | node recall |
|---|---|---|---|
| pilkwang single seed | 0.9118 | — | 0.9930 |
| detected corpus alone (matched control) | 0.9060 | 56 | — |
| detected + margin, batch 1, no anchor, 250 steps | 0.9154 | 47 | 0.9932 |
| **detected + margin + averaging + anchor, 1 epoch** | **0.9266** | 46 | **0.9968** |
| best own-weights of the campaign (symmetric axis) | 0.9278 | 43 | 0.9945 |

**+0.0112 from the training setup alone**, with the method untouched. Node recall 0.9968 is the highest
recorded here; the masked detection anchor is what produced it.

## What did NOT change: the shape

```
init  0.8476 (64 mislinks) -> 400  0.8531 (52) -> 800  0.8388 (59) -> 1600  0.8513 (48) -> 1785  0.8306 (62)
```

The peak moved up but the curve still falls after it — and it OSCILLATES rather than decaying, which is the
signature of a noisy objective, not a converged optimum. So the mechanics were part of the story and not all
of it. Note step 1600: 48 mislinks against the init's 64, the best association of any arm, at a score below
its own peak — score and association do not move together here.

## The remaining signature, and what it points at

```
              init     400      800      1600     1785
P_true        0.191    0.161    0.143    0.096    0.099
P_chosen      0.572    0.560    0.544    0.488    0.438
inverted      0.91     0.94     0.95     0.96     0.95
```

Both probabilities fall together — the model FLATTENS its softmax column rather than re-ranking it. That is
the failure `zni` was refuted for, arriving here through the loss's REDUCTION rather than through its form.

The mechanism is a class imbalance the detection loss beside it already refuses. `BalancedBCE` weighs
detection positives `1/n_pos` and negatives `w/n_neg` per frame. `SoftmaxFocalBCE` averaged over active
CELLS, so the single true candidate is outnumbered by its rivals — up to 38 to 1 on this corpus
(targets/gap mean 8.3, max 39) — and the gradient is mostly "not that one", with only the focal power
pushing back by an amount nobody chose. An objective that is mostly negative pressure drives every
probability down, which is what the table shows.

`SoftmaxFocalBCE.of(..., balanced=True)` reads the same per-cell focal BCE one DECISION at a time — the
positive mean plus the rival mean, then averaged over decisions — which is `BalancedBCE`'s rule at row level.
No constant enters; the balance is the two class counts, which the data supplies.

**Pre-registered falsifier:** if class imbalance drives the flattening, P_true should stop falling while
P_chosen keeps falling (the gap closing), and mislinks should not climb past the peak. If both keep falling
together, the flattening has another cause and this fix is not it.

## A test that could not have caught any of this

`_trained_bytes` reads the BEST-checkpoint, and `save_best` writes the INIT whenever no window improves — so a
byte-identity assertion over that file can compare two untrained models and pass. It proves a knob is inert
exactly as loudly as it proves the run never saved, and the accumulation test hit precisely that: identical
bytes for a knob the weights prove is live. `_trained_weights` reads the resume snapshot instead, which is
written every window regardless of score. Three existing `*_off_is_byte_identical` tests are suspect for the
same reason and have not yet been re-verified.

## What this does and does not license

It does NOT retract the individual refutations. It says the horizon argument behind them was measured on a
loop running at batch one with no anchor, so any arm whose verdict rested on "it degraded with more
training" is a candidate for re-running rather than a settled result.

It does NOT produce a submittable model: 0.9266 single-seed is still below the shipped dual-seed 0.9334. The
gap is now 0.007 rather than 0.018.

## Open, in priority order

1. The balanced link loss (running) — the pre-registered test above.
2. `symmetric_links` + margin + balanced together. Symmetric ALONE gave 0.9278; it has never been combined
   with either of the others, and all three now compose from one command.
3. LR warmup. A warm-started model meets a FRESH AdamW state, so the first updates are taken with no
   second-moment estimate — standard practice would warm up, and we never have. Untested.
4. Corpus scale. The detected corpus is 20 of 191 train videos (1785 pairs). Whether the oscillation is
   small-corpus noise is a one-flag experiment.
