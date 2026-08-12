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

### The falsifier fired, and the balance is REFUTED

```
step:       400      800      1200     1600     1785      held out
balanced    0.8516   0.8323   0.8602*  0.8472   0.8269    0.9045  (53 mislinks)
unbalanced  0.8531   0.8388   —        0.8513   0.8306    0.9265  (46 mislinks)
```

P_true fell exactly as before (0.167 -> 0.102) and the gap to P_chosen was unchanged at ~0.39, so **class
imbalance is not what drives the flattening**. And the held-out number is decisive in the other direction:
0.9045 against 0.9265, i.e. **-0.022**, far outside any floor. `balanced_links` stays default-off and the
mechanism claim is withdrawn.

Two things worth keeping from a dead arm:

* **The selector preferred the worse model, by a lot.** Balanced peaked HIGHER in-run (0.8602 vs 0.8531) and
  lost by 0.022 held out. The peaks differ by 0.0071 — about 3x the selector's repeatability floor — so this
  is not selector noise, it is the four validation movies disagreeing with the four test movies. The balanced
  arm's peak also sat at step 1200 against the unbalanced arm's 400, so the selector rewarded a
  *more-trained* checkpoint that generalised worse. Any future arm judged on the in-run peak alone inherits
  this failure mode.
* **The re-baseline was negligible.** `joint_det_acc8` reads 0.9265 with divisions in `shipped()` against
  0.9266 without — 0.0001. The division stage costs the edge term essentially nothing on these movies, so
  older proxy numbers remain usable in practice even though they are not formally comparable.

The flattening therefore still has no established cause. What is now excluded: the reduction's class
imbalance (this arm), and the loss's *form* (the margin term is invariant to adding a constant to a row, and
flattening happened under it anyway).

## Three arms, and the selector ranks them backwards

All three share the fixed mechanics (averaged updates, masked detection anchor, one epoch, detected corpus,
hard-negative margin) and differ only in the link loss. Held out on the untouched test four:

| arm | in-run peak | held out | mislinks | node ratio |
|---|---|---|---|---|
| margin only | 0.8531 | **0.9265** | 46 | +0.015 |
| margin + symmetric axes | 0.8490 | 0.9231 | **44** | +0.041 |
| margin + balanced reduction | **0.8602** | 0.9045 | 53 | +0.005 |

**The in-run ranking is the exact inverse of the held-out ranking.** The selector's spread (0.011) is larger
than its repeatability floor, so this is not noise — the four densely-annotated VALIDATION movies genuinely
disagree with the four TEST movies about which association model is better. Every checkpoint this campaign
chose on that selector is therefore suspect in the same way, and an arm reported on its in-run peak alone
should not be believed.

**Symmetric axes improve ASSOCIATION and pay for it in DETECTION.** 44 mislinks is the best association of
any arm here, and its node ratio nearly triples (+0.041 against +0.015) — the metric charges that, so it
scores lower despite linking better. Both heads read one trunk, so a link objective that pushes harder on
association drags detection toward over-firing. That is the same coupling the contrastive term showed, at a
much smaller magnitude, and it means association and detection cannot be tuned independently here.

Best own-weights remains the margin-only arm at 0.9265, still under the shipped dual-seed 0.9334.

## Can the training data teach association and division at all?

Measured, because the answer differs completely between the two.

**Association: the labels are adequate; we were using the wrong corpus.** The annotated corpus poses 0.99
in-gate candidates per source and is 1.9% contested — almost no discrimination problem — while inference
faces 3.80 candidates and is 95.4% contested. Detection-matched pairs (8.3 targets/gap) are what moved
0.9060 -> 0.9265. The rivals never need labels; they only need to be PRESENT as competitors. What remains
looks like a representation limit rather than a data limit, since two independently-initialised seeds fail on
the SAME pairs.

**Division: the labels are insufficient by three orders of magnitude.** Counted over all 199 train videos:
133318 annotated nodes, **151 division events** (0.113%), per-video median 0, max 5, and **112 of 199 videos
contain none at all**. No learned mitosis detector is trainable on that. It also explains after the fact why
the division gain that DID land came from parameter-free geometry — 151 examples can support a geometric
argument and nothing more.

The synthetic corpus inverts that ratio: 2174 sequences, 4056226 nodes, **165267 division events** (4.07%) —
1094x more events than the real corpus, labelled by construction. And the reason synthetic FAILED for
association does not carry over: it failed because the warm-started head already solved synthetic pairs
(synthetic edge loss 0.004-0.009 against 0.20-0.53 on real), so the synthetic share diluted a budget that had
better work to do. Dilution needs an existing signal to dilute; for division there is none.

The design that follows, and the asymmetry is the whole point: **151 real divisions is a hopeless training
set and a perfectly good test set.** Train on synthetic, hold out every real division as the judge.

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
