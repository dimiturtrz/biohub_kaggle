# We trained on choices that were never contested

Thirteen arms have failed to move the dense mislinks. Every one of them assumed the model was being asked the
right question and answering it badly. This is the measurement that says it was rarely being asked at all.

## The number

Mean candidate targets inside the shipped linker's own 10 µm gate, per source
(`python -m celltrack.training.corpus_audit`):

| population | sources/frame | **mean in-gate** | median | p90 | share contested |
|---|---|---|---|---|---|
| synthetic GT (complete labels) | 296 | 2.79 | 3 | 5 | 77.8% |
| **real GT annotated — what training sees today** | 12.3 | **0.99** | 1 | 1 | **1.9%** |
| real detections, dense movie — what we choose among | 744 | 3.80 | 4 | 6 | 95.4% |

**One candidate per source.** In 98.1% of training sources there is no alternative to reject — the step asks
*"is this the successor?"* when the answer cannot be anything else. At inference, 95.4% of sources face a rival.

The head is trained on uncontested choices and deployed on contested ones. Discrimination is the thing the
score depends on, and it is the thing the training signal almost never exercises.

## Why this was invisible

The corpus is built from the ground-truth graph, so sources, targets, and every *negative* are annotated cells.
Annotation covers ~1.8% of the population, so annotated cells are sparse — and sparse points rarely land within
10 µm of each other. The candidate set is thin **because** the labels are sparse. Nothing in the code looks
wrong; the emptiness is a property of the sampling, not a bug in it.

This is the population trap that produced an edge AUC of 1.0000 on GT pairs, an inverted threshold ranking on
the sparse proxy, and a base rate of 0.9807 in the shortcut probe. Those were bad *measurements*, caught within
hours. This one sits in the **training data**, where nothing prints a base rate and no assertion fires.

## The synthetic fix is a clean null

Synthetic sequences label every cell, so candidate sets are complete by construction. Fixed-fraction mixing,
matched regime, three arms from an identical init:

| arm | peak clamped | dense mislinks | node recall |
|---|---|---|---|
| control (0% synthetic) | 0.8598 | 41 | 0.9902 |
| 25% | 0.8456 | 42 | 0.9959 |
| 50% | **0.8320** = its init, never improved | 43 | 0.9927 |

Monotone damage in the dose. It differed from the earlier `lna` refutation on all four counts I designed
against — backbone unfrozen, masked detection loss, mixed rather than synth-only, complete-candidate framing
rather than teach-the-task — and reproduced it anyway.

**The mechanism is in the loss terms, not the scores.** Synthetic edge loss ran **0.0041–0.0095** against
**0.20–0.53** on real pairs, at every window of both mixed arms. The warm-started head *already solves* the
complete-candidate synthetic pairs. So the complete candidate set supplied almost no association gradient, and
a quarter to half the step budget went to an already-solved task. Dilution — which predicts the monotone shape
exactly.

## My kill-check guarded the wrong risk

Before the run I pre-registered the failure I thought most likely: synthetic candidate sets might be far
*smaller* than real ones, so completeness would buy nothing because the crowding that causes our failure would
not be present. I asked for the number specifically to catch that.

It did not fire. Synthetic carries 2.79 in-gate rivals against real inference's 3.80 — the same regime. The
225-vs-743 per-frame gap is mostly **volume**, not density.

The arm failed for a reason I had not named:

> **Completeness is necessary but not sufficient. The negatives must also be hard.**

Synthetic near-neighbours are trivially separable even at 2.79 rivals per source. Ours are not — that is the
entire content of "the affinity is 97.4% inverted on the mislinks".

A pre-registered check earns its keep by being *checkable*, not by being right. This one was cheap, it was
answered, and the answer redirected the conclusion instead of confirming it.

## What the three rows actually point at

The corpus with **both** properties is our own dense movie's detections:

- **complete by construction** — every real rival is present, because they are the same detections the linker
  chooses among;
- **hard by construction** — all 38 mislinked partners are unannotated detections the current features cannot
  separate.

That closes the 2×2×2. `zni` had the right population and the wrong objective with a frozen backbone. The
hard-negative margin arm had the right objective and an unfrozen backbone on the wrong population. Real
detections + margin + unfrozen has never been run, and it is the first configuration where all three axes are
right at once.

## Honest limits

**The arms are confounded.** A fixed fraction at a fixed step budget trades real steps for synthetic ones, so
the 25% arm saw 750 real steps to the control's 1000. "Adding synthetic" and "fewer real steps" are not
separated by this design. Dilution is the better reading because the synthetic loss is already at zero — the
mechanism, not the arm ordering, is what carries it.

**Synthetic is not dead.** Node recall rose 0.9902 → 0.9959 at 25% — the detection half's only real movement
in the experiment. It may be a detection asset even though it is not an association one.

**This does not predict a score.** It explains why thirteen arms could all be sound and all fail, and it names
one configuration none of them occupied. Whether the head can learn what the features cannot express is a
separate question, and `a1v` already measured the features as lacking it at this resolution.

## The lesson

We had a rule from three prior burns: *state the population before quoting any model-quality number.* It was
written for measurements. The same failure in a training corpus is quieter, more expensive, and lasts longer —
no number reads 1.0000 to tip you off.

The corresponding rule: **before training on a corpus, measure what a step actually asks.** Here the question
is one line — the mean number of candidates the deployed gate admits per source — and it should be printed at
startup beside the pair count, not discovered thirteen arms in.
