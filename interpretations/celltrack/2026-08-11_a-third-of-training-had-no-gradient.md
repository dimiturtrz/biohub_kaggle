# A third of our association training had no gradient

`SoftmaxFocalBCE` normalises the source→target logits across the **source** axis — the frontier's form, and
the one our inference scorer reads. It states "each target has one parent".

With **one source** in the matrix, `softmax(dim=0)` returns 1.0 whatever the logits. The gradient is exactly
zero, and the loss is a constant:

```
one decidable source:   loss=66.6667  grad=[0, 0, 0]   |grad| = 0
four decidable sources: loss=0.1449   grad[0]=[-0.142, +0.0071, +0.0071]
```

Over 30 train videos / 2776 GT pairs: **799 single-source (28.8%)**, and 61.7% carry two sources or fewer.

## Three consequences, and they compound

1. **~29% of association steps were no-ops.** Every own-weights run this campaign paid for them.
2. **Every training-loss curve we read was dominated by that constant.** The "loss 57" on the detected corpus
   was blamed on the detection head colliding with the association centres. That collision is real, but it was
   not what produced the number.
A third consequence I asserted — that the difficulty sampler, ranking by loss, would preferentially mine the
dead pairs — is **wrong**, and I filed a bead for it before reading the code. `_PairOutcome.difficulty()` is
the fraction of annotated sources whose true successor the logits fail to rank top-1, taken as `argmax(dim=1)`
over **raw logits**. It is immune to the degeneracy.

The correction is more interesting than the claim. That argmax is over **targets** — so the sampler was
already asking *"which successor for this source"*, the exact question the source-normalised loss could not
express. The instrument was right about what a mislink is; the objective was the half that could not act on it.

## The deeper form, present even when the matrix is healthy

At four sources the true target gets −0.142 while its rivals in the same row get **+0.0071** — twenty times
weaker. That asymmetry is structural, not a weighting choice: a rival target's probability can only fall if
some **other supervised source** claims its column. Every dense mislink we have measured picks an
**unannotated** detection, which no source claims.

So the axis the linker chooses along — *which successor for this source* — has never been directly supervised.
Our objective has been shaped by annotation density rather than by the decision the tracker makes.

This sits underneath the whole association record. Seven independent attacks read flat; the corpus audit
showed training sees 0.99 in-gate candidates per source against 3.80 at inference. The instrument was fine and
the methods did little, but neither observation explains *why* nothing moved. A loss that cannot express the
failure does.

## The fix, and the one it replaced

The obvious repair is a **slack row** — one extra logit meaning "no source here parents this target", mirroring
the linker's zero-cost skip. It needs a scale: our edge logits sit near −11, so a zero-logit slack takes
essentially all the mass, and a learned scalar is a parameter to defend.

Normalising over **targets** as well needs nothing. For a single-source matrix, `softmax(dim=1)` over that row
is already a real distribution over the rivals — gradient non-zero immediately — and it states "one child per
source", which is the axis the linker chooses along. Zero new parameters, and the default is unchanged so a run
that does not ask is byte-for-byte yesterday's run.

It also rhymes with the only leaderboard gain we have. Bidirectional harmonic fusion (+0.004, LB 0.895) works
because mutual consistency pays under a global solver — and the recorded mechanism for *why* is that
softmax-over-sources bakes one-parent-per-target into the probability, deflating a far true parent that
competes against near false ones. This puts that symmetry in the **logits** instead of buying it back with a
second forward pass at inference.

## The lesson

We had audited the training **corpus** twice — candidate density, coverage, contestedness — and never audited
the **objective** computed over it. A loss with a degenerate case is not a slow learner; it is a no-op wearing
a large number, and a large number reads as "this pair is hard".

Check that a loss has a gradient on every shape its data actually takes. `SoftmaxFocalBCE` had a unit test from
the day it was written, and that test used a 2×2 matrix.
