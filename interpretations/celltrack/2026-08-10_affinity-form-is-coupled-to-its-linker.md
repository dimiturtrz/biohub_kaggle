# The affinity's shape belongs to whoever consumes it

Three times this campaign has copied a component from a stronger public pipeline, measured it alone, and
found it flat or worse: Yusuke's 14 µm edge gate (LB-flat), appearance-in-the-linker, and his motion relink
(which lost outright to our linker on the proxy). Each time the conclusion recorded was "that knob doesn't
transfer". A fourth attempt today produced the same shape of result and, this time, an explanation.

## What was tried

The pipeline currently sitting at **0.915 public** uses *bidirectional harmonic probability fusion*: score
each frame gap forward, score it again with the temporal pair swapped, and combine the two with a harmonic
mean. The harmonic mean is dominated by the smaller of its arguments, so a pair only scores high when both
directions agree — mutual consistency rather than one-sided preference.

The mechanism fits our documented failure exactly. Our dense mislinks are **100 % affinity-inverted**: the
transformer scores a wrong *near* neighbour at 0.686 against the true far successor at 0.134. A wrong near
neighbour ought to win going forward and lose coming back, because that near cell has its own better parent.
And unlike every previous attempt to improve association, this needs no retraining — so none of the three
refuted retrain results (`lna` synthetic, `lna` real-positives, `zni` hard negatives) apply to it.

It measured **−0.0027** pooled, and **−0.0010 on the dense movie the argument is about**. Its own target got
worse.

## The bonus sweep ruled out the obvious explanation

A harmonic mean of two values in `[0, 1]` is never larger than either, so fusion systematically shrinks every
probability. Our linker's cost is `distance − bonus·P` with `bonus` tuned against *forward* probabilities, so
the learned signal was plausibly just under-weighted. That predicts a higher bonus recovers it.

| bonus | 20 | 40 | 60 |
|---|---|---|---|
| pooled | 0.9307 | 0.9256 | 0.9253 |

Monotonically worse. The optimum sits at or below the current value, so the fusion is not producing
rescaled-but-equally-ordered probabilities — it is producing probabilities our linker **ranks worse**. That
distinction is what pointed at the consumer rather than the scale.

## The consumer, not the affinity

Our `EdgeTransformerScorer` soft-maxes over **sources**. That is not a formatting detail — it encodes "one
parent per target" *into the probability itself*, and `AssignmentLinker`'s cost is built around that
asymmetry. A harmonic fusion of both directions produces a **symmetric** score: it re-states a constraint our
linker already imposes structurally, while discarding the directional sharpness the cost depends on.

A constrained lineage ILP — what the 0.915 pipeline links with — is the mirror image. It imposes
one-parent/one-child as explicit global constraints, so it wants a symmetric, calibrated per-pair affinity and
has no use for directional normalisation.

If that reading is right, the fusion should stop hurting once the consumer is global. Our `FlowLinker` is a
global min-cost-flow solver, so the prediction is directly testable — with one trap: **at
`disappearance_cost = 0` the flow linker provably reduces to the assignment linker edge-for-edge**, and the
first cross-test was run there. It reproduced the assignment numbers exactly and tested nothing. A control has
to be verified to actually vary the thing it controls for.

At a boundary cost of 3, where the solver behaves globally:

| linker | forward | + bidirectional |
|---|---|---|
| assignment (per-frame) | 0.9334 | **0.9307** (−0.0027) |
| flow (global) | 0.9349 | **0.9353** (+0.0004) |

**The sign flips.** Neither cell clears our noise floor and neither is a win — the meaningful quantity is the
~0.0031 *differential* between consumers for an identical affinity. The hypothesis made a falsifiable
prediction and survived a test that could have killed it: had the fusion hurt the global solver equally,
"shaped for a different consumer" would be dead.

## What follows

**Port bundles, not parts.** Three cherry-picks failed alone while their bundle scored, and now there is a
mechanism for why rather than a shrug: components co-vary with what consumes them. The frontier's fusion and
its ILP are one design decision expressed in two files.

**Make the coupling checkable.** The affinity's *form* is chosen in the scorer but determined by the linker,
and nothing in the types says so — a legal configuration can pair a symmetric affinity with a cost built for
an asymmetric one, and you discover it as a small negative number weeks later. The first step landed today:
linkers that cannot read an affinity now refuse an `affinity_bonus` outright and warn when a mounted affinity
is handed to them (four of six were silently discarding both). The larger step is for a linker to declare the
affinity form it consumes, and for a mismatch to be refused when the config is built.

**A cheap probe beat an assumption again.** `pyscipopt`, `ilpy`, `motile`, `tracksdata` and `rustworkx` are all
absent from the Kaggle image, so the ILP half of that bundle needs five offline wheels — the same stack that
already failed there once. But `networkx` is present, so *our* global solver needs no packaging at all. The
expensive port is now gated on a free experiment rather than on faith.

## Correction: how different the reverse pass really is

Written after reading the backbone rather than assuming it. `TemporalUNet3D`'s `_TemporalAttention` is a
per-voxel `MultiheadAttention` over the time axis with **no positional encoding**, so it is permutation-invariant
over time: feeding `[t+1, t]` returns the same features as `[t, t+1]`, swapped. The backbone cannot tell which
frame came first; order reaches the pipeline only through the edge transformer's time-fraction node feature
(0.0 for the source frame, 1.0 for the target).

That weakens one claim above. The reverse pass is *not* a wholly new question put to the network — the features
are the same and only the head's source/target roles and time fractions swap. It is still not a pure
re-normalisation of the forward logits, but it sits far closer to one than argued here, which is a plausible
part of why it behaved like the renormalisation `e9b` refuted. It also means the U-Net recompute in the reverse
pass is avoidable: the forward features could be reused with the fractions swapped, halving the cost.

The consumer-coupling result itself is unaffected — the sign flip between the per-frame and global linkers was
measured, not inferred from this reasoning.
