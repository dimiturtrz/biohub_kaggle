# Half our errors were the same error, counted twice

The competition metric is `jaccard = TP / (TP + FP + FN)`, so a false positive costs exactly as much as a
false negative. For the whole campaign we measured only the false negatives.

Not by choice — by instrument shape. `dense_diagnosis` classifies the **fate of each ground-truth edge**: it
walks the 1183 true edges and asks what happened to each. A *predicted* edge with no true edge to attach to is
never enumerated, because there is no fate for it to have. The instrument answers *"what did we lose"* and is
structurally blind to *"what did we invent"*.

## The counts

A prediction-centric decomposition — walk our **emitted** links instead, and classify each by why the metric
charged us for it:

| movie | annotated | TP | FP | FN | MISLINK | TARGET_ORPHAN | SOURCE_ENDS | SYNTHETIC | EXTRA_CHILD |
|---|---|---|---|---|---|---|---|---|---|
| 6bba_05db0fb1 (dense) | 1.8% | 1132 | 71 | 51 | **38** | **33** | 0 | 0 | 0 |
| 6bba_05b6850b | 13.5% | 837 | 11 | 8 | 6 | 5 | 0 | 0 | 0 |
| 44b6_0113de3b | 0.2% | 48 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| 44b6_0b24845f | 0.16% | 49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## They are one event, charged twice

**All 38 mislink targets are unannotated.** The wrong partner is never another annotated cell — it is an
unannotated detection that outranked the true successor.

**All 33 target-orphans stole a lost target.** The target's own annotated in-edge is one we already failed to
reproduce. Every single one is the *second charge on a false negative we already knew about*.

So the dense movie's entire error is one repeated event: source `s` loses successor `t` to an unannotated
detection (charge one), and `t` acquires a parent from a different unannotated detection (charge two).
**71 false positives over 38 mislinks — 1.87 charges per event.**

## What that changes

**Fixing one mislink pays ~2.9 points, not 1** (−1 FN, −1.87 FP, +1 TP). The campaign's spend on association
was not misallocated; it was **underpriced by about 3×**. Every arm we judged marginal deserves re-reading
against that multiplier.

**Our noise floor is exactly one mislink.** One repair moves dense jaccard by ~0.0023 and the floor is ~0.002.
So a method fixing one mislink is indistinguishable from noise, a method needs ~5 to be clearly detectable, and
"a tie at the floor" now means something precise: *fixed at most one*. That retro-sizes the night's
refutations — ranker at b=3 (−0.0007), evidence ramp (−0.0014), seed alignment (−0.0001) each moved
zero-to-one mislink, and the ramp's own mislink count confirmed it directly (38 → 39). The instrument was
adequate; the methods did nothing.

**The prize is computable for the first time.** Repair the mislinks and nothing else: TP 1170, FP 0, FN 13
(the residual skips and endpoint-missing) → `1170/1183 = 0.989` against today's `1132/1254 = 0.903`. **+0.086
on the dense movie** — four times the entire gap to the reproducible public frontier. This is a *ceiling*, not
a forecast; all 38 are affinity-inverted and thirteen axes have failed to move them. But it is the right
denominator for judging whether an expensive idea is worth the wall-clock. Against 0.020 an expensive attempt
looked marginal. Against 0.086 it is the obvious place to spend.

## Three readings of mine that the counts killed

**"The unclassified FPs are a separate lever."** They are the shadow of the FN term. There is no second pile of
errors to go after.

**"Sparse annotation manufactures phantom FPs."** I could not square that with the proxy *overshooting* the
leaderboard, and the resolution runs the other way: linking on past a track end into unannotated tissue is
**uncountable — free**. A sparse annotation therefore charges *fewer* FPs than a dense one, which predicts
overshoot, which is what we see. The gradient is the evidence: 0 FPs at 0.16–0.2% annotation, 11 at 13.5%, 71
at 1.8% dense. FP structure is a **crowding** phenomenon, not a sparsity one.

**"The boundary-cost sweep is over-linking seen from the other side."** `SOURCE_ENDS = 0` *structurally*, so it
cannot be. A higher disappearance cost forces the flow to continue tracks, so it accepts lower-affinity
partners — it **manufactures** mislinks. Same curve, different mechanism.

## The specification error that nearly hid it

I asked for edges whose **both** endpoints match ground truth. `EdgeCounts._countable` is an **OR** — a link is
charged if its source has an annotated out-edge *or* its target has an annotated in-edge. Both-matched would
have excluded `TARGET_ORPHAN` entirely: the class that *is* the finding. The specification was corrected rather
than executed, which is the only reason the answer exists.

Worth keeping as a rule: when commissioning a decomposition of a metric, **read the metric's own predicate**
first. I described what I assumed it charged, and I was wrong about it in exactly the way that would have
confirmed my prior.

## What is not here

No repair. Zero mislinks fixed. The score is unchanged, and every number above is measurement or arithmetic —
the +0.086 in particular assumes an oracle. What the work bought is a price, a floor with a physical meaning,
and a mechanism check (`--charges`) that can now say whether a score moved *for the reason claimed*.
