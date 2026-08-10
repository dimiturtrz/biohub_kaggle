# Five times in one day, the answer was "the pipeline already does that"

A day of proposing additions, and the recurring finding was not that the additions were wrong — it was that
the thing they added already existed somewhere in the pipeline, sometimes in the exact place I was about to
put it. Recording the pattern, because the cost of missing it is not a wasted hour; it is a *measured null*
that looks like a refuted idea.

## The five

**1. Motion features.** I was about to add per-pair displacement to the edge transformer. Reading the external
head first showed it already computes `rel = (coords_t − coords_t1) / 100.0` and concatenates it before the
pair MLP. The model was never geometry-blind. What was actually missing was one step narrower — *history*,
the previous displacement — which reframed the task from "add motion" to "add memory".

**2. Parental softmax.** The corrective research listed "normalise over the competing-candidate set" as the
verified fix for wrong-near-neighbour errors, and ranked it a top-3 action. `SoftmaxFocalBCE` has soft-maxed
over the source axis since it was written.

**3. Partial-label masking.** I filed the detection loss labelling ~98% of real cells as background as a P0
and briefed it as a new mechanism. `SoftmaxFocalBCE` already did exactly that on the edge side —
`active = rows-or-columns the annotation touches`. The fix was a **consistency repair**, not an invention,
which is a much safer thing to ship.

**4. Mutual-agreement gate.** Built, swept, null (+0.0004). The reason is redundancy: under a global flow
solver, "these two agree" is a constraint the program *already imposes* structurally. The identical quantity
worked as a **score** (+0.004 on the leaderboard) because that changes what the solver optimises rather than
restating what it enforces.

**5. Resolution.** The direction I rated highest when asked where a much larger upside could come from,
argued from three genuine findings — every mounted detector added zero complementary recall, misses produce no
response at any threshold, misses are "crowd-buried at (1,4,4)". I never checked the miss *count*. It is **4,
out of 1229**. Detection is at 99.7% and there is nothing left to find.

## What distinguishes the good outcomes from the bad ones

In (1), (2) and (3) the check happened **before** the work: an audit, a research cross-reference, a read of the
sibling loss. Each converted a planned addition into either a sharper task or a smaller, safer change.

In (4) and (5) the check happened **after**: a built-and-swept feature, and a direction I argued for at length.
Both cost real time, and worse, both produced a *null that reads like a refutation*. "Mutual agreement doesn't
help" is a false lesson — it helps, as a score, on the leaderboard. "Resolution is a dead end" is also false
as stated; the truthful version is "there is nothing left to recover on the axis we can measure."

That is the actual cost of skipping the check: not the wasted effort, but a **mislabelled negative** that a
future session will read as settled.

## The cheap check that would have caught (5)

Before arguing that our representation is the bottleneck, count the phenomenon. Four misses in 1229 is a
number available from one detector pass and a matcher we already own — it took nine minutes once I stopped
theorising. An earlier version of the same probe sampled ten frames, found 75 of 75, and reported *zero*
misses: a sample too small to contain a 0.3% effect. **Sizing the sample to the effect was the whole
experiment.**

## The rule, stated so it is checkable

Before adding a component, answer two questions in writing:

1. **Where would this already be, if it existed?** Then read that file — not a grep for its name, the file.
   `rel` is not called "motion"; the edge loss's masking is not called "partial labels".
2. **How big is the phenomenon I am attacking?** If the answer is unknown, measure it first. A mechanism that
   explains 0.3% of the data cannot be worth a multi-day build, however sound it is.

Both questions are cheaper than any of the five investigations above, and either one would have redirected two
of them before they started.
