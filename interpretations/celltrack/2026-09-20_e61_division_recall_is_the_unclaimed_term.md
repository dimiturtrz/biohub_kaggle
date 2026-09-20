# E61 — the division term is worth 0.1, we bank 0.003, and every sweep that "closed" the axis could only ever move 0.001

**Date:** 2026-09-20 · **Run:** `python -m celltrack.eval.proxy_eval --divisions 4 --per-movie` ·
**Log:** `logs/ruler_divisions4.log` · **Issue:** bd `hq93`

## The reading

The division-bearing ruler (bd `brod`) put the banked 0.947 champion on eight held-out movies carrying 25 GT
divisions — the first set where the division term is measurable at all. Per movie, `div=tp/fp/fn`:

| movie | raw_jac | adj_jac | nodes | ratio | tp/fp/fn |
|---|---|---|---|---|---|
| 44b6_a21120c2 | 0.9259 | 0.9333 | 21204 | −0.080 | 0/1/2 |
| 44b6_c50204e0 | 0.8122 | 0.8298 | 36867 | −0.217 | 0/1/2 |
| 44b6_d2f34f90 | 0.9347 | 0.9753 | 13616 | −0.434 | 0/0/2 |
| 44b6_d5e7d891 | 0.8467 | 0.8534 | 41300 | −0.079 | 0/2/2 |
| 6bba_48816121 | 0.9304 | 0.9267 | 24916 | +0.040 | 0/1/5 |
| 6bba_09961292 | 0.9069 | 0.9111 | 29671 | −0.046 | 0/3/4 |
| 6bba_afb141ff | 0.9181 | 0.9236 | 5688 | −0.060 | 0/1/4 |
| 6bba_cdcfe533 | 0.9845 | 0.9908 | 27904 | −0.063 | 1/1/3 |

Totals: **tp=1, fp=10, fn=24.** `division_jaccard = 1/35 = 0.0286`.

Per acquisition: **44b6 = 0.8798, 6bba = 0.9420.**

## What that is worth

The metric is `score = adjusted_edge_jaccard + 0.1 * division_jaccard` — `core/metrics/score.py:134-138`,
with no 1.1 normalisation. So the division term is worth **0.1** and the champion banks **0.0029** of it.

| scenario | division_jaccard | Δ score |
|---|---|---|
| champion, as measured | 0.0286 | — |
| remove ALL ten false forks | 0.0400 | **+0.0011** |
| double the false forks | 0.0222 | −0.0006 |
| find half the missed ones | 0.3714 | **+0.0343** |
| perfect | 1.0000 | **+0.0971** |

## Why "DIVISION axis CLOSED" does not follow from the evidence that closed it

The jaccard is dominated by `fn=24`. Every division experiment on record moved the **fp** side — fork budget,
pruning, post-processing, knob sweeps on the real board. The ceiling on all of them *together* is **+0.0011**,
an order of magnitude below the noise floor.

So those sweeps did not read flat because the axis is dead. They read flat because **they could not read
anything else.** A knob that is structurally incapable of moving the number more than 0.001 returning 0.000
is not evidence about the term; it is a measurement of the knob's own reach. The old flat result is now
*explained* rather than believed, and it never licensed the conclusion that was drawn from it.

This is the denominator mistake in a new costume: the fp side is a subset of the term, and a rate computed
inside the subset you happened to sweep is not a rate over the term.

## What actually moves it, and what it costs

Only **recall**. Half the missed divisions is +0.0343 — over twice the 1.5% bar, and bigger than any other
lever currently open (Campaign A's offset head prices at roughly two edges).

The caveat that has to travel with it: **a fork is not free.** Emitting one rewrites edges, so a recall push
that fires wrongly pays in `adjusted_edge_jaccard`. That is a plausible mechanism for the global division-ILP
landing at −0.055 — it may have bought division recall with edge damage and been charged for it. Any gain
here must therefore be read on the **combined** score, never on `division_jaccard` alone.

Note also that the champion's division stage is already generating plenty of candidates — 399 to 7534
proposals per movie, budgeted down to 7–56 forks, of which 1 in 11 is right. The raw material is not
obviously the scarce thing; the selection over it might be.

## The oracle answered: it is SELECTION, not the detector

`python -m celltrack.eval.proxy_eval --divisions 4 --division-reach`, same eight movies, same operating point:

| movie | recovered | nodes_missing | no_fork | fork_rejected |
|---|---|---|---|---|
| 44b6_a21120c2 | 0 | 0 | 2 | 0 |
| 44b6_c50204e0 | 0 | 0 | 2 | 0 |
| 44b6_d2f34f90 | 0 | 0 | 2 | 0 |
| 44b6_d5e7d891 | 0 | 0 | 2 | 0 |
| 6bba_48816121 | 0 | 0 | 5 | 0 |
| 6bba_09961292 | 0 | 0 | 4 | 0 |
| 6bba_afb141ff | 0 | 2 | 2 | 0 |
| 6bba_cdcfe533 | 1 | 0 | 3 | 0 |
| **TOTAL** | **1** | **2** | **22** | **0** |

**22 of the 24 false negatives are reachable over nodes we already predict.** The parent is detected and both
daughter lineages are present within the match radius; nothing forked on them. Only 2 are a detector cost.

`fork_rejected = 0` is the sharper half of the reading. Not one fork lands on a true parent in a shape the
annotation then refuses — so this is not a topology or gate problem at all. The forks are emitted *somewhere
else entirely*.

This contradicts nothing in "division postproc dead, root is detector-temporal" except its conclusion: the
nodes are there, and the earlier note never had this measurement.

## Why widening the budget does not collect it — priced, not run

The derived budget is `node_count * 0.00113` = 25 forks for a 22k-node movie, and the stage proposes ~950. So
~200 forks were emitted across the eight movies and **only 10 scored as false positives** — the other ~190 sit
on unannotated cells and are invisible to the metric (E53). That makes a wider cut look nearly free.

It is not, because fp grows about ten times faster than tp. Uncapped was already measured at 1 TP / 807 FP per
movie. Extrapolating the measured 5% scored-fp rate at multiplier `m` (forks `25m`, scored fp `~10m`, tp
linear to 22 at the uncapped end):

| multiplier | tp | fp | division_jaccard | Δ score |
|---|---|---|---|---|
| 1 (shipped) | 1 | 10 | 0.029 | — |
| 4 | ~3 | ~40 | 0.046 | +0.0018 |
| 8 | ~6 | ~80 | 0.057 | +0.0028 |
| uncapped | 22 | ~320 | 0.067 | +0.0038 |

Every row is under the noise floor, and `max_added_forks` already exists to bracket the derivation, so no arm
was run. **The budget is not the binding constraint — the RANKING is.** A method has to put true parents into
the top 25 of 950, where today roughly 1 in 200 emitted forks is right.

## The next oracle, one level down

The nodes are reachable; the question is now whether the *candidate list* is. For each of the 22, is the true
(parent, daughter) pair among the ~950 proposals the stage gates down, or does candidacy drop it before any
ranking sees it?

- In the proposal list ⇒ a pure ranking problem, and the cheapest +0.03 on the board.
- Not in it ⇒ candidacy (`RunnerUpCandidacy` consults the head's runner-up only; `GeometricCandidacy` proposes
  every orphan). Switching candidacy is then the arm, and it is a flag, not a model.

That check instruments the stage rather than the score, so it is CPU-only over one cached run. Nothing gets
built until it answers.

## Secondary reading: the ruler itself

The same run is the first test of whether division-bearing movie selection explains our top-end proxy offset.
It does **not**. The champion reads ~0.91 here against its known LB **0.947** — still under-reading by
roughly 0.037, where the public 0.942 kernel's equivalently-built ruler tracked its own LB to ~0.000.

Two differences remain uncontrolled, so this is not yet a verdict on the ruler: their number is their pipeline
on their selection, not ours; and these eight movies are still *sparsely* annotated (296–1950 GT nodes) where
the hidden evaluation annotates densely. The divisions defect is fixed; the sparsity defect is not. E51's
in-sample argument also still stands untouched.

What the ruler unambiguously did buy is this page: the division term was invisible on `TEST_MOVIES` (3
divisions, three of four movies at zero), and it is not invisible now.
