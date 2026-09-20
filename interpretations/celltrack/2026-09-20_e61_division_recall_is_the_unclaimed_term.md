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

## The levers already exist, and were only ever arbitrated on the wrong half of the term

An audit of `AffinityDivisionConfig` against the donor kernel (`research/frontier_kernels/evg0942`,
public 0.942) finds every frontier division cue already built and reachable by flag:

| flag | what it is | ever measured on recall? |
|---|---|---|
| `division.ranking` ∈ {probability, geometry, symmetry} | the three orderings over proposals | no |
| `division.candidacy=geometric` | propose every unparented target, not just the head's runner-up | no |
| `division.require_persistence` | penalise a daughter that does not survive as a track | no |
| `division.require_mutual_nearest` | candidate must be the nearest unparented node **to the existing child** | no |
| `division.require_c3_divergence` (2.25um) | the two daughters must **separate** from t+1 to t+2 | no |
| `division.kernel_faithful` | byte-faithful replica of the donor's whole `add_safe_divisions_postlink` | no |

Every prior arbitration of these ran on the LB or on `fp`, i.e. on the ±0.001 half of the term that E61
prices out. None was ever read against `recovered`.

`require_c3_divergence` is the one with a mechanism the shipped rankings structurally lack: it is
**multi-frame**, where probability, geometry and symmetry are all pairwise. Two daughters move apart after
mitosis and a mis-linked lookalike pair does not, which is exactly the discrimination the confusor work
found pairwise cues cannot supply.

The reading that makes these ranking levers rather than precision knobs: under a budget cut of 25 out of
~950, a veto that deletes false proposals **promotes true ones into the cut**. A filter and a ranking are
the same instrument once a budget binds.

## The first sweep: four arms, four times `recovered=1`

| arm | proposals (movie 1 / 2) | forks | recovered |
|---|---|---|---|
| `ranking=symmetry` (shipped) | 950 / 7534 | 25 / 53 | 1 |
| `ranking=geometry` | 950 / 7534 | 25 / 53 | 1 |
| `ranking=probability` | 950 / 7534 | 25 / 53 | 1 |
| `require_persistence=true` | 950 / 7534 | 25 / 53 | 1 |
| `candidacy=geometric` | **1493 / 15914** | 25 / 53 | 1 |

The candidacy arm is the one that proves the instrument: the proposal pool doubles, so the override
demonstrably reaches the stage, and the flat `recovered` is a real null rather than an inert flag. Every
extra proposal it buys is discarded by the same budget cut — which is the ranking constraint restated, not
a candidacy answer.

The three rankings agreeing exactly is the sharper reading. Probability, geometry and symmetry are all
**pairwise** functions of the same two distances, so they are three orderings of one cue. That they select
the same top-25 out of 950 says the cue itself, not its parameterisation, is what fails to separate. A
fourth pairwise ranking would be a fifth measurement of the same null.

## The second sweep: the cue that is not pairwise quadruples recall

| arm | proposals (movie 1 / 2) | forks | recovered |
|---|---|---|---|
| shipped (`symmetry`) | 950 / 7534 | 25 / 53 | 1 |
| `parent_gate_um=1.0` — the plumbing assert | **0 / 0** | 0 / 0 | **0** |
| **`require_c3_divergence=true`** | **119 / 741** | 25 / 53 | **4** |
| `+ require_mutual_nearest=true` | 111 / 606 | 25 / 53 | 3 |
| `kernel_faithful=true` | 110 / 598 | **79 / 146** | 4 |

The gate arm is the assert the earlier nulls needed: a 1um parent gate emits nothing at all, and the one
division the shipped stage recovers duly disappears (`missed` rises 24 → 25). The instrument moves when the
mechanism moves, so the four flat rankings were a real null.

**`require_c3_divergence` takes `recovered` from 1 to 4.** It deletes 87% of the proposals — 950 down to
119 — and the true ones are in what survives. That is the whole reading: with the budget cut unchanged at
25, a filter that removes false proposals *is* the ranking lever, exactly as predicted, and the cue that
works is the one the three flat rankings structurally lack. Two daughters separate from t+1 to t+2 and a
mis-linked lookalike pair does not; pairwise distance cannot express that, and no reparameterisation of it
ever will.

Two arms of the same family are dominated and should not be carried forward:

- **mutual-nearest *costs* a division** (4 → 3). It is one-directional — the candidate must be nearest to
  the existing child — and a genuine sister in dense tissue sometimes is not.
- **kernel-faithful matches c3's recall with three times the forks** (79 vs 25 on movie 1). Same recall,
  strictly more false positives; the replica's value was never its selection.

## Scored on the combined metric: +0.0123, and the edge term went UP

`--divisions 4 --per-movie --set division.require_c3_divergence=true` — same eight movies, same operating
point, so the only difference is the cue:

| term | shipped | c3 | Δ |
|---|---|---|---|
| division tp/fp/fn | 1 / 10 / 24 | **4 / 5 / 21** | — |
| `division_jaccard` | 0.0286 | **0.1333** | **+0.0105** on the score |
| mean `adjusted_edge_jaccard` | 0.9180 | 0.9198 | **+0.0018** |
| **combined** | | | **≈ +0.0123** |

The caveat this page has carried from the start — a fork rewrites edges, so recall must be paid for in the
edge term — resolves the other way. Six of the eight movies' edge jaccard *improves*, because c3 halves
the false forks (10 → 5) at the same time as it quadruples the true ones. It is not a recall method that
buys divisions with edge damage; it is a precision filter that happens to raise recall, which is what a
veto under a binding budget always was.

+0.0123 is under the 1.5% bar and must be reported as directional. But it is free (a post-processing flag
on the banked champion, no retraining, no runtime), it is mechanism-named rather than swept, and it is the
largest single move measured since 0.947. Per the standing rule it is kept and composed, not claimed.

The standing warning applies to the submission decision and not to this measurement: the proxy blinds above
0.900, and these movies read ~0.91. What licenses reading this one is that the gain is located in the
division term, which is the part the proxy was shown to measure faithfully and the LB-tracking public
kernel never contested.

## c3's own ceiling: 6, and the budget is still not the constraint

Uncapped under c3 (`max_added_forks=100000`), `recovered` is **6** — against 4 at the shipped budget of 25.
Getting those two extra divisions costs ~700 forks a movie instead of 25. So the budget does not bind under
c3 either: it already admits 4 of the 6 that c3 can reach at all.

The sharper number is the 17 that stay `no_fork` even with every proposal accepted. They are not outranked
and not outbudgeted — **c3's own test deletes them, or `runner_up` candidacy never proposed them.** Two
things follow, and both are flags rather than builds:

- **c3 × geometric candidacy is untested.** The filter and the wider proposal pool have only ever been run
  apart. Geometric candidacy alone recovered nothing because the budget threw its extras away; under c3 the
  extras are pre-filtered, which is the configuration in which it could finally pay.
- **c3's veto rejects on `divergence is None`** — a daughter whose track gaps or branches at t+2 is deleted
  whatever its geometry — and `2.25um` is a donor constant, never derived for our spacing. Sweeping the
  threshold to 0.0 separates *must be measurable* (the structural single-successor requirement) from *must
  be large* (the magnitude). If 0.0 holds the recall, the magnitude was never the working part.

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
