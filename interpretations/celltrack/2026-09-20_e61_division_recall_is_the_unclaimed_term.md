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

- **c3 × geometric candidacy: run, and it recovers 4 — exactly c3 alone.** The filter and the wider
  proposal pool had only ever been run apart, and the hypothesis was that geometric candidacy recovered
  nothing only because the budget threw its extras away. It does not hold: with the extras pre-filtered by
  c3 and the budget no longer spending itself on them, `recovered` does not move (`nodes_missing=2`,
  `no_fork=19`, unchanged). **The candidacy pool is not what withholds the 17.** Whatever `runner_up`
  fails to propose, `geometric` does not propose either — which leaves c3's own veto as the single named
  cause, and makes the threshold sweep the only live question of the two.

  Re-run at `max_added_forks=100` it recovers **5** — again exactly what c3 alone recovers at that budget.
  The null therefore holds at two different cut sizes, so it is a property of the proposal pool and not an
  artifact of the budget happening to truncate in the same place twice.
- **c3's veto rejects on `divergence is None`** — a daughter whose track gaps or branches at t+2 is deleted
  whatever its geometry — and `2.25um` is a donor constant, never derived for our spacing. Sweeping the
  threshold to 0.0 separates *must be measurable* (the structural single-successor requirement) from *must
  be large* (the magnitude). If 0.0 holds the recall, the magnitude was never the working part.

## The threshold sweep: loosening the veto loses recall, which is the mechanism proving itself

| `c3_divergence_um` | recovered |
| --- | --- |
| 0.0 — structure only, any measurable separation passes | 2 |
| 1.0 | 2 |
| **2.25 — shipped, the donor's constant** | **4** |
| 4.0 | 3 |

The curve is not monotone and it does not fall away from the magnitude — it falls away from the strictness.
At 0.0 the requirement is purely structural (both daughters must have a single successor at t+2, so a
divergence is computable at all) and recall collapses to 2, barely above the ungated baseline of 1.

That is the predicted re-inflation, and it is the strongest evidence yet for *why* c3 works. Admitting every
measurable divergence refills the proposal pool, the budget of 25 binds again, and the true forks are pushed
back out of the cut — the same failure that made `probability`, `geometry` and `symmetry` read identically
flat. **Under a binding budget the veto is not a filter that happens to precede a ranking; it IS the
ranking.** Its strength is the parameter, and weakening it returns the system to the null.

Two consequences:

- **The `divergence is None` relaxation is priced out without being built.** Readmitting candidates whose
  t+2 is a gap or a branch loosens the veto in exactly the direction the sweep already measured as losing.
  The 19 `no_fork` divisions it would target are not reachable this way.
- **Do not tune 2.25.** Four, three and two recovered divisions out of 25 is integer scatter well inside the
  noise floor; the only difference here that clears it is c3 on (4) against c3 off (1). The constant is the
  donor's and was never fitted by us, which is the one thing keeping this from being a sweep-selected peak —
  and it should stay that way. The finding is the flag, not its value.

## The budget-100 arm, scored: the fifth division costs more than it pays

Widening the budget under c3 buys a fifth recovered division. Scored on the combined metric it loses on
**both** terms:

| arm | score | edge | division jaccard | forks emitted |
| --- | --- | --- | --- | --- |
| c3, shipped budget | **0.9359** | 0.9226 | 0.1333 | 36 / 7 / 33 |
| c3, `max_added_forks=100` | 0.9312 | 0.9205 | 0.1064 | 94 / 51 / 100 |

The division jaccard *falls* while its true positives rise, because the false positives rise faster — which
is the arithmetic that priced budget widening out in the first place, now measured rather than argued. The
edge term falls too, so the extra forks are also rewriting edges badly. Both halves of the metric agree, and
the shipped budget stands.

This closes the budget axis under c3 from both ends: uncapped reaches 6 divisions and is unscoreable at ~700
forks a movie, 100 reaches 5 and is measurably worse, and 25 reaches 4 and is the best of the three.

## CORRECTION: the champion already runs c3, so +0.0123 is a gap-closure, not a lever

The banked 0.947 is a harvested **notebook** (`kaggle/kernels/celltrack-public-0947/`), not one of our
python kernels, so it never mounts `TrackerConfig.shipped()`. Reading its own constants:

```
SAFE_DIV_REQUIRE_DIVERGENCE = os.environ.get('BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE', '1')
SAFE_DIV_DIVERGE_UM'] = '2.25'
```

**The champion already has the C3 divergence gate on, at the same 2.25µm.** Everything measured above was
our own tracker, whose `shipped()` had the flag off. So `+0.0123` is our pipeline closing a gap to the
champion — it is NOT available on top of 0.947, and there is no submission in it.

Two things I got wrong, in order:

1. **I priced a flag without reading the champion's constant for it first.** That is the explicit rule from
   E52 ("check the champion's constant first"), and skipping it turned a replication into a claimed win for
   several hours of arms.
2. **"Free on the banked champion" was the wrong denominator**, the same error CLAUDE.md names: the numbers
   came from `proxy_eval` running *our* recipe, and I attributed them to the champion because our recipe is
   the thing I call shipped.

What survives, and it is the part worth keeping:

- **The term arithmetic still stands.** Even at the c3-on counts (4/5/21, jaccard 0.1333) only **0.0133 of
  an available 0.1** is claimed. The division term is still ~0.087 unclaimed, now measured on a
  configuration the champion actually runs rather than on one it does not.
- **Every closure above still holds** — budget (both ends), candidacy (both budgets), ranking, threshold.
  They were measured on the c3-on configuration, which is the champion's configuration, so they transfer.
- **The `tp=1, fp=10, fn=24` figure that opened this page is retired.** It describes our tracker with the
  gate off, and no longer describes anything we ship or bank.

The honest one-line summary of E61 is now: *the division term is the large unclaimed term, our replica was
missing the champion's divergence gate and is now fixed, and nothing here beats 0.947.*

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

## The rest of the replica gap: the champion's gate constants do not transplant

The correction above asked the obvious follow-up — if our replica was missing the champion's divergence
gate, what else is it missing? The champion's `SAFE_DIV` bracket answers it:

| constant | champion notebook | our `shipped()` | our `kernel_faithful` |
|---|---|---|---|
| parent gate | `SAFE_DIV_MAX_UM` 4.7 | 10.0 (inherits the linker gate) | 8.0 |
| sister gate | `SAFE_DIV_SISTER_MAX_UM` 7.2 | 20.0 (2x parent) | 11.0 |
| existing-child gate | `SAFE_DIV_EXISTING_CHILD_MAX_UM` 7.8 | `inf` | 10.0 |
| per-frame cap | `SAFE_DIV_FRAME_FRAC_CAP` 0.008 | budget from `_DIVISION_RATE` | 0.0076 |
| global cap | `SAFE_DIV_GLOBAL_FRAC_CAP` 0.004 | — | 0.00375 |

Three different bracket, three ways. Our `kernel_faithful` path is faithful to *a* donor, but not to this
one — which is itself worth recording, because `kernel_faithful` is named as though it replicates the
champion and does not.

Adopting the champion's numbers **loses recall**:

| arm | recovered |
|---|---|
| c3, our gates (10.0 / 20.0 / inf) | **4** |
| c3, champion gates (4.7 / 7.2 / 7.8) | 2 |
| c3 + mutual-NN, champion gates | 2 |

Two of 25 against four of 25 is integer scatter by our own noise rule, so the honest claim is *no gain,
direction negative* — not "their gates are worse". But it is enough to kill the transplant, and it settles
the `require_mutual_nearest` question that the wide-gate arm left open: under the tight bracket mutual-NN is
**inert** (2 either way), where under ours it cost a division. It was never an independent gate; it was
removing proposals the gates were already removing.

The generalizable half: **a donor constant transplants only with the component it was fitted against.**
`c3_divergence_um = 2.25` moved over because it is a threshold on a physical quantity — how far two
daughters separate in one frame — which our detector measures the same way theirs does. The gate bracket did
not move over, because it is fitted to the champion's *candidate distribution*: their detector and affinity
put proposals at different distances than ours do. Reading a replica gap therefore has to be done per
constant with a mechanism attached, never wholesale by diffing the env block. Diffing finds the candidates;
only the mechanism says which are gaps and which are co-adaptations.

## The instrument that reopens the axis: 11 of 19 are still in reach

Every candidacy in `AffinityDivisionRecovery` draws its proposed second daughter from the orphan pool —
`RunnerUpCandidacy` gates on `gap.orphan[runner_up]`, `GeometricCandidacy` on `gap.orphan[index]`. So a
division whose daughter lineages the linker has already claimed is outside this stage at **every** setting of
its gates, ranking, candidacy and budget. That is the mechanical explanation for the candidacy null recorded
above: geometric and runner_up widen the search *inside the same pool*, which is why 1493 proposals against
950 bought exactly zero recall.

It also predicted something, so `DivisionReach.unproposable` was added to price it: the count of `no_fork`
divisions where no daughter lineage contains an unparented node. **The prediction was that all of them would
be unproposable. It is refuted:**

```
TOTAL recovered=4 nodes_missing=2 no_fork=19 (unproposable=8) fork_rejected=0
```

**8 of 19** are structurally out of reach — the daughter is already claimed, and only a re-parent could take
her back, which is an ILP-level swap and not something this stage can express. That connects to
[E54c](../../../interpretations/celltrack): every thief is an unannotated real track, so the swap is contested
rather than free.

**10 of 19 are not.** The orphan is present, the parent has its single child, the gates would admit a
proposal — and no fork was emitted. `fork_rejected=0` throughout, so it is not that a fork was emitted in a
shape the annotation refused. Those 10 are the tightest target left on the division axis, and they are worth
roughly `14/(14+5+11) = 0.47` division jaccard against the current 0.1333 — about **+0.033 combined** at a
perfect, FP-free recovery. The FP-free part will not hold, but even a third of it clears the 1.5% bar.

The open question is which stage drops them. The c3 veto deletes 87% of proposals (950 to 119), so the first
suspect is the gate itself vetoing true divisions — checkable by reading `unproposable` with c3 off, since
the pool membership the count measures does not depend on the veto, while what survives to a fork does.

### Two hypotheses for the 10, both refuted — and one instrument bug found in my own count

The first suspect was the c3 veto deleting true divisions, since it removes 87% of proposals. Reading the
count with the gate off settles it: `unproposable` is **8 with c3 on and 8 with c3 off**, exactly as it
should be — the count is a property of the linker's output, not of the veto — while `no_fork` goes 19 to 22.
So the gate **converts three** proposable-but-lost divisions into recovered ones. It is not vetoing truth.

The second suspect was the parent side. `_gap_proposals` skips any parent whose out-degree is not one, so a
division whose parent track the linker ended is invisible however many orphans surround it. `no_kept_child`
prices that at **0 of 19, on every movie.** There is always a forkable node on the parent side. Refuted.

The third thing found was a flaw in my own instrument, which mattered more than either hypothesis. The reach
oracle builds a daughter lineage as `{child, *successors[child]}`, so a division whose daughter was claimed
but whose *grandchild* was an orphan counted as proposable — while the stage can only ever propose the node
one frame past the parent. Restricting the check to the lineage HEADS moves the count 8 to **9**, and the
proposable set 11 to **10**. One division in the headline number was an artifact of the ruler. The finding
survives, slightly smaller, which is the only reason it is worth reporting at all.

What the 10 are NOT, now: not claimed (that is the 9), not parentless (that is 0), not c3-vetoed (the gate
helps), not fork-rejected (0 throughout), and **not budget-lost** — c3 uncapped recovers 6, not 16, so
admitting every proposal does not find them. That leaves the candidacy's own filters and `_within_gates` as
the only places left in the proposal path.

## The proposal path, priced to its ceiling: 7 of 25, and the other 10 need a different stage

The last untried corner was candidacy crossed with an unbound budget. Geometric candidacy read
*identical* to runner-up at budget 25 (4) and at budget 100 (5), so the earlier arms said candidacy
does not bind. Uncapped it reaches **7**, where uncapped runner-up reached 6.

```
arm                                    forks emitted   recovered   spurious
c3, shipped budget 25                      25 / 53          4          5
c3, budget 100                                 —            5          —
c3, uncapped, runner_up candidacy          ~700/movie       6          —
c3, uncapped, geometric candidacy         3114 total        7         59
```

So candidacy and budget **interact**: geometric proposes reachable daughters that lose the ranking at
every finite cut. That is a ranking fact inside a wider pool, not a pool-membership fact — and it is
the opposite of what two cut sizes predicted on their own. Recorded as a caution about reading an
interaction off single-axis arms.

It is not shippable and was never going to be. 3114 emitted forks buy seven true divisions and 59
*annotated* spurious ones; division jaccard falls to `7/(7+59+18) = 0.083` against the shipped c3 arm's
`4/(4+5+21) = 0.133`. The 3114 also contains a large unannotated remainder that the division term
cannot see at all ([[E53]]) but that the adjacency term pays for — which is exactly why the budget-100
arm scored 0.9312 against 0.9359. The arm is an instrument, not a candidate.

**What it prices is the stage's ceiling.** With the veto on, candidacy widened and the budget removed —
every knob in the proposal path at its most permissive — the stage reaches 7 of 25. Of the 16 that
remain, **10 are `unproposable`**: their second daughter is already parented by another predicted
track, so no orphan-pool proposal can ever name it, at any setting. The residual **6** are proposable
and still unreached, leaving only `_within_gates` and the candidacy's own probability filters as
suspects — the whole rest of the path is now measured and eliminated.

That splits the remaining division recall into two mechanisms, not one:

- **6 divisions** are a gate/threshold problem inside a stage that already exists. Cheap to test.
- **10 divisions** (40% of every miss) require *taking* a daughter from a track that currently claims
  it. That is not a recovery stage at all — it is a re-parenting decision, and it belongs inside the
  linker's own objective. [[E54c]] found the same shape from the other end: every "thief" was an
  unannotated but genuinely long track, so the slot really is claimed and the ILP really is right
  under its current objective. A division-aware objective is the only thing that reaches these.

A perfect FP-free recovery of the 6 plus the 7 already reachable values at
`13/(13+5+12) = 0.433` jaccard against the shipped 0.133, i.e. **~+0.030 combined** — and the FP-free
part will not hold. The honest read is that the reachable half of the axis is worth a fraction of a
bar, and the half that could carry it needs a linker change, not a post-processing change.

## The residual six are a RANKING failure, not a gate: the wrong daughter wins

With the gates opened to 1000 µm (parent, sister and existing-child all), candidacy geometric, budget
removed and the veto on:

```
TOTAL recovered=7 nodes_missing=2 no_fork=11 (unproposable=10) fork_rejected=5
```

Recovery does not move — still 7. What moves is where the misses sit: `no_fork` falls 16 → 11 and
`fork_rejected` rises 0 → 5. Opening the gates by two orders of magnitude does not find one extra
division; it emits a fork at five of the right mothers and picks the **wrong second daughter** at every
one. The gates were never what stood between us and those five.

That closes the last suspect on the proposal path and splits all 25 missed divisions cleanly:

| count | cause | what could reach it |
|-------|-------|---------------------|
| 2 | daughter lineage never detected | the detector |
| 7 | already recovered under c3 + uncapped + geometric | — |
| 10 | second daughter already parented by another track | a displacing (steal) stage |
| 5 | fork emitted at the right mother, wrong daughter chosen | the RANKING |
| 1 | still unexplained | — |

Two levers, each with a named mechanism and neither of them a knob on the existing stage. The 10 need a
pool the stage cannot see; the 5 need a cue that separates a true sister from a near one, which is the
same discrimination [[confusor]] measured to be pairwise-unresolvable — and exactly what the multi-frame
c3 divergence already supplies as a veto rather than as a score. Ranking BY divergence, rather than
vetoing on it, is the cheap arm that follows from this table.

## The C3 veto is not free: it deletes six true divisions to buy its +0.0123

The arm that removes the veto at maximum permissiveness — geometric candidacy, uncapped budget,
`require_c3_divergence=false` — reads:

```
TOTAL recovered=13 nodes_missing=1 no_fork=11 (unproposable=10 no_kept_child=0) fork_rejected=0
```

against the same arm WITH the veto, which read `recovered=7 … fork_rejected=5`. Six divisions move
from lost to recovered, and `fork_rejected` empties completely.

**This overturns the reading in the section above.** I took `fork_rejected=5` under wide gates as
evidence that the stage emits a fork at the right mother and picks the wrong second daughter — a
RANKING failure. It is not. `FORK_REJECTED` is assigned post hoc, on the produced graph: it means a
fork exists somewhere on this division's parent side whose topology does not match the annotation.
Under gates opened by two orders of magnitude the stage emits 726 forks on one movie, so *some* fork
lands near almost any mother. The stage that actually lost those five was the C3 gate, which vetoed
the true pair before the ranking ever saw it.

So the causal table splits differently:

| count | cause | what could reach it |
|-------|-------|---------------------|
| 1 | daughter lineage never detected | the detector |
| 7 | recovered under the veto | — |
| 6 | **true fork vetoed by C3** | divergence as a SCORE, not a gate |
| 10 | second daughter already parented (`unproposable`) | the steal stage |
| 1 | still unexplained | — |

The instrument's lesson generalises: **a stage label read off the OUTPUT graph is not a record of what
the pipeline decided.** `fork_rejected` cannot distinguish "the true candidate was proposed and
rejected" from "a different fork happened to land nearby". Only the A/B against the suspected stage
separates them, which is why the veto-off arm is the measurement and the label was the hypothesis.

### Why divergence belongs in the ranking

The veto earns +0.0123 by suppressing false forks, and it costs six true ones. Those are not in
tension if the same quantity orders the set instead of censoring it: a pair that does not move apart
sorts last and the budget evicts it, while a true pair the gate would have deleted for falling a few
tenths short of 2.25um stays reachable. `DivergingDaughterRanking` (commit `1c10595`) is that term,
dimensionless so it can be weighed against the geometry without a fitted scale.

The arm that tests it is `prefer_divergence=true` with `require_c3_divergence=FALSE` — soft where the
champion is hard. Running the veto-on-plus-rank arm first only prices what the preference adds
INSIDE the censored set, which is the smaller question.

## Divergence-as-a-score, measured inside the veto: 3 recovered against the shipped 4

The first arm off the new `prefer_divergence` term kept `require_c3_divergence=true` — shipped candidacy,
shipped budget, the preference added on top. It recovers **3** where the shipped champion setting recovers
**4**. Directionally negative, and a one-division difference on 25 is well inside what a reordering can move
by accident, so the number is not the finding. The reason it cannot be anything else is.

The veto admits only pairs whose separation grows past 2.25µm. Inside that admitted set, divergence is a cue
that has already fired for every member: the score is being asked to order candidates that are all on the
same side of it, so what it mostly reads there is the residual spread of a saturated variable. Adding it to
the ranking buys nothing and costs whatever the base ranking knew. The arm prices what the preference adds
*inside the censored set*, which is the small question — and it answers it no.

The arm that can say anything is the one where the score replaces the gate rather than decorating it:
`require_c3_divergence=false` with `prefer_divergence=true`, so the six true forks the veto deletes are back
in the set and the score is the thing that sorts them. That needs its own matched control —
`require_c3_divergence=false` alone, same candidacy and budget — because dropping the veto changes the
recovered set by itself, and an uncontrolled soft arm would credit the preference for what the relaxation did.
Both are running.
