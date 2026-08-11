# Divisions were closed on evidence that could not see them

The score is `adjusted_edge_jaccard + 0.1 · division_jaccard`. We earn **nothing** from the second term, and
we closed it twice: once on 2026-08-02 as *"detection-limited, low ceiling, deprioritise"*, and again this
morning when a re-test measured −0.0039 and I recorded it as refuted.

Both closures were made on a proxy that structurally cannot see the thing being judged.

## The proxy holds three divisions

Measured, not assumed:

| movie | annotated divisions |
|---|---|
| 44b6_0113de3b | 0 |
| 44b6_0b24845f | 0 |
| 6bba_05b6850b | 0 |
| 6bba_05db0fb1 | **3** |

Three targets, all in one movie. On the other three, **any** fork we emit is a pure false positive — there is
no true positive available to offset it. Meanwhile the hidden annotation is denser, and the EDA counted ~125
divisions in 6bba and ~26 in 44b6.

That makes the FP:TP ratio structurally hostile in a way the hidden set's is not. Eight hundred forks against
3 targets is ~250:1 and `div_jac` collapses to 0.0009 no matter how good the forks are; the same budget
against ~125 targets is nearer 6:1. **The division term can differ in sign between proxy and leaderboard, not
merely in magnitude.** This is the threshold situation again — an axis the local proxy cannot arbitrate — but
arrived at by arithmetic rather than by a hypothesis about annotation density.

And no division configuration has *ever* been submitted. The kernel was written on 2026-08-02 with its
threshold deliberately pinned so the delta would be attributable to divisions alone, and never pushed.

## The metric is far more forgiving than we believed

Our research corpus recorded *"requires strict topology, parent → 2 daughters, matched via bipartite
matching"*. Reading our own ported scorer says otherwise:

> Each ground-truth division opens a local window — grandparent, dividing parent, children, grandchildren —
> and the prediction is matched against that window on its own. A predicted fork recovers the division when it
> sits on the parent side and **reaches two of the ground-truth daughter lineages down two different branches
> of its own.**

Not exact edges. Not the exact frame. **Reachability inside a ±2-generation window.** Placement precision was
never the wall the corpus said it was.

## We forfeit the term by construction

`FlowLinker` splits every detection into in/out nodes joined by a capacity-1 arc. Each node can have at most
one outgoing link, so `dividing_rows()` is empty and `division_jaccard = 0` **exactly**. The prediction-centric
error decomposition confirms it from the other side: `EXTRA_CHILD = 0` on all four movies. We are not losing
the division term to detection limits. We never contest it.

## What actually binds — and I had it backwards

Three gates stand between us and division 232, the one case our own diagnosis calls recoverable (both
daughters detected, second is an orphan, parent→daughter 6.08 µm, sisters **13.5 µm apart** because *"they
separate onto opposite sides of the mother"*):

| gate | ships at | 232 needs |
|---|---|---|
| `parent_gate_um` | 4.7 | ≥ 6.08 |
| `sister_gate_um` | 7.2 | ≥ 13.5 |
| `min_second_prob` | 0.5 | 0.124 |

I argued the probability floor was the hard one, because >1200 parents have an orphan target at P ≥ 0.25 while
the true division sits at 0.124. **Measurement says otherwise:** at the shipped gates the recovery proposes
only **141 candidates at P ≥ 0.25 and 187 at P ≥ 0.05** — dropping the floor five-fold adds 46. The
**geometry** binds. The ~800-fork flood came from *widening* the geometry, which is precisely what reaching
232 requires.

So the tension is exact: **we cannot reach the reachable division without opening the gate that floods.**

## The cap is what costs, not the gates

| arm | clamped |
|---|---|
| control, no division | 0.9353 |
| reachable gates, cap ~297 forks | 0.9313 (−0.0040) |
| reachable gates, cap ~50 forks | **0.9348 (−0.0005)** |

At a tight absolute cap the whole stage costs **less than the noise floor**. That converts the submission from
a gamble into a bounded bet: measured downside on the term that transfers, unknown-but-arithmetically-real
upside on the term the proxy cannot see. It also corrects this morning's reading, where I attributed −0.0039
to "divisions" rather than to the budget.

The cap and the ranking are coupled. With `D` true divisions, `F` forks and precision `p`,
`div_jac = pF / (F + D − pF)`: more forks pay **only** if precision holds.

## Appearance cannot supply the precision

The literature offered a cheap discriminator — nine static 3D features and an SVM separate mitotic from
interphase nuclei at >90% in 3D developmental tissue. Measured on our data, as per-mother percentiles inside
two controls (three positives cannot support an AUC):

| feature | 232 | 624 | 771 |
|---|---|---|---|
| centroid offset | 82 | **5** | **97** |
| intensity mean | 18 | 75 | 12 |
| volume | 36 | 80 | 14 |
| elongation | 71 | 41 | 23 |

No feature puts all three in a tail. On every one, 624 and 771 sit on opposite sides — scatter, not a class.

**And the control fired in the confound direction**, which is why the null is trustworthy: median intensity is
1544 for false candidates, 1388 for the linked baseline, **1263 for annotated cells**. Brightness separates
*is a recovery candidate* from *is annotated* — not *is mitotic*. Without that control an intensity claim
would have read as a finding.

## What the frontier actually does

Source-read from seven byte-identical forks at public 0.915: **no learned verifier** (their DeepCenter veto is
switched off), an orphan candidacy restriction identical to ours, and then — the difference — **ranking by
geometry**, `parent_dist + 0.15 · sister_dist` ascending, under frame and global caps. They emit 311 forks
over 119,005 edges.

**We rank by the edge probability.** The one signal measured incapable of separating. That is why capping
evicts the true fork: it sits near the bottom of a P-ranking.

## What is left

Geometry ranking, plus `daughter_angle_cosine` — true daughters separate to **opposite sides** of the mother,
which is literally how 232 is described, while a false sister sits at a random angle. The frontier does not use
it. If that fails, the remaining candidate is the Hirsch/Linajea learned cell-state head — a five-frame 3D
patch stack, **7.5× fewer false-positive divisions on isotropic light-sheet**, MIT-licensed — which the
appearance null does not touch, because it reads a temporal stack rather than static features off a detection
centre.

## The lesson that generalises

Divisions were closed twice on measurements that could not observe the quantity in question: once against a
proxy holding three targets, and once under a configuration whose gates excluded the only reachable case. Both
readings were *correct about what they measured* and wrong about what they were taken to mean.

Before closing an axis, ask what the measurement could have shown if the thing were true. Here the answer was
"almost nothing" — and that was knowable in advance from the annotation counts.
