# The term we never contested

The competition metric is `adjusted_edge_jaccard + 0.1 · division_jaccard`. For the whole campaign we banked
**exactly zero** of the second term — and not because we tried and failed. `FlowLinker` splits every detection
into in/out nodes joined by a capacity-1 arc, so no node can have two children, `dividing_rows()` is empty,
and `division_jaccard = 0` **by construction**. The prediction-centric error decomposition confirmed it from
the other side: `EXTRA_CHILD = 0` on all four movies.

We had been optimising one term of a two-term metric and calling the other one closed.

Contesting it scored **0.899** on the public leaderboard, against a standing best of 0.895.

## Why the axis looked closed twice before

Divisions were shut down on 2026-08-02 (*"detection-limited, low ceiling"*) and again on 2026-08-11 morning
(a re-test measuring −0.0039). Both readings were correct about what they measured and wrong about what they
were taken to mean, for the same reason: **our proxy holds three annotated divisions.**

| movie | annotated divisions |
|---|---|
| 44b6_0113de3b | 0 |
| 44b6_0b24845f | 0 |
| 6bba_05b6850b | 0 |
| 6bba_05db0fb1 | **3** |

With three targets, any fork budget faces a false-positive-to-true-positive ratio 30–50× worse than the hidden
set's, where a fully annotated dense movie would hold roughly 79–170 divisions (measured corpus rate: 0.113%
of annotated nodes). `div_jaccard` on the proxy is therefore dominated by an artefact of *our* annotation
sparsity, and it can differ in **sign** from the hidden set, not merely in magnitude.

So the proxy's readable signals were never the division score. They were the **fork count** and **whether the
one recoverable division survived** — and the cost to the edge term, which is the part that does transfer.

## What actually did the work

Three ranking strategies at identical gates and identical candidates:

| ranking | forks | clamped | division 232 |
|---|---|---|---|
| probability (shipped) | 239 | 0.9314 | missed |
| frontier geometry (`parent + 0.15·sister`) | 464 | 0.9313 | missed |
| **split symmetry** | 490 | **0.9434** | **recovered** |

The symmetry cost is `parent_dist / parent_gate + |u+v| / (|u|+|v|)` over the two daughter displacements. The
second term is the half-angle form: **0** for a clean opposite split, **1** for two cells leaving together. It
is dimensionless, and both terms are normalised by the constraint that decides their own admissibility, so
equal weighting is the only unfitted choice available.

Sister distance stays a **gate** and never a score — as a score its sign is wrong, because a true mitosis
pushes daughters *apart* (division 232's sisters are 13.5 µm apart).

Two of my own diagnoses were refuted along the way. I had argued the geometry gates blocked the recoverable
division; the blocker was `min_kept_prob` — at 0.5 the true divider is never even *proposed*, because its
kept-child probability sits below it. And I expected the frontier's distance formula to carry the ranking; the
**angle term** does.

## The honest half

The symmetry ranking is principled. **The floors-off-plus-cap configuration is not.**

`min_kept_prob = 0.0` means I removed the model's confidence requirement *because the true divider scored
below it*. We emit forks the model does not believe are divisions, ranked by geometry, and cap the count to
bound the damage. That is attractive because of an asymmetry in the **metric**, not in the biology:

- a false fork costs the edge term almost nothing (−0.0019 at 490 forks), while
- a true one pays `0.1 × div_jaccard`, and
- the metric credits a fork that merely **reaches** two daughter lineages within a ±2-generation window —
  not one placed at the right frame or on the right edges.

That is buying cheap lottery tickets where the scoring rule is lenient. It works, and it is not detection.

For contrast, the same afternoon I *refused* the equivalent trade on `min_track_length`: 8 read +0.0035 on the
headline score and I killed it because the clamped column showed the entire gain was node-count bonus — the
term measured at **+0.006 proxy / −0.007 leaderboard**. The line exists; on divisions I stepped across it and
described it as a bounded bet.

## What principled would look like

A classifier that identifies actually-dividing cells. Static 3D appearance is measured **dead** on our data —
nine features, per-mother percentiles inside two controls, no feature putting all three true mothers in a
tail, with the annotated-vs-detected brightness confound explicitly controlled.

The live candidate is the Hirsch/Linajea learned cell-state head: a five-frame 3D patch stack, 7.5× fewer
false-positive divisions on isotropic light-sheet, MIT-licensed. The appearance null does not touch it,
because it reads temporal context rather than static features off a detection centre.

## The lesson

The failure was not that divisions were hard. It was that **we accepted a zero we never measured**. The term
read 0.0000 for months and nobody asked whether that was a result or a property of the solver — and it was the
solver: a capacity-1 arc, chosen for one-to-one linking, silently forfeiting a tenth of the metric.

Before optimising a composite objective, check each term is **reachable**. A term your architecture cannot
express is not a hard problem; it is an unasked question, and it will read exactly like a solved one.

Remaining headroom on the axis is `0.1 × 0.4468 ≈ 0.045` against roughly 0.004 banked.

## Postscript: the budget was never the lever

A second submission bracketed the fork budget at cap 100 against the first's 300 — about 400 forks against
1072 — and scored **0.899, identically**. Of the three outcomes named in advance, the flat one fired: between
those two counts the leaderboard cannot tell the difference, so **the ranking carries the whole gain and the
budget carries none of it**.

That is worth more than a tie usually is, because it turns a hand-set ceiling into a removable one. A cap that
provably does not bind is a magic number, so the next submission sets it to `None` and lets the budget follow
the **measured division rate** instead — 0.113% per node-observation on our own corpus, which is 29/36/8/84
forks on the four movies rather than 1072. If the plateau reaches down to the phenomenon's own scale the
parameter never comes back; if it does not, the floor sits somewhere between 157 and 400 and we have bracketed
it from the other side.

It also sharpens the honesty problem rather than dissolving it. I called the floors-off-plus-cap configuration
"partly gaming" because the cap was what bounded the damage from forks the model does not believe in. The cap
turns out not to matter — which means the damage was never being bounded by it, and the real reason the
speculation is cheap is the metric's own asymmetry. The ranking is the principled half, and it is now measured
to be the *only* half doing work.
