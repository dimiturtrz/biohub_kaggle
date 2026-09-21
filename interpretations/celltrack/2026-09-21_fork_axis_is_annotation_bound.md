# E64 — the fork axis prices at +0.126 with the annotation and +0.008 without it

2026-09-21. Instruments: `celltrack/postproc/division_reparent.py`, `celltrack/eval/fork_rank_report.py`,
scored through `celltrack/eval/official_ruler.py` on the six test movies. Champion control = **0.8463**.

## What was measured

Four matched arms through the same ruler, all on the champion's own predictions:

| arm | edge_jac | divJ | score | Δ |
|---|---|---|---|---|
| champion (control) | 0.8440 | 0.0228 | **0.8463** | — |
| oracle re-parent only | 0.8453 | 0.0415 | 0.8494 | +0.0031 |
| blind fork cull (no GT) | 0.8497 | 0.0000 | 0.8497 | +0.0034 |
| oracle cull + re-parent | 0.8726 | 1.0000 | **0.9726** | **+0.1263** |

The oracle arm is the largest number this project has measured on the local official ruler. It is also
entirely annotation-bound, and this file is the record of why.

## The re-parent is edge-free, and it is still small

The official metric charges a predicted link when EITHER endpoint matches an annotated node the annotation
continues through — `pred_valid = out_valid(source) | in_valid(target)` in `biohub_tracking.metrics`, the same
OR as `core/metrics/edges.py`'s `ChargedLinks._countable`. A matched daughter is such a node, so the thief's
link into her is already a false positive: moving her to her annotated parent removes an FP and adds a TP.
Measured: eTP +8, eFP −4, eFN −8, zero collateral. The mechanism is confirmed.

It buys **+0.0031**. E61's table had implied ~25 recoverable divisions; the six-movie ruler set holds **11
annotated divisions in total**, six of which the champion already gets. The projection was 10× too high
because it was read at a different operating point — the E61 lesson, repeating on me.

## The 0.126 splits into two sub-problems, and both are chance-bound

Decomposing the oracle: the division term contributes **+0.0977** (0.1 × (1.0 − 0.0228)) and the edge term
**+0.0286** (edge_jac 0.8440 → 0.8726). The blind arm reaches only +0.0034 of the whole thing, so all of it
is DISCRIMINATION, none of it the cut.

**Division term.** `fork_rank_report` ranks all 7185 emitted forks under the five `fork_ranking` strategies
and their compositions, and reports where the true-division forks land. Best combination
(`symmetry+surviving+diverging`) reaches divJ 0.10 at a keep-budget of 85, worth **+0.008**; on the metric's
real denominator of 11 divisions it is ~+0.0036. Every other cue is below that. The term is a ratio
`k / (11 + m)`, so it cannot be bought by deletion alone — the blind cull removes all 254 charged false forks
and still scores 0.0000, because it keeps none of the 11.

**Edge term.** This is the sharper grade and the one that closes the axis. Of 7185 forks, **245** are resolved
by the annotation to exactly one branch — those are the only ones a keep-one rule can be graded on.

| keep-one rule | branch accuracy | annotated edges lost of 245 |
|---|---|---|
| probability (what the blind arm does) | 0.5878 | 101 |
| survival (`frames_ahead`) | 0.5551 | 109 |
| survival + probability | 0.5551 | 109 |
| nearest (µm) | **0.6000** | **98** |
| nearest + survival | 0.5551 | 109 |

Chance on a two-branch fork is 0.50. The entire spread between the shipped rule and the best alternative is
**3 edges of 245** — noise. The cues fired (they disagree with each other and reorder the set), and the
denominator is stated.

## Why this is coherent rather than a null to explain away

`survival` was the cue with an argument behind it: E39 found fork-unique nodes are BLIPS, so the surplus
branch should die within a frame or two while the real continuation runs on. It is the WORST of the five
here. That is not a contradiction of E39 — E39's population is fork-unique *nodes* across the whole graph,
not the branches of forks the annotation resolves — but it does say the blip shape is absent from this
population. At a fork the annotation resolves, the surplus branch looks like a continuation.

Which is the confusor keystone, again, from a new direction: *confusor pairwise-unresolvable → needs
multi-frame* (rival nearer 97.3% of the time, appearance 0.115 against chance 0.260). A fork's two branches
are exactly a pairwise decision on a rival, and they price out the same way.

## Verdict

- **Fork PRECISION via post-processing: CLOSED.** ≤ +0.008, sub-floor, on a correctly-sized denominator.
- **Division RECALL via re-parenting: CLOSED at this operating point.** +0.0031. Mechanism proven and
  edge-free, so the asset survives — what died is pricing it offline on a set with 11 divisions in it.
- **The 0.126 is not a lever.** It is the value of the annotation, and it now sizes what a model-side
  division/branch discriminator would be worth if one existed: +0.126 is the ceiling, and nothing readable
  off the linked graph gets past +0.008 of it.
- **Do not re-run either arm as an improvement.** Both are kept as instruments.

## Kept assets

`plan_reparents` (edge-free daughter re-parenting, proven), `plan_fork_cull` (with a blind arm that is
shippable and worth +0.0034 — free, mechanism-sound, sub-floor, so directional only), `branch_accuracy` (the
245-fork grader: any future model-side branch scorer is priced against 0.5878 in one CPU run).
