# E59 — a re-solve prices perfect localization at two edges, and the 76/76 ceiling was near-tautological

2026-09-20 · CPU only, one run, no GPU, no submission · `kaggle/localization_resolve.py`

## What was claimed

E57 concluded that **localization is the only channel left**, on two legs:

1. a **broken** GT edge's endpoints sit a median **4.08 µm** from the cell against **1.46 µm** for a
   recovered one, and
2. `cue_oracle.py`'s fourth row — distance to the target's **true GT position** ranks the true successor
   first **76/76**, against 10/76 for plain distance and 10/76 for oracle velocity.

Leg 2 is what turned a correlation into a lever: if the simplest discriminator there is becomes perfect
once the coordinates are right, the coordinates are the bug. E57 itself flagged that this could not be
converted into a score — snapping a node onto its GT coordinate is a **no-op for the metric by
construction**, because the broken/recovered verdict depends only on the node matching, which the snap
improves directly, and the solver has already run — and it asked for a **re-solve**. This is that re-solve.

## The instrument

Arms differ in exactly one thing: **the coordinates the linker is allowed to see**. Every arm is scored on
the **original** predicted positions, so the node matching, the node-count term and the estimated cell
count are identical across arms and the entire delta lands on edges. `shrink` is the fraction of each
matched node's localization error handed to the linker: 0.0 none, 0.5 half — the realistic target for a
trained head — 1.0 all. The linker is a plain per-frame Hungarian on distance at the champion's own
14 µm output radius, the same in every arm.

**The harness reproduces the number it has to.** Scored on the champion's own edges it returns
**2022 recovered / 105 broken**, E54's figures exactly, on the four public-GT movies.

## The measurement

| arm | GT edges recovered | broken | score |
|---|---|---|---|
| champion edges (reference level) | 2022 | 105 | 0.8932 |
| re-solve, shrink 0.0 (control) | 1951 | 176 | 0.8043 |
| re-solve, shrink 0.5 | 1946 | 181 | 0.7990 |
| re-solve, shrink 1.0 (perfect coordinates) | 1953 | **174** | 0.8059 |

**Perfect coordinates buy two edges and +0.0015.** Half the correction buys −0.0053. Both sit under the
0.01–0.02 noise floor and the pair is non-monotone, which is what a null looks like. The ranking oracle
implied 76 of 2127 edges, about **+0.036**; the re-solve delivers **1/38th of it**.

## Why leg 2 was never evidence

Row four ranks candidates by distance to `gt_xyz[target]`. The predicted node the scorer matched to that
target is, by construction of the matching, within 7 µm of that very point — and the matching is a
Hungarian assignment that would mostly have preferred any rival that were nearer. So the row asks "is the
node matched to this cell the node closest to this cell?" and answers yes. It is **near-tautological**, and
it measures the matcher, not a cue a tracker could use. Leg 1 survives as a correlation — broken edges
really do sit on worse-localized nodes — but a correlation is what it stays: crowded regions produce both.

## What this does and does not close

It does **not** close the centre-offset head, and the reason is the instrument's own limit, one level up
from the one E57 named. Only the **2172 of 122808** predicted nodes the scorer can match have a GT
coordinate to be snapped onto — 1.8 %. E54c established that the edge is stolen by an **unannotated real
track**, and that thief keeps its wrong position in every arm here. A trained head corrects the whole
field, including the thief; this oracle corrects one side of a two-body problem. Its +2 is therefore a
**lower bound from a one-sided correction**, not the head's ceiling.

What it does close is the idea that the head can be **priced offline at all**. There is no GT for the
nodes whose displacement actually does the damage, so no snap, no shrink and no ranking built from the
annotated subset can size it. The honest options are to train the head and measure, or to leave it.

**Net for the plan:** "localization is the only channel left" loses its causal support and returns to
being a plausible mechanism with a correlation behind it. Nothing offline distinguishes it from the
crowding story ([[celltrack-true-bottleneck-is-dense-regime-model-gap]]) any more — which was always the
other reading of the same 4.08 µm.

## Method note

State the denominator: the 2172 snapped nodes are 1.8 % of the predicted set, and 105 broken edges are
4.9 % of 2127 annotated ones. The re-solve's own control (shrink 0.0, 176 broken) is **not** the champion
(105) — a weaker linker fails on a partly different edge set, so the delta is read within the arm family
and never against the champion's level.
