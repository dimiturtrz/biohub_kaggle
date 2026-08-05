# An unswept smoothing constant was a dense-matching bias (2026-08-05)

Sense-making of a parameter audit. `LinefitSmoother.strength` — how far each linear-track interior node is
pulled toward the straight line between its neighbours — shipped at 0.8, an unswept magic number (the linker
constants audit `i0a` had flagged it "cosmetic, unswept"). It is neither cosmetic nor correctly valued.

## The sweep

Proxy score at thr0.99 across the smoothing strength:

| strength | 0.0 (off) | 0.3 | 0.4 | 0.8 (shipped) | 1.0 |
|---|---|---|---|---|---|
| proxy | 0.9417 | **0.9426** | 0.9426 | 0.9344 | 0.9340 |

A broad peak at 0.3–0.4, and the shipped 0.8 sat well down the far slope — **−0.008** versus the peak. The
whole effect is on the one crowded dense movie `6bba_05db0fb1`; the three sparse movies are unchanged to four
decimals:

| movie | smooth 0.8 | smooth 0.3 |
|---|---|---|
| dense 6bba_05db0fb1 | 0.8919 | **0.9059 (+0.014)** |
| 44b6_0113de3b / 44b6_0b24845f / 6bba_05b6850b | 0.9600 / 1.0000 / 0.9697 | unchanged |

## Mechanism

The smoother moves a node's coordinates toward the midpoint of its neighbours. The metric matches predicted to
annotated centres within a 7 µm radius, so a *little* de-jitter recovers matches a wobbling detector would
lose — which is why 0.3 beats 0.0 (off) by a hair. But at 0.8 the pull is heavy, and a genuinely *moving* cell
— accelerating, turning — is dragged off its true path far enough to fall **outside** the 7 µm radius, losing
the match. Crowded tracks in the dense movie have the most such interior nodes and the least slack, so they
take the entire hit; the sparse movies, whose annotated tracks are short and well-separated, are untouched.

## Why it matters

This is a *recall-side* gain — more predicted centres land within the match radius of an annotated one — read
off the metric's own matching, not the node-count adjustment. That is the class of change that transfers to the
hidden leaderboard (unlike node-count bonus-farming, which anti-transferred). The dense movie is the pooled
drag, so a +0.014 on its raw Jaccard is the first real recall gain on that bottleneck since the NMS-window fix
— and it came from auditing an unswept constant, not a new method.

Fixed the default to 0.3 (kept the smoother: 0.3 still beats off). The lesson generalises: every unswept magic
number is a candidate bias; sweep them before trusting them.

## Correction (LB refuted it)

The prediction above — that this recall-side gain would transfer — was **wrong**, and the leaderboard said so
cleanly. Two submissions identical but for this constant: smooth0.8 (55250898) = **0.892**, smooth0.3
(55251767) = **0.889**. The heavier smoothing the sparse proxy penalised is what the *denser hidden
annotation* rewards; the +0.014 dense raw-Jaccard on our four sparse movies was a proxy artifact, not an LB
signal. Default reverted to 0.8.

The real lesson is the opposite of the one first drawn here: at the 0.89 tier a proxy raw-Jaccard improvement
on the dense movie is **not** evidence of an LB gain — recall-side proxy wins (NMS un-merge, this smoother)
have now failed to transfer as reliably as node-count farming did. Only the detection threshold moves the LB.
"Sweep the unswept constant" still holds; "trust a recall-side proxy win" does not — those must be arbitrated
on the leaderboard before shipping as a default. Shipping 0.3 was a 0.003 LB regression, caught and reverted.
