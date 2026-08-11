# A sweep is only one variable if nothing upstream depends on it

The ported association ranker — the frontier's primary association signal, carrying 0.85 of its evidence —
measured **negative at every weight** in our pipeline. Four arms, all below control, the worst at −0.0229. I was
one write-up away from recording it as the sixth frontier component that fails alone on our pipeline.

It was my experiment.

## The confound

The ranker needs a *context graph*: six of its 22 features (`edge_prob`, `has_learned_edge`, the degrees,
`has_prev`/`has_next`, `target_best_next_prob`) describe an emitted edge list, not the dense affinity. So the
tracker links once, featurises that graph, and re-links. Sound.

I swept the split of a fixed evidence budget: `a + b = 20`, with `a` on the transformer and `b` on the ranker.
And pass 1 was built with `without_ranker()`, which keeps the arm's own `a`. So at the 85/15 split, **pass 1
linked its context at `affinity_bonus = 3` with no ranker to compensate**. At that bonus a link costs
`6.6 − 3·P` µm against a boundary cost of 3, so the global flow terminates tracks rather than linking them and
the link rate halves.

The ranker's own input graph degraded in lockstep with the weight under test. `b` and pass-1 quality were the
same variable wearing two names.

The measurement that showed it, same code and same detections, only the pass-1 cost differing:

| | 85/15 arm | control | trained |
|---|---|---|---|
| `source_out_degree` | 0.584 | 0.956 | 0.968 |
| `source_has_prev` | 0.579 | 0.939 | 0.942 |
| `target_best_next_prob` | 0.492 | 0.753 | 0.770 |
| **features off-distribution** | **8 / 22** | **3 / 22** | |

## The port was faithful all along

Nothing was wrong with the 22 features. Every micrometre feature sits on the training distribution
(`edge_dz/dy/dx` z ≤ 0.03, `edge_dist_um` z = 0.15, `motion_dist_um` z = 0.21), `t_norm` z = 0.03. The piece
the port had to *invent* — a per-source candidate list, which a global min-cost flow never materialises —
came out at **2.67 / 4.35 against the trained 2.73 / 4.46**. The ranker's output is neither constant nor
saturated: `frac > 0.5 = 0.290` against a base-rate ceiling of `1/3.81 = 0.263`.

The only cell with an undamaged context, `a=20 / b=2`, scored **−0.0019** — a tie at the noise floor. The
ranker is **untested**, not refuted, and it took a diagnostic to tell those apart.

## Why this class of error is hard to see

A knob sweep looks like the safest experiment there is. The failure needs three things to line up, and they did:
the pipeline has a stage whose *input* is produced by an earlier stage; the knob under test participates in
that earlier stage; and the metric degrades smoothly, so the result looks like a dose-response curve rather
than a broken control.

That last part is what makes it dangerous. **−0.0019, −0.0043, −0.0031, −0.0229 reads as evidence.** It has a
shape. I even described it as monotone before checking, and had to withdraw that — it never was.

The structural fix is not vigilance. Pass 1's config must be *derived* so it is invariant to the split, with a
test asserting two different splits of the same budget produce an identical context graph. A test can hold that
invariant; an intention cannot.

## The instruments are compounding

Three negatives tonight turned out to be measurement errors, and each was caught by an instrument built in
response to an earlier one:

1. The **clamped score** (built after the node-count bonus was found to reward under-detection) caught the
   selector saving a checkpoint whose recall had fallen 0.966 → 0.938.
2. The **base-rate guard** (built after my own probe scored the ground-truth graph at base rate 0.976 and
   reported a meaningless AUC of 1.0000) now refuses to print a separability number on a degenerate population.
3. The **feature-distribution check** (built to ask whether our ranker features were faithful) found this
   confound instead — one level above the question it was asked.

None of the three was the cheapest thing to do at the time. All three paid within hours. The pattern worth
keeping: after a surprising negative, the highest-value next move is usually an instrument that could tell a
real negative from a broken measurement — not another arm of the same sweep.

## What is still open

A real gap survives the correction: our `frame_count` is **747 ± 52** against the ranker's trained
**318 ± 219**, corroborated by 7 µm density 0.57 vs 0.26. At threshold 0.97 we run roughly 2.3× the detection
density this asset was trained against. That is not a bug in the port; it is a genuine distribution shift
between our operating point and theirs, and it is the next thing to test — the artifact's own manifest asks for
it to be used as *"a constrained local association tie-breaker, not a global edge veto."*
