# Velocity-referenced attention bias (kn9u) — REFUTED on a cheap CPU gate

**Date:** 2026-08-20
**Bead:** kn9u
**Verdict:** refuted (this USE); geometry exhausted as a confusor lever.
**Cost:** one CPU diagnostic (~6 min), zero GPU, zero LB submission — the cost-tier gate did its job.

## The lever under test

The dense-movie confusor (`6bba_05db0fb1`, ~85% of the LB gap, n=37 mislinks) is a
**fast-mover directional inversion**: the true successor moves FAR along the cell's
motion, while a slower/nearer rival sits close to the source. A radial attention bias
keyed on `|candidate − source|` rewards the near rival — the exact axis the rival wins
(radial rival-nearer 97.3%).

kn9u proposed keying the bias on `|candidate − (source + source_velocity)|` instead:
predict where the cell is *going* from its previous-gap displacement, and measure
nearness to that predicted point. If motion is locally constant, the true far-successor
lands near `source + velocity` and the near rival lands far from it — the inversion flips.

## The gate and its result

`VelocityReferencedGap` (celltrack/analysis/localisation.py) re-measures the rival-distance
gap from `source + GT-predecessor-velocity` — **GT** velocity, i.e. the oracle ceiling for
this idea. If the constant-velocity prediction can't separate the confusor even with the
true velocity handed to it, no learned velocity estimate will.

```
mislinks vs chosen   radial       n=37    rival nearer=97.3%
mislinks vs chosen   velocity-ref n=35    rival nearer=100.0%  coverage=94.6%
correct  vs nearest  velocity-ref n=1058  rival nearer=0.8%    coverage=96.0%
```

Velocity-referencing made the confusor **WORSE**: rival-nearer 97.3% → **100.0%**, under
**oracle** GT velocity. On the correct-links control it stays healthy (0.8%), so the
instrument is honest — it just says the confusor is not where constant-velocity helps.

## Which link broke: IDEA (not harness)

- **Data** — clean: real confusor subset, 94.6% honest coverage (the 2 uncovered = source
  with no GT predecessor, correctly reported uncovered).
- **Test** — clean: GT-oracle velocity is the ceiling; control reads correct; the
  radial comparator reproduces the known 97.3%.
- **Plumbing** — clean: instrument produced numbers on both arms, unit-tested (29 pass).
- **Idea** — **this is the break.** The constant-velocity assumption fails on exactly the
  confusor subset. These fast movers are not moving in a straight line at constant speed
  from the previous gap; `source + GT_velocity` overshoots or veers, landing farther from
  the true successor than the near rival sits. First-order motion does not model them.

## What this does NOT rule out (leave re-testable)

- **Velocity as a head FEATURE**, not a geometric bias — a learned association head could
  use velocity nonlinearly where a linear `source + v·Δt` prediction fails. Untouched.
- **Higher-order motion** (acceleration, curved tracks) — not tested; the refutation is
  specific to first-order constant-velocity.
- The confusor is **not first-order-motion-separable** → it is a **model/affinity gap**, not
  a geometry gap. This is consistent with the proxy-saturated / affinity-quality-bound
  finding: the remaining lever is better affinity DISCRIMINATION of same-frame rivals, not
  a smarter cost geometry.

## Where the lever moves next

Affinity/appearance discrimination — a contrastive head whose negatives are the OTHER cells
in the same frame (the sibling rsna_knee_kaggle agent independently pointed at the same
mechanism: hard-negative mining, not reconstruction/domain-adaptation, is what separates
near-identical instances). Constrained by my own prior refutations to full-finetune OR
fully-stock-frozen — never adapt-lower-then-freeze — and read the loss curve before the metric.
