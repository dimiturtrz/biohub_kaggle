# Relative-PE is radial, but the confusor is directional (fast-mover flow)

**Date:** 2026-08-20 · **Task:** celltrack · **Instrument:** `celltrack.analysis.localisation` on the dense
movie `6bba_05db0fb1` (the movie holding ~85% of the LB gap), CPU pass, `runs/localisation_meps_cpu.log`.

## The question this settles, before spending GPU

Lever 1 of the frontier corrective is spatial relative-PE in the edge transformer's cross-attention. It is
built and mounted (`DistanceAttentionBias`, bead `biohub_kaggle-8owx`) as a **radial** bias: a per-head learned
function of the pair separation *magnitude* `|Δ|` in micrometres (`relative_position_bias.py:98`,
`separation_um → .norm(dim=-1)`). Before committing the GPU training arm, does that mechanism match the
measured association failure?

## What the confusor actually is (n=37 mislinks, dense movie)

| quantity | mislinks (chosen) | correct (nearest rival) |
|---|---|---|
| `identical delta` (true vs rival, on lattice) | **0.0%** | 0.0% |
| `rival nearer` | **97.3%** | 0.2% |
| distance gap `d(src,true) − d(src,rival)` median | **+6.439 um** | −6.182 um |
| within noise-flip `|gap| < 2.37 um` | 8.1% | 2.4% |
| annotated step (motion) median | **3.275 um** (max 9.69 ≈ gate) | 1.675 um |

Reading, in order:

1. **Geometry is not exhausted.** `identical delta = 0%`, L1 median 7 lattice voxels — the true and rival
   candidates never hand the head the same displacement. Position *can* separate them; a PE premise is sound.
2. **It is not quantization noise.** Only 8.1% of mislinks have a gap small enough for localisation error to
   have flipped the ordering; the median gap is a large +6.4 um. The rival is *genuinely* the nearer cell.
3. **The mislinks are fast movers.** Annotated step median 3.3 um (double the 1.7 um typical), up to the 9.7 um
   gate. The true successor moved *far*; a slower / neighbouring cell sits *near* the source.
4. **On correct edges the true successor is the nearest** (`rival nearer = 0.2%`, gap −6.2 um). Distance is a
   *good* cue almost everywhere — which is why the linker follows it — and a *misleading* one on exactly the
   fast-mover subset that constitutes the gap.

## Why a radial bias is mismatched to this

Cross-attention keys the query node at `t` against candidate keys at `t+1`. A radial bias reweights those keys
by `|Δ from source|`:

- To rescue the 37 fast-movers a head must learn **"prefer far"** — but the same per-head scalar-of-distance
  hits all 1132 correct edges, where the true successor is the *nearest*. "Prefer far" trades 37 against 1132;
  net negative.
- The bias has **no per-source velocity input**. A fast source (far = correct) and a slow source (far = wrong)
  with the same candidate geometry receive the *same* bias. Radial `|Δ|` cannot condition on which regime a
  source is in — and the regime is the whole distinction.
- On the confusor subset a radial focus is **actively counterproductive**: it concentrates attention on the
  keys *near* the source, which is precisely the rival, reinforcing the wrong link.

Radial relative-PE is therefore a weak-to-negative attack on its own named bottleneck. It is not killed — it is
zero-init (cannot move the model far) and may regularise elsewhere — but it is **demoted**: do not prioritise
the GPU arm on the expectation that it closes the dense-movie gap.

## The instrument the mechanism points to: velocity-referenced (directional) bias

The failing structure is *far along the motion*, not *far*. The right reference is the cell's own predicted
next position, not its current one. Key the bias basis on

  `| candidate_position − ( source_position + source_velocity ) |`

reusing `PriorVelocity` (already built — `Σ_i P[i,j]·(c_j − c_i)`, the affinity-weighted incoming
displacement, `prior_velocity.py:106`) as `source_velocity` under a constant-velocity assumption. This puts the
zero-residual basis on the *true far successor* and penalises the *near rival* — attacking the measured
inversion directly, and still zero-init-safe (residual reduces to radial `|Δ|` when velocity is zero, e.g. the
first gap).

**Caveat carried forward** ([[celltrack-velocity-is-warmstart-not-fromscratch]]): from-scratch `PriorVelocity`
is confident-wrong (chicken-and-egg), so a velocity-referenced bias inherits that dependency and needs the
GT-prev-link velocity warmup annealed to self-derived — not a naive from-scratch run.

## Verdict / next

- **Mechanism named:** confusor = fast-mover directional inversion (true far-along-flow vs near rival), not a
  distance-magnitude or quantization problem. `idea` link confirmed sound (position separates); the built
  `radial` *use* is mismatched.
- **Radial arm (`--relative-position`, bead 8owx):** demoted, still re-testable as a zero-init regulariser; run
  only opportunistically, not as the gap-closer.
- **New lever:** velocity-referenced directional position bias (new bead). Build is non-GPU; test is GPU-bound
  and gated on the velocity warmup being correct.
