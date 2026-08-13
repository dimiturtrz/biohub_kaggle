# Motion, put into the model and into the solver — two clean results, one wall

2026-08-13. The premise this session tested: the pipeline is motion-blind (the backbone's time attention has no
positional encoding; the linker cost is first-order and prices no trajectory), and cell motion is real
(MSD exponent 1.64, velocity correlation ~3 frames). So give the model motion sight (temporal positional
encoding, **dtgd**) and give the solver a direction-persistence cost (**jf5g**). Both are physically argued and
neither existed. Both are now resolved, and the resolution is the same physical fact from two sides.

## dtgd — temporal positional encoding is real but recipe-masked, and sub-competitive

The published `TemporalUNet3D` attends each voxel across the window's frames with **no positional signal**, so
it is permutation-invariant over time and cannot represent direction. `TemporalPositionAttention` wraps each
temporal block, reuses its trained `norm`/`attn`, and adds a learned per-frame-index embedding — zero-init, so
byte-identical to the published block at step 0 (verified end-to-end: init proxy 0.8466 == baseline). Off by
default, persisted in the checkpoint.

Matched A/B (warm-start seed1, masked detection, lr 1e-5 cosine, 5000 steps), held-out **test-4**, single-seed,
shipped flow+bidirectional+division pipeline:

| model | validation-4 | held-out test-4 | mislinks |
|---|---|---|---|
| pilkwang single (untrained) | — | 0.9118 | ~38 |
| control (recipe, TPE **off**) | 0.8729 | 0.9064 | 56 |
| dtgd (recipe, TPE **on**) | 0.8731 | 0.9099 | 54 |

Two facts, and neither is "inert":

1. **The masked warm-start recipe degrades held-out by −0.0054** (control 0.9064 < untrained pilkwang 0.9118).
   Fine-tuning a converged model on the GT-pair corpus pulls it off the published optimum — the campaign's
   most-repeated pattern, here measured cleanly against the untrained baseline.
2. **TPE genuinely helps, matched: +0.0035 held-out** (0.9099 vs 0.9064), ~3× the run-to-run floor — even
   though validation was a tie (0.8731 ≈ 0.8729). The dense validation four could not see the benefit; the
   test four could. The motion feature carries real held-out value.

The trap avoided: on validation both arms climb identically to 0.873, which reads as "TPE does nothing." Only
the held-out split separates them. Selecting on the dense validation set would have discarded the difference.

But the net dtgd run is 0.9099 — below untrained pilkwang, far below best-own single (0.9265) and shipped dual
(0.9334). TPE recovers most of the recipe's damage, not all. And its ceiling is bounded: even if a non-degrading
regime (from-scratch, or frozen-detection fine-tune) let the full +0.0035 land on top of an undamaged base, the
result (~0.9153 single-seed) is still below best-own 0.9265. **So TPE is a real capability with no path to a
competitive number in this regime** — kept, byte-identical-off, available for a from-scratch test, but not a
lever for the score.

## jf5g — direction-persistence is refuted where it would act

The first-order flow cost prices no trajectory: a track that reverses 180° every frame costs the same as a
straight one with equal step lengths. jf5g would price the turn between consecutive steps on the line graph.

Two findings killed it before any code:

- **Formulation.** The naive line-graph (node = arc `i→j`, arc = triple `i→j→k`) breaks the 1-to-1 detection
  constraint: a detection `j` appears in every transition-node `A_{ij}` and `A_{jk}`, and "use `j` once" becomes
  a constraint across those nodes that a plain node-capacity cannot express. Strict 1-to-1 with turn costs is
  the known hard core (Butt & Collins 2013) — it needs **Lagrangian relaxation**, not the clean polynomial flow
  the bead assumed. Solve time was fine (measured: pure `network_simplex` 64.5s on the dense movie, ~40 min
  Kaggle submission — viable); the correctness was not.

- **Mechanism (the cheap gate).** On the 38 dense mislinks, does a turn penalty prefer the true successor (the
  straighter continuation of the source's GT incoming velocity) over the wrong one?

  | history | cos(v, true) | cos(v, wrong) | true straighter |
  |---|---|---|---|
  | 1-step | −0.10 | **+0.39** | 42% |
  | 3-step | +0.30 | +0.16 | 46% |

  No. At one step the penalty is **actively wrong** — the near stationary decoy is the straighter continuation,
  because the true fast-mover *changed direction*. At three steps the median leans true but per-mislink it is a
  coin flip (46%). A turn penalty rescues ≤46% and hurts the rest.

The cheapest test saved a multi-hour Lagrangian build. Stage-gated rigor as intended.

## The wall, from eight angles

The 38 dense mislinks are now refuted from **eight independent directions**: raw distance, learned affinity,
full-resolution appearance, bidirectional agreement, global flow, drift-field subtraction, oracle
trajectory-prediction, and now turn-angle persistence. The physical reading is consistent and complete: **these
are fast-movers that change direction between frames at the (1,4,4) resolution**, so the true successor lands
far while a near decoy sits close, and no motion, appearance, or persistence cue separates them. Persistence is
real (Newton, MSD 1.64) — but the cells that fail are precisely the ones that violate it. The physics is not
wrong; it does not apply to the failing population.

## Where this leaves the score

Model-side own-weights is exhausted (every retrain, TPE included, lands below the mount). The dense mislinks are
an eight-angle wall. The solver's direction lever is refuted where it acts. Every *mechanism-sound, single*
lever is now closed. The two paths that remain are structural, not tweaks:

1. **Bundle synergy.** The reproducible frontier (0.915) beats us (0.899) by a margin no single ported piece
   reproduces (1 for 7 alone). The pieces may only pay *together* — which is the case for replicating the bundle
   end-to-end and building past it, rather than cherry-picking parts that individually die.
2. **Better assets.** Independent detector/edge seeds (not fine-tuned descendants, which add no ensemble
   diversity) or finer resolution — the thing that could actually move the eight-angle wall, since it is a
   resolution/asset limit, not a config one.

Both are larger than a knob. Neither is a plateau to accept — they are the honestly-remaining shape of the
problem after the cheap and mechanism-sound levers are spent.
