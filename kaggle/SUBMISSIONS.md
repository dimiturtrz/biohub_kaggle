# Submission registry — the stage grid

A submission is one choice per pipeline stage. Its name is the tuple of those choices, in fixed order, so
the name alone says the exact method and the coverage grid below says what is tried vs untried. **No
free-form mnemonics** — every token comes from the closed vocabulary here; adding an option means adding a
row here first.

## Naming rule

```
celltrack-<S1>-<S2>-<S3>
```

Positional: first token = detection, second = linking, third = post-processing. Kaggle slugs are dash-only,
so each stage token is a single word (no internal dash). One kernel per tuple, under `kernels/<slug>/`.

## Stage vocabulary (closed — extend by adding a row, never inline)

### S1 — detection (per-frame cell centres; the recall wall)
| token | spec |
|-------|------|
| `dog` | multi-scale Difference-of-Gaussian band-pass (classical, CPU) |
| `unetaug` | our `DetectionUNet`, width-16, BCE, aug recipe, 8k steps (`detector_bce_aug_8k.pt`) |
| `unetw64bg` | our `DetectionUNet`, width-64, aug + bg-crops 0.3 + weight-decay 0.01, 8k steps (`detector_bce_w64_aug.pt`) — **refuted** |
| `pilktunet` | pilkwang public `TemporalUNet3D` + detect-head (`edge_predictor_best.pth`, split_0) |
| `pilktunet99` | `pilktunet` read out at threshold 0.99 via the shared `TemporalUNetDetector.detections` (PeakExtractor plateau-collapse) |
| `ourstunet` | our-trained `TemporalUNet3D` (planned) |

### S2 — linking (nodes → tracks; the edge-Jaccard the metric scores)
| token | spec |
|-------|------|
| `nn` | nearest-neighbour one-to-one Hungarian, gated |
| `motion` | motion-predicted two-pass Hungarian (tight 6 µm / loose 10 µm, half-velocity) |
| `ilp` | ILP global optimum (motile/SCIP) |
| `division` | division-aware (NN base + daughter assignment) |

### S3 — post-processing (track cleanup)
| token | spec |
|-------|------|
| `raw` | none |
| `st` | short-track filter (min length 3) |
| `stlf` | short-track (3) + linefit smoother (0.8) |
| `divstlf` | division recovery (parent 10.5 µm / sister 18 µm) + short-track (3) + linefit (0.8) |
| `stlfg` | short-track + linefit + gap-close (planned) |

## Coverage grid (fold-0 local pooled / Kaggle LB)

| S1 | S2 | S3 | fold-0 | LB | notes |
|----|----|----|--------|----|-------|
| `dog` | `nn` | `raw` | — | **0.427** | classical baseline (`celltrack-classical-submit`, legacy slug) |
| `unetaug` | `nn` | `stlf` | 0.472 | 0.442 | 55146849 (legacy slug `celltrack-aug8k-nn`) |
| `unetw64bg` | `motion` | `stlf` | 0.246 | — | width/bg/wd refuted; not submitted |
| `pilktunet` | `nn` | `stlf` | 0.877ˢ | — | subset; NN loses the sparse fast-movers |
| `pilktunet` | `motion` | `stlf` | **0.877** (0.927ˢ) | **0.859** | 55148762 — local→LB offset −0.018 |
| `pilktunet99` | `motion` | `stlf` | **0.9074** | pending | thr 0.99 + PeakExtractor; +0.011 over 0.5, +0.030 over the 0.859 kernel's path |
| `ourstunet` | `motion` | `stlf` | 0.887ᶜ | — | warm-start=pilkwang wts (fine-tune erodes); NOT yet genuinely ours |

ˢ = fold-0 stratified subset (2/prefix), not full-41. The `pilktunet` linker delta is large: `motion` beats
`nn` +0.050 subset (+0.229 on sparse 44b6) — the better the detector, the more the linker decides the score.

ᶜ = **calibrated** operating point. Full-41 `pilktunet`/`ourstunet` threshold sweep (motion+stlf+TTA) is
monotone in the read-out threshold — trimming over-detection farms the count-ratio bonus:

| thr | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|-----|-----|-----|-----|-----|-----|------|
| pooled | 0.8742 | 0.8771 | 0.8794 | 0.8815 | 0.8837 | 0.8861 |
| nodes | 1.02M | 985k | 952k | 929k | 892k | 855k |

That table predates the `PeakExtractor` peak read-out (this session's `tunet.detections` rewrite); on the
current shared path the same sweep is **higher and clears 0.9** — 0.5 → **0.8961**, 0.99 → **0.9074** — the
plateau-collapse recovers one centre per saturated blob where the old naive `logits == maxpool` over-counted.
The detector ceiling is ~**0.887** (rising, plateauing) — short of 0.9 by operating point alone. `ourstunet`
warm-start fine-tune (lr 1e-4, our fold-train) *erodes* the score (best = init, late collapse to 0.80 @2k
steps): the warm-start already fits, so there is no gradient, and our fold-train is likely disjoint from the
(leaked) videos pilkwang's 0.877 rests on. A genuinely-ours detector needs from-scratch training over many
epochs (batch=1 single-frame is ~1 epoch/47min — shape-bucketed batching is the throughput fix; `cdg`).

Untried cells worth filling: `pilktunet`×`ilp`, any S3=`stlfg` (gap-close) — the remaining non-detector lever
to 0.9 once the detector is fixed.

## Divisions — the term we score zero on

`MotionHungarianLinker` is a one-to-one assignment, so no node can ever have two children and
`division_jaccard` is **structurally 0.0000**, forfeiting the metric's whole `0.1 × division_jaccard` term.
Measured on fold-0 ground-truth nodes with `celltrack.bracket --arm ceiling-motion[div]`:

| arm | edge_jaccard | division_jaccard |
|-----|--------------|------------------|
| `ceiling-motion` | 0.9947 | **0.0000** |
| `ceiling-motiondiv`, frontier gates (10.5 / 8.0) | 0.9950 | 0.2593 |
| `ceiling-motiondiv`, tuned (10.5 / 18.0) | 0.9951 | **0.4468** |

≈ **+0.045** score on GT nodes, and the edge Jaccard *rises* too — a recovered daughter edge is a real GT
edge the one-to-one pass was dropping, so true forks pay on both terms and only false ones trade against
the large one. The parent gate's fold-0 optimum is 10.5 µm, independently reproducing the frontier's value
(14 µm already falls to 0.2558). The cap never binds here at any fraction from 0.02 to 0.5.

**Refuted end to end (2026-08-02).** On *real* pilkwang detections the pass regresses — the ground-truth
ceiling does not carry (`scratchpad/div_eval.py`, full-41, `TemporalUNetDetector.detections`):

| thr | nodiv | div (sister 8) | div (sister 18) | div_jaccard (18) |
|-----|-------|----------------|-----------------|------------------|
| 0.5 | **0.8961** | 0.8870 | 0.8874 | 0.0130 |
| 0.99 | **0.9074** | 0.8982 | 0.8975 | 0.0144 |

On annotated nodes an orphan is a true lost daughter; on real detections it is mostly a false-positive or
missed-link — the recovered fork is false, and each false parent edge costs the edge Jaccard (~0.009) far
more than the near-zero `division_jaccard` gain (0.013 × 0.1). Neither sister gate wins and the fraction cap
only converges toward `nodiv`, so it is not a gate-width miss. The frontier gates its post-link divisions
behind a **DeepCenter veto** (a centre-prior net that confirms each fork) we have not mounted; that veto is
the missing precondition. Code and tests are kept; `divstlf` is **not shipped**. Divisions wait on the veto.
