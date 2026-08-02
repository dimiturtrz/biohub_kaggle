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
| `stlfg` | short-track + linefit + gap-close (planned) |

## Coverage grid (fold-0 local pooled / Kaggle LB)

| S1 | S2 | S3 | fold-0 | LB | notes |
|----|----|----|--------|----|-------|
| `dog` | `nn` | `raw` | — | **0.427** | classical baseline (`celltrack-classical-submit`, legacy slug) |
| `unetaug` | `nn` | `stlf` | 0.472 | 0.442 | 55146849 (legacy slug `celltrack-aug8k-nn`) |
| `unetw64bg` | `motion` | `stlf` | 0.246 | — | width/bg/wd refuted; not submitted |
| `pilktunet` | `nn` | `stlf` | 0.877ˢ | — | subset; NN loses the sparse fast-movers |
| `pilktunet` | `motion` | `stlf` | **0.877** (0.927ˢ) | **0.859** | 55148762 — our best; local→LB offset −0.018 |
| `ourstunet` | `motion` | `stlf` | 0.887ᶜ | — | warm-start=pilkwang wts (fine-tune erodes); NOT yet genuinely ours |

ˢ = fold-0 stratified subset (2/prefix), not full-41. The `pilktunet` linker delta is large: `motion` beats
`nn` +0.050 subset (+0.229 on sparse 44b6) — the better the detector, the more the linker decides the score.

ᶜ = **calibrated** operating point. Full-41 `pilktunet`/`ourstunet` threshold sweep (motion+stlf+TTA) is
monotone in the read-out threshold — trimming over-detection farms the count-ratio bonus:

| thr | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 |
|-----|-----|-----|-----|-----|-----|------|
| pooled | 0.8742 | 0.8771 | 0.8794 | 0.8815 | 0.8837 | 0.8861 |
| nodes | 1.02M | 985k | 952k | 929k | 892k | 855k |

The detector ceiling is ~**0.887** (rising, plateauing) — short of 0.9 by operating point alone. `ourstunet`
warm-start fine-tune (lr 1e-4, our fold-train) *erodes* the score (best = init, late collapse to 0.80 @2k
steps): the warm-start already fits, so there is no gradient, and our fold-train is likely disjoint from the
(leaked) videos pilkwang's 0.877 rests on. A genuinely-ours detector needs from-scratch training over many
epochs (batch=1 single-frame is ~1 epoch/47min — shape-bucketed batching is the throughput fix; `cdg`).

Untried cells worth filling: `pilktunet`×`ilp`, any S3=`stlfg` (gap-close) — the remaining non-detector lever
to 0.9 once the detector is fixed.
