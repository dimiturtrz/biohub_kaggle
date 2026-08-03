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
| `stlf6` | short-track (min length 6, frontier `OUTPUT_MIN_TRACK_LEN`) + linefit (0.8) |
| `divstlf` | division recovery (parent 4.7 / sister 7.2 / existing-child 7.8 µm) + short-track + linefit |
| `stlfg` | short-track + linefit + gap-close (reuse built/inert; synthetic needs DeepCenter) |

## Coverage grid (fold-0 local pooled / Kaggle LB)

| S1 | S2 | S3 | fold-0 | LB | notes |
|----|----|----|--------|----|-------|
| `dog` | `nn` | `raw` | — | **0.427** | classical baseline (`celltrack-classical-submit`, legacy slug) |
| `unetaug` | `nn` | `stlf` | 0.472 | 0.442 | 55146849 (legacy slug `celltrack-aug8k-nn`) |
| `unetw64bg` | `motion` | `stlf` | 0.246 | — | width/bg/wd refuted; not submitted |
| `pilktunet` | `nn` | `stlf` | 0.877ˢ | — | subset; NN loses the sparse fast-movers |
| `pilktunet` | `motion` | `stlf` | **0.877** (0.927ˢ) | **0.859** | 55148762 — local→LB offset −0.018 |
| `pilktunet99` | `motion` | `stlf` | **0.9074** | pending | thr 0.99 + PeakExtractor; +0.011 over 0.5, +0.030 over the 0.859 kernel's path |
| `pilktunet99` | `motion` | `stlf6` | **0.9090** | pending | short-track min length 3 -> 6 (frontier `OUTPUT_MIN_TRACK_LEN`); +0.0016 |
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

**Detection-limited on real detections (2026-08-02).** The ground-truth ceiling does not carry. First error
found and fixed: the "10.5 / 18" gates above were a **misread** of the frontier's config — its
`add_safe_divisions_postlink` uses a *tight* parent gate **4.7 µm**, sister **7.2 µm**, an **existing-child
gate 7.8 µm** (the mother's own link must be short, rejecting mislinked mothers), score
`parent + 0.15·sister`, and per-frame 0.8 % / global 0.4 % caps. Ported faithfully to
`celltrack.division_recovery` (all five gates, tested). With the correct gates on real detections
(`scratchpad/div_gate_sweep.py`, full-41 @0.99):

| gates | score | div_jaccard | forks |
|-------|-------|-------------|-------|
| nodiv | 0.9074 | 0.0000 | 0 |
| our old 10.5 / 18 | 0.8975 | 0.0144 | (floods) |
| frontier 4.7 / 7.2 / 7.8 | **0.9076** | 0.0114 | 1725 |

The tight parent gate flips it from −0.010 to **+0.0002**. But `div_jaccard` reaches only 0.0114 (vs 0.4468
on GT nodes): **most true second daughters are never detected**, so the fork the metric scores cannot be
placed. The gain is real but negligible, and once `stlf6` prunes short false chains, adding divisions turns
net-negative (0.9090 → 0.9085). The existing-child gate is inert at parent 4.7 (never binds). Divisions are
now *correct* but capped by detector recall — the frontier's dual-seed blend + DeepCenter recover and confirm
those daughters. Kept, not shipped.

**Gap-close reuse** (`celltrack.gap_closer`, frontier 5.8 / 3.2 / 5 %): bridges a one-frame dropout by reusing
an existing isolated node at the midpoint (the geometry-only half of the frontier's `gap_close`; synthetic
insertion needs DeepCenter). **Inert here** (0.9090 → 0.9090) — at 0.99 there are too few isolated detections
near a midpoint to reuse. Built + tested, waits on the synthetic + veto half.

Both post-proc stages therefore converge on the same missing piece: the **DeepCenter centre-prior model**
(`biohub-deepcenter-unet3d-center-prior-v1`) — for recall (detect the daughters / gap cells) and precision
(veto false forks / synthetic nodes). That is the next real lever (`gsm`).

## 2026-08-03 — global linker goes SCIP-free, and the proxy proves out on the board

The frontier's global-ILP linker was blocked on Kaggle (`SCIP: unspecified error!`). It didn't need SCIP:
`ILPLinker(division=False)` is `MaxParents(1)+MaxChildren(1)`, whose constraints never couple two frame gaps,
so the program **decomposes into an independent min-cost matching per gap** — `scipy.linear_sum_assignment`
with a zero-cost skip per node reproduces it **edge-for-edge** (dense-movie edge jaccard 1.0). `AssignmentLinker`
(`celltrack.assignment_linking`) is that, pure numpy/scipy, no motile/ilpy/pyscipopt.

Champion = dual-seed logit blend + two-seed edge-transformer blend (`0.8·seed1+0.2·seed2`, logit space) +
`AssignmentLinker` + density gap-bridge + `stlf6` + linefit. Proxy build-up (4 test movies):

| config | proxy | note |
|--------|-------|------|
| dual-seed + neural edge-bonus (greedy) | 0.8989 | greedy Hungarian |
| + global `AssignmentLinker` (SCIP-free ILP) | 0.9125 | ooi |
| + density gap-bridge | 0.9134 | |
| + edge-seed blend `0.8/0.2` | 0.9154 | 3th |
| + affinity bonus `2·gate` (was `1·gate`) | 0.9227 | 88x — bonus underweighted the learned signal |
| + threshold `0.995` (node-count bonus) | 0.9287 | vbn — held pending LB confirm |

Derivations (no magic): gate `10um` = max annotated single-frame displacement (9.96um, proxy plateau 10–15);
bonus `2·gate` = crowding over-trust (scale-match autobalance undershoots ~2.5×, refuted in 88x); threshold
`0.995` banks the node-count bonus (every movie already under-detects at 0.99).

**LB confirms the proxy.** dual-seed `@0.99` = proxy 0.8876 → **public 0.882** (55212692), single-seed 0.8731 →
0.871: the 4-movie proxy predicts the board within ~0.005. New best public **0.882** (was 0.873). The SCIP-free
Assignment champion (0.9227, 55218808) and the neural greedy (0.8989, 55216476) are scoring; the `0.995`
candidate is prepped, held until the Assignment core confirms the linker itself transfers.
