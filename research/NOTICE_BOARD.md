# Research Notice Board

## Open Questions & Investigation Status

### Online Recheck: Kaggle Notebooks, Discussions, SOTA Models (Aug 2026)

**Status**: 🔶 **PARTIAL** (2026-08-04; notebooks >0.90 unverified, forum activity recent but unsummarized, SOTA models listed)
**Deep-dive**: [`2026-08-04_kaggle_online_recheck.md`](deep_dives/2026-08-04_kaggle_online_recheck.md)

**Question**: (1) Any public notebooks >0.90? (2) Recent forum posts (since 2026-08-01) on methods? (3) Available pretrained 3D models?

**Findings**:
- **Q1 — Notebooks >0.90**: UNVERIFIED. No scores >0.90 detected in indexed search; known notebooks (kaiwalyaatulraut, xiaoleilian, pilkwang) have no disclosed scores. Kaggle leaderboard is JavaScript-rendered; direct visit required. [S1, S2]
- **Q2 — Recent posts**: Confirmed recent activity (José Freitas, Davit Khantadze, Aug 3–4 2026). Topics: synthetic dataset, training time, ensemble diversity. Specific method details NOT indexed; forum visit needed for post titles/content. [S3]
- **Q3 — Pretrained SOTA**: CONFIRMED. Ultrack (Nature Methods 2025, torch weights downloadable); Cell-TRACTR (Zenodo weights); CellSeg3D (HuggingFace); SAM2-based (zero-shot); CELLECT (2025, 50× faster). Cellpose3D (generalist, slower). StarDist3D (high-precision, limited 3D weights). [S4–S11]

**Next**: Manual Kaggle leaderboard + forum visit for notebook scores + specific post titles. Ultrack + Cell-TRACTR merit testing on fold-0 for ensemble gain.

---

### Kaggle Metric Scoring & Test Set Split

**Status**: 🔶 **PARTIAL** (2026-08-03; metric formulas confirmed, test overlap unresolved)
**Deep-dive**: [`2026-08-03_kaggle-deep-metric-split.md`](deep_dives/2026-08-03_kaggle-deep-metric-split.md)

**Question**: (1) Does public/private split use same test movies with annotation subset vs different movies? (2) Do the 4 test movies (44b6_*, 6bba_*) appear in training data? (3) What is `estimated_node_count` (T_true) and how is it computed?

**Findings**:
- **Metric formulas CONFIRMED**: `score = adjusted_edge_jaccard + 0.1 * division_jaccard`. Adjusted Jaccard = `max(0, jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))` where T_true is per-movie total node estimate. [S2]
- **Division Jaccard exploit CLOSED**: Pre-patch, synthetic hub node at (−10000, −10000, −10000) with edges merging all track components + synthetic forks inflated score ~0→1.0 (+0.1 bonus). Patch implemented, all submissions re-scored. [S1, S2]
- **T_true is per-movie TOTAL**, not per-timeframe; no source found on computation method (actual GT count, statistical estimate, or field-specific default). [OPEN]
- **Test set overlap UNCONFIRMED**: No primary source found confirming whether 4 test movies appear in train data with sparse annotations. Public LB = 29%, private = 71% also unverified. [OPEN]

**Risk**: Your working assumption (test movies in train with annotation split) is plausible but unconfirmed. If false, model generalization risk is higher.

**Next**: Direct Kaggle forum inquiry or data inspection via Kaggle notebooks to verify test-train overlap status; impacts stratification strategy.

---

### What the public notebook frontier actually does

**Status**: ✅ **SETTLED** (2026-08-02)
**Deep-dive**: [`2026-08-02_public-notebook-frontier.md`](deep_dives/2026-08-02_public-notebook-frontier.md)

**Question**: What is the current public top method, and what does it do that we do not?

**Findings**:
- The public frontier is **one pipeline, forked** — the top ~0.913 notebook, pilkwang's two-seed blend and
  the "3rd no-hack" notebook share an identical config block and `EXPERIMENT_TAG`. Read one, read all.
- We are at **0.859 LB** (rank 1122; top 0.943). Local fold-0 0.877 → LB 0.859, a **−0.018** offset: our
  local evaluator is a trustworthy proxy.
- The gap is **post-detection**, not architecture: gap-close with synthetic nodes, topology repairs,
  post-link safe divisions, a dual-seed logit blend, a DeepCenter veto model — and `DET_THRESHOLD=0.99`
  against our 0.5.
- Our `MotionHungarianLinker` **matches the frontier's motion relink exactly** (tight 6.0 / loose 10.0 /
  half-velocity), arrived at independently. The linker is not the weak link.

**Next**: threshold sweep first (cheapest), then gap-close (`1ro`), then topology repairs.

---

### Detection + Tracking SoTA for Zebrafish 3D Embryo Tracking

**Status**: 🔶 **SUPERSEDED IN PART** (2026-07-31; tiering outdated as of 2026-08-02)
Method survey (Cellpose/StarDist/Trackastra/motile/Ultrack, Gurobi-in-Kaggle) remains valid; the
0.857 / 0.897 notebook tiering is stale — see the 2026-08-02 frontier dive above.

**Deep-dive**: [`2026-07-31_celltrack_sota_3d.md`](deep_dives/2026-07-31_celltrack_sota_3d.md)

**Question**: What are the state-of-the-art methods for 3D cell detection + linking in fluorescence microscopy that work offline in Kaggle kernels?

**Findings**:
- **Detection**: Cellpose3/cyto3 (simplest, pretrained, offline) or Ultrack (SOTA, slower)
- **Linking**: Trackastra (transformer, division-aware) + motile ILP (CBC solver, offline)
- **Jitter fix**: Soft-argmax or Gaussian subpixel refinement; distance-transform peak-finding insufficient
- **Solver**: CBC is Kaggle-compatible (Gurobi needs WLS licence in cloud)
- **Metrics**: Edge Jaccard + Division Jaccard (weighted 0.1) for competition

**Next**: Implement Path A (Cellpose3 + Trackastra) or Path B (Ultrack) depending on segmentation quality baseline.

---

### Linking Disambiguation: Appearance + Learned Association Methods

**Status**: ✅ **SETTLED** (2026-08-03)
**Deep-dive**: [`2026-08-03_linking_disambiguation_methods.md`](deep_dives/2026-08-03_linking_disambiguation_methods.md)

**Question**: What concrete methods can disambiguate multi-object linking in crowded 3D cell microscopy? Which require training, which are analytical, and when does global (ILP) beat greedy Hungarian?

**Findings**:
- **Analytical features** (free, offline): intensity/texture (LBP, Hu moments), shape (eccentricity), local descriptors (SIFT/ORB)
- **Learned linkers** (no retraining needed): Trackastra (transformer, pretrained CTC weights), CELLECT (contrastive embedding, cross-modal pretrained)
- **Global optimization** (5–15% gain on dense frames): ILP/min-cost-flow via motile+CBC (Kaggle-compatible) or Ultrack; permits division + backtracking; greedy commits locally
- **Production stack**: Trackastra greedy or ILP mode; CELLECT for embedding space; analytical features as cost matrix terms

**Next**: Test Trackastra ILP on fold-0; measure edge Jaccard lift vs baseline Hungarian. Then decide: add appearance cost-weighting (0-cost improvement) or full ILP (requires solver tuning).

---

### Training Discriminative Association Models for Dense 3D Cell Tracking

**Status**: ✅ **SETTLED** (2026-08-03)
**Deep-dive**: [`2026-08-03_dense-association-training.md`](deep_dives/2026-08-03_dense-association-training.md)

**Question**: How do SOTA methods train association models to disambiguate crowded successors? What loss functions, hard-negative mining strategies, and features make the difference?

**Findings**:
- **Hard-negative mining** (core technique): Sample spatially-proximate wrong neighbors as negatives; forces model to learn "which of N candidates" not just pairwise matching
- **Loss functions**: Trackastra uses parental softmax (enforces ≤1 parent, biological constraint); TWiX uses bidirectional contrastive loss (forward+backward consistency)
- **Metric learning**: Triplet loss with focal weighting (down-weights easy negatives, emphasizes hard boundary cases)
- **Features beyond distance**: Appearance embeddings (via metric learning), optical-flow motion priors, neighborhood attention context, morphological shape similarity
- **Implementable levers** (prioritized): (1) hard-negative mining, (2) parental softmax + weight matrix, (3) bidirectional loss, (4) appearance embedding head, (5) motion-corrected linking via optical flow

**Next**: Implement hard-negative mining + parental softmax as low-cost gains; measure edge Jaccard lift. Then add motion prior if gain plateaus.

---

### What separates the 0.930–0.947 pack from 0.913 frontier?

**Status**: 🔶 **PARTIAL** (2026-08-03; competition ongoing, no winner writeups available)
**Deep-dive**: [`2026-08-03_pack_decode.md`](deep_dives/2026-08-03_pack_decode.md)

**Question**: What concrete levers do top teams (0.930–0.947) use to exceed the public 0.913 frontier? Which are detector, linker, ensemble, metric-exploitation, or post-processing?

**Findings** (Evidence-based; some speculative):
- **Undertrain gap (HIGH CONFIDENCE)**: Baseline explicitly states "not trained to convergence — expect gains." Extended training is likely +0.01–0.03 and nearly free.
- **Edge-centric linkers (HIGH CONFIDENCE)**: HOCT (Biohub-authored, July 2026) addresses "cell division entanglement" + "weak edge topology" problems in node-centric approaches. 59% error reduction on fine-tuning; state-of-the-art on CTC benchmarks.
- **Multi-detector ensemble + TTA (HIGH CONFIDENCE)**: Standard in 3D detection leaderboards; empirically +9.69% AP gains. Zero retraining for TTA; 2–3 detector seeds yield +0.01–0.06.
- **Division-aware topology (MEDIUM CONFIDENCE)**: Recent Cell-TRACTR and division literature show linkage confusion (mother→child vs same-cell) as primary error. Dedicated division head + topology enforcement likely +0.01–0.025.
- **Graph post-processing (MEDIUM CONFIDENCE)**: Gap-filling, tracklet refinement, Kalman smoothing shown in ARGUS and tracking literature; +0.01–0.02 orthogonal gains.
- **Metric exploitation (LOW CONFIDENCE; HIGH RISK)**: Conservative node prediction can game Jaccard formula (10% penalty threshold), but brittle and regresses on private data if annotation densities differ.

**Metric structure**: Adjusted Jaccard = max(0, jaccard × (1 − 0.1 × (Npred − Ntrue) / Ntrue)); 0.1× weight on divisions. Over-prediction penalty is mild (10%), favoring cautious edge-first strategies.

**Next**: Prioritize (1) train to convergence, (2) multi-detector + TTA, (3) HOCT-style edge attention if first two plateau. Competition ends 2026-09-29; no public 0.93+ solutions found yet.

---

### Mountable 3D Detectors for Kaggle Kernels + Complementarity

**Status**: ✅ **SETTLED** (2026-08-03)
**Deep-dive**: [`2026-08-03_mountable-detectors-sota.md`](deep_dives/2026-08-03_mountable-detectors-sota.md)

**Question**: Which offline-mountable 3D detectors complement your existing TemporalUNet3D? What's the complementarity of different architectures on crowded embryo tissue?

**Findings**:
- **Foundation models** (2025–2026): Cellpose-SAM v2 (June 2026, robust to contrast), CellposeDINO (lightweight), Omnipose (morphology-agnostic) — all offline, auto-download, 3D-capable.
- **Complementarity benchmark**: StarDist3D (high precision 0.81, lower recall 0.63) vs U-Net (0.65–0.69 precision, 0.67–0.77 recall) — precision-recall tradeoff on dense nuclei; ensemble both for +0.02–0.04 Jaccard.
- **Best ensemble for your case**: StarDist3D (high-precision) + Cellpose-SAM v2 (recall + generalization) → expected gain +0.01–0.04 edge Jaccard on crowded frames.
- **Joint detection+tracking**: CellTracker-GNN (ECCV 2022, graph neural network, learned associations), TrackMate v7 (Fiji plugin, pluggable detectors).
- **Practical Kaggle**: All fit T4 (~2–4 GB VRAM); PyTorch/TensorFlow deps self-install; weights auto-cached after first download.

**Next**: Test StarDist3D + Cellpose-SAM v2 ensemble on fold-0 validation; measure precision lift vs baseline. If >+0.005 Jaccard, integrate into submission pipeline.

---

## Legend

- ✅ **SETTLED**: Findings complete, actionable, moved to SUMMARY
- 🔶 **PARTIAL**: Some answers found, remaining questions documented in Open Questions
- ❓ **OPEN**: Not yet investigated
