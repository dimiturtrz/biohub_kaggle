# Research Notice Board

## Open Questions & Investigation Status

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

## Legend

- ✅ **SETTLED**: Findings complete, actionable, moved to SUMMARY
- 🔶 **PARTIAL**: Some answers found, remaining questions documented in Open Questions
- ❓ **OPEN**: Not yet investigated
