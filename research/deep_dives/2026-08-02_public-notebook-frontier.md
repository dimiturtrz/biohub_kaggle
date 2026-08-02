# The public notebook frontier — one forked pipeline, and what it does that we do not

**Date**: 2026-08-02
**Status**: settled
**Scope**: Every public notebook attached to the competition, ranked by vote and by recency; the recipe
behind the current public top score; our true leaderboard position.
**Method**: `python kaggle/survey.py notebooks --since <date>` / `pull` / `leaderboard` / `submissions`
(needs `KAGGLE_TOKEN`). 478 public notebooks enumerated, 8 pulled and read.

## TL;DR

The public frontier is **not a field of competing methods — it is one pipeline, forked.** The top public
notebook, pilkwang's two-seed blend, and the "3rd place no-hack" notebook are the *same source* with
different constants: identical `BIOHUB_PRESET`, identical `EXPERIMENT_TAG`, identical 102-constant config
block. Reading one reads all of them.

That pipeline is at **~0.913 public**. We are at **0.859** (rank 1122; LB top 0.943). The gap is not the
detector architecture — it is **five post-detection stages we have not built**, plus a detection threshold
we set 0.49 too low.

This supersedes the tiering in [`2026-07-31_celltrack_sota_3d.md`](2026-07-31_celltrack_sota_3d.md), which
recorded the learned tier at ~0.897 and is now two weeks stale on a fast-moving board.

---

## Our actual standing (was previously believed "pending")

`kaggle/survey.py submissions` — all four submissions are **complete and scored**, not pending:

| submitted | LB | method |
|---|---|---|
| 2026-07-31 23:38 | **0.859** | `pilktunet-motion-stlf` — fold-0 local 0.877 |
| 2026-07-31 20:55 | 0.442 | `unetaug-nn-stlf` — fold-0 local 0.472 |
| 2026-07-31 18:43 | 0.001 | keep-250/frame NMS (the plateau bug) |
| 2026-07-23 19:14 | 0.427 | classical `dog-nn-raw` |

**Local fold-0 0.877 → LB 0.859 (−0.018).** The generalisation gap is small, which is the useful news: our
local evaluator is honest and the leak-inflation in the borrowed weights is real but modest. Local fold-0 is
a trustworthy proxy — a measured −0.018 offset, not a coin flip.

## The frontier pipeline, stage by stage

Constants below are the notebooks' own defaults. **Bold = we do not have this.**

### Detection
- **Dual-seed logit blend** — two `TemporalUNet3D` checkpoints (`biohub-tracking-support-pack-50ep-v1`,
  which is the one we grafted, plus `biohub-temporal-unet3d-seed314159-v1`), logits averaged before the
  sigmoid. A third model, `biohub-deepcenter-unet3d-center-prior-v1`, is mounted separately.
- `DET_THRESHOLD = 0.99`. **We run 0.5.**
- **DeepCenter veto** — a separate centre-prior network scores each candidate point by max-pooling its
  heatmap over a small `(z, y, x)` window. Used to *confirm or veto* detections and repairs from the main
  detector. A second opinion, not a second detector.

### Linking
- ILP (`USE_ILP=1`) with explicit `edge=-1.0`, `appearance=0.1`, `disappearance=0.1`, `division=1.0` weights.
- **Motion relink** at `tight=6.0 µm`, `relaxed=10.0 µm`, `velocity_weight=0.5`.
  **This is exactly our `MotionHungarianLinker`** — same two-pass tight/loose structure, same gates, same
  half-velocity damping, arrived at independently. Our linker is not behind; it is the public state of the art.
- **`MOTION_RELINK_LEARNED_BONUS = 0.75`** — the learned edge probability is folded into the motion cost, so
  geometry and the transformer's association score decide jointly. Ours is pure geometry.

### Post-processing — the bulk of what we lack
- **Gap close** (`GAP_CLOSE_MAX_GAP=1`, `6.0 µm`) — bridges a one-frame detection dropout, and where the
  bridge implies a missing cell it **inserts a synthetic node** at a refined midpoint
  (`refine_synthetic_midpoint`, DeepCenter-confirmed). Capped at `3.8%` added nodes / 1650 absolute.
- **`recover_strict_gap2`** — the same recovery across a two-frame gap, under stricter conditions.
- **`add_safe_divisions_postlink`** — divisions proposed *after* linking, geometry-gated
  (`parent ≤ 10.5 µm`, `sister ≤ 8.0 µm`), DeepCenter-confirmed, under a global fraction cap.
- **Topology repairs** (`filter_output_graph`) — `enforce_next_frame`, `single_parent_repair`,
  `prune_isolated`, `edge_max_um = 14.0`.
- Short-track filter + linefit smoother — **we already have both**, and they are unchanged from the public
  classical recipe we reimplemented.

## What this means for us

Ranked by (expected gain) / (cost), all CPU-side and doable without the GPU box:

1. **Raise the detection threshold.** We run 0.5; the frontier runs 0.99. Our own full-41 sweep was already
   monotone upward to 0.95 (0.8742 → 0.8861) and we never went further. Cheapest possible experiment — a
   threshold sweep on cached responses, no retraining.
2. **Gap close with synthetic nodes** (`1ro`, S3 token `stlfg`). Already filed, still untried, and it is in
   *both* the classical 0.857 recipe and the learned frontier. The one post-proc stage everyone has and we
   do not.
3. **Topology repairs** — cheap, deterministic, no model.
4. **Dual-seed blend** — mount the second checkpoint, average logits. Small code change, no training.
5. **Learned bonus in the linker** — needs the edge transformer we have not grafted.
6. **DeepCenter veto** — a third model to mount; largest change, most machinery.

One caveat, and it is not the one to expect. **Borrowing is not a rules problem** — `rules.txt` §2.6(b)
makes external models "acceptable unless specifically prohibited by the Host", no such prohibition exists,
and all four pilkwang packs are **CC0**, compatible with the MIT winner licence. There is no from-scratch
requirement. A pretrained model is a dependency.

The real caveat is strategic: a **frozen** borrowed detector cannot be improved, so every future gain has
to come from the stages around it. That is what makes `80e`/`cdg` worth doing — not integrity, leverage.
The recipe is no longer unknown either: ×4 Y/X downsample, per-video quantile norm at 0.001/0.999, W=2
window, per-frame voxel BCE, architecture in `external/`. Fine-tuning the public weights on our fold-train
is the cheaper half of the same lever.

Mirror the weights into our own kit (`build_kit.py` already does) so reproducibility does not depend on an
upload staying live — CC0 is irrevocable as a licence, but the file could still be deleted.

## Also on the board, not part of the fork

- `aaaa1597/s1-06-stardist-btrack-pipeline` (2026-08-02) — StarDist + btrack, a genuinely different stack.
  Untested by us; the only visible alternative to the pilkwang lineage.
- `josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking` (2026-08-01) — synthetic training
  data generation. Relevant to `cdg`: our from-scratch run is data-starved at ~1.3 epochs.
- `xiaoleilian/biohub-ct-mix-divaug` (150 votes) — division augmentation.
- The metric hack (`kirneo`, `outwrest`) remains present on the board. Already tested against our faithful
  local metric and found near-inert; no change.

## Sources

- Survey tooling: `kaggle/survey.py` (this repo)
- `saitejabandaruin/biohub-top-notebook-0-913` — public ~0.913, fork of the below
- `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` — the lineage head, 163 votes
- `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` — 182 votes, gap recovery
- `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` — 75 votes, "target 0.930" preset
- `indarkarhana/biohub-dual-seed-frame-retention-guard-v1` — 67 votes
- Public model weights: `pilkwang/biohub-tracking-support-pack-50ep-v1`,
  `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
