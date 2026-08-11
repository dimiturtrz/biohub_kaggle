# Public frontier recheck — the 0.915 clean notebook, a new CC0 ranker, and where 0.895 actually sits

**Date**: 2026-08-11
**Status**: settled for the notebook/LB half; unresolved for the public/private split question
**Supersedes**: [`2026-08-04_kaggle_online_recheck.md`](2026-08-04_kaggle_online_recheck.md) §1–2 (that
document's "no public notebook above 0.90 — unverified" is wrong: it was search-based, never pulled a
kernel, and the notebooks were at 0.913 on the day it was written). Extends
[`2026-08-02_public-notebook-frontier.md`](2026-08-02_public-notebook-frontier.md), which remains
structurally correct — one pipeline, forked — but is nine days stale on constants.

**Method**: `kaggle kernels list/pull` (17 kernels pulled and read as source, not descriptions),
`kaggle competitions leaderboard -d` (full 2225-row public CSV), `kaggle datasets download/metadata`
(the ranker artifact and the pinned official scorer downloaded and read). Web only where the API has no
endpoint.

---

## TL;DR

1. **The public notebook frontier is 0.915, and it is a single lineage — "Biohub 162" by
   `yusuketogashi`, forked ~15 ways in the last week.** We are at 0.895. **A free public notebook
   currently beats our pipeline by 0.020.** [F1, F2]
2. **The one asset we do not have is `pilkwang/biohub-local-association-ranker-unet300-v1` — CC0, 20 KB,
   a 22-feature MLP that re-ranks association candidates inside the linker cost.** It is the delta
   between the 0.908-era notebooks and the 0.915 one. Mountable offline; the artifact ships its own
   feature-name list, feature stats, and training config. [F3]
3. Two more mechanisms in the 0.915 config are new to us: **four-view Jensen–Shannon-reliability
   logarithmic-opinion-pool TTA on the *edge* head** (we TTA only the detector), and a **three-frame
   forward acceleration lookahead** bonus in the assignment cost. [F4, F5]
4. **The 0.93+ tier moved**: top public went 0.943 → **0.949**, and ≥0.93 grew from ~8 to 15 teams. The
   0.91 band is now 608 teams thick — that is the public notebook's fork-mass, not 608 independent
   methods. At 0.895 we rank roughly **990 / 2225**. [F2]
5. **No new primary evidence on the public/private split.** The official metric is confirmed exactly
   (α = 0.1 node-count penalty, division weight 0.1, 7 µm matching) from the pinned scorer source, but
   neither the scorer, the host repo, nor the competition pages state a public %, and nothing confirms or
   refutes the sparse-train/dense-hidden hypothesis. **We should stop repeating "public = 29% of test" as
   though it is sourced.** [F6]

---

## Question

What has the public side of `biohub-cell-tracking-during-development` disclosed since our 2026-08-04
scan: notebooks above 0.895 (especially ones using the same pilkwang weights we mount), leaderboard
movement, newly mountable offline weights and their licences, techniques we have not tried, and any
primary evidence about the metric or the public/private split.

---

## Findings

### F1 — The public frontier is one notebook at 0.915, and its whole config is in the clear

`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` ("🧬 Biohub 162 | Three-Frame Forward
Acceleration Lookahead | No Hack", last run 2026-08-05, 132 votes) is the lineage head. Its own header
states: *"Fixed baseline: Biohub 159B, clean leaderboard `0.915`; Immediate target: `0.916+`"*. Yusuke
Togashi's public LB row is **0.916** (154 entries, 2026-08-05). [S1, S3]

Everything else new on the board this week is a fork of it with one flag changed. Fourteen kernels pulled
carry the **byte-identical 45-line `os.environ` block**; the only diffs are the deliberate single-axis
ablations. Their authors sit at exactly **0.915** on the LB — Lonnie (`lonnieqin`), tangaii (`tangai1`),
Cluckin-VV, Rayk Kretzschmar — i.e. the notebook reproduces its claimed score for anyone who runs it.
`pilkwang` himself is at **0.921**. [S1, S3]

**The disclosed 0.915 config, verbatim** (from `lonnieqin/biohub-gap2-joint-node-budget`, lines 82–127;
identical in the other thirteen except where noted):

```
DET_THRESHOLD                    = 0.96875        # we: 0.97   — a tie
MOTION_RELINK_LEARNED_BONUS      = 1.0            # was 0.75 in the 0.913 era
ILP_APPEARANCE_WEIGHT            = 0.0            # births free
ILP_DISAPPEARANCE_WEIGHT         = 1.5            # premature endings expensive  <-- ASYMMETRIC
ILP_DIVISION_WEIGHT              = 1.0
ILP_EDGE_WEIGHT                  = -1.0
GAP_CLOSE_MAX_GAP                = 2              # was 1
GAP_CLOSE_UM                     = 5.8            # was 6.0
GAP_DENSITY_ADAPTIVE             = 1              # NEW: gate shrinks in crowded neighbourhoods
GAP_DENSITY_REFERENCE_UM         = 6.5
GAP_DENSITY_GAIN                 = 0.040
GAP_DENSITY_MAX_STEP_DELTA_UM    = 0.125
GAP_DENSITY_NEIGHBORS            = 3
OUTPUT_MIN_TRACK_LEN             = 6              # we: 6 — confirmed, 7 was never theirs
OUTPUT_KEEP_DIVISION_COMPONENTS  = 1
SAFE_DIV_MAX_UM                  = 4.66
SAFE_DIV_SISTER_MAX_UM           = 8.5
SAFE_DIV_EXISTING_CHILD_MAX_UM   = 7.65
SAFE_DIV_FRAME_FRAC_CAP          = 0.0076
SAFE_DIV_GLOBAL_FRAC_CAP         = 0.00375
ADAPTIVE_SHORT_TRACK_RESCUE      = 0              # OFF in the scoring config
USE_DEEPCENTER_VETO              = 0              # OFF in the scoring config
BIDIRECTIONAL_EDGE_WEIGHT        = 0.20           # we: 0.8/0.2 — a tie
BIDIRECTIONAL_FUSION_MODE        = harmonic_probability     # we have this
EDGE_TTA_MODE                    = js_reliability_log_pool  # NEW to us
EDGE_TTA_VIEWS                   = 4
USE_LOCAL_ASSOCIATION_RANKER     = 1              # NEW to us
LOCAL_RANKER_MODE                = full_motion_assignment
LOCAL_RANKER_FULL_WEIGHT         = 0.85
LOCAL_RANKER_PRIMARY_RETAIN_WEIGHT = 0.15
LOCAL_RANKER_MARGIN_UM           = 0.35
LOCAL_RANKER_MIN_ADVANTAGE       = 0.15
LOCAL_RANKER_MAX_BONUS           = 0.20
USE_FORWARD_ACCELERATION_LOOKAHEAD = 1            # NEW to us
FORWARD_LOOKAHEAD_MAX_ACCEL_UM   = 4.0
FORWARD_LOOKAHEAD_MAX_BONUS      = 0.20
SECONDARY_EDGE_WEIGHT            = 0.15
SECONDARY_DETECTION_WEIGHT       = 0.475          # near-balanced, not 0.5
SECONDARY_LINK_MODE              = low_margin_consensus
SECONDARY_LOW_MARGIN_MAX         = 0.35
DUAL_SEED_EDGE_THRESHOLD         = 0.48
```

Two of those deserve calling out because they contradict things we assumed:

- **`USE_DEEPCENTER_VETO = 0`.** The DeepCenter centre-prior network — which the 2026-08-02 doc ranked as
  a thing to build — is **switched off in the config that scores 0.915**. `tangai1` ran it as a
  deliberate ablation (`biohub-clean-deepcenter-gap-confirmed-v1`, threshold 0.25, expected_epoch 500);
  it is a candidate, not part of the baseline. Do not spend effort mounting it.
- **`ADAPTIVE_SHORT_TRACK_RESCUE = 0`** and `min_track_len = 6`. Our refutation of "min_track_length 7 +
  short-track rescue" is consistent with the frontier: they do not use it either. `tangai1`'s
  `joint-recall-rescue-v1` turns it on with `min_mean_edge_prob 0.82 / max_mean_edge_dist 3.25 µm /
  caps 0.018 frac, 180 abs` as an *experiment*, not a default. Old news, correctly refuted.

**Same weights, isolated method.** This lineage mounts exactly the packs we mount —
`biohub-tracking-support-pack-50ep-v1` and `biohub-temporal-unet3d-seed314159-v1` — so the 0.895 → 0.915
gap is **pure linking/post-processing config plus the ranker**, with the detector held constant. That is
the same shape of finding that the 0.897 disclosure gave us, and it is again the highest-signal fact in
this scan. [S1]

### F2 — The leaderboard: top 0.949, the 0.93 tier thickened, and the 0.91 band is a fork-plateau

Full public CSV, pulled 2026-08-11 00:10 UTC — **2225 teams**. [S3]

| threshold | teams ≥ |
|---|---|
| 0.945 | 3 |
| 0.940 | 6 |
| 0.935 | 10 |
| 0.930 | **15** |
| 0.920 | 44 |
| 0.910 | **608** |
| 0.900 | 892 |
| **0.895 (us)** | **990** |

Top ten: 0.949 Mark Cooper (2026-08-07) · 0.946 Soheil Ayati · 0.945 TWEAK · 0.943 Amin · 0.942 htnhtn ·
0.942 yuto083 · 0.939 Changye Li · 0.939 enddl22 · 0.938 Matt Goldfield · 0.935 Umesh Arampath.

Movement since the snapshot embedded in `prvsiyan`'s notebook (2026-08-02: top 0.943, ≥0.93 ≈ 8 teams):
**the ceiling rose 0.006 and the 0.93 tier roughly doubled.** The pack is still 0.930–0.947 as we
believed, now with a 0.949 head. None of the top-15 teams has a public notebook — the 0.949 method is
undisclosed.

The 44 → 608 cliff between 0.92 and 0.91 is the public notebook's fork-mass. It means two things: our
0.895 is behind ~990 teams *most of whom ran someone else's notebook*, and 0.915 is a **floor available
for free**, not a target.

### F3 — NEW mountable asset: `pilkwang/biohub-local-association-ranker-unet300-v1` (CC0)

Downloaded and read (18.5 KB total). This is the single most actionable find. [S4]

| property | value |
|---|---|
| licence | **CC0-1.0** (Kaggle dataset metadata) — no NC restriction, unlike Cellpose/CellTracker-GNN |
| files | `ASSOCIATION_RANKER_MANIFEST.json`, `local_association_features.stats.json`, `model/local_association_ranker.pt` (20 KB), `model/model_info.json`, `model/history.json` |
| architecture | MLP, `hidden = 64`, `dropout = 0.05`, 22 inputs |
| training | 30 epochs max, `lr 1e-3`, `weight_decay 1e-4`, `patience 4`, `seed 2026`, **best epoch 7**, `val_fraction 0.15`, `split_column = "dataset"` |
| data | 199 datasets, **385,648 rows in 126,705 candidate groups** (108,876 train / 17,829 val groups), 126,828 positive rows |
| reported val | `best_score = 0.9777` (group-level; the manifest does not name the metric) |
| author's own note | `"runtime_intent": "Use only as a constrained local association tie-breaker, not as a global edge veto."` |

**The true 22 features** (from `ASSOCIATION_RANKER_MANIFEST.json` — note this is *not* the list hard-coded
as `_RANKER_FALLBACK_FEATURES` in the notebooks, which is a stale guess the loader overrides with
artifact metadata; if we reimplement, use this one):

```
edge_prob, has_learned_edge, source_out_degree, target_in_degree,
source_frame_count, target_frame_count, source_density_7um, target_density_7um,
candidate_rank_dist, candidate_count,
edge_dz_um, edge_dy_um, edge_dx_um, edge_dist_um, edge_xy_um, edge_abs_z_um,
motion_dist_um, motion_gain_um,
source_has_prev, target_has_next, target_best_next_prob, t_norm
```

Every one of those is computable from state our linker already has. Nothing needs the image.

**How it is applied** (`tangai1/biohub-clean-joint-recall-rescue-v1`, lines 2756–2766) — this is the whole
mechanism, in one line:

```python
evidence = 0.85 * ranker_prob[i, col] + 0.15 * primary_edge_prob[i, col]
cost[i, col] = motion_dist[i, col] + 0.05 * raw_dist[i, col] - 1.0 * evidence
```

then `scipy.optimize.linear_sum_assignment(cost)`. Two things follow. First, the learned evidence entering
the cost is **85% ranker / 15% the transformer edge probability** — the ranker has largely *displaced* the
affinity transformer as the association signal, with the transformer surviving only as a feature
(`edge_prob`) inside it. Second, the alternative mode `low_margin_top2_rescue` (apply the ranker only when
the top-2 cost margin < 0.35 µm and the ranker prefers the runner-up by ≥ 0.15, bonus capped at 0.20) is
the author's stated intent — but **`full_motion_assignment` is what the 0.915 config actually runs.**

Caveat for us: this cost sits inside a per-frame Hungarian assignment; **our linker is a global
min-cost-flow.** The evidence term is additive and portable (our cost is `distance − 20·P`; this becomes
`distance − 20·P − w·ranker_prob`), but the `candidate_rank_dist` / `candidate_count` features are
defined relative to a per-source candidate list, so a faithful port needs that list constructed even if
the assignment stays global. That is bookkeeping, not research.

### F4 — NEW technique: four-view JS-reliability logarithmic opinion pool on the *edge* head

`yusuketogashi/no-hack-biohub-cell-another-approch-3rd`, lines ~1690–1755. We do 4-fold flip TTA on the
**detector**. This does TTA on the **edge/affinity head** and — the actual idea — fuses the views with
*label-free reliability weights* rather than averaging:

1. Four views: identity, `flip_x`, `flip_y`, `transpose`; each produces a harmonic-fused per-source
   probability distribution over targets, un-flipped back to canonical orientation.
2. Consensus `p̄ = mean over views`. For each view v and each source, the **Jensen–Shannon divergence**
   `JS(p_v ‖ p̄)` is computed *over the target axis* — one scalar per (view, source).
3. Weights `w_v = 1 / (1 + JS_v / median_v(JS))`, normalised over views. A view that is a spatial outlier
   for that particular source is smoothly down-weighted; the weight is **one coherent number per view per
   source**, deliberately not per-edge ("rather than selecting different views independently for every
   edge candidate").
4. **Logarithmic opinion pool**: `log p_pooled = Σ_v w_v · log p_v`, softmaxed. Their comment gives the
   mechanism: a log pool "rewards associations supported across multiple reliable views and avoids the
   graph inflation caused by a single optimistic spatial view" — i.e. it is a soft AND, where an
   arithmetic mean is a soft OR.
5. Pooled logits are re-centred/re-scaled onto the identity view's logit scale (ratio clamped to
   [0.5, 2.0]) so the downstream threshold and ILP see an unchanged numeric range.

This is not "TTA expansion to 8-fold" (which we refuted). It is 4 views on a head we do not TTA at all,
combined by a rule that is geometric-mean-like rather than arithmetic. The re-scaling trick at step 5 is
also why it can be dropped in without re-tuning the candidate threshold.

The same notebook shows the detector-side TTA the secondary seed gets: flips **plus** `rot90(k=1,3)`,
transpose, and rot90∘transpose — 8 views — with the secondary's logits mean/std-aligned to the primary's
(scale ratio clamped to [0.5, 2.0]) before the `0.475` blend. Our dual-seed blend does not do the
moment-alignment; on two independently trained nets it is a real correction, not a cosmetic one.

### F5 — NEW technique: three-frame forward acceleration lookahead

Same notebook, lines 1496–1544, full source; the entire mechanism:

```python
current_velocity = target_pos - source_pos
residual = min over next_id in frame t+2 of ‖(next_pos - target_pos) - current_velocity‖
           # only candidates with ‖next_pos - target_pos‖ <= 10.0 µm are considered
support  = max(0, 1 - residual / 4.0)
cost[i, col] -= 0.20 * support
```

A candidate edge is rewarded when *some* continuation into t+2 exists that keeps velocity constant. It is
a constant-acceleration prior evaluated one frame beyond the edge being scored — cheap (it reuses the
existing frame index), and unlike "motion relink" (which we refuted) it does not re-decide links, it only
shades the cost of the link under consideration. Their notebook ships a semantic unit test asserting a
straight continuation gets the full 0.20 and a sharp turn gets less.

This is the *only* change between Biohub 159B (0.915) and Biohub 162 (target 0.916). The author's own
target for it is **+0.001 — below our 0.01 noise floor.** Take the mechanism seriously; do not take the
claimed gain seriously.

### F6 — Metric confirmed exactly; the public/private split remains unsourced

`t2better/biohub-patched-scorer-v1` (2026-08-08) ships the **official `tracking_cellmot` source pinned at
commit `075fc5f5a52d11077f9dc2b074644618f26939e2`**, licence field "unknown" on Kaggle but the pack
includes the upstream `LICENSE` and `metrics.md`. Read directly from
`src/tracking_cellmot/metrics.py` [S5]:

- `ADJUSTMENT_ALPHA = 0.1`, `SCORE_DIVISION_WEIGHT = 0.1`, `max_distance = 7.0` µm default.
- `J_adj = max(0, J · (1 − α · total_node_ratio))`; `score = adj_edge_jaccard + 0.1 · division_jaccard`.
- Node matching is optimal bipartite assignment on centroid distance, ≤ 7 µm.
- Host `metrics.md` states plainly: *"Our ground-truth annotations are sparse: we haven't annotated every
  cell in the videos."* [S6]

**What is NOT in any primary source I could reach**: any public/private split percentage; any statement
that the four test movies also appear in train; any statement that the hidden evaluation is *densely*
annotated while the released annotations are sparse. The host repo README, `metrics.md`, and the scorer
source are all silent. The Kaggle Overview/Data/Discussion pages are client-rendered and return only a
title to WebFetch, and the Kaggle API exposes no discussions endpoint — so "absent from my sources" here
means **not checked**, not **refuted**.

**Recommendation**: demote "public = 29% of test" and the sparse-train/dense-hidden claim to *unverified
working assumptions* in any doc that states them as fact. The sparse→dense hypothesis has a real
consequence — it predicts that recall-adding post-processing is under-rewarded locally and over-rewarded
on the hidden set — so it is worth resolving, but it must be resolved by reading the Overview/Evaluation
tab in a browser, not by another API pass.

### F7 — Other newly published mountable artefacts (licences checked)

| dataset | date | licence | note |
|---|---|---|---|
| `pilkwang/biohub-local-association-ranker-unet300-v1` | 2026-07-10 | **CC0-1.0** | F3. The one that matters. |
| `dmitriygerasimov/biohub-tracklet-transformer-v1` (+ `-tracklet-min3-v1`) | 2026-08-08/09 | **CC0-1.0** | 20 MB. A *tracklet*-level transformer — association above the edge level. Not read; no notebook I pulled mounts it. |
| `qualzz/biohub-division-pair-verifier-v1` | 2026-08-07 | **CC BY 4.0** | 4.8 MB, 45 downloads. A division-pair verifier — the second-opinion model for divisions that DeepCenter was being used for. Attribution required. |
| `musculer/biohub-hoct-general-v0-official` | 2026-08-05 | **unknown** | 23 MB "HOCT general v0 official public weights". Unknown licence = do not ship. |
| `bigbag1983/biohub-t2-neg01-detector` | 2026-08-05 | — | another detector pack, unverified |
| `t2better/biohub-patched-scorer-v1`, `-audit-code-v1` | 2026-08-08 | unknown/upstream | pinned official scorer, useful as a local-eval cross-check |

All are Kaggle-hosted, therefore mountable in an offline kernel. Nothing here needs internet at run time.

---

## What we have not tried

Checked against the refuted list in the brief (gate 14 µm, appearance-in-linker, motion relink,
min_track_length 7 + rescue, boundary-truncated rescue, division recovery, synthetic pretraining,
Trackastra, Ultrack, cellpose/StarDist/DeepCenter as complementary detectors, 8-fold TTA, hard-negative
edge finetune, contrastive trunk, mutual-agreement gating, position-aware boundary priors) — these are the
ones that survive as genuinely new:

1. **Learned local-association re-ranker in the linking cost** (F3). Mount the CC0 artifact, or train our
   own on the same 22 features — 385k rows, 64-hidden MLP, converged at epoch 7, this is minutes of CPU.
   Training our own also gives us the feature pipeline we would need anyway, and side-steps any doubt
   about what the borrowed net saw.
2. **JS-reliability log-pool fusion of edge-head TTA views** (F4). Distinct from both "8-fold TTA" and
   "mutual-agreement gating": the novelty is per-view-per-source reliability weights from JS divergence
   feeding a *logarithmic* pool.
3. **Moment-alignment (mean/std, ratio clamped [0.5,2]) of the secondary seed's logits before blending**
   (F4). Small, mechanical, and our current 0.8/0.2 blend does not do it.
4. **Forward acceleration lookahead** (F5) — constant-velocity support from t+2, as a bounded cost shade.
5. **Density-adaptive gap-close gate** (F1) — gap distance shrinks with local neighbour density
   (ref 6.5 µm, gain 0.040, ≤0.125 µm per step, 3 neighbours). Our gap bridge uses a fixed gate. This is
   *not* "gate widening", which we refuted; it is the opposite, applied conditionally.
6. **Asymmetric appearance/disappearance priors: births free (0.0), premature endings expensive (1.5)**
   (F1). Our track-boundary cost is a symmetric 3. Given the metric's node-count penalty
   (α=0.1 · node ratio), an asymmetry that permits genuine entries while punishing fragmentation has a
   plausible mechanism, and it is a one-constant change.
7. **`GAP_CLOSE_MAX_GAP = 2` with a shared synthetic-node budget** across gap-1, gap-2 and safe-division
   insertions (`SHARED_SYNTHETIC_NODE_BUDGET_FRAC 0.05 / ABS 2000`, in
   `lonnieqin/biohub-gap2-joint-node-budget`). A *joint* budget is the interesting part — it makes the
   three recovery stages compete for one node allowance rather than each spending its own.
8. **Tracklet-level transformer** (`dmitriygerasimov/...-tracklet-transformer-v1`, CC0) — association
   above the edge level, unexplored by every notebook I pulled. Speculative; listed for completeness.

Explicitly **not** worth doing, on this evidence: mounting DeepCenter (off in the 0.915 config, F1);
`min_track_length 7` / short-track rescue (off in the 0.915 config, and we already refuted it).

---

## Corrections to prior docs

1. **`2026-08-04_kaggle_online_recheck.md` §1 is wrong.** It reports "no public notebooks above 0.90 —
   unverified" on a day when the public lineage was at 0.913 and climbing. Root cause: it searched *about*
   notebooks instead of pulling them. Every finding in this document that matters came from
   `kaggle kernels pull` + reading source. **The Kaggle CLI is the tool; web search is not.**
2. **`2026-08-02_public-notebook-frontier.md`** listed "DeepCenter veto" as a frontier component to build
   (its item 6). As of the 0.915 config it is **disabled by default**. Its constants there
   (`DET_THRESHOLD 0.99`, `GAP_CLOSE_MAX_GAP 1`, `GAP_CLOSE_UM 6.0`, `MOTION_RELINK_LEARNED_BONUS 0.75`,
   `GAP_CLOSE_MAX_ADDED_FRAC 0.038 / ABS 1650`, `SAFE_DIV` geometry `10.5/8.0`) are all superseded — see
   the block in F1.
3. **"public = 29% of test" and "test movies are sparsely annotated in train but densely annotated in the
   hidden eval"** are, as far as any primary source I reached goes, **unsourced**. Neither confirmed nor
   refuted. Flag them as assumptions wherever they appear.
4. The `_RANKER_FALLBACK_FEATURES` list that appears in all fourteen forked notebooks is **not** the
   ranker's real feature contract — the artifact manifest's list is (F3). Anyone copying the feature list
   out of a notebook rather than the manifest gets eight features wrong.

---

## Sources

- [S1] `yusuketogashi/no-hack-biohub-cell-another-approch-3rd` — "Biohub 162", last run 2026-08-05,
  132 votes. Pulled 2026-08-11 00:12 UTC. Header claim "Biohub 159B, clean leaderboard 0.915"; JS-pool at
  lines ~1690–1755; forward-lookahead at 1496–1544.
- [S2] Forks pulled and read 2026-08-11 00:09 UTC (15 kernels): `lonnieqin/biohub-gap2-joint-node-budget`,
  `tangai1/biohub-clean-{forward-lookahead,harmonic,gap2-joint-budget,short-track-rescue,detector-recall-09625,joint-recall-rescue,deepcenter-gap-confirmed,frame-retention-guard}-v1`,
  `cluckinvv/biohub-{no-hack-ranker-public-v2,division-two-stage-v1}`,
  `sushanthtiruvaipati/biohub-divgeom-filter-single-child-repair-v1`,
  `kevin250304/biohub-nohack-reproduction-20260810`, `nikitagajbhiye30/biohub-11`,
  `maulikgajera/biohub-cell-tracking-truncated-track-rescue-patch`,
  `prvsiyan/biohub-clean-public-frontier-lineage-tracker` (documents an owned 0.913 row, 237,298 rows,
  SHA-256 `8c1605b5944d…`), `raykkretzschmar/biohub-harmonic-bidirectional-association-v1`.
- [S3] `kaggle competitions leaderboard biohub-cell-tracking-during-development -d`, CSV timestamped
  2026-08-11T00:10:11 UTC, 2225 rows. Named rows cited: Mark Cooper 0.949; pilkwang 0.921 (2026-08-07);
  yusuketogashi 0.916 (2026-08-05); lonnieqin / tangai1 / cluckinvv / raykkretzschmar 0.915.
- [S4] `kaggle datasets download pilkwang/biohub-local-association-ranker-unet300-v1`, downloaded and read
  2026-08-11 00:12 UTC. `ASSOCIATION_RANKER_MANIFEST.json`, `model/model_info.json`,
  `local_association_features.stats.json`. Licence CC0-1.0 per `kaggle datasets metadata`.
- [S5] `t2better/biohub-patched-scorer-v1`, downloaded 2026-08-11 00:13 UTC.
  `SCORER_COMMIT.txt` = `075fc5f5a52d11077f9dc2b074644618f26939e2`;
  `src/tracking_cellmot/metrics.py` lines 28–34, 276, 420.
- [S6] `royerlab/kaggle-cell-tracking-competition`, `metrics.md` and `README.md`, fetched 2026-08-11
  00:12–00:14 UTC. Confirms the metric; silent on the split.
- [S7] `kaggle datasets list -s biohub --sort-by published` + per-dataset `metadata`, 2026-08-11
  00:13 UTC, for the licences in F7.
- **Not obtained**: Kaggle Discussion threads and the Overview/Evaluation tab — client-rendered, no API
  endpoint, WebFetch returns title only. Any claim about the public/private split needs a browser.
