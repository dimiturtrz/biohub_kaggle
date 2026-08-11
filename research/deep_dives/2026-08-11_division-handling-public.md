# Division handling in the public solution space — source-level read

**Date:** 2026-08-11 · **Method:** `kaggle datasets download` / `kaggle kernels pull`, then read the actual
source. Every claim below is tagged **[SRC]** (I read the file) or **[META]** (inferred from metadata /
listing only, not opened) or **[CALC]** (arithmetic on [SRC] values).

**Why this pass exists.** A prior research pass searched *about* notebooks instead of pulling them and
reported "no public notebooks above 0.90" on a day the frontier was 0.913. This pass pulls bytes.

**Scoring context.** `score = adjusted_edge_jaccard + 0.1 * division_jaccard`. Our ported scorer
(`core/metrics/divisions.py`) credits a division by *reachability* — a predicted fork on the parent side of
the grandparent→grandchildren window reaching two GT daughter lineages down two different branches. Not
exact edges, not the exact frame. **So the binding constraint is false-positive precision, not placement.**

**Our binding measurement.** Dense movie, 3 annotated divisions. Uncapped affinity recovery placed the true
fork but emitted **807 false forks**; any cap then evicted the true one.

---

## 0. What I could and could not open

| Artifact | Status | Evidence |
|---|---|---|
| `qualzz/biohub-division-pair-verifier-v1` | **OPENED, full source** — 4 `.py` (1,926 lines), 2 Keras `model.json`, 2 `.weights.h5`, `geometry_prefilter.joblib`, 2 `summary.json`, `history.csv` | [SRC] |
| `cluckinvv/biohub-division-two-stage-v1` | **CANNOT OPEN — 403 Forbidden.** Not private-to-me; the dataset does not exist publicly. `kaggle datasets list --user cluckinvv` → **"No datasets found"** (zero public datasets for that user). Their public kernel `biohub-no-hack-ranker-public-v2` mounts *no* such dataset (sources are 4× `pilkwang/*`). | [SRC] on the 403 + listing + kernel metadata |
| `sushanthtiruvaipati/biohub-divgeom-filter-single-child-repair-v1` | **CANNOT OPEN — 403 Forbidden.** `datasets list --user sushanthtiruvaipati` returns 3 datasets, **none biohub-related** (jigsaw / LLM models). | [SRC] on the 403 + listing |

> **Correction to our own notice board:** artifacts 2 and 3 should be treated as **non-existent public
> artifacts**, not as "unopened". Do not re-queue them. Note that their *names* describe things that DO
> exist as code in the frontier notebook (`OUTPUT_DIVISION_GEOMETRY_FILTER`, `OUTPUT_SINGLE_CHILD_REPAIR`) —
> i.e. the names look like they were synthesised from frontier constant names rather than observed.

Frontier lineage pulled and converted (7 notebooks, all [SRC]):
`yusuketogashi/no-hack-biohub-cell-another-approch-3rd` (4,032 lines), `cluckinvv/biohub-no-hack-ranker-public-v2`,
`tangai1/biohub-clean-forward-lookahead-v1`, `nikitagajbhiye30/biohub-11` (all byte-identical size 210,651),
`lonnieqin/biohub-gap2-joint-node-budget`, `raykkretzschmar/biohub-harmonic-bidirectional-association-v1`,
`navazshfathi/best-score-biohub-...`.

---

## 1. THE HEADLINE ANSWERS

**Q: What separates a true division from a crowded false fork?**

Two *different* public answers, and they disagree completely:

| | Frontier lineage (LB 0.915) | `qualzz` pair verifier (not on any public LB) |
|---|---|---|
| Mechanism | **Pure geometry + an orphan requirement + hard caps.** No learned verifier. | **Learned 3D CNN verifier** (image + geometry), gated by a learned GBDT geometry prefilter. |
| Learned component | The optional DeepCenter veto exists but is **switched OFF** in every fork. | Two Keras CNNs + one `HistGradientBoostingClassifier`. |

**The single most load-bearing rule at the frontier is not a gate at all — it is a candidacy restriction:**
a new daughter must be a node with **no incoming edge** (an orphan). [SRC]

```python
candidate_ids = [node_id for node_id in child_frame_ids
                 if node_id not in incoming and node_id not in used_targets]
```
`add_safe_divisions_postlink`, line 3143.

This is the mechanism we lack. Our affinity recovery proposes a second child from *all* neighbours; the
frontier only ever proposes a second child that **nothing else claimed**. In a crowded frame every plausible
false sister is already the target of some other track's edge, so it is structurally ineligible. Divisions
are only ever *added*, never *stolen* — no edge is re-parented, so a fork cannot be created by breaking an
existing good link. That converts "807 false forks" from a ranking problem into a candidacy problem.

**Q: How many forks do they emit?** **311**, on 123,088 nodes / 119,005 edges (the notebook's own
`_known_output_signature`, line 3874-3879). [SRC] That is **0.26 % of edges**.

**Q: Before or after the short-track filter?** **BEFORE**, and the short-track filter is explicitly made
division-blind. `add_safe_divisions_postlink` line 3559 → `filter_short_track_components` line 3611, with
`OUTPUT_KEEP_DIVISION_COMPONENTS=1` rescuing any component containing a fork from the length-6 cull. [SRC]
This is exactly the ordering that our own measurement found flips divisions from net-negative to net-positive —
the public frontier independently landed on it, plus the rescue clause we do not have.

---

## 2. Frontier lineage — every division constant, verbatim

All values [SRC]. Environment overrides are set at notebook lines 28-44; defaults at 114-220. **All seven
forks set byte-identical division values** — this is a monoculture, one author's tuning replicated.

### 2.1 `SAFE_DIV_*` — the gates on adding a fork

| Constant | Default in code | Override set by all 7 forks | Unit | Role |
|---|---|---|---|---|
| `OUTPUT_SAFE_DIVISIONS` | `1` | (not overridden) → on | bool | master switch |
| `SAFE_DIV_MAX_UM` | `4.7` | **`4.66`** | µm | max parent→**new** daughter distance |
| `SAFE_DIV_SISTER_MAX_UM` | `7.2` | **`8.5`** | µm | max **sister–sister** distance |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | `7.8` | **`7.65`** | µm | max parent→**existing** child distance (else parent is disqualified entirely) |
| `SAFE_DIV_FRAME_FRAC_CAP` | `0.008` | **`0.0076`** | fraction | per-frame cap, **fraction of single-child sources in that frame** |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | `0.004` | **`0.00375`** | fraction | global cap, **fraction of total edges** |
| `DEEPCENTER_SAFE_DIV_VETO` | `1` | **`'0'` → OFF** | bool | learned appearance veto — **disabled at the frontier** |
| `DEEPCENTER_SAFE_DIV_THRESHOLD` | `0.12` | (unused, veto off) | heatmap prob | veto threshold |

Cap arithmetic [CALC on SRC]:
- `global_cap = max(1, round(len(edges) * 0.00375))` → `round(119005 * 0.00375)` = **446 forks max**.
- Observed divisions = 311 < 446, and 311 also includes forks surviving from the ILP/motion stage.
  **The global cap is not binding — the geometry + orphan gates bind first.** [CALC]
- `frame_cap = max(1, round(len(source_ids) * 0.0076))` where `source_ids` = nodes in frame `t` with
  **exactly one** outgoing edge. A 500-cell frame → cap 4; a 130-cell frame → cap `max(1, 1)` = 1. [CALC]

### 2.2 The decision rule, quoted

Per frame `t`, for each source with exactly one outgoing edge (line 3142), and each orphan candidate at
`t+1` (line 3143):

```python
child_dist = edge_distance_um(source, existing_child)
if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:   continue     # 7.65 um
parent_dist = edge_distance_um(source, candidate)
if parent_dist > SAFE_DIV_MAX_UM:                 continue     # 4.66 um
sister_dist = edge_distance_um(existing_child, candidate)
if sister_dist > SAFE_DIV_SISTER_MAX_UM:          continue     # 8.5 um
if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(...): continue   # OFF
score = parent_dist + 0.15 * sister_dist
```
lines 3156-3182. Then `proposals.sort(key=lambda item: item[0])` — **ascending, lowest score wins** —
and admission stops at `global_cap` / `frame_cap`, with `used_targets` guaranteeing each node is claimed
as a new daughter at most once (lines 3187-3206).

Note what is **absent**: no appearance feature, no temporal consistency, no velocity/split-alignment term,
no learned score. `score = parent_dist + 0.15 * sister_dist` is the entire ranking function. The
existing child's edge probability is never consulted.

### 2.3 The other division knobs — and which are dead code

| Constant | Value | Live? |
|---|---|---|
| `OUTPUT_KEEP_DIVISION_COMPONENTS` | `1` (explicitly set) | **LIVE** — rescues fork-bearing components from the short-track cull |
| `OUTPUT_MIN_TRACK_LEN` | `6` (nodes) | LIVE — short-track threshold |
| `OUTPUT_FILTER_SHORT_TRACKS` | `1` | LIVE |
| `OUTPUT_SINGLE_PARENT_REPAIR` | `1` | **LIVE** — post-link, keeps best edge per target → enforces in-degree ≤ 1 (line 3525-3534) |
| `OUTPUT_SINGLE_CHILD_REPAIR` | `0` (default, not overridden) | **OFF** — so ILP-stage forks survive into the output |
| `OUTPUT_DIVISION_GEOMETRY_FILTER` | `0` (default, not overridden) | **DEAD CODE at the frontier** |
| `DIV_PARENT_MAX_UM` | `10.5` µm | dead (guarded by the above) |
| `DIV_SISTER_MAX_UM` | `8.0` µm | dead |
| `DIV_DROP_TO_SINGLE_IF_BAD` | `1` | dead |
| `ILP_DIVISION_WEIGHT` | `1.0` | LIVE but **untuned** — passed as `--ilp-division-weight` to an external tracking binary (line 1745); left at 1.0 in all 7 forks |
| `ADAPTIVE_SHORT_TRACK_RESCUE` | `0` | off |

> The "divgeom filter + single-child repair" that artifact 3's *name* advertises is precisely the pair of
> knobs the frontier leaves **switched off**. [SRC]

### 2.4 Pipeline order (function call order, lines 3505-3612) [SRC]

| # | Stage | Line | Division effect |
|---|---|---|---|
| 1 | `motion_relink_edges` | 3518 | may emit multi-child |
| 2 | single-**parent** repair (in-deg ≤ 1) | 3525 | ON |
| 3 | single-**child** repair (out-deg ≤ 1) | 3536 | **OFF** — forks preserved |
| 4 | `close_single_frame_gaps` | 3549 | gap closure (DeepCenter gap veto ON, thr 0.10) |
| 5 | `recover_strict_gap2` | 3558 | |
| 6 | **`add_safe_divisions_postlink`** | **3559** | **adds forks here** |
| 7 | division geometry filter | 3569 | **skipped** (flag 0) |
| 8 | prune isolated nodes | 3603 | |
| 9 | **`filter_short_track_components`** | **3611** | culls <6-node components **except** fork-bearing ones |
| 10 | `linefit_smooth_output_graph` | 3612 | |

**Divisions are added at step 6 and protected at step 9.** LB context: `KNOWN_CLEAN_LB = 0.915`,
`TARGET_LB = 0.916` (lines 1314-1315). [SRC]

---

## 3. `qualzz/biohub-division-pair-verifier-v1` — a genuinely learned verifier

**Licence: CC BY 4.0 — attribution required.** This is *not* CC0 like the association ranker. If we mount
or derive from these weights we must credit "Biohub Division Pair Verifier V1 by qualzz (CC BY 4.0)" in the
notebook and any README. Data-attribution obligation survives into our submission notebook. [SRC: Kaggle
CLI licence line + `README.md`]

`README.md` in full [SRC]: *"Learned weights, geometry prefilter, and minimal inference utilities for
classifying one parent plus two daughter candidates. Trained only from the Biohub competition training
split. No test predictions or raw competition volumes are included."*

### 3.1 Three models, in series

| Stage | Artifact | Type | Input | Output |
|---|---|---|---|---|
| 1 | `geometry_prefilter.joblib` | `HistGradientBoostingClassifier` | 10 geometry features | `geometry_probability` |
| 2 | `best.weights.h5` + `model.json` (`division_pair_verifier_tf`) | Keras 3D CNN | image `(17, 64, 64, 8)` + geometry `(10,)` | `pair_probability` |
| 3 | `parent_gate.weights.h5` + `parent_gate.model.json` | Keras 3D CNN | image `(17, 64, 64, 9)`, **no geometry** | `parent_probability` |

Prefilter hyperparameters, read off the unpickled estimator [SRC]:
```
HistGradientBoostingClassifier(class_weight='balanced', l2_regularization=2.0,
                               learning_rate=0.05, max_iter=200,
                               max_leaf_nodes=15, random_state=42)
```
**`feature_names_in_` is `None`** — the model takes a **positional** 10-vector. The order is fixed by
`GEOMETRY_FEATURES` in `division_verifier.py` lines 29-40. **Do not guess this order** (this is the exact
failure mode that burned us on the association ranker manifest).

### 3.2 The 10 geometry features — verbatim, with their normalisation denominators

From `candidate_geometry_values`, `division_verifier.py` lines 47-92. Positions are voxel coords multiplied
by `SCALE_ZYX_UM = (1.625, 0.40625, 0.40625)` µm/voxel **before** any distance. `v1 = d1-p`, `v2 = d2-p`,
`split = d2-d1`, `r1=|v1|`, `r2=|v2|`.

| # | Name | Expression | Denominator | Unit after norm |
|---|---|---|---|---|
| 0 | `parent_child_near` | `min(r1, r2)` | `/ 12.0` | µm/12 |
| 1 | `parent_child_far` | `max(r1, r2)` | `/ 12.0` | µm/12 |
| 2 | `sister_distance` | `|split|` | `/ 18.0` | µm/18 |
| 3 | `parent_midpoint_distance` | `|(d1+d2)/2 - p|` | `/ 10.0` | µm/10 |
| 4 | `daughter_angle_cosine` | `v1·v2 / max(r1·r2, 1e-6)` | — | cosine, −1..1 |
| 5 | `daughter_distance_asymmetry` | `|r1 - r2|` | `/ 10.0` | µm/10 |
| 6 | `sister_z_distance` | `|split[0]|` | `/ 12.0` | µm/12 |
| 7 | `sister_xy_distance` | `|split[1:]|` | `/ 18.0` | µm/18 |
| 8 | `incoming_speed` | `|p - previous|` | `/ 10.0` | µm/frame /10 |
| 9 | `split_velocity_alignment` | `|split·velocity| / max(|split|·speed, 1e-6)` | — | cosine, 0..1 |

Order-invariant by construction (min/max, abs). Features 8-9 are **0.0** when `previous` is missing/non-finite.
`previous` = the parent's unique predecessor, only used when in-degree is exactly 1 (`pair_postprocess.py`
line 140: `previous = nodes.get(previous_ids[0]) if len(previous_ids) == 1 else None`). [SRC]

**Features 4 and 9 are the two the frontier does not have** — `daughter_angle_cosine` (a real division has
daughters on roughly opposite sides, cosine → −1) and `split_velocity_alignment` (the split axis vs. the
parent's incoming motion). These are the cheapest thing we could bolt onto our own recovery without any
learned model.

### 3.3 The CNN's 8 input channels

`DivisionCandidateDataset.__getitem__`, lines 354-384 [SRC]. Crop `(17, 80, 80)` in the dataset default but
**`(17, 64, 64)` at inference** (`pair_postprocess.py` line 579), centred on the parent.

| Ch | Content |
|---|---|
| 0-4 | raw uint16 volume at `t-2 … t+2` (`temporal_radius=2`), quantile-normalised `(v - q0.001)/(q0.999 - q0.001)`, clipped to `[0, 2.5]` |
| 5 | Gaussian marker at the **parent**, `sigma_um=1.25` |
| 6 | Gaussian marker at **both daughters** (max-combined), `sigma_um=1.25` |
| 7 | `abs(raw[t+1] - raw[t])` temporal difference |

The parent gate uses 9 channels: same 5 raw + parent marker only + three difference planes
(`current`/`previous`/`future` derived), no daughter marker, no geometry vector (`division_parent_gate.py`
lines 108-118, 208). [SRC]

Architecture (identical in the PyTorch reference and the Keras deployment): 4 `ConvBlock`s
(Conv3d-GroupNorm(8)-SiLU ×2) with strides `(1,1,1) → (1,2,2) → (2,2,2) → (2,2,2)`, channels 16→32→64→96,
global average pool → concat with a 10→48→32 geometry MLP (LayerNorm, dropout 0.10) → 64 → 1 logit,
dropout 0.20. [SRC `division_verifier.py` 402-450, `pair_postprocess.py` 30-90]

### 3.4 Candidate generation and the fork-count caps

`_sequence_candidates`, `pair_postprocess.py` lines 265-341 [SRC]:

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| `radius_um` | **18.0** | µm | kNN ball around the parent, searched at `t+1` |
| `alternatives_per_parent` | **16** | count | nearest-N alternatives kept per single-child parent |
| `proposals_per_frame` | **20** | count | after prefilter, top-N by `geometry_probability` per `(sequence, parent_t)` |

```python
proposed = (proposed.sort_values("geometry_probability", ascending=False)
            .groupby(["sequence", "parent_t"], sort=False)
            .head(proposals_per_frame))
```
lines 553-557. So the funnel is: all pairs within 18 µm → 16/parent → **20/frame** (learned rank) → CNN → threshold.

Two candidate types are produced: `existing_fork` (parent already has 2 children — audited for *removal*)
and `proposed_fork` (parent has 1 child — audited for *addition*).

### 3.5 The decision thresholds — extreme, and that is the point

`apply_pair_edits`, lines 655-661 [SRC]:

| Threshold | Value | Applies to |
|---|---|---|
| `add_proposed_threshold` | **0.999** | add a new fork |
| `keep_existing_threshold` | **0.9923445582389832** | retain an existing fork |

`keep_existing_threshold` is not a round number: it is `metrics.candidate_best_jaccard.threshold` copied
verbatim out of `summary.json`, i.e. the held-out Jaccard-optimal operating point. [SRC both files]

`0.999` for additions is *stricter* than the Jaccard-optimal point — the author deliberately
biases additions toward precision and only relaxes for forks the base linker already believed in.

### 3.6 Held-out performance — the honest numbers

`summary.json`, sequence-held-out, natural plausible-pair distribution [SRC]:

| Metric | Pair verifier | Parent gate |
|---|---|---|
| Training manifest rows / positives | 14,271 / 151 | 128,732 / 151 |
| Candidate rows / positives | 2,454 / 31 | 26,242 / 31 |
| Average precision | **0.5929** | 0.2038 |
| ROC-AUC | 0.9782 | 0.9237 |
| Best-Jaccard threshold | 0.99234 | 0.99993 |
| @ that threshold | **tp 17, fp 10, fn 14** (P 0.630, R 0.548, J 0.415) | tp 9, fp 17, fn 22 (P 0.346, R 0.290, J 0.188) |
| Frame-level AP | 0.6116 | 0.2258 |
| Training time | 7,833 s | 10,530 s |

Parent-gate fixed-threshold sweep [SRC] — the false-positive curve we care about:

| Threshold | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|
| 0.5 | 18 | **183** | 13 | 0.090 | 0.581 |
| 0.9 | 15 | 86 | 16 | 0.149 | 0.484 |
| 0.99 | 13 | 51 | 18 | 0.203 | 0.419 |
| 0.999 | 9 | 26 | 22 | 0.257 | 0.290 |
| 0.9999 | 9 | **18** | 22 | 0.333 | 0.290 |

**Read this carefully.** A parent-only appearance model (no pair geometry) at threshold 0.5 emits 183 FP for
18 TP — that is our 807-false-forks failure in miniature. Adding the *pair* (two daughters + 10 geometry
features) moves AP from 0.204 → 0.593 and gets FP down to 10 at 17 TP. **The pair structure, not the
appearance model, is what buys precision.** The parent gate's only genuinely good number is
`parent_recall_at_5 = 0.968` / `parent_median_rank = 1.0` — it is a decent *ranker* of where divisions are,
and a bad *detector*.

### 3.7 The fork-removal rule (this is a real "single-child repair")

When an existing fork scores below `keep_existing_threshold`, one child is dropped — chosen by
`_continuity_cost`, lines 619-652 [SRC]:

```python
predicted = parent_point + (parent_point - previous_point)   # constant-velocity extrapolation
cost = |child_point - predicted|                              # um
# plus, if the child itself has exactly one successor:
cost += 0.35 * |outgoing_velocity - incoming_velocity|        # um, acceleration residual
remove_child = max(costs, key=costs.get)                      # drop the worse-continuing child
```
Weight **0.35** on the acceleration term; constant-velocity prediction falls back to `parent_point` when the
parent's in-degree ≠ 1. This is strictly better motivated than the frontier's distance-only ranking.

Invariants enforced after every edit (lines 805, 834, 838) [SRC]: `out_degree ≤ 2`, `in_degree ≤ 1`, every
edge must satisfy `target_t == source_t + 1`, no duplicate edges, node cardinality unchanged — all raise
`RuntimeError` rather than silently repairing. Worth copying as assertions in our own linker.

### 3.8 Caveat — the GEFF anchoring in the same file is a leak-shaped thing

`run_postprocess` (lines 843-887) wraps the division work in `apply_geff_anchors` /
`apply_hard_geff_anchors`, which open `competition_root/train/{sequence}.geff` and **force matched
ground-truth edges — including division branches — into the submission** (`match_radius_um=7.0`,
`max_edge_um=8.0`, `stability_margin_um=0.25`). [SRC lines 344-522]

This is the sparse-annotation exploitation the "no-hack" frontier notebooks explicitly avoid. **The verifier
models are independently useful and are trained only on the train split; the anchoring functions in
`pair_postprocess.py` are not something to port.** Take `score_pairs` / `apply_pair_edits` /
`_continuity_cost` / the geometry features; leave `apply_geff_anchors` and `apply_hard_geff_anchors`.

---

## 4. Side-by-side: the two public approaches vs. our situation

| Axis | Frontier (0.915) | qualzz verifier | Us (current) |
|---|---|---|---|
| Base linker emits forks? | Yes (ILP), single-child repair OFF | Yes (consumes any submission) | **No** — min-cost-flow, strictly 1-to-1, **zero forks** |
| Fork source | ILP + `add_safe_divisions_postlink` | audits an existing submission both ways | n/a |
| Candidate pool | orphans at `t+1` within 4.66 µm | all nodes at `t+1` within 18.0 µm | all neighbours |
| Ranking signal | `parent_dist + 0.15·sister_dist` | GBDT(10 geom) → CNN(image+geom) | affinity |
| Threshold | none — pure gates + caps | 0.999 add / 0.99234 keep | n/a |
| Caps | frame 0.76 % of single-child sources; global 0.375 % of edges | 16/parent → 20/frame | uncapped → 807 FP |
| Learned verifier? | **No** (DeepCenter veto present but OFF) | **Yes**, 3 models | no |
| vs. short-track filter | **before**, + fork-components rescued | n/a (post-hoc on a final CSV) | measured net-negative when after |
| Forks emitted | **311** / 119,005 edges (0.26 %) | held-out: 17 TP + 10 FP per fold | 0 |

### What the 807-false-fork failure actually was

Ranked against the frontier, our recovery is missing **three** independent precision mechanisms, and the
caps are the *least* important of them:

1. **Orphan-only candidacy** [SRC line 3143] — the second daughter must have no incoming edge. This alone
   removes most crowded-frame false sisters, because in a dense frame the false sisters are all already
   claimed. We do not have this. **Highest value, ~zero cost, no model needed.**
2. **A sister-distance gate at all** — 8.5 µm, and an *existing*-child gate at 7.65 µm that disqualifies the
   whole parent before any candidate is considered. A parent whose current link is already long is not
   allowed to divide.
3. **Ordering + rescue** — add forks before the short-track cull, and exempt fork-bearing components from it
   (`OUTPUT_KEEP_DIVISION_COMPONENTS=1`, `OUTPUT_MIN_TRACK_LEN=6`). We measured this ordering effect
   ourselves; the frontier also pairs it with the rescue clause, which we do not have.

Only *after* those does the cap matter — and at the frontier the global cap (446) is **not even binding**
against 311 observed forks. **Caps are a backstop, not the mechanism.** Our "cap evicts the true fork"
symptom is the signature of using a cap to do a candidacy filter's job.

### Cheapest ranking improvement that needs no learned model

Add `daughter_angle_cosine` and `split_velocity_alignment` (§3.2 features 4 and 9) to the ranking score. The
frontier's `parent_dist + 0.15·sister_dist` uses neither, and a real division has daughters at cosine → −1
about the parent, with the split axis uncorrelated-to-perpendicular to the incoming velocity. Both are ~5
lines of numpy on coordinates we already have.

### On mounting the verifier

The prefilter (`geometry_prefilter.joblib`) is a 10-feature sklearn GBDT — CPU-cheap, no image reads, usable
as a *ranker* over our own fork proposals with no TensorFlow dependency at all. The two CNNs need TF 2.20 +
native-resolution zarr crops (`(17,64,64)`, 5 frames each) which is a real inference cost inside the Kaggle
time budget. **Staged plan: prefilter-only first (cheap, tests the geometry hypothesis), CNN only if the
prefilter moves the number.** Attribution required either way (CC BY 4.0).

---

## 5. Verification status

| Claim | Basis |
|---|---|
| All `SAFE_DIV_*` values, caps, ordering, 311 divisions, LB 0.915/0.916 | [SRC] `yusuketogashi_...py` lines cited inline |
| All 7 forks identical on division params | [SRC] grep across all 7 converted notebooks |
| DeepCenter veto OFF at frontier | [SRC] line 44 `= '0'` in all 7 |
| `OUTPUT_DIVISION_GEOMETRY_FILTER` dead | [SRC] default `"0"`, no override in any of the 7 |
| Verifier architecture, features, thresholds, held-out metrics, licence | [SRC] all 4 `.py` + both `summary.json` + unpickled joblib |
| Prefilter has no `feature_names_in_` | [SRC] unpickled and printed |
| Artifacts 2 & 3 do not exist publicly | [SRC] 403 + per-user dataset listings + frontier kernel metadata |
| Global cap = 446; frame cap arithmetic | [CALC] on [SRC] constants |
| **Not verified:** whether 311 splits into ILP-forks vs. safe-div-adds | the notebook's `safe_divisions_added` stat is printed at runtime only; we have no run log |
| **Not verified:** the verifier's effect on LB | it appears on no public leaderboard notebook; `summary.json` is held-out CV only |

Pulled artifacts are in the session scratchpad, not committed (data stays out of the repo).
