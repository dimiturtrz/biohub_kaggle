# celltrack structure audit — miscategorisation, division scatter, naming, grab-bags

Date: 2026-08-10 · Scope: `celltrack/` package layout only (analysis + proposal; nothing moved).

## Method

Read every module's docstring + primary class; pulled fan-in/out from
`devtools.graph.fitness core celltrack` and traced actual import edges. Gate state at time of audit:
**clean** — no god-file (all < 750 line ceiling; largest is `postproc/gap_closer.py` at 460),
no god-module (`bottleneck_degree = 8`; the top bottleneck `TemporalUNetDetector` is only 6-in / 5-out),
the one import cycle is in `core.data.split` (out of scope). So this audit is about *cohesion and
naming*, not gate violations — the proposals below improve legibility, they do not un-break a red gate.

## Shipped-path map (what a move costs)

`__init__.py` files are all **empty** — no re-exports, so every import is a direct module path and a
rename propagates to every consumer, including the Kaggle kernels which import celltrack module paths
verbatim (`from celltrack.detectors.pipeline import ...`, `from celltrack.linkers.motion_linking import ...`).
The kit bundles `tracker.py` + the packages it reaches, so **any rename on tracker's import closure =
kit rebuild**. Kernel-referenced modules found: `detectors.pipeline`, `detectors.response_cache`,
`linkers.motion_linking`, `linkers.ilp_linking`. These are the expensive ones to touch.

Layer contract (import-linter, must not invert):
`affinity` < `detectors` < `edges | linkers | postproc` < `tracker` < `eval` < `training`.
`edges/linkers/postproc` are **siblings at one layer**, so moving a file *between* them never inverts
the contract — only its consumers' import lines change.

---

## Findings

### 1. MISCATEGORISED — `edges/affinity_division.py` is a postproc stage, not an edge scorer  ★ top pick
`edges/` holds affinity **producers**: `EdgeTransformerScorer` and `BlendedEdgeTransformerScorer` both
emit an `EdgeAffinity`. `affinity_division.py` is the odd one out: `AffinityDivisionRecovery` is a
post-link **graph→graph recovery stage** that *consumes* an affinity to re-attach a second daughter —
structurally identical to `postproc/division_recovery.py` (`DivisionRecovery`), which is the geometry-gated
version of the exact same operation. They are strategy variants of one concept ("recover the fork a
1-to-1 linker dropped") living in two different packages. It belongs in `postproc/`, beside its twin.
- Imports only `affinity` + `core` → moving to `postproc` keeps the layer contract valid (postproc > affinity).
- **Only one consumer**: `tracker.py:22` (`AffinityDivisionConfig`). No kernel imports it directly.
- Shipped path: yes (tracker → kit rebuild) but **1 import line** to edit. Cheapest high-value move.

### 2. CROSS-CUTTING SCATTER — DIVISION lives in 3 packages; the fix is exactly finding #1
Division appears as `linkers/division_linking.py` (`DivisionAwareLinker`),
`postproc/division_recovery.py` (`DivisionRecovery`), `edges/affinity_division.py`
(`AffinityDivisionRecovery`). This is **not** a case for a new `division/` package — the three are
genuinely different *kinds*:
- `DivisionAwareLinker` emits forks *during* assignment → it is a **linker**, correctly in `linkers/`.
- `DivisionRecovery` + `AffinityDivisionRecovery` are both **post-link recovery stages** → both belong in
  `postproc/`.
So the scatter fully resolves by moving `affinity_division` into `postproc` (finding #1). After that,
each division artefact sits with its own *kind*, and the two recovery variants are adjacent (inviting a
later shared base / strategy unification — a follow-up, not this audit).

### 3. NAMING — `linkers/linking.py` vs `linkers/linkers.py` is a homophone trap, and `linking.py` holds two subjects
- `linkers/linking.py` = the `Linker` **Protocol** (the family contract) **+** `NearestNeighbourLinker`
  (the baseline concrete). Two subjects: a shared vocabulary and one implementation of it.
- `linkers/linkers.py` = `LinkerConfig` + `build(...)` — the **factory/registry** that dispatches a name
  to a concrete.
- `linkers/{assignment,flow,ilp,motion,division}_linking.py` = the concrete impls.

Two problems: (a) `linking.py` and `linkers.py` are near-indistinguishable by name yet do opposite jobs
(contract+baseline vs factory); (b) the `Linker` Protocol is shared vocabulary several producers speak —
the same role `affinity.EdgeAffinity` plays at the package root — but it is buried in a file that also
ships a concrete. Proposal:
- Split `linking.py` → `linkers/linker.py` (`Linker` Protocol alone — the family contract) +
  `linkers/nearest_neighbour_linking.py` (`NearestNeighbourLinker`, matching the `*_linking.py` sibling scheme).
- Rename `linkers/linkers.py` → `linkers/registry.py` (or `build.py`) — says "factory", kills the homophone.
- Consumers to update: `tracker.py`, `eval/proxy.py`, `eval/bracket.py`, `eval/proxy_eval.py`,
  `linkers/division_linking.py`, the registry itself, `training/tunet_detector.py`. Shipped path (tracker) →
  kit rebuild. **Medium churn, medium value.** The registry rename is the cheap independent half — do it
  alone if the split feels too broad.

### 4. GRAB-BAG — `training/` is 5 kinds flat in one directory
Flat contents mix losses (`balanced_bce`, `softmax_focal_bce`), data
(`crops`, `targets`, `tunet_dataset`, `joint_dataset`), a model (`joint_model`), trainers
(`tunet_detector`, `edge_finetune`), and loop infra (`tracked_run`, `early_stop`). Propose internal
subpackages:
- `training/losses/` ← `balanced_bce.py`, `softmax_focal_bce.py`
- `training/data/` ← `crops.py`, `targets.py`, `tunet_dataset.py`, `joint_dataset.py`
- (optional) `training/loop/` ← `tracked_run.py`, `early_stop.py` — only 2 files; marginal, list but don't push.
- Leave `joint_model.py`, `tunet_detector.py`, `edge_finetune.py` at `training/` root (the model + the
  two trainer entrypoints).
- **Training-only, NO shipped path, no kernel touches training.** Churn is a handful of *intra-training*
  imports (`crops→targets`, `edge_finetune→softmax_focal_bce`, `tunet_detector→{balanced_bce,early_stop,
  tunet_dataset}`, `tracked_run→early_stop`) plus test-mirror paths. **Low risk, moderate value.**

### 5. Detector "infra" — mostly fine, one mild note
- `detectors/peaks.py` (`PeakExtractor`): the **shared readout** both the classical and learned detectors
  end on (response volume → suppressed centres). It is detection **vocabulary**, correctly scoped inside
  `detectors/` (the sub-package analogue of `affinity.py`). **Fine as-is.**
- `detectors/response_cache.py` (`ResponseStore`/`ResponseCache`): genuinely **persistence infra**, not a
  detector — but it is *detector-response* caching and sits at the base layer with no lower home to move to.
  Kernel + tracker import it → rename = kit rebuild for near-zero gain. **Leave; noted only.**
- `detectors/pipeline.py` (`BlendDetectorScorer`): a composite dual-seed **detector**; the filename
  `pipeline` is vaguer than its content. Rename to `blend_detector.py` would read truer, but kernel +
  tracker import it → kit rebuild. **Low value; skip unless touching the kit anyway.**

### 6. Minor / acceptable
- `eval/proxy.py` (`TestMovieProxy`, reusable scorer) vs `eval/proxy_eval.py` (`TrackerProxyEval`, CLI) —
  slight homophone, but library-vs-entrypoint is a real, documented distinction. Acceptable; a
  `proxy_sweep_cli.py` rename is cosmetic-only.
- `detectors/center_prior.py` (6 classes) and `postproc/gap_closer.py` (many classes, 460 lines) are dense
  but each is one **cohesive** concept (a recipe→confirm→veto→scorer chain; a gap-closing pass with its
  private helpers) and both pass the gate. **Do not split** — that is churn against a green gate.

### 7. Explicitly FINE — do not churn
- Detector strategy family (`detection` / `dog_detection` / `tunet` / `center_prior`) — clean variants.
- Edge **scorers** (`edge_scoring`, `blended_edge_scoring`) — coherent producers.
- `postproc/` stages (gap_closer, linefit_smoother, short_track_filter, topology_repair) — one pass each.
- `celltrack/affinity.py` at the root — exemplary shared-vocabulary placement; the model the linkers'
  `Linker` protocol should follow (finding #3).
- The layer contract itself — direction is clean, no celltrack-internal cycles, no god-modules/files.

---

## Ranked proposals (cheap high-value first)

| # | Change | Rationale | Shipped? | Risk (imports to edit) |
|---|--------|-----------|----------|------------------------|
| 1 | `edges/affinity_division.py` → `postproc/affinity_division_recovery.py` | It's a post-link division-recovery stage (consumes affinity), twin of `postproc/division_recovery.py`; de-scatters DIVISION | Yes (tracker → kit rebuild) | **1** line (`tracker.py:22`) |
| 2 | Rename `linkers/linkers.py` → `linkers/registry.py` | Kills the `linking.py`/`linkers.py` homophone; names the factory | Yes (tracker) | 4 (`tracker`, `proxy_eval`, `tunet_detector`, self) |
| 3 | `training/` → add `losses/` + `data/` subpackages | Splits a 5-kind grab-bag into cohesive homes | No (training-only) | ~6 intra-training imports + test mirrors |
| 4 | Split `linkers/linking.py` → `linker.py` (Protocol) + `nearest_neighbour_linking.py` (baseline) | One subject per file; lifts the `Linker` contract to sit like `affinity.py`; baseline matches `*_linking` scheme | Yes (tracker, eval) | 5 (`tracker`, `proxy`, `bracket`, `division_linking`, registry) |
| 5 | Rename `detectors/pipeline.py` → `blend_detector.py` | `pipeline` is vaguer than `BlendDetectorScorer`; do only when the kit is already being rebuilt | Yes (tracker + a kernel → kit rebuild) | 2 + 1 kernel |

Not ranked (leave as-is): `detectors/peaks.py`, `detectors/response_cache.py`, `eval/proxy*` naming,
`center_prior.py`/`gap_closer.py` multi-class density, `linkers/division_linking.py` placement (it's a
linker and belongs in `linkers/`).
