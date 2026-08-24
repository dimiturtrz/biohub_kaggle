# Architecture reorder — plan (P1–P4)

**2026-08-24 · source: sdlc-scaffold-b4 audit (peer) · status: FILED + PLANNED, moves ON HOLD**

Reference distribution: `D:/work/ai` (owner's hand-crafted, liked). Scheme = top level is ML-pipeline
stage; second axis inside `training/` is task × method; every leaf repeats a quartet
(`hyperparams/learning_method/model/train`); cross-cutting lifted to `training/shared/`. Cross-audited vs
`rsna_knee_kaggle` (kneemri).

## What celltrack already does well — DO NOT churn

- `losses/` — best loss layer of the three repos: eight modules named **by form, not use** (`info_nce`,
  `sigreg`, `softmax_focal_bce`, `balanced_bce`, `hard_negative_margin`). Keep; extend the principle outward.
- `detectors/ edges/ linkers/ postproc/` — real strategy families with swappable members (7 linkers,
  4 detectors). Discipline is genuinely there.
- `core/` (task-neutral) vs `celltrack/` (task) two-tier split. Good; worth enforcing with import-linter.

## Axis decision (peer recommendation, NOT owner-ratified — plan against, don't move)

**ML lifecycle owns the top level; the task-specific algorithm chain nests under `inference/`.**
Reasons: (1) two of three repos already do it — converging costs celltrack one move, not three;
(2) the lifecycle axis is stable across projects, an algorithm chain is not (classification has no chain) —
a scaffold can only encode the axis that survives the next project; (3) `detectors/edges/linkers/postproc`
are what `tracker.py` calls at runtime — they ARE inference by definition.

**The seam that makes it low-risk:** the NETWORK stays in `models/` (`temporal_unet_detector.py`,
`edge_transformer.py`); the RUNTIME STRATEGY WRAPPER moves under `inference/` (`detectors/tunet.py` =
the peak read-out over the net). Trained things do not relocate — only the read-outs get filed where they
belong. Target import contract: **`inference` imports `models`; `models` never imports `inference`.**

### Import-direction check (read-only, run 2026-08-24)

`grep` for `celltrack.(detectors|edges|linkers|postproc)` imports:
- **`models/` imports the chain: NONE** — the model layer is already clean below the seam. ✅
- **`training/` imports the chain: YES** — `training/joint_detector.py`, `training/tunet_detector.py`.

So the contract is NOT already true: the training arms pull in the runtime chain wrappers. **That
`training → chain` edge is the actual P1** — the real work is severing it (training should import the net
from `models/`, not the detector/linker wrapper), not just renaming folders.

## The four findings

| # | Finding | Target | Proving constraint | Bug-reduction | Blocker |
|---|---------|--------|--------------------|---------------|---------|
| **P1** `wsg5` | mixed top-level axis (chain vs lifecycle are siblings) | lifecycle top level; chain under `inference/`; sever `training→chain` | import-linter contract `inference→models`, `models⊥inference` GREEN; full test + tracker smoke | indirect (stops wrong-home drift) | axis go + shipped-code go |
| **P2** `sttw` | 6 root leftovers incl `tracker.py` (shipped pipeline, filed nowhere) | `inference/` home: `tracker.py` + chain + `kernel_runtime`/`multi_gpu_submission`/`operating_point`/`precision`/`affinity` | root has no loose logic modules; arch-gate god-module clean; kernel submit smoke | indirect (leftovers bucket is where god-modules/cycles form) | falls out of P1 + shipped-code go |
| **P3** `8b54` | 4 training arms, no shared leaf contract; `joint_config.py` 550 LOC | `training/<arm>/{config,method,model,cli}.py` over shared `tracked_run` driver | each arm's CLI still runs; `joint_config` split under file_max; test green | **HIGH** — dedup kills cross-arm drift; splits config god-file | live directional-PE arm runs `joint_*` — wait for it to complete |
| **P4** `uwov` | `joint_model.py` fuses backbone+heads; TTA in `edges/`; preprocessing 1 file | split `backbone.py`/heads; move `blended_/flip_view_edge_scoring` to `inference/`; `preprocessing/` pkg | backbone unit-testable in isolation; TTA off the model layer | modest (clearer boundaries/testability) | `joint_model.py` loaded by live arm |

**Order:** P3 first if greenlit (needs no axis decision, biggest LOC win, but wait for the live arm) →
P1+P2 together (axis then homes, shipped-code go) → P4 last.

## Does this reduce bug chance?

- **Fixes existing bugs: no.** Code ships at 0.900; moving modules changes no logic.
- **Reduces future bug RATE: yes, mostly via P3.** Four arms rolling their own config/CLI is the drift-bug
  factory — fix a flag bug in one, the other three stay wrong. A shared contract makes it single-source.
  P1/P2 convert a growing root leftovers bucket (where god-modules and cycles incubate) into an
  import-linter-enforced boundary, so the arch-gate catches erosion instead of review. P4 buys testability.
- **The act carries near-term risk.** P1/P2 move `tracker.py` + the kernel/submit path; P3/P4 touch the
  files the live experiment runs. Net: worth doing for maintainability and future-bug-rate, sequenced
  safely (tests + arch-gate + smoke as the oracle) — NOT a "fix bugs now" move, and not while the live arm
  holds the training files.
