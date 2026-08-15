# biohub-association-ranker (Kaggle dataset)

Staging tree for the Kaggle dataset **`dimiturnt/biohub-association-ranker`** (CC0-1.0). Uploaded as-is;
Kaggle preserves the `model/` subdir.

## What this is

A **local association tie-breaker** — a small MLP (`hidden=64`, `dropout=0.05`, 22 input features) that
re-scores candidate edges the linker already proposes. Applied as
`evidence = 0.85 * ranker_prob + 0.15 * primary_edge_prob`; author's own note: *"Use only as a constrained
local association tie-breaker, not as a global edge veto."* Reported val `best_score = 0.9777` (group-level,
best epoch 7). The 22 features are all computable from linker state — nothing needs the image. Feature list
and applied mechanism documented in [`research/deep_dives/2026-08-11_public-frontier-recheck.md`](../../../research/deep_dives/2026-08-11_public-frontier-recheck.md) (§F3).

## Provenance — third-party, NOT ours

These weights are **not trained by any in-repo code**. They originate from the CC0 Kaggle dataset
`pilkwang/biohub-local-association-ranker-unet300-v1` (byte-identical manifest: `best_epoch=7`,
`best_score=0.9777398513390436`, 126705 candidate groups). We re-host them under our own account as
`dimiturnt/biohub-association-ranker` (CC0 permits this). There is no producing command in this repo — the
`.pt` is a **sole local copy** of an external artifact, so it is committed here (with a
`!kaggle/datasets/**/*.pt` negation in `.gitignore`) rather than left ignored. 20 KB.

## Publish status

Published — `kaggle datasets status dimiturnt/biohub-association-ranker` → `ready` (checked 2026-08-15).

## Re-upload

```bash
# from repo root; the venv kaggle entrypoint (the repo's kaggle/ dir shadows the installed package)
.venv/Scripts/kaggle.exe datasets version -p kaggle/datasets/biohub-association-ranker -m "<message>"
# first-time create instead of version:
# .venv/Scripts/kaggle.exe datasets create -p kaggle/datasets/biohub-association-ranker
```

## Consumers in-repo

- [`celltrack/edges/association_ranker.py`](../../../celltrack/edges/association_ranker.py) — loads + applies.
- [`celltrack/kernel_runtime.py`](../../../celltrack/kernel_runtime.py) — kernel-side wiring.
- [`celltrack/analysis/ranker_distribution.py`](../../../celltrack/analysis/ranker_distribution.py) — the
  `mean`/`std` tensors inside the `.pt` ARE the per-feature training-normalisation stats.
