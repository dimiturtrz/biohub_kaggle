# Champion provenance & reproducibility — banked 0.924 LB

2026-08-26 · consolidation day (understand the wall-break, fortify against loss)

## What the champion IS

| field | value |
|---|---|
| **LB score** | **0.924** (public) |
| **Kaggle submission id** | 55779061 |
| **Kernel** | `dimiturnt/celltrack-ilp-faithful-0923` (notebook `biohub-0-927-lb.ipynb`) |
| **Datasets** | `pilkwang/biohub-tracking-support-pack-50ep-v1`, `.../biohub-deepcenter-unet3d-center-prior-v1`, `.../biohub-temporal-unet3d-seed314159-v1` |
| **Competition** | `biohub-cell-tracking-during-development` |
| **Linker** | **tracksdata `ILPSolver`** (global min-cost, whole-track) — NOT celltrack `LinkerConfig(name="ilp")` (motile/SCIP) |
| **Artifacts** | dual-seed fusion (thr 0.96875) + DeepCenter veto + divsub, fed to the global ILP |

## Why it wins (the wall-break, 0.902 → 0.924, +0.022)

Clean single-variable LB A/B, same three artifacts both arms:
- per-frame greedy Hungarian linker → **0.902** (subs 55774589 / 55774416)
- global tracksdata ILP linker → **0.924** (sub 55779061)

The confusor bottleneck was never (only) a representation problem — it was the **per-frame greedy
assignment STRUCTURE**. A global min-cost solver over the whole track defeats the confusor by coupling
frames temporally; a greedy per-frame Hungarian commits locally-optimal wrong edges it can never revise.
Full mechanism: `2026-08-26_global_ilp_linker_breaks_wall_0924.md`. This vindicates the poe-refuted
"flow global coupling is the lever" line and resolves the `name="ilp"` memory contradiction (that memory
was WRONG; the tracksdata ILPSolver is the winning path).

## The loss risk, and the fix applied today

**Risk:** the champion's actual code was ONLY in the gitignored `.ipynb` (repo policy: no notebooks in
git, `.gitignore:66-67`). Kaggle servers back it up (confirmed via `kaggle kernels list`, last run
2026-08-25 20:41), so not at imminent total loss — but it was NOT reproducible or reviewable from git.

**Fix (this commit):**
1. `kernel-metadata.json` committed — the git-tracked repro pointer (slug + 3 datasets + competition).
2. `champion_submission.py` committed — the notebook's code cells extracted VERBATIM (all 10 parse
   clean, `ast.parse` per cell). Reviewable/diffable in git; satisfies the no-notebooks policy while
   keeping a git-resident copy of the tracksdata assembly glue (which does not import locally).
3. `extract_champion.py` committed — reproducible re-extraction after any kernel edit.

**Recover the runnable notebook** (if the Kaggle copy is ever needed locally):
```
uv run kaggle kernels pull dimiturnt/celltrack-ilp-faithful-0923 \
    -p kaggle/kernels/celltrack-ilp-faithful-0923
uv run python kaggle/kernels/celltrack-ilp-faithful-0923/extract_champion.py   # refresh the .py
```

## Honest standing

0.924 is a strong result, NOT first place (winning cluster 0.945–0.953, #1 0.962). The genuine
linker/recovery/floor-raiser stack caps ~0.926–0.93; the remaining ~0.02 is a dense-regime
REPRESENTATION gap with no public code. Goal remains open. See
`2026-08-26_win_path_ranked_levers.md` for the forward levers.
