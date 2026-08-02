# Session handoff — what this session established

*2026-08-02. Written as a document because `bd` is not installed on this machine, so none of it could be
filed as beads issues or memories. Fold into beads when a machine with `bd` picks this up.*

## Four things changed in what we believe

**1. We are at 0.859, and we thought we were pending.** `kaggle/survey.py submissions` shows all four
submissions complete and scored. `SUBMISSIONS.md` recorded the best one as "in flight" and the
`unetaug` one as "pending"; both had landed on 2026-07-31. Corrected in the table.

**2. Local fold-0 is a trustworthy proxy.** 0.877 local → 0.859 LB is a **−0.018** offset. Small and
consistent, which means the leak-inflation in the borrowed weights is real but modest, and — more
usefully — we can iterate locally without spending any of the 5 daily submission slots.

**3. The public frontier is one pipeline, forked.** The ~0.913 top notebook, pilkwang's two-seed blend
and the "3rd place no-hack" notebook share an identical 102-constant config block and `EXPERIMENT_TAG`.
Reading one reads all of them. Full recipe in
[`research/deep_dives/2026-08-02_public-notebook-frontier.md`](../../research/deep_dives/2026-08-02_public-notebook-frontier.md).

**4. Borrowing is settled, and it is not a rules problem.** `rules.txt` §2.6(b) makes external models
"acceptable unless specifically prohibited by the Host"; there is no such prohibition. All four pilkwang
packs are **CC0**, compatible with the MIT winner licence (§1.6). No from-scratch requirement exists —
§2.8's carve-out lets a winner *identify* rather than deliver third-party components procurable without
undue expense. A pretrained model is a dependency.

That reframes `80e`/`cdg`. The argument for training our own detector is **leverage, not integrity**: a
frozen borrowed detector cannot be improved, so every future gain has to come from the stages around it.

## The one number that stands out

**Our division Jaccard is exactly zero, and it is structural.** `MotionHungarianLinker` uses
`linear_sum_assignment` — strict one-to-one, so no node can ever have two children, so no fork can ever
exist. The metric's division term is worth **0.1** and we score none of it.

Two corroborations already in the repo:

- `2026-07-23_division-linking.md` measured precisely this: NN linker → division Jaccard **0.000**,
  "structural, not a tuning miss". A division-aware second pass reached **0.333** ≈ **+0.033** score.
- `ShortTrackFilter` carries a carve-out preserving components that contain a division. **That branch has
  never fired.**

`celltrack/division_linking.py` exists and is not in the shipping pipeline. Of the 0.054 gap to the
frontier, this looks like the largest identifiable single chunk, it is pure CPU graph work, and it needs
no data root.

## An unresolved contradiction in our own docs

`2026-07-23_operating-point-sweep.md` concludes: *"Do not chase the node-count bonus. Operate at the
estimate."* Measured on a jittery, low-precision detector, where trimming costs real recall.

The 2026-08-01 full-41 sweep on the *strong* detector runs the other way — monotone upward as nodes are
shed (0.8742 @ thr 0.3 → **0.8861** @ thr 0.95, 1.02 M → 855 k nodes) — and the frontier notebooks run
`DET_THRESHOLD=0.99` against our 0.5.

Both findings are correct in their own regime: with a precise detector, trimming farms the count bonus;
with a jittery one it does not. **Only a beads memory records the reconciliation, and the interpretation
document reads as settled.** This is the stale conclusion most likely to misdirect the next decision, and
it sits directly under the cheapest available experiment.

## Where the project's knowledge actually lives

`research/` held exactly **one** deep dive before today. The accumulated corpus is instead:

- **`.beads/issues.jsonl` — 21 `memory` records.** qc5, the graft, the motion A/B, calibration refuted,
  width-64 refuted *and its later retraction*. Dense, dated, cross-linked, and invisible outside `bd`.
- **`interpretations/tracking/` — 6 documents, all 2026-07-23.** The work continued nine more days.

The research practice drifted into the issue tracker. Promoting those memories into dated deep dives is
filed under Housekeeping in the roadmap.

## Environment notes for the next session

- **`bd` is not installed here.** Nothing this session is filed as issues; the roadmap carries it instead.
- **`bd dolt push` remains broken** — the remote is misconfigured as a `git+https` URL, not a dolt remote.
  Issue state survives via the git-tracked `.beads/issues.jsonl`.
- **This laptop has no data root, no `paths.yaml`, no `.venv`, no `uv`,** and a 6 GB RTX 3060 Laptop GPU.
  Anything touching the dataset or the detector needs the GPU box.
- **`kaggle/survey.py` needs `KAGGLE_TOKEN`** (a `KGAT_*` bearer token) in the environment. Env-only by
  design; no credential path is baked into the repo.
- The repo's own `kaggle/` directory still shadows the installed `kaggle` package — call the venv
  entrypoint directly, or run from outside the repo root.
