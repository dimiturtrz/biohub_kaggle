# Division recovery on the edge-bonus champion — the 0.1 term is not reachable net-positive

**Task:** biohub_kaggle-dga. Recover cell divisions to claim the metric's `0.1 * division_jaccard` term,
which the shipped 1-to-1 linker scores 0 on. Proxy: the 4 real test movies (`scratchpad/test_movies_edge.py`,
gitignored). Champion = dual-seed blend nodes + `MotionHungarianLinker(tight=6, loose=10, affinity=<edge
transformer>, bonus=10)` + `ShortTrackFilter(6)` + `LinefitSmoother(0.8)`.

## Champion baseline reproduced

`score=0.8989  adj=0.8989  div_jac=0.0000`. All divisions live in `6bba_05db0fb1`, which the proxy scores
`adj=0.8272`, `div(tp/fp/fn)=0/0/3`. The truth holds **3** annotated divisions in that movie (not 4).

## The three divisions, dissected (`scratchpad/div_diag.py`)

| truth divider | @t | both daughters detected? | 2nd daughter status | edge-P(parent→2nd) | parent→2nd | sister dist |
|---|---|---|---|---|---|---|
| 232 | 24 | yes | **orphan**, parent's **rank-1** target | 0.124 | 6.08 um | **13.5 um** |
| 771 | 62 | yes | already linked to another parent (in_deg=1) | 0.0246 (rank 2, beaten by a non-daughter) | 8.29 um | 7.45 um |
| 624 | 52 | **no** — one daughter never detected | — | — | — | — |

- **624** is detection-limited: the blend never produces the second daughter. Unreachable by any linker.
- **771** is structurally out of reach: its second daughter is already someone's child, and the edge head
  ranks it *third* (below a non-daughter), so neither geometry (orphan-only) nor the edge head identify it.
- **232** is the only reachable one. Its second daughter is an orphan and the parent's rank-1 edge target,
  but the two daughters sit **13.5 um apart** (they separate onto opposite sides of the mother) — far past
  any precision-preserving sister gate, and the frontier `DivisionRecovery` misses it purely because its
  parent gate (4.7 um) is tighter than the 6.08 um the orphan sits at.

**Detection-limited count: 2 of 3 divisions have both daughters detected.**

## The ceiling exists — and is only reachable with an oracle

Injecting *only* the 232 fork through the full chain (short + smooth): `6bba div 0/0/3 → 1/0/2`, and
`6bba adj 0.8272 → 0.8280` (the recovered edge helps the edge term too). That single fork would give
`div_jac = 1/3 ≈ 0.333` → **pooled ≈ 0.932** (+0.033). The metric credits it cleanly; the wall is
placement precision, not the scorer.

## Why no gate reaches it — the edge head is not discriminative on dense tissue

Built `celltrack/affinity_division.py` (`AffinityDivisionRecovery`): the literal "a division is a parent
whose two top edge-targets are both children" reading — the kept child must be the head's top target, the
second daughter must be its **runner-up** and an orphan clearing a probability floor and parent/sister gates.

Every configuration that admits 232 floods `6bba_05db0fb1`:

| config | 6bba div (tp/fp/fn) | 6bba adj | pooled |
|---|---|---|---|
| champion | 0/0/3 | 0.8272 | **0.8989** |
| geometry DivisionRecovery, parent gate 7.5 | 0/8/3 | 0.8267 | 0.8941 |
| affinity runner-up, floor 0.12, kept≥0.6, sister≤14 (capped) | 0/63/3 | 0.8206 | 0.8870 |
| affinity runner-up, same, **uncapped** | **1**/807/2 | 0.8100 | 0.8801 |

Uncapped, the true fork *is* placed (TP=1) — but the gate that admits it admits **807 false forks** on the
dense movie. `div_jac` collapses to 0.0009 and `adj` craters. The edge transformer's softmax-over-sources
normalisation inflates the second-choice probability wherever detections are missing, and orphans are
everywhere on dense tissue, so 232's P=0.124 is not distinctive: >1200 parents have an orphan target with
P≥0.25. The global cap then evicts 232 (bottom of the P ranking) below the false crowd — which is why every
*capped* run scores exactly `div_jac=0.0000`.

## Verdict — KILL for shipping, keep the stage opt-in

The `0.1` division term is **not reachable net-positive** on this test set:
- 1/3 detection-limited (624), 1/3 structurally unreachable (771), 1/3 (232) recoverable in isolation but
  indistinguishable from ~800 dense-tissue false forks by any available feature (edge-P, orphan status,
  parent/sister distance).
- Every real configuration scores `div_jac=0.0000` (capped) or ~0.001 (uncapped) and pooled **0.880–0.894,
  below the 0.8989 champion** — net negative, the same after-`ShortTrackFilter` trap the prior geometry
  finding hit (`divisions-refuted-end-to-end`).

`AffinityDivisionRecovery` is a working, tested mechanism (it *does* recover 232) — kept as an opt-in
`GraphStage`, not wired into the champion. Per the project rule that a physically-valid method regressing on
a finite test means the test lacks that axis: the recall ceiling here is the detector (2nd daughters not
detected) and the edge head's dense-tissue precision, not the recovery logic.

## Files

- `celltrack/affinity_division.py` — the edge-head runner-up division stage (new).
- `tests/unit/celltrack/affinity_division.py` — 8 gate-equivalence unit tests (new).
- `scratchpad/test_movies_edge.py`, `scratchpad/div_diag.py` — the proxy harness + per-division diagnostic
  (gitignored, local to the worktree).
