# The fork-and-tune line — its measured ceiling, and the ~0.006 private we left on it

**This is the line we were on.** We forked the public 0.953 notebook (`anvithpothula/biohub-x138`),
probed environment switches, and finished public 0.953 / private 0.918 at rank 562. Two other teams
posted their measurements of exactly the same line. Together they pin its ceiling and name the cheap CPU
switches inside it that we did not take.

Sources: topic **#744512** (Santanu Banerjee, 288th of 4,020 — 22 leaderboard probes), topic **#744498**
(Nikolce — reproducing the x138 V1284 head), topic **#744491** (zephyr — the public/private split).

| line | public | private | rank |
|---|---|---|---|
| Nikolce, best of the line | ~0.954 | **0.924** | **161** of 4,020 |
| 288th (x138 + two env switches) | 0.955 | **0.921** | 288 |
| **us** (x138 + cyto rescue) | 0.953 | **0.918** | **562** |
| 288th's own from-scratch detector + linker | 0.786 | 0.760 | — |

> "The whole public lineage lands at ~0.92 private, while the top 20 hold 0.94–0.98. That gap looks like
> better detection and linking, not better thresholds." — 288th

## 1. The two switches that were worth +0.002 private, and cost nothing

288th's full table (public → private), one environment switch per submission:

| submission | public | private |
|---|---|---|
| x138 + re-admit 0.90 + **detection threshold 0.955** | 0.955 | **0.921** |
| x138 + re-admit 0.90 + **DeepCenter division veto 0.15** | 0.955 | **0.921** |
| x138 + re-admit 0.90 | 0.955 | 0.919 |
| x138 + re-admit 0.85 | 0.954 | 0.919 |
| x138 + flow re-link tight gate 8 µm | 0.951 | 0.919 |
| x138, exact fork | 0.953 | 0.917 |
| public 0.940 + candidate edge threshold 0.25 | 0.938 | 0.913 |
| public 0.940, exact fork | 0.940 | 0.910 |
| public 0.940 + ILP division weight 0.6 | 0.926 | 0.898 |
| their own detector + linker | 0.786 | 0.760 |

- **Re-admit 0.965 → 0.90**: +0.002 public, +0.002 private. *"Weaker real peaks close to where a track
  stops pay off."* Reaching further (4 → 6 µm) lost; weaker still (0.85) lost.
- **Detection 0.965 → 0.955** and **DeepCenter division veto 0.25 → 0.15**: each tied on public and gave
  their best private. Their exact fork of x138 scored 0.953/0.917 — **the same as ours** — so the three
  switches together are the +0.004 private between their fork and their best, and Nikolce's head lifts the
  line ~0.003 further.
- A wider flow-relink tight gate (7 → 8 µm, beyond the ~7.6 µm cell spacing) lost 0.004 public, *entirely
  from the densest movie, where neighbours stole each other's links.*

**Our E66 arm moved the veto 0.20 → 0.15 and read flat public 0.946 — on a weaker base.** The direction was
right; we applied it to the wrong world and never re-priced it on the 0.953 base. Read against the veto
author's own count: over 50 movies this pipeline family found **17 true divisions vs 19 false and 56
missed ≈ 47 % precision against an ~11 % break-even** — the veto was simply too strict.

## 2. The ILP cannot fork — a plumbing fact, not a tuning result

> **"The ILP never divides. There were zero forks in all 36 raw ILP graphs.** tracksdata minimises cost,
> and at the preset `division_weight=1.2`, `appearance_weight=0`, a second daughter starting a new track is
> free while a division costs more than the extra edge can earn. The preset's comment says the value was
> raised to *encourage* divisions, which is the opposite of what it does. Every division in these
> submissions comes from a geometric repair stage."

Third independent confirmation (12th place: zero forks on all 199, all 352 divisions are geometric grafts;
14th place proves it for their weights). **Our flat division-cost sweeps were therefore reading a knob with
no consumer** — a plumbing artefact, not evidence about division cost.

Fixing it does not help either: weight 1.2 → 0.6 produced **473 new ILP forks and cost 0.013 public**.
*"They displaced correct continuation links, and false forks were far more expensive on the hidden set than
on sparse local GT."* Division headroom over 49 local GT divisions: parent detected 49, both daughters 36,
inside the repair stage's geometry 25, **recovered 10**. Loosening the geometric gates added **3 true forks
for 55 false**.

## 3. The metric notes (independently derived, and they match 12th place)

- An edge counts only if its source matched an annotated node **with a successor**, or its target one
  **with a predecessor**. *"Edges between unannotated cells are free, however wrong."*
- **Link iff P(correct) > J/(1+J) ≈ 0.48.** A wrong link is worse than no link.
- **Adding forks of precision p raises division J only if p > TP/(2·TP + FP + FN)** — ~11 % at 10 TP /
  29 FP / 39 FN — *"but the hidden set punished extra forks far more than that suggests."* (12th place's
  version of the same break-even: J/(1+J) = 32 % public, ~25 % private.)
- The node-count term `J × (1 − 0.1 × ratio)` **has no ceiling**, but shedding short tracks for it (min
  track length 6 → 10) **lost 0.002 — those tracks still carry scoring edges.** Independent confirmation of
  our prune closure, and of 12th/18th place's node-deletion trap.
- Edge errors on the public pipeline's own four validator movies: **95.6 % of GT edges recovered, 2.4 %
  fragmented (both ends detected, no link), 2.0 % lost to detection, essentially zero wrong
  associations.** *"The links it makes are right; the ones it misses are the problem."* This is E54's
  wrong-association = 0 and E43/E44's fragmentation-is-selection, measured by someone else.
- **Public differences ≤ 0.002 did not predict private ones, in either direction.** With ~29 % of the test
  set on the public board, a single 0.001–0.002 is not evidence. Two of their pairs invert: the candidate-
  threshold probe lost 0.002 public and won 0.003 private.
- **Local CV on train movies is in-sample for the public models** (trained on all 199), *"so it cannot rank
  changes that shift trust between the learned links and the heuristics."* Motion re-link OFF gained
  **+0.020 adjusted J locally** and lost 0.001 public — *"learned links look better than they are on movies
  the models were trained on."*

## 4. The V1284 coordinate head, reproduced — and why its residual does not rank it

The 0.953 notebook hard-failed without a private dataset holding `v1284_head.pt`. Nikolce reproduced it
from material the notebook itself publishes.

**What it is:** a **7.3k-parameter** coordinate refiner, `Linear(224,32) → SiLU → Linear(32,3)`, output
bounded to **2 µm**. The 224 features are the UNet's 32-channel map sampled at a detection centre and its
**6 face neighbours**, encoded as `[centre] + [neighbour − centre]`. It nudges detection centres closer to
truth *before* association.

**Why reproducible:** the consuming module is written inline by the notebook and documents three modes —
`candidate`, `zero`, **`capture`**. Capture mode dumps exactly the `(coords, features)` pairs needed. Train
= run the pipeline in capture mode over training movies, match each detection to its nearest GT node, fit.

**Result:** their head (40 movies, 17,975 detection–GT pairs) scored **0.954 public vs the author's
0.953**, on an otherwise byte-identical pipeline — the two runs differ by one 34 KB file.

**The negatives, which are the useful part:**

| head, held-out residual | public |
|---|---|
| −10.5 % | 0.953 |
| **−22.7 %** | **0.954** |
| −26.0 % | 0.952 |
| −28.7 % (all 199 movies) | 0.951 |

- **More data made it worse on the LB.** There is an interior optimum: *"the downstream gates (relink
  radii, safe-division distances, gap-close steps) are tuned against the original centre-error
  distribution; a large calibration change makes them mis-fire."*
- **The held-out residual does not rank heads.** Head-to-head spread is ±0.002 against a 0.001 LB quantum;
  two independent 40-movie draws gave 0.954 and 0.952, *"so a single head's score is mostly noise."*
- Re-enabling the runtime `PPSWEEP` (x138 ships it off) changed nothing — best candidate +0.0005 proxy,
  under the notebook's own 0.001 margin, so it selected base and produced a byte-identical submission.

**Unit gotcha:** captured coords index the detector's **feature** grid = native ÷ (1,4,4); GT `geff` coords
are native. Compared raw, the nearest GT node sits **346 µm** away; with the downsample applied, **1.62
µm**. *"Training on the raw version gives a confident, useless head."*

> "This was a fork-and-tune line and it had a ceiling: ~0.954 public, **0.924 private, 161/4020**. The
> winners were far beyond it, so treat the above as a note on one mechanism rather than a solution
> writeup."

## 5. 288th's from-scratch attempt, and the six silent bugs their harness found

Their own detector (temporal 3D UNet, centre heatmap on a 4× XY-pooled isotropic 1.625 µm grid, CenterNet
focal loss with **ignore regions around unannotated nuclei**, parabolic sub-voxel refinement) plus a
Hungarian linker in µm (null column per target, squared cost, velocity extrapolation at 0.3 = the measured
autocorrelation of GT displacements, physical NMS at 6 µm) reached only **0.786 public / 0.760 private**.

Their decomposition: with **perfect** detection of the annotated cells their linker reached **0.856**
adjusted edge J; with the public detector **0.825**; with theirs **0.673**. *"The bottleneck was detector
precision, and fixing it needed more GPU than our shared 30 h/week quota allowed."*

Their local official-metric harness found **six silent bugs, each worth more than any hyper-parameter**:

1. a loss with **no negatives** (`<` vs `<=` on a flat background);
2. centroid sub-voxel refinement **worse than none**;
3. a linker forced to make `min(n, m)` matches — **116 of 143 missed edges had been stolen by a
   distractor**;
4. checkpoint selection on recall instead of recall², which **picked epoch 3 of 22**;
5. NMS that **never ran in production**;
6. a **unit double-scaling in the harness itself** — which produced two wrong recommendations before they
   caught it. *"Synthetic tests passed because they were self-consistent in the wrong units."*

(Our own record carries #4 almost verbatim — the frozen-detector checkpoint that ranked the donor's metric —
and #6 is the family our "assert the flag LANDED" rule exists for.)

## 6. Their "what we'd do differently"

- **Track the public ceiling from day one** (`kaggle kernels list --competition … --sort-by
  scoreDescending`) **and re-base the moment it moves.** *"The biggest gain was re-basing onto the public
  ceiling. Forking x138 was worth +0.007 private, more than two weeks of knob probing on the older 0.940
  notebook combined."*
- **Don't promote on +0.001 public.** Keep mechanism-backed ties in the final stack, and keep one robust
  pick.
- **Spend GPU on models, not post-processing.**
- Automation: their daily probe loop ran on a schedule via Claude Code. *"It worked when the machine was
  on. Four of the last six days it wasn't, which cost 21 submission slots, more than any knob was worth."*
  For code-competition submissions the CLI success text is `N submissions remaining today.`, and *"did it
  submit?"* is answered by `kaggle competitions submissions`, not a log.

Code, notes and their full public/private table: <https://github.com/SanTanBan/biohub-cell-tracking>

## 7. What this line's ceiling means for our post-mortem

- The fork-and-tune line tops out near **0.924 private / rank 161**. We finished 0.918 / 562. **~0.006
  private was available inside the line we were already on**, from three CPU-only switches (re-admit
  0.965 → 0.90, veto 0.25 → 0.15, detection 0.965 → 0.955) and one 7.3k-parameter head we could have
  trained in capture mode without a GPU.
- Even the top of the line is **0.022 short of the bronze cut and 0.053 short of the winner**. The
  0.94+ teams ([12th](2026-09-30_corwin_12th_place.md), [18th](2026-09-30_ymg_aq_18th_place.md)) got there
  by adding *downstream models* to the same base — not by tuning it.
- Three of our standing conclusions are independently reconfirmed here (wrong-association ≈ 0, prune axis
  loses real scoring edges, the local proxy is in-sample and cannot rank trust-shifting changes) and one is
  overturned: **the division-cost knob had no consumer.**
