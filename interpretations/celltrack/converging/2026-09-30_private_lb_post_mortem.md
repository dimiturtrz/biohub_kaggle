# Post-mortem — private leaderboard, Biohub Cell Tracking During Development

**Final: public 0.953 · private 0.918 · rank 562 of ~4000 teams.** Competition closed 2026-09-29.

This is the sense-making file for *our own* result. The four solutions it compares against are written up
separately and faithfully in `research/solutions/`:

| write-up | private | why it matters to us |
|---|---|---|
| [3rd — yu4u](../../../research/solutions/2026-09-30_yu4u_3rd_place.md) | 0.967 | own detectors, sparse-label loss, joint LP: the frontier's ablation ladder |
| [12th — Corwin](../../../research/solutions/2026-09-30_corwin_12th_place.md) | **0.946** | **tuned the public models, added repair — no new detector.** Exact metric replay |
| [18th — ymg_aq](../../../research/solutions/2026-09-30_ymg_aq_18th_place.md) | **0.942** | **~9 MB of tabular heads on Harmonic Fusion v30 — no new detector** |
| [the fork-and-tune line](../../../research/solutions/2026-09-30_the_fork_and_tune_line.md) | 0.921–0.924 | 288th + Nikolce measured *our own line*: its ceiling and the switches we skipped |

Verdict tags follow the [conclusion tree](2026-08-27_conclusion_tree.md): **BANK** · **REFUTED** ·
**HARNESS** · **AUDIT-FLAT** · **OPEN**.

> **Revision note (2026-09-30, after reading the full discussion sweep).** The first version of this
> file was written from two writeups and got three verdicts wrong. Six more solutions posted the same
> day contradict them. The corrections are inline below and marked **[REVISED]**.

---

## 1. What the private board did

The public board's top was a single open notebook (`anvithpothula`/`amanatar`/`crystalbaby`, 0.953) and
several hundred forks of it. On the private embryo the entire fork field collapsed into one band:

| private rank | team | private |
|---|---|---|
| 1 | Sergio Alvarez | 0.977 |
| 2 | Soheil Ayati | 0.970 |
| 3 | yu4u | **0.967** |
| 4 | z7777 | 0.962 |
| 5 | Tang | 0.954 |
| 10 | Kaggle Agent | 0.948 |
| 12 | Corwin | 0.946 |
| 18 | ymg_aq | 0.942 |
| 20 | Yurinchi | 0.940 |
| 100 | Filip Strzałka | 0.928 |
| 161 | Nikolce | 0.924 |
| 200 | maco-macoo | 0.923 |
| 288 | Santanu Banerjee | 0.921 |
| 350 | Congcong Bi | 0.920 |
| 500 | mikelou1 | 0.918 |
| **562** | **Dimitar Terziev (us)** | **0.918** |
| 600 | Kamal Kadakara | 0.918 |

Ranks ~200 to ~600 span **0.923 → 0.918**: four hundred places inside five thousandths. Another forker
(`zephyr`, topic #744491) says it in one line:

> At the 21% test-set, I was only +0.001 ahead of the highest public notebook score of 0.953.
> Eventually, we may all finalized at 0.920.

**Our own fleet, public → private:**

| submission | arm | public | private |
|---|---|---|---|
| 56673516 | cyto rescue on the 0.953 base | 0.953 | **0.918** |
| 56673514 | plain public 0.953 | 0.953 | 0.917 |
| 56668584 | cyto-all, 43 edges | 0.945 | 0.917 |
| 56462929 | E66 veto OFF | 0.946 | 0.917 |
| 56462928 | E66 veto 0.20→0.15 | 0.946 | 0.917 |
| 56668912 | cyto-dense, 13 edges | 0.947 | 0.916 |
| 56679148 | horizon + cyto (the slot we'd have picked) | 0.906 | 0.892 |

**The headline finding is the flatness.** Seven submissions spanning 0.945–0.953 public land in
0.916–0.918 private — a 0.002 spread, i.e. our own declared noise floor. Every knob we turned in the
last two weeks was invisible on the embryo that counted. The ~0.05 gap to the frontier lives in none of
them.

Two corollaries worth stating out loud:

- **Autoselection outperformed our judgment.** Letting Kaggle keep the two best *public* scorers picked
  our best private arm (0.918). The composed arm we had staged for the final slot scored **0.892** —
  our worst by a wide margin, and the only one outside the band.
- **[E67's dose-monotonicity did not reproduce — HARNESS.]** Public said the cyto rescue was
  *dose-dependently harmful* (13 edges → 0.947, 43 edges → 0.945). Private says the opposite by the
  same margin (13 edges → 0.916, 43 edges → 0.917). Both readings are ±0.001. The correct verdict is
  **tie, on a knob too small to measure on either embryo** — and our "monotone in dose, so every
  proposed edge is net-negative" wording was over-read from two points inside the noise floor. The
  *conclusion* (edge proposal is not a lever) survives; the *mechanism claim* does not.

## 1b. [REVISED] Two existence proofs against our central thesis

Our standing model of the gap was *"TRUE bottleneck = dense-regime model gap"* — i.e. reaching 0.94+
required a detector we could not train. **Two teams reached 0.942 and 0.946 private starting from the
same public notebooks we forked, with no new detector.**

- **12th (gold, 0.946).** *Tuned* `pilkwang`'s public models — four knobs — replicated `prvsiyan`'s
  public chain, then added consolidation, a mitosis specialist, a nine-rule repair engine, a learned
  recentring CNN and an identity-link cutter. Their own accounting: the public-chain replica cost
  −0.016 public→private; knob tuning and consolidation widened the drop to 0.031; **the stages after
  their base champion read +0.006 public and +0.013 private.**
- **18th (0.942).** Harmonic Fusion v30 (public 0.9116) + **~9 MB of added models**: LightGBM edge
  rescorer, per-axis coordinate regressors, a 60,449-param division CNN trained on **external**
  zebrafish data, CatBoost daughter-pair models, TabPFN 3.5 stacking. CV 0.9116 → 0.9528.
- **Our own line's ceiling** ([fork-and-tune file](../../../research/solutions/2026-09-30_the_fork_and_tune_line.md))
  is ~0.954 public / **0.924 private / rank 161**. We landed 0.918 / 562. **~0.006 private sat inside the
  line we were already on**, reachable with CPU-only switches: re-admit 0.965 → 0.90 (+0.002 private),
  DeepCenter division veto 0.25 → 0.15, detection 0.965 → 0.955 (each +0.002 private, tied on public), and
  a 7.3k-param coordinate head trainable from the notebook's own `capture` mode.

**The honest reading:** the detector was not the binding constraint at our score. ~0.024–0.028 of private
score was available *downstream* of it, and two independent teams collected it. Our thesis was not wrong
about the detector being the frontier's advantage — it was wrong about it being *ours*.

## 2. Why the board shook: the two test embryos differ in density

yu4u *probed* the split with their own detector — counts in four evenly spaced frames per video,
percentiles across videos:

| prefix | videos | split | p10 | median | p90 |
|---|---|---|---|---|---|
| `44b6` | 71 | train | 134 | 355 | 627 |
| `6bba` | 128 | train | 54 | 114 | 445 |
| `fdad` | 60 | **public test** | [40, 60) | **[160, 260)** | [260, 420) |
| `ea36` | 106 | **private test** | [40, 60) | **[60, 100)** | [160, 260) |

The public test embryo is roughly **2–3× denser at the median** than the private one, and the private
board is **79%** of the data. The public field tuned on the dense embryo and graded on the sparse one.

**This is a direct hit on our own standing model of the task.** Our bottleneck memory read *"TRUE
bottleneck = dense-regime model gap"*, and an entire axis family — finer decode, detection-merge,
confusor separation, crowding — was justified by it. That model was inferred from public-LB behaviour,
which means **we were aiming at the half of the test set worth 21%**. Not wrong about the dense embryo;
wrong about which embryo decided the competition. The plain denominator question — *which videos is
this score computed over?* — was never asked of the split itself. Our own
`unsourced-split-assumptions` note had already flagged that the 29% public fraction had no primary
source; the fix was to go **probe** the split, which is exactly what 3rd place did and we did not.

**[REVISED] Density is not cleanly separable from embryo identity.** 12th place reweighted their
in-sample videos toward the private density and predicted private **above** public by ~0.02; it came out
0.024 below. Their reweighting puts 81% of its weight on `6bba` and 75% on copy videos, so *"density,
embryo and copies are confounded: reweighting an in-sample world does not correct an in-sample bias."* A
density-matched proxy built from training videos would not have saved us either.

## 2b. [REVISED] The metric, in exact currency — and why our 1.5% bar pointed the wrong way

12th place restored the host scorer and **replay it to 1e-12**. Score = node-adjusted edge Jaccard +
0.1 × division Jaccard, and with no term clipped the deficit splits exactly:

```
1.1 − score = FN_e/E_tot + FP_e/E_tot + node term + 0.1·FN_d/D_tot + 0.1·FP_d/D_tot
```

| quantity | value |
|---|---|
| one missed or false **edge** | **7.47e-6** (1 / 133,795) |
| one missed **division** | **5.5e-4 — the price of 74 edges** (14th place independently: +0.000585) |
| their base champion's deficit, in-sample | edges 0.0759 · **divisions 0.0768 (51%)** · node term −0.0022 |
| break-even precision for added divisions | **32% public, ~25% private** (J/(1+J)); or p > TP/(2TP+FP+FN) ≈ **11%** |
| break-even for adding an **edge** | P(correct) > J/(1+J) ≈ **0.48** |

**Our "1.5% bar ≈ 35 edges" was edge-currency only.** In division currency 1.5% ≈ **27 divisions**, and
divisions were the *larger half* of the deficit. Every edge-hunting campaign we ran (E37, E47, E52–E59,
E67) was chasing the smaller half at 1/74 the unit price, and we never once computed the break-even
precision an added division needs. The three mechanisms behind **83% of the edge deficit** are exactly
what E57 diagnosed: a neighbour within 7 µm takes the link (0.0295), a GT endpoint has no node within
7 µm (0.0211), link swaps (0.0122).

## 2c. [REVISED] The division-cost knob had no consumer — our sweeps were plumbing, not evidence

Three teams found this independently:

> "With division cost 1.0 against appearance 0.0, the public ILP never forms a fork on the 199 videos:
> all 352 divisions of this stage on the visible videos are geometric grafts." — 12th (14th confirms for
> their weights)

> "The ILP never divides. There were zero forks in all 36 raw ILP graphs. … The preset's comment says the
> value was raised to *encourage* divisions, which is the opposite of what it does." — 288th

`tracksdata` minimises cost; with free track appearance, a second daughter starting its own track is free
while a fork costs more than the extra edge earns. **Our flat `division_cost` sweeps were reading a knob
nothing consumed** — a `celltrack-e6-packet-grouping-is-inert` repeat, and the exact failure mode our own
"assert the flag LANDED" rule exists for. Downgrade `celltrack-champion-swept-division-on-real-LB-flat`
and the constant-sweep kills from REFUTED to **HARNESS**.

Fixing the weight does not rescue it: 288th's 1.2 → 0.6 produced **473 new ILP forks for −0.013 public**.
The division lever was never the cost — it was a **witness**. 12th: their exact joint MILP found the right
fork for 22 of 96 missed divisions against 67 with oracle cues — *"the limit was the division witness, not
the solver."*

## 3. Grid — our verdicts against the frontier's evidence

yu4u's ablation (same OOF detections, fixed folds and scorer) plus the two fork-line existence proofs are
the cleanest external ruler we will get for our own kill list.

| # | our verdict | still correct? | the frontier's number |
|---|---|---|---|
| E52 | component prune priced out; champion's min component already 6 | **BANK — confirmed three times over** | A5 = **+0.0017**, their constant is also < 6 nodes. **[REVISED]** And the *reverse* is a trap: 18th deleting 10% of nodes raised CV but cost **−0.015 public / ~−0.06 private**; 12th's same filter erased **279 real cells and 9 of the 28 absent daughters of GT divisions**; 288th lost 0.002 raising min length 6 → 10 *"because those tracks still carry scoring edges."* New sub-lesson we did not have: **run repair BEFORE the filter** — 12th's ran after, too late. |
| E55 | gap repair cannot score: all GT edges are dt=1, a t−1→t+1 bridge matches no GT edge | **BANK on the edge — REFUTED on the fix** | A4 = **+0.0051**. They close gaps by **inserting the interpolated node**, so both resulting edges *are* dt=1. Our premise was right and we drew the wrong conclusion from it. `gaploose`'s 925 edges for zero was 925 of the wrong object. |
| E57/E59 | localization error is real (4.08 µm on broken edges vs 1.46 µm) but the snap oracle prices it at ~2 edges | **[REVISED] HARNESS — and this is the single most-built lever in the field** | A6+A7 = **+0.0065** (global affine motion field). **Three more teams built a learned coordinate corrector**: 18th's per-axis LightGBM (86 features, Huber, applied shift = 0.5 × prediction capped at 3 µm) = **+0.020 CV**; 12th's recentring 3D CNN (13×41×41 native crop, Laplace scale, LightGBM stacker, 930k samples, ≤2 µm/axis, never within 4 µm of another node, topology frozen) = **+0.003 private**; and the **V1284 head was already inside the 0.953 notebook we forked** — 7.3k params, `Linear(224,32)→SiLU→Linear(32,3)`, ≤2 µm, features = the UNet's 32-ch map at the centre plus its 6 face neighbours as `[centre]+[nbr−centre]`. Our snap oracle priced *per-edge* correction and could not see any of these. |
| E58/E37/E53/E47 | CPU post-processing axis closed on the champion | **[REVISED] REFUTED on the private embryo** | A4–A7 = +0.0131 on a 0.964 base, *and* 12th's post-champion repair stages read **+0.006 public / +0.013 private** — double on the graded embryo. Their repair is nine fused rules + an identity-link cutter + learned recentring, all CPU/light-GPU. The axis is not closed; **our champion-relative pricing of it was done on the wrong embryo.** |
| E54b | motion continuity oracle-refuted; ORACLE velocity recovers 10/61 broken edges | **REFUTED — self-selected denominator** | A1 dense 3D flow = **+0.0117**. We measured motion on *the 61 edges our own champion broke*, a set already filtered by a pipeline whose failures are elsewhere. Their flow is not a cue added to a ranker, it is a **field that redefines the candidate set** before matching. |
| E54/E57 | wrong-association ≈ 0; fragmentation is selection, not candidates | **BANK — independently reproduced** | 288th on the public pipeline's validator movies: **95.6% of GT edges recovered, 2.4% fragmented (both ends detected, no link), 2.0% lost to detection, essentially zero wrong associations.** Same numbers, someone else's harness. |
| E61 | division recall unreachable from post-processing; **"remaining lever = re-parenting inside the LINKER"** | **BANK in scope — and it named the right thing** | A3 = **+0.0625**, division Jaccard 0 → 0.535. We then priced its three legs at zero runs and **deleted two-pass tracklet ILP from the conclusion tree on 09-22** as "the donor's lever, doesn't transplant". |
| division family (`hq93`, global division-ILP, constant sweeps) | post-processing and constant-sweep division work is dead; `div_jac = 0.0000` in every arm | **[REVISED] HARNESS on the sweeps, BANK on post-processing, and it is the whole gap** | See §2b–2c: the cost knob had **no consumer** (zero forks at the preset), divisions are **51% of the deficit**, one division = **74 edges**, and we never computed the ~25–32% break-even precision. yu4u's 0.540 division J is **+0.054 of total score** — 3.5× our bar, from one component. `div_jac = 0.0000` meant *"we have no division model"*, not only *"the proxy is blind"*. |
| HOCT / Trackastra as association heads | HOCT refuted as an association head (detection-side) | **[REVISED] BANK on the head — AUDIT-FLAT as a feature** | 18th feeds **HOCT `ctc_v0` and `general_v1` and Trackastra `ctc` as columns into a LightGBM edge rescorer** (341 features, ~1.4M rows) worth **+0.021 CV**. We built HOCT, refuted it as a head, and never tried it as a feature into a ranker. |
| ensemble axis | closed — "a pool from ONE recipe saturates" | **BANK in scope, AUDIT-FLAT beyond it** | yu4u: EfficientNetV2-L 2.5D U-Net (0.3) + EfficientNet-B7 2.5D (0.4) + MONAI SegResNet **3D** (0.3) — cross-*architecture*, with a 3D member. 18th's ensemble is of **tabular heads over one detector**, which is the cheap version we never tried. |
| pmkf | "trains GT-sparse NOT detected-crowd" (~17/fr GT vs ~980 FP/fr) | **BANK — correct diagnosis, no cure built** | yu4u: DoG candidates at low threshold, **exclude ±6 µm around *unmatched* candidates from the loss**, DoG within 2 µm of a GT centre counts as duplicate (no exclusion), positives stay supervised inside exclusion zones, pos/background MSE averaged separately with background weight 0.5. We diagnosed the exact failure and never wrote the mask. *(12th tried a 10-epoch fine-tune with such a mask, lost recall, and note the full sparse-label recipe was never tried from scratch.)* |
| E53 | "unannotated FPs are metric-invisible" | **BANK — and two teams monetised it** | yu4u: two probabilities per candidate, `a` = P(evaluated at all), `q` = P(correct \| evaluated); expected TP = `a·q`, FP = `a·(1−q)` — sparse annotation **in the objective**. **[REVISED]** 18th's version is one line of labelling: train the edge rescorer on *"every edge the official metric can judge, not only GT edges"* — treating only GT edges as positives teaches the model that unscored edges are negatives. We found the property and never spent it. |
| proxy discipline | proxy faithful below ~0.89, blind/inverting above 0.90; gate on mechanism | **BANK — and understated** | **[REVISED]** Quantified by two teams. 18th, over their 34 late submissions: public↔private **r = 0.09** (ρ −0.23), CV↔public 0.03, CV↔private **0.34** — *their CV was the only useful ruler and the public board was noise*. 12th: in-sample bench over-read private by **~0.03**, their OOF world under-read by **~0.06**, yu4u's OOF CV over-read by 0.011; out of fold position errors beyond 7 µm grow **×3.8**; one change read +0.0143 in-sample, −0.012 public, −0.003 private. 288th: *"public differences ≤ 0.002 did not predict private ones, in either direction."* |

## 4. What we actually missed

### 4a. Mechanisms (originally four; now nine)

1. **A division model.** Three trained CNNs (parent / pre / post) seeing `[t−1, t−1→t aligned, t,
   t+1→t aligned, t+1]` at (1,2,2). Worth **+0.054 total** at 0.1× weight. We had concluded the division
   axis was denominator-bound; it was **witness-bound**.
2. **Joint link-and-division optimisation against a surrogate of the metric.** Fix ordinary links first
   and a correct daughter gets assigned to another parent, so the division can never be added afterwards.
   Linearised surrogate of the competition metric from expected TP/FP/node counts, re-linearised up to
   three rounds, HiGHS LP → MILP.
3. **Dense self-supervised flow as a field, not a cue.** Photometric + feature consistency after warping,
   forward-backward consistency, smoothness; synthetic pairs with known deformations; annotated links used
   **only** for checkpoint selection. **+0.0117**, and structurally invisible to our oracle experiment.
4. **Node insertion for gaps.** The dt=1 constraint we correctly discovered is satisfiable — by adding the
   node, not bridging over it. **+0.0051**.
5. **[REVISED] External data.** 151 annotated divisions is too few for an image model, so 18th used
   **Linajea zebrafish 160328** (70 divisions, 419 negatives), **OrganoidTracker 2**'s division CNN,
   CELLECT, self-generated synthetic 1→2 images, and **TabPFN 3.5** public weights (licence explicitly
   permits Kaggle-style competitions). 12th initialised their division reader on **Zebrahub** and used a
   frozen **DINOv2 ViT-S/14**. **We never considered external data at all.** It is the only route we can
   see to a division witness from 151 positives.
6. **[REVISED] A learned coordinate corrector.** Built by four teams, in two forms (global affine field;
   per-node learned offset with a magnitude cap and a no-topology-change guard). One was already in the
   notebook we forked. See the E57/E59 row.
7. **[REVISED] Tabular heads over a frozen detector.** 18th's whole +0.04 CV is LightGBM / CatBoost /
   TabPFN on hand-built features, ~9 MB of models, no detector training. This is the cheapest thing in the
   entire field and it was open to us for the whole competition.
8. **[REVISED] Integer coordinates.** GT coordinates are integer voxel indices. 12th: the same world
   scored **0.950 with integers vs 0.942 with floats** public (private 0.911 vs 0.906) — **free
   +0.008/+0.005 at write time.** *Check whether our submission rounded.*
9. **[REVISED] Per-embryo annotation conventions and dataset artefacts.** 6bba GT sits **+0.675 plane**
   above the nucleus z centre, 44b6 **+0.115**; a one-plane z lift of ~23% of nodes read −0.001 public and
   **+0.004 private** — *"the convention depends on the embryo: hedge it, do not tune it on the public."*
   **947 frames in 114 of 128 6bba videos are byte-identical to their predecessor**, and those videos carry
   **100 of the 151 divisions**; 80 pairs of 44b6 crops are byte-identical (2 of the 26 44b6 divisions are
   one event counted twice) — **video-level folds can leak.** Whole-stack one-frame jumps of 3.3–8.1 µm in z
   affect 52 GT frames in 36 6bba videos. We never audited the raw data for any of this.

## 5. Lessons that generalise

- **Probe the test split before modelling the task.** One detector pass over the test videos and a
  percentile table would have told us the private embryo is sparse. We spent months on a dense-regime
  thesis derived from a 21% slice. *Which rows is this number computed over* applies to the split, not
  just to our own eval subsets. **[REVISED]** And do not try to fix it by reweighting an in-sample world —
  12th proved that fails (density, embryo and copies are confounded).
- **[REVISED] Price a lever in the metric's own currency before running it.** One division = 74 edges. We
  set a 35-edge bar and then spent the campaign in edge-space while divisions were 51% of the deficit. The
  break-even *precision* for an added division (~25–32%) is one line of algebra we never wrote.
- **[REVISED] A knob that reads flat may have no consumer.** Three teams proved the public ILP cannot
  fork at the shipped weights. Our division-cost sweeps were a plumbing result wearing a refutation's
  clothes — the exact failure our own "assert the flag LANDED" rule names.
- **A ceiling measured through your own pipeline's failures is not a ceiling.** E54b and E59 both
  priced a mechanism on the set of edges our champion happened to break. A mechanism that changes the
  candidate set upstream cannot be priced on the downstream leftovers of the pipeline it replaces.
- **"The proxy is blind to X" and "we have no X" read identically in the logs.** `div_jac = 0.0000` in
  every arm was evidence for both, and we only ever wrote down the first.
- **A correct in-scope kill is not a closed axis.** Every division verdict we filed was true for what
  it tested. The axis that was actually open was the one thing none of them tested, and E61 *named it*
  before we priced it at zero runs and deleted it.
- **[REVISED] Ordering is a mechanism.** A filter that is correct in isolation destroys the input of a
  repair stage that runs after it. 12th's short-component filter erased 279 real cells and 9 of 28 absent
  daughters *before* their repair could use them.
- **[REVISED] Cheap models over a frozen detector are a real axis, and we never opened it.** 18th's
  ~9 MB of gradient-boosted heads bought +0.04 CV and 0.942 private. We had the detector, the features and
  the GPU-free budget for it all along, and spent the time on constants instead.
- **[REVISED] Read the raw data for artefacts before designing folds.** Copied frames, duplicated crops,
  per-embryo annotation offsets, integer-vs-float coordinates: four findings, none needing a GPU, one worth
  +0.008 public on its own.
- **Forking the top public notebook buys a rank, not a score.** +0.006 of public for free, and four
  hundred teams inside 0.005 on private. **[REVISED]** But *inside* that line ~0.006 private was still
  available from three CPU switches and one 7.3k-param head — 288th and Nikolce measured it, and we left
  it. Also, 288th: *"track the public ceiling from day one and re-base the moment it moves"* — re-basing
  0.940 → 0.953 was worth +0.007 private, more than two weeks of their knob probing.
- **The public fraction is 21%, not 29%** (per zephyr, second-hand but sourced; our 29% never had a
  primary source at all).

## 6. Corrections to the record

- `celltrack-true-bottleneck-is-dense-regime-model-gap` — the dense regime was the **public** embryo, and
  **[REVISED]** ~0.024 of private score was reachable downstream of the public detector (12th 0.946, 18th
  0.942, both fork-based). Amend to: the detector was the *frontier's* advantage, not our binding
  constraint at 0.918.
- `celltrack-e55-gap-repair-cannot-score-dt1` — premise stands, conclusion inverted. Gap closing is
  worth +0.0051 **via node insertion**.
- `celltrack-e54b-motion-continuity-oracle-refuted` — downgrade from REFUTED to **HARNESS**
  (self-selected denominator; a field-level method is worth +0.0117).
- `celltrack-cyto-rescue-is-metric-invisible-and-scoring-latency-is-not-a-queue` — drop
  "dose-dependently harmful"; private reverses the dose ordering at the same ±0.001. It is a tie.
- Conclusion-tree third seat: the deleted **two-pass tracklet ILP** node should be restored as
  **OPEN — division model + joint LP**, since the frontier's largest row is exactly it.
- **[REVISED]** `celltrack-champion-swept-division-on-real-LB-flat` and the division constant-sweep kills
  → **HARNESS**: the public ILP produces **zero forks** at the shipped weights, so the knob had no
  consumer.
- **[REVISED]** `celltrack-e58-champion-already-holds-the-edge` / `celltrack-postproc-repair-ceiling-on-0947`
  ("CPU post-processing axis CLOSED") → **REFUTED on the private embryo**: 12th's post-champion repair read
  +0.006 public and **+0.013 private**.
- **[REVISED]** `celltrack-e59-snap-oracle-cannot-price-localization` — keep the finding that *offline
  per-edge pricing* is dead, but record that a **learned coordinate head is the field's most-replicated
  lever** (4 teams; +0.020 CV / +0.003 private / already inside our own fork as V1284).
- **[REVISED]** `celltrack-hoct-association-head-refuted-detection-side` — add: as a **feature into a
  tabular edge rescorer** (18th, with Trackastra), that axis is untested by us and worth +0.021 CV there.
- **[REVISED]** New entry needed: **external data was legal and load-bearing** (Linajea, Zebrahub,
  OrganoidTracker 2, CELLECT, DINOv2, TabPFN 3.5). We never evaluated it.
- **[REVISED]** New entry needed: **integer coordinates at write time** (+0.008 public / +0.005 private).
  Verify whether our submission rounded.
