# Post-mortem — private leaderboard, Biohub Cell Tracking During Development

**Final: public 0.953 · private 0.918 · rank 562 of ~4000 teams.** Competition closed 2026-09-29.

This is the sense-making file for *our own* result. The 3rd-place solution it repeatedly compares
against is written up separately and faithfully in
[`research/solutions/2026-09-30_yu4u_3rd_place.md`](../../../research/solutions/2026-09-30_yu4u_3rd_place.md).
Verdict tags follow the [conclusion tree](2026-08-27_conclusion_tree.md): **BANK** · **REFUTED** ·
**HARNESS** · **AUDIT-FLAT** · **OPEN**.

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
| 20 | Yurinchi | 0.940 |
| 100 | Filip Strzałka | 0.928 |
| 200 | maco-macoo | 0.923 |
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

## 3. Grid — our verdicts against the frontier's ablation

yu4u's ablation (same OOF detections, fixed folds and scorer) is the cleanest external ruler we will
ever get for our own kill list.

| # | our verdict | still correct? | frontier's number |
|---|---|---|---|
| E52 | component prune priced out; champion's min component already 6 | **BANK — confirmed** | A5 = **+0.0017**, and their own constant is **< 6 nodes**. Identical choice, identical size. |
| E55 | gap repair cannot score: all GT edges are dt=1, a t−1→t+1 bridge matches no GT edge | **BANK on the edge — REFUTED on the fix** | A4 = **+0.0051**. They close gaps by **inserting the interpolated node**, so both resulting edges *are* dt=1. Our premise was right and we drew the wrong conclusion from it: the answer to "that edge can never be a GT edge" is *add the missing node*, not *abandon the axis*. `gaploose`'s 925 edges for zero was 925 of the wrong object. |
| E57/E59 | localization error is real (4.08 µm on broken edges vs 1.46 µm) but the snap oracle prices it at ~2 edges | **HARNESS — magnitude right, method wrong** | A6+A7 = **+0.0065**, from a *global affine motion field* per frame pair with probability-weighted links and a quadratic trade-off against the original detection — not a per-edge snap. Right diagnosis, and offline pricing of a single-cell correction could not see a field-level fix. |
| E58/E37/E53/E47 | CPU post-processing axis closed on the champion | **BANK** | A4+A5+A6+A7 together = **+0.0131** on a 0.964 base. Real, and far under our 1.5% bar for any one of them. |
| E54b | motion continuity oracle-refuted; ORACLE velocity recovers 10/61 broken edges | **REFUTED — self-selected denominator** | A1 dense 3D flow = **+0.0117**, the second-largest single row. We measured motion on *the 61 edges our own champion broke*, a set already filtered by a pipeline whose failures are elsewhere. Their flow is not a cue added to a ranker, it is a **field that redefines the candidate set** before matching. The denominator trap named three times in `CLAUDE.md`, once more. |
| E61 | division recall unreachable from post-processing; **"remaining lever = re-parenting inside the LINKER"** | **BANK in scope — and it named the right thing** | A3 = **+0.0625**, of which division Jaccard 0 → 0.535. The lever we correctly identified is exactly what they built. We then priced its three legs at zero runs and **deleted two-pass tracklet ILP from the conclusion tree on 09-22** as "the donor's lever, doesn't transplant". |
| division family (`hq93`, global division-ILP, constant sweeps) | post-processing and constant-sweep division work is dead; `div_jac = 0.0000` in every arm | **BANK — and it is the whole gap** | Division Jaccard is **0.1× weight**, so their 0.540 is **+0.054 of total score** — 3.5× our bar, from one component. We read `div_jac = 0.0000` in arm after arm and treated it as *the proxy is blind*. It was also *we have no division model*. |
| ensemble axis | closed — "a pool from ONE recipe saturates" | **BANK in scope, AUDIT-FLAT beyond it** | Their detector is EfficientNetV2-L 2.5D U-Net (0.3) + EfficientNet-B7 2.5D U-Net (0.4) + MONAI SegResNet **3D** (0.3). Cross-*architecture*, with a 3D member. Our finding about one-recipe pooling stands; we never ran the experiment it does not cover. |
| pmkf | "trains GT-sparse NOT detected-crowd" (~17/fr GT vs ~980 FP/fr) | **BANK — correct diagnosis, no cure built** | Their answer: DoG candidates at low threshold, **exclude ±6 µm around *unmatched* candidates from the loss**, DoG within 2 µm of a GT centre counts as a duplicate (no exclusion), positives stay supervised inside exclusion zones, pos/background MSE averaged separately with background weight 0.5. We diagnosed the exact failure and never wrote the mask. |
| E53 | "unannotated FPs are metric-invisible" | **BANK — and they monetised it** | Two probabilities per candidate: `a` = P(evaluated at all), `q` = P(correct \| evaluated). Expected TP = `a·q`, expected FP = `a·(1−q)`. Sparse annotation put **into the objective** rather than worked around. We found the same property empirically and had no machinery to spend it. |
| proxy discipline | proxy faithful below ~0.89, blind/inverting above 0.90; gate on mechanism | **BANK — and understated** | Their own writeup: *"agreement with the public leaderboard also does not establish generalization to the private test embryo."* The private result says the public **board** was the saturated proxy, for everyone. |

## 4. What we actually missed — four mechanisms, none of them a knob

1. **A division model.** Not a division cost, gate, or sweep — three trained CNNs (parent / pre / post)
   that see `[t−1, t−1→t aligned, t, t+1→t aligned, t+1]` at (1,2,2) and score "does this cell split".
   Worth **+0.054 total** at 0.1× weight. Trained from **151 annotated divisions** in the whole
   training set, with 25/25/50 positive/hard-negative/ordinary frame sampling to make that work. We had
   concluded the division axis was denominator-bound; it was model-bound.
2. **Joint link-and-division optimisation against a surrogate of the metric.** Their stated reason is
   the one our E61 almost wrote down: fix ordinary links first and a correct daughter gets assigned to
   another parent, so the division can never be added afterwards. Objective = a linearised surrogate of
   the competition metric built from expected TP/FP/node counts, re-linearised around the current
   solution for up to three rounds, accepted only on surrogate improvement, HiGHS LP → MILP.
3. **Dense self-supervised flow as a field, not a cue.** Self-supervised on real pairs (photometric +
   feature consistency after warping, forward-backward consistency, smoothness) plus direct supervision
   on synthetic pairs with known deformations; annotated links used **only** for checkpoint selection.
   One field serves both correspondence search and division-image alignment. **+0.0117**, and the thing
   our oracle experiment was structurally unable to see.
4. **Node insertion for gaps.** The dt=1 constraint we correctly discovered is satisfiable — by adding
   the node, not by bridging over it. **+0.0051**.

## 5. Lessons that generalise

- **Probe the test split before modelling the task.** One detector pass over the test videos and a
  percentile table would have told us the private embryo is sparse. We spent months on a dense-regime
  thesis derived from a 21% slice. *Which rows is this number computed over* applies to the split, not
  just to our own eval subsets.
- **A ceiling measured through your own pipeline's failures is not a ceiling.** E54b and E59 both
  priced a mechanism on the set of edges our champion happened to break. A mechanism that changes the
  candidate set upstream cannot be priced on the downstream leftovers of the pipeline it replaces.
- **"The proxy is blind to X" and "we have no X" read identically in the logs.** `div_jac = 0.0000` in
  every arm was evidence for both, and we only ever wrote down the first.
- **A correct in-scope kill is not a closed axis.** Every division verdict we filed was true for what
  it tested — post-processing, constants, global ILP without a model. The axis that was actually open
  was the one thing none of them tested, and E61 *named it* before we priced it at zero runs and
  deleted it.
- **Forking the top public notebook buys a rank, not a score.** +0.006 of public for free, and four
  hundred teams inside 0.005 on private. Real separation came from a pipeline nobody could fork.
- **The public fraction is 21%, not 29%** (per zephyr, second-hand but sourced; our 29% never had a
  primary source at all).

## 6. Corrections to the record

- `celltrack-true-bottleneck-is-dense-regime-model-gap` — the dense regime was the **public** embryo.
  Amend, don't delete: the dense gap was real, it just wasn't the graded one.
- `celltrack-e55-gap-repair-cannot-score-dt1` — premise stands, conclusion inverted. Gap closing is
  worth +0.0051 **via node insertion**.
- `celltrack-e54b-motion-continuity-oracle-refuted` — downgrade from REFUTED to **HARNESS**
  (self-selected denominator; a field-level method is worth +0.0117).
- `celltrack-cyto-rescue-is-metric-invisible-and-scoring-latency-is-not-a-queue` — drop
  "dose-dependently harmful"; private reverses the dose ordering at the same ±0.001. It is a tie.
- Conclusion-tree third seat: the deleted **two-pass tracklet ILP** node should be restored as
  **OPEN — division model + joint LP**, since the frontier's largest row is exactly it.
