# CORRECTION 2026-08-26 05:52Z — the "metric hack" thesis is REFUTED against the OFFICIAL scorer

VERIFIED against the organizers' reference scorer `scratchpad_kout/tracking_repo/src/biohub_tracking/`
(`metrics.py` + `division_metrics.py`, code-cited below). b101ac traced it; I independently confirmed the
exact lines. **The kirneo `augment_dataset` off-image div-forks farm ZERO division-Jaccard and are
inert-to-negative on the edge term — the 0.945–0.953 cluster (#1 0.962) is GENUINE MODEL PLAY, not the exploit.**

Code-exact mechanism (division credit is exactly zero, MAGNITUDE-INDEPENDENT):
- `division_jaccard = div_TP/(div_TP+div_FP+div_FN)` (`metrics.py:436`, weight `SCORE_DIVISION_WEIGHT=0.1` `:34`).
- Off-image fakes @(−10000): no GT within `DistanceMatching(max_distance=7.0)` (`division_metrics.py:100`)
  → `MATCHED_NODE_ID` null → **div_TP=0** (score_divisions is GT-side) AND **div_FP=0** —
  `count_matched_pred_divisions` (`:395-408`) iterates ONLY pred nodes with a non-null matched GT node, so
  unmatched fakes cannot become FP (`fp = max(0, matched_pred_divs − tp)`, `:449`). div_FN is GT-side,
  unchanged. ∴ **division_jaccard cannot move via off-image forks — exactly zero, not small.** The hack's
  entire premise (farm 0.1·divJ free) is FALSE under this scorer.
- The fakes only HURT the edge term: `num_pred_nodes` (`:244`) is the RAW count incl. fakes → raises
  `total_node_ratio = (num_pred_nodes − n_total)/n_total` (`:353-356`) → LOWERS
  `adj_edge_jaccard = max(0, edge_J·(1 − 0.1·total_node_ratio))` (`:360-365`); hub→root invalid edges tax
  edge_J further. Net inert-to-negative.

**CONSEQUENCE — 0.93 is NOT a genuine ceiling.** The gap to 0.945+ is a REAL, closeable MODEL gap
(re-validates [[celltrack-true-bottleneck-is-dense-regime-model-gap]], which the hack thesis had wrongly
superseded). Two independent routes now agree: old LB-gap analysis + fresh official-scorer trace.

## Re-ranked levers (post-correction)
1. **[TOP] Replicate the 0.945–0.953 winners' dense-regime model/method** — highest gain, GPU-bound,
   days-scale. Scope FIRST: WHICH winner, code public (kernel) or paper-only? (b101ac research lane.)
2. **Motion-in-ILP + divsub-C3** — DEMOTED to cheap FLOOR-RAISERS to ~0.93. Still run the free CPU A/B
   (no slot, pure win). Mechanics in the (superseded-framing) sections below.
3. centroid-refine + D4-TTA — ride free.

The sections below are RETAINED for the lever mechanics (motion/divsub still valid as floor-raisers); their
"genuine ceiling ~0.93 + hack" FRAMING is SUPERSEDED by this correction.

---

# [SUPERSEDED FRAMING] 0.924 → genuine ceiling: ranked levers, and the 0.945+ verdict (owner-values)

2026-08-26. Re-synthesis of the win path AFTER the global-ILP wall-break (banked 0.924,
[[celltrack-linker-is-the-gap]]). Owns the "what closes 0.924→0.945" question; pairs with b101ac's
build+GPU lane. Sources: `research/frontier_kernels/SYNTHESIS_2026-08-25_genuine_vs_hack.md`, two code
audits (division-ILP regime + ILP objective cost, cited inline).

## The gap is not one lever — it is (genuine ceiling ~0.93) + (metric hack)

Two independent GENUINE kernels reproduce **0.923–0.927** (`metric_hack_used: False`, pass lineage
invariants). The winning cluster 0.945–0.953 (#1 = 0.962) is **LIKELY genuine-ceiling + the kirneo metric
hack** (`augment_dataset`: off-image hub@t=−1000 + FORKS=32 fabricated divisions per movie, farming
0.1·division-Jaccard; INVISIBLE on our faithful `core/metrics/divisions.py`, only the LIVE board credits
it). So the honest research target is **0.924 → ~0.93** via composable axis-adds; **0.945+ is an
owner-values call, flag-don't-fire** ([[celltrack-win-path-genuine-ceiling-vs-hack]]).

## KEY STRUCTURAL FINDING — the 0.924 linker is tracksdata `ILPSolver`, NOT celltrack's `"ilp"` linker

`grep -r "tracksdata\|ILPSolver" celltrack/` = **zero hits**. The tracksdata `ILPSolver` credited with 0.924
lives ONLY in `scratchpad_kout/tracking_repo/scripts/predict_unet_transformer.py:556-560` and
`research/frontier_kernels/kirneo…_code.py:894-900`. The celltrack `LinkerConfig(name="ilp")` mounts a
DIFFERENT solver — **motile/ilpy/SCIP** `ILPLinker` (`celltrack/linkers/ilp_linking.py:60-70`), cost
`distance − affinity_bonus·P` (`ilp_linking.py:105-107`).

Two objectives, do not conflate:
| solver | where | edge objective | division |
|---|---|---|---|
| **tracksdata `ILPSolver`** (= 0.924) | `scratchpad_kout/…predict_unet_transformer.py:556`, kirneo kernel | `edge_weight·EdgeAttr("edge_prob")`, `edge_weight=−1.0` → pure `−P` | `division_weight=1.0` **objective term** |
| celltrack `"ilp"` (motile) | `celltrack/linkers/ilp_linking.py` | `distance − bonus·P` | `MaxChildren(2)` **constraint** |

**Consequence for the build:** the divsub-C3 and motion grafts attach to the graph produced by the
**tracksdata script**, not celltrack's operating-point path. b101ac's "graft onto the bespoke 0.924 kernel"
must target that actual script — VERIFY which one produced sub 55779061 before building. This is a second,
independent confirmation of the route-through refutation.

## Ranked honest levers (0.924 → ~0.93)

### [1] divsub-C3 post-hoc safe-division — REAL axis-add; the −0.055 refutation does NOT transfer
- **Mechanism.** Our 0.924 base has ~zero division-J (structurally absent / unfired). The frontier's
  candidacy is a DIFFERENT gate than our refuted attempts: parent mid-track + mutual-nearest-orphan
  daughters + caps (parent≤8µm / sister≤11µm / child≤10µm) + DeepCenter-confirm + **C3 divergence**
  (daughters separate ≥2.25µm MORE at t+2 than t+1 — post-mitotic signature).
- **Q2 verdict (why −0.055 is out of scope).** The recorded "global-ILP division REFUTED −0.055" was the
  **celltrack motile `ILPLinker(division=True)` / `MaxChildren(2)`** regime (`.beads/issues.jsonl:129,489,693`),
  a DIFFERENT solver from the tracksdata `ILPSolver` (0.924), measured on a "(1,4,4) motion-base + saturated
  inverting proxy = POSSIBLY broken harness" (`.beads/issues.jsonl:55`). It failed because `MaxChildren(2)`
  made forks CHEAP (2-edge splits) and DOUBLE-DIPPED with div-recovery postproc. A **C3-gated post-hoc pass**
  has none of that failure mode — it is orthogonal to the linker solver. NOT pre-refuted.
- **Size / cost.** Frontier-measured **+0.0046 CV, divJ 0→0.0625**. Small but real, on an axis the base
  lacks; composes. CPU-buildable — we HAVE `celltrack/postproc/affinity_division_recovery.py`
  `_build_kernel_faithful` (`:605-625`, C1 gate implemented). ~30min + GT-free fire-check.
- **VERIFY FIRST.** Confirm banked 0.924's div-J is actually ~0 (is divsub already firing in the stack?).
  If already non-zero, this add shrinks. GT-free: count C3-gated forks on the 0.924 linked graph, confirm
  they land on DeepCenter-confirmed real peaks.

### [2] motion/velocity term in the tracksdata ILP objective — candidate genuine-ceiling lever, TEMPERED
- **Mechanism.** The 0.924 tracksdata objective is pure `−edge_prob` (+ appearance/disappearance/division
  weights) — **no distance, no motion term**. The genuine 0.926 kernels' lift rides a motion-relink cost
  (velocity-predicted distance `pos + 0.5·(pos−prev)`, `celltrack/linkers/motion_linking.py:145,204-206`).
  Adding a velocity term to the GLOBAL objective = global coupling (our +0.022) × motion model (their
  genuine lift) — an untried combination (neither genuine kernel does both: they use motion + per-frame
  greedy Hungarian; we use pure-affinity + global ILP).
- **Size.** Plausibly 0.924 → 0.926–0.93 IF motion adds on top of global.
- **TEMPER (why #2 not #1).** (a) `motion_distance` was previously **sub-noise** on the crowded axis
  ([[celltrack-motion-distance-crowded-lever-aggregate-small]]) — but under the OLD greedy/flow regime, not
  the global ILP. (b) motion & global-coupling may be partly **REDUNDANT** — both fight the confusor; the
  global solver may already resolve what motion fixes for a myopic greedy linker. Counter-hope: the confusor
  is a velocity-INVERTED fast-mover ([[celltrack-relative-pe-radial-confusor-directional]]), so a motion
  term COULD disambiguate what pure-affinity global can't — but that is a hypothesis. Higher build cost
  (touches the solver objective) + uncertain payoff.
- **Structure.** Velocity needs prior linkage (chicken-egg, [[celltrack-velocity-is-warmstart-not-fromscratch]])
  → **TWO-PASS**: initial solve → velocity from selected edges → re-cost `dist_pred = ‖target−(pos+0.5·vel)‖`
  → re-solve. Not a one-line cost tweak.
- **FREE LOCAL INSTRUMENT (the ratio-changer, b101ac 05:39).** `import tracksdata` FAILS locally so the
  byte-exact 0.924 repro is offline-blocked, BUT the celltrack motile `ILPLinker` (`ilp_linking.py:60`,
  SCIP/ilpy) imports fine → the motion-in-ILP MECHANISM is **CPU-fire-checkable NOW, no GPU, no slot**:
  swap static `distance` (`:105`) for the two-pass velocity-predicted distance in the motile objective, GT-score
  vs static on annotated movies, count confusor edges that flip. Only the exact-0.924 number needs tracksdata.
  This instrument directly retires the two tempers above (sub-noise? redundant-with-global?) at ~zero cost —
  so it is the highest-ratio next action even though the lever ranks behind divsub on priors.

### [3] centroid-refine + D4-TTA — cheap recall adds, ride free into any build
- 8-view D4 detection TTA + centroid refinement to local intensity peak (raises node-match under the 7µm
  gate — proxy-blind / LB-visible). Both genuine kernels use them; orthogonal to the linker. Individually
  small (bundled in the genuine +0.026), cheap, compose free.

### [LOW] reuse-graft — ~null, do not build
- b101ac's orthogonality argument (concurred): ILP changes ASSIGNMENT, reuse changes RECALL@thr0.99, which
  was LB-refuted at 0.898 pre-ILP; the base-swap 0.900→0.924 does not change whether reuse helps. Its own
  pre-registration predicts null. Fire would activate (necessary-not-sufficient) but fire ≠ benefit
  ([[celltrack-recovery-stack-lb-refuted]]).

## The 0.945+ verdict — owner-values, flag-don't-fire
Genuine ceiling of the top public methods ≈ **0.93**. The +0.02 to 0.945–0.953 = the kirneo metric hack
(off-image fabricated div-forks, invisible on our faithful metric, only the live board credits it). Owner
directive = "working baselines + our own research sum to a winning score" → the hack is neither. A single
probe sub COULD measure the live hack lift on our own forest (composable on any `submission.csv`) but it is
worthless tracking signal + un-validatable on our instrument. **Owner call only.** The only non-hack route
to 0.945+ is an un-replicated dense-regime model
([[celltrack-true-bottleneck-is-dense-regime-model-gap]], days-scale, unproven) — out of scope for a knob.

## Build order for b101ac (honest stack, floor 0.924) — instrument-first, let the free A/B set slot order
The two grafts are near-tied on priors (divsub certain-but-small vs motion bigger-but-uncertain), and BOTH
have free CPU fire-checks that need no GPU and no slot. So order by cheapest-de-risk, not by prior:
1. **Motion-in-ILP LOCAL A/B first (CPU, no slot, no GPU wait).** motile `ILPLinker`, two-pass
   velocity-predicted distance vs static `distance`, GT-scored on annotated movies. This is the free
   instrument on the PROVEN assignment axis (+0.024) — if it moves global linking above the 0.01 floor it
   becomes the **first slot candidate ahead of divsub**; if flat, divsub takes the slot. Runs now, GPU-idle.
2. **divsub-C3** GT-free fire-check in parallel (CPU): confirm banked 0.924 div-J is ~0, then count
   C3-gated forks on the linked graph landing on DeepCenter-confirmed peaks. Certain +0.0046 floor add;
   −0.055 out of scope (different solver). First slot if motion is flat.
3. **centroid-refine + D4-TTA** ride free alongside whichever wins.
4. VERIFY which script produced sub 55779061 (tracksdata path) before any exact-repro / slot submission.
5. reuse-graft: parked (~null).

Status: banked **0.924**; honest genuine ceiling ~0.93 is the reachable target; 0.945+ = owner-values.
[[celltrack-win-path-genuine-ceiling-vs-hack]] [[celltrack-0027-carrier-is-deepcenter-recovery-stack]]
