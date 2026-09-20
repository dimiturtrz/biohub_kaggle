# Conclusion tree — the one living synthesis (supersedes scattered per-run memories)

2026-08-27. **Purpose:** the memory base exploded to ~80 one-off leaves, each an experiment, no synthesis
layer. This is the tree: observations → conclusions → the remaining levers, per config axis. When a new
result lands, fold it into the matching node here and update the leaf verdict — do NOT spawn another
isolated memory. Per-run detail stays in `interpretations/celltrack/<date>_*.md` + git history; this doc is
the index that decides *what to run next*.

Verdict tags: **BANK** (shipped/true) · **REFUTED** (killed on real board or sound mechanism) ·
**HARNESS** (killed but the kill was our setup — undertrained / wrong grid / wrong ruler; re-testable) ·
**AUDIT-FLAT** (mechanism-bounded before building) · **OPEN** (live, un-run or un-scored).

---

## NEWS 2026-09-20 05:58Z — E57: it was LOCALIZATION all along, and E54b had already proved it

Read at the official 7 µm ruler, over the same 105 broken and 2022 recovered GT edges: a **broken** edge's
endpoints sit **median 4.08 µm** from the cell (46 % beyond 5 µm), a **recovered** edge's **1.46 µm** (2 %).
Three times worse, one denominator, no ruler swap. The official matcher forgives 7 µm — about one cell
width — so a node half a cell off still counts as *detected*, and its dead edge is filed under
**selection**. The detection/selection split E54 reported is a property of the **tolerance**, not of the
tracker: 26/76 at 7 µm, 641/20 at 2.87 µm.

The closer was banked five hours earlier and under-read. `cue_oracle.py`'s fourth row —
**distance to the true GT position of the target ranks the true successor first 61/61**, against 10/61 for
plain distance and 10/61 for oracle velocity. **With correct coordinates the simplest discriminator there
is, is perfect.** E54c's "structure right, discriminator missing" therefore restates as **the
discriminator is present and is being fed nodes a cell diameter off**. E54b even named the mechanism and
left it: 15 more true successors fall *outside* the 10 µm gate because "matched predicted positions carry
up to 7 µm localization slack each". `gate_um = 10` is the maximum **GT** step — right for clean
coordinates, systematically undersized for the predicted ones it is actually applied to.

A detector with **no weights at all** reaches part of the gap. `kaggle/classical_blindspot.py` harvests
the multi-scale DoG from `fabriciodasilva/biohub-dodecatiad-cell-tracking` (CPU-only) — the one detector
family that cannot share the learned pool's bias, which is what E36's *flat-across-nine-caches* 0.824
demanded. Over all 292 frames of the four public-GT movies: champion recall **0.7747** at cell separation
but **0.9863** at the official ruler; of the 494 cells missed at 2.87 µm, a model-free detection lands
within ruler on **0.1336** against **0.0486** for a frame-shifted matched null → **excess +0.0850 = 42
cells**, at **0.7× the champion's detection count**. Fewer detections, not more, so the rescue is not
bought with false positives; and stride-5 gave +0.0909, so it is not a sampling artifact.

**Node RECALL at the scorer's ruler is CLOSED** — six misses in 2193, model-free excess of one cell. The
open channel is **LOCALIZATION**, which the 7 µm matcher is built not to see and which edges die on.
Cheapest first: **widen `gate_um` and re-solve** (a constant, no training) · then a center-offset head
priced against the 42 · **never** swap the detector wholesale (finer122 graft 0.7503, ILP swap 0.6555).
Do not price this as 42 recovered edges: the snap-to-GT oracle is a **no-op by construction** — the
broken/recovered verdict depends only on the node matching and the ILP has already run — so causality
needs a re-solve. Three CPU runs, ~25 min, zero GPU, zero submissions.
`interpretations/celltrack/2026-09-20_e57_localization_not_selection.md` · bd `n2xg`.

---

## NEWS 2026-09-20 05:36Z — E56: the broken-edge count is ANTI-correlated with the board

Full write-up: `interpretations/celltrack/2026-09-20_e56_cross_tracker_edge_agreement.md`.

- Built `error_budget.py --rival`: compares two cached submissions **GT edge by GT edge** over the same
  2127 public-GT edges. **fp16 is the control and it behaves** — its broken set is **100 % nested** inside
  the champion's, confirming E38's 0.9956 union at edge resolution. The instrument discriminates a
  same-family variant from a decorrelated one.
- The v1329 family rescues **39 champion-only edges**: ruler-stable (44/38/39/37 at 2.87/5/7/10 µm),
  **33/39 selection** (17 swap, 9 source-rival, 7 target-rival), **33/39 inside the dense movie**
  `6bba_05db0fb1`.
- **CORRECTION to commit `0d7dc7d`**, which called v1329f and v50 "two unrelated recipes". They are not:
  v50's `v1329_submission.csv` is **byte-identical** to v1329f's submission and both carry node count
  123485. v50 = v1329 + one stage. **n = 1 family witness, not 2.**
- **The deflation.** Broken GT edges vs LB across all four banked submissions:
  105 → **0.947** · 104 → 0.945 · 78 → 0.939 · 77 → 0.907. **The champion breaks the MOST and scores the
  BEST**; the ordering is exactly reversed. Not the E51 saturation story — these are the *same four
  movies the board scores*. Our pooled harness sides with the count, not the board (champion 0.8932 vs
  v1329f 0.9263, 0.033 the wrong way against the board's 0.008).
- The confound is visible and uncontrolled: the rivals also carry **677 more nodes**, which E53's signed
  node-count term prices. v50 sizes the sensitivity — same node count as v1329f, 90 fewer edges,
  **−0.032**. So the reading is not "recovering GT edges hurts" but **"the score is not dominated by
  GT-edge recovery, and this harness cannot separate the two effects."**
- **E54's bar survives with an explicit clause: recover 35 GT edges *without adding nodes*.** That was
  implicit before and is now measured. **A cross-tracker merge is NOT licensed by an edge-count win** —
  adopting a rival's edge means adopting endpoints the champion lacks, i.e. the exact term the board
  punishes. Never price a cross-tracker change with E54's ~2.3 units/edge; it was derived *within* one
  tracker with node count held fixed.
- **ADDENDUM, same tick — the 33 selection fixes are node-free BY CONSTRUCTION, so the axis is alive.**
  `classify_broken` tests rules in order with the detection rules first, so a swap/rival label is only
  reachable when **both** GT endpoints already matched a champion node. All 33 connect nodes the champion
  **already has**; importing them costs **zero** nodes. **The 677-node penalty belongs to the rival's
  DETECTOR, not to the edges** — the two were confounded and separate cleanly. Swap fixes are edge-neutral
  too, and E54c put the collateral at zero (0/126 thieves annotated). That returns the 33 to the
  within-tracker regime where ~2.3 units/edge holds: **≈1.4%, just under the bar**, oracle-assumed →
  **directional, composes.** **A RE-RANKING axis, not a merge axis** — the champion lacks the
  discriminator (E54c) and v1329 is an existence proof one exists; transfer needs no GT at test time.
  Affordable only because **twrp** freed 4010s (rival predict ~930s). bd `x6bd` blocks `kiw1`.
- **SECOND ADDENDUM, same tick — AXIS CLOSED. Node-free is not the same as free.** `kaggle/edge_transfer.py`
  asks the transfer question the way a *submission* must: align rival nodes onto champion nodes by
  **distance alone, no GT**, then sort every rival edge. At the official 7 µm ruler: unaligned 3831,
  redundant 112643, **CLEAN 13**, **CONTESTED 2405** (at 2.87 µm: 7070 / 110273 / 11 / 1538). Clean
  additions are **11–13 across four movies** = sub-noise, worth nothing. **And the 33 are not among them**
  — swap / source-rival / target-rival each mean a champion slot is ALREADY TAKEN, so all 33 are contested
  **by definition**. The harvest is not "add 33 edges", it is **"pick the right 33 out of 2405"** = a
  **1.4 % base rate**, each wrong pick breaking a GT edge *and* planting an FP (E54, one axis). Exactly
  E48/E49's spec, now binding on transfer proposals: their GBM hit FP-recall 0.0846 vs 0.75 required on an
  *easier* base rate. **Borrowing a rival's edges does not supply the discriminator — it supplies 2405
  suggestions, 98.6 % wrong.** Corrects the first addendum: node-free right, "alive" too generous.
- **Also banked from that census: 94.7 % of the rival's edges are REDUNDANT** (112643 / 118892). Two
  separately-trained families agree on nineteen edges in twenty — the ensemble-saturation closure measured
  at **edge resolution** for the first time, rather than through a score. Cost to close the whole axis:
  **one CPU run, zero submissions, zero GPU.**

---

## NEWS 2026-09-20 05:17Z — E55: gap repair cannot score (GT is all dt=1), and the LB already said so

- E54 measured **0 GT edges with `dt != 1`**. So a gap-repair bridge connects `t-1` to `t+1` and
  **matches no GT edge, ever** — best case metric-invisible (E53), worst case a clean FP when both
  endpoints are annotated. It does not restore the two real GT edges either. **E37's repair axis and the
  `max_gap = 2` lever are both retired by argument, for zero submission slots.**
- **The board already told us:** `gapfill` = **0.947, an exact tie with the base champion**. That tie was
  sitting there unread. **Pre-registered:** `gaploose` (filler 240 → 1165) returns **≤ 0.947**; above it
  would refute this and force a re-measure of the dt=1 count.
- **`fp16` = 0.945**, below the champion — E38's "no slot" call from the 0.9956 edge union, confirmed.
  **Final-2 stays 0947 + v1329f.** det096 / synthsec / gaploose still pending.
- **Standing question for every edge-adding proposal from now on: *can the edge it adds ever BE a GT
  edge?*** Anything spanning more than one frame answers no. (Recovering a missed detection's NODE is a
  different proposal — the 26/105 detection slice.)

---

## NEWS 2026-09-20 05:13Z — E54c: the error is a SPLICE with an unlabelled REAL track

- **0 of 126 slot-stealing rivals is an annotated cell** (`error_budget.py --thieves`). On E54's 76
  selection failures: source slot 63 unannotated / 13 free / **0 annotated**; target slot identical.
- Harness suspect cleared first: the thieves sit **median 6.92 µm from the nearest GT node of any kind,
  0 % within 2 µm** — one cell-separation. Not mislocalized labelled cells.
- **And they are REAL CELLS, measured, not junk.** Predicted track length: thieves **median 61 frames,
  0 % ≤3 frames, 67 % ≥50** vs GT-matched 74 and all-predicted 44. Not one blip; they out-persist the
  average predicted node.
- **SELF-CORRECTION, recorded deliberately.** I first read "0 annotated thieves" as "no conflict exists,
  so re-costing cannot help". The track-length measurement refutes that: the slot **is** claimed, by the
  thief's own real trajectory, which GT merely does not label. The ILP constrains it correctly already.
- **So: structure right, DISCRIMINATOR missing.** The failure is a splice between a labelled track and an
  unlabelled real one, and position, appearance (E46, below chance) and motion (E54b, oracle) all fail to
  separate two real cells.
- **Narrows the standing conclusion from "dense detector" to "identity representation".** The thief
  *should* be detected — better detection does not remove it. And a same-feature re-solve cannot either.

---

## NEWS 2026-09-20 04:52Z — the error budget in metric units: ONE axis, 105 broken GT edges

- **E54: `wrong-association = 0` on all four movies, verified by a direct recount.** Predicted edges with
  BOTH endpoints matched to annotated GT nodes but not a GT pair: 0 / 0 / 0 / 0. Among cells the
  annotators labelled, the global ILP links them to each other correctly **every time**. Every one of the
  141 FP edges is single-endpoint-matched — an annotated cell continued into a detection GT lacks.
  **Retires a framing: the confusor is not a competition between two tracked cells.**
- **99 % of FPs (139/141) are incident to an endpoint of a GT edge the tracker failed to recover.** FP and
  FN are not independent axes — they are two views of the same **105 broken GT edges**, each costing
  ~**2.3 metric units** (105 FN + 139 FP). E47's ~2× asymmetry, derived, and general to every missed link.
- **Budget:** pooled adjusted **0.8932** (raw 0.8915), tp 2022. **83 of the 105 are on the one dense
  movie.** Perfect-fix ceilings OVERLAP and must not be added: all-105 → 0.9991; fp→0 alone +0.059;
  frag alone +0.035; detection alone +0.012.
- **The 105, partitioned (`--shapes`): 76 (72 %) are PURE SELECTION** — every endpoint detected, the
  candidate present, the solver ranked another partner above it. **50 are outright swaps** (both endpoints
  linked elsewhere) + 13 source-rival + 13 target-rival. Only **26 detection**, **3 division**, and
  **0 GT gaps** (`dt ≠ 1` never occurs). 76 vs a 35-edge bar → **`kiw1` licensed with 2× headroom.**
- **Broken edges move 4.08 µm median vs 1.46 µm recovered; 46 % > 5 µm vs 2 %** — E45's "outranked band is
  fast cells" now holds on the champion's metric-visible errors.
- **MOTION CONTINUITY IS ORACLE-REFUTED** (`kaggle/cue_oracle.py`, same tick). Ranking every in-gate
  candidate in the next frame: distance **10/61** · predicted-tracklet velocity **8/61** ·
  **ORACLE velocity from the cell's TRUE GT trajectory 10/61** · true-GT-position ceiling 61/61.
  **Handing the ranker the actual past trajectory buys nothing over plain distance** → the estimate is not
  the problem, and no Kalman / tracklet second pass recovers these. I wrote the motion premise into E54
  and refuted it 20 min later; the doc carries the correction.
- **Cue inventory now CLOSED: position, appearance (E46, below chance) and motion all fail on these 61.**
  The information is absent from the detected representation → pairwise-unresolvable keystone
  re-confirmed with motion added AND an oracle. **Do not build `kiw1` on the motion argument.** What
  remains is a joint-assignment argument (50/76 are swaps) — but the champion already solves a global
  ILP, so the COSTS are wrong, not the structure. Re-points at the dense-regime DETECTION model.
- **15 of the 76 have their true successor outside the 10 µm gate** — unreachable by any solver; matched
  predicted positions carry up to 7 µm localization slack each. **GATE WIDENING PRICED AND CLOSED same
  tick**: 12 µm → 3 unreachable, 15 µm → 0, but **distance still ranks exactly 10 first at every gate**
  (10/61 → 10/73 → 10/76). Widening admits the edges without making them RANKABLE, while adding
  confusors everywhere else. No-op at best.
- **Consequences.** (1) Do NOT chase FP suppression as its own programme — 99 % are symptoms; deleting an
  FP without supplying the right link converts a 2.3-unit error into a 1.3-unit one at best. (2) Every
  future proposal must state **how many of the 105 it moves**; the 1.5 % bar ≈ **35 recovered GT edges**.
  (3) Detection is genuinely smallest (26/105), consistent with E40/E41. (4) `kiw1` (two-pass tracklet
  ILP) is the axis, now sized.
- 85.8-99.8 % of predicted edges are metric-**invisible** — E53 restated as a measurement.
- `kaggle/error_budget.py` · `interpretations/celltrack/2026-09-20_e54_error_budget_in_metric_units.md`.
  Caveat: E51 in-sample, magnitudes are ceilings; the shape is structural.

## NEWS 2026-09-20 04:45Z — the prune axis was never a tracking axis (read from the scorer's source)

- **E53: an edge between two UNANNOTATED detections is neither TP nor FP.** `metrics.py:55-114` and the
  champion's faithful copy (`champion_submission.py:3584-3620`): an edge counts only if an endpoint
  matched a GT node. A spurious detection away from annotated cells is **metric-invisible** except
  through `adjusted = jac * (1 - 0.1*(t_pred - t_true)/t_true)`, where `t_true` is the geff's **estimated
  true node count** and the term is SIGNED — `t_pred < t_true` multiplies the jaccard UP.
- **Champion already sits at/below `t_true` on 3 of 4 movies** (25637/25755, 20729/32795, 6152/6362,
  70290/69800) → no over-detection penalty left to recover.
- **Re-reads the prune axis:** E47's +0.1139 oracle is mostly that multiplier, not recovered edges;
  E48/E49's target label was not "is this an FP" but "did an annotator pick this cell" (98.4 % of their
  rows were unlabelled REAL cells); E52's cut buys ≤ +0.009, under the noise floor, against ~2× that in
  collateral. **Axis closed for a better reason than E48/E49 gave.**
- **The label is partly predictable** — on the two sparsest movies annotation is a thin slab in y (GT std
  5.7 / 8.4 µm vs 30.7 / 24.9 for all detections). **Not building it**: deleting real cells for the
  node-count bonus games sparse GT and improves no tracking. Any future prune must name its channel —
  recovered edges (real) or the multiplier (not). Real FP headroom = edges TOUCHING annotated cells, i.e.
  the mislink/confusor problem (E45/E46).
  `interpretations/celltrack/2026-09-20_e53_prune_axis_is_an_annotation_artifact.md`

## NEWS 2026-09-20 04:40Z — a harvested element was already in the champion; and a submission can't price a prune

- **E52: SHORT5 is a NO-OP on our output.** `howonkang/biohub-0947-short5-prepp-r1` deletes every linked
  component of ≤5 nodes from the raw geff. Minimum component size in the champion's submission is
  **exactly 6** on all four movies — `OUTPUT_MIN_TRACK_LEN = 6` already binds, and no short division
  component survives for `OUTPUT_KEEP_DIVISION_COMPONENTS = 1` to spare. Cuts at ≤2/≤3/≤5 remove **0
  nodes**. Its only remaining claim is as a PRE-postprocessing pass (i.e. "repair is net-harmful on short
  components"), which sits inside E37's ~1.1 % repair ceiling → sub-bar, no slot owed.
  `interpretations/celltrack/2026-09-20_e52_short5_is_a_noop.md`
- **The denominator rule this exposed — general.** Only **1.39 %** of predicted nodes match any GT cell at
  2.87 µm (2.21 % at 7 µm). That is annotation SPARSITY (E51's fracs 0.0016–0.135), not a 98 % FP rate.
  So **a cached submission cannot measure a pruner's FP-recall** — its negative class is "unannotated",
  not "false". Collateral stays valid (denominator = matched GT). Price prunes on a candidate set with GT
  correspondence, as E47–E49 did.

## NEWS 2026-09-20 04:30Z — the proxy is in-sample, and its own split cannot say how much

- **E50 (`g89y` precondition): both gates FAIL under an intensity proxy, but the proxy is confounded.**
  Containment @2.87 µm **0.7498 < 0.824** detector; best FP-recall @2 % collateral **0.0203 vs 0.75**
  (E47), with `height` a flat 0.0000 at every cut. Per the pre-registration this is NULL-leaning, NOT a
  refutation of `g89y` — the real build uses a trained contour head. A fixed `FOREGROUND_QUANTILE = 0.98`
  first produced a *fake* containment of 0.6047 by cutting above every GT cell on one bright-background
  movie; `proxy_validity()` now refuses such a movie on its own arithmetic. Otsu admits the cells and
  costs >3 min/frame — **no threshold is both valid and affordable.**
  → `interpretations/celltrack/2026-09-20_e50_hierarchy_precondition.md`
- **E51: the public affinity model trained on ALL 199 videos → every proxy movie is affinity-in-sample.**
  Manifest is literally `unet_transformer_alltrain_seed314159_v1`, its 40 "test" videos a strict subset of
  its 199 train. DeepCenter held out 128 — but the four proxy movies it DID train on are exactly the four
  `44b6` ones, which are also the four sparsest annotations (frac ≤0.029). In-train adj_jac 0.9189 vs
  held-out 0.8996 **attributes to nothing**: in-train ≡ `44b6` ≡ sparsest is perfectly collinear on this
  split. **Mechanism for a rule we held only empirically** — local association headroom is optimistic, so
  an offline association win need not transfer (dw0 +0.0267 local / −0.028 LB is the shape). It
  *strengthens* E49: AUC 0.6380/0.6622 were in-sample, i.e. an optimistic ceiling that still failed 9×.
  → `interpretations/celltrack/2026-09-20_e51_proxy_contamination.md`

---

## NEWS 2026-09-20 03:30Z — the board compressed; the last standing lever is runtime-clear

- **The field moved and we did not.** Public LB re-listed: 0.974 / 0.973 / 0.969 / 0.968 / 0.967 /
  0.966 — and **0.964 is only about rank 12**. Our banked 0.947 no longer sits near the top; the gap to
  the front is ~0.027, which is far above the 0.01–0.02 noise floor and is the same dense-regime model
  gap E42–E49 keep arriving at from new directions. Entry deadline **09-22**, final submission **09-29**.
- **Donor `fabriciodasilva/biohub-dodecatiad` REJECTED on inspection, no run spent.** Read in full: DoG
  blob detector + Hungarian/velocity linker + gap-close + a second-daughter division heuristic, on
  numpy/scipy/zarr only. **No ultrack, motile, tracksdata, trackastra; no hierarchy, contour, watershed,
  ILP, appearance, embedding or affinity anywhere in the file.** Every element is a strict subset of what
  we already run, and its one detector idea is the axis E40 closed (DoG union 0.139 vs 0.450, n=3206).
  Recorded in bd `r5ce` as do-not-re-harvest. Einstein criterion paid for itself: a read, not a run.
- **`g89y`'s runtime gate is FOREGROUND-DEPENDENT, and my earlier "closed in its favour" was wrong.**
  The 4.5 s/frame figure holds only at a 2 %-of-voxels foreground. That threshold is *invalid on some
  movies*: where the background is bright and hazy (raw median 1115 vs 114), the GT cells sit BELOW the
  cut, so 0 % of them can be contained — the number measures the threshold, not the hierarchy. Admitting
  those cells (Otsu) inflates foreground to 21–48 % and costs **>3 min/frame** — ~20–40× — which blows
  the ~27000 s headroom by 4×. So the honest statement is: hierarchy cost scales with foreground volume,
  and the cheap regime is the one that misses cells. Budget is an argument against `g89y` again until
  the trained contour head shows it can be both tight and cell-containing.
- **Ultrack's own published mechanism is exactly g89y's claim** (Nature Methods 2025): candidate
  segmentations from *multiple* algorithms, with temporal consistency used to **select** among them.
  That is the one axis E48/E49 explicitly could not reach, because selection scores nested contours our
  candidate set never contained. E50 (`kaggle/hierarchy_precondition.py`) is measuring it now.

## NEWS 2026-09-20 00:49Z — where the campaign stands

- **Board:** ours 0.947 (`celltrack-public-0947`, tied by `-gapfill`), then readmit 0.946, v1329f 0.939,
  v50 0.907, our own banked 0.924. LB top 0.974. Deadline 09-29. **5 slots, reset 00:00 UTC.**
- **Pending LB:** det096 (56370794), synthsec (56370924), fp16 (56372434). **fp16 is now decided
  offline** — 0.9956 of the edge union with 0947, 257 disputed edges (E38) — its score is a formality.
- **Post-processing on the 0947 base is priced out (E37).** The repair the E36 residue asks for is
  already shipping in `-gapfill`; its ceiling is ~1.1 % more edges because only 935 bridgeable
  end→start pairs exist, and 39 % of dangling ends are cells leaving the imaged volume. Two free
  instrumented kernels (`gapdiag`, `gaploose`) are in flight to settle which gate discards the rest;
  **neither is a submission candidate** at that ceiling.
- **The final-2 hedge is real and keeps its slot (E38):** 0947 and v1329f dispute ~16 % of the edge
  union (10356 / 10700 unique edges), so they are two genuinely different graphs, not one in two hats.
- **The live lever is unchanged in SIZE and corrected in TARGET (E40/E41/E41(b)).** E36's 2.5 % of edges
  is independently reproduced — 3.30 % on a cache whose recall is 0.697–0.750 vs E36's 0.824 — so the
  lever is real and detection is still the axis. What was wrong is *which cells it points at*. E36's
  mechanism (a GT node borrows a **neighbouring** cell's detection) fails: missed cells sit 24.8 µm from
  the nearest GT neighbour, the same as found cells, and a matched null (frame t+50) shows the detection
  3.83 µm away is the cell's **own**, displaced — own-frame closer than chance in 0.807 of cases, and
  only 3 % of these cells lack any local intensity excess. So the 2.87 µm population (n=3864, 40.3 % of
  edges — impossible against a 0.92 LB) is **localization scatter**, while the population that actually
  costs score is the **288 nodes blind at the official 7 µm ruler (2.25 %)**, 13× smaller. Aim a recovery
  pass at those, not at the 2.87 µm set. A learning-free DoG union is separately dead on the conditional
  (0.156 vs its own 0.414 marginal). The card stays parked by explicit instruction; nothing starts
  without asking.
- **The second addressable block is SOLVER-side and mostly spent (E43/E44).** Edge fragmentation is a
  *selection* failure, not a candidate one: 99.51 % of fragmented GT edges already carry a proposed
  candidate with an affinity (only 59 = 0.48 % have none, and those are median 11.72 µm — outside the
  10 µm gate). So it needs no detector pass. But the obvious knob is spent: sweeping termination cost
  shows a non-monotone curve peaking at 1.0–2.0, and raising it to d5er's proposed 3/4/6 **loses**
  recall (0.8589 → 0.8293) because expensive termination makes nodes grab FP rivals. What stays live is
  the **11.95 % of true edges the affinity head ranks below a rival, by a median margin of 0.0985** —
  an affinity-*ranking* problem, reachable by a global solver or a better head, never by a cost knob.
- **That surviving band is PAIRWISE-bound, and the next lever is tracklet-level (E45/E46).** The band is
  a fast-cell population (5.7× enriched); on it, distance picks true 0.1767, oracle velocity buys +0.065,
  and raw appearance reads **0.4853 vs a 0.8230 mutual-best control** — 0.4345 at matched displacement,
  *below chance*. The confusor is a near-static lookalike and is an **unmatched FP in 0.9698** of
  contests: every local pairwise cue selects it, and it has no continuation for the pair to see. So the
  argued levers are **segment-SELECTION ILP (bd `g89y`, prune the duplicate)** and the **two-pass
  tracklet ILP (bd `kiw1`, out-compete it with a sequence)** — never a better edge feature. (E46 is a
  replication of a 2026-08-23 memory that the index had dropped; see the E46 block.)
- **The OFFLINE PRUNE AXIS IS CLOSED — the joint ceiling is 9× short (E49).** E48 killed two scores one
  at a time; E49 gave the inference its best shot: eight features (intensity, the affinity field's
  best-out/best-in/margin/degree, crowding, temporal support t±1), a GBM free to combine them, **split by
  MOVIE**. Joint **AUC 0.6622 — only +0.026 over its best single feature — FP-recall 0.0846 at the
  required 2 % collateral.** Not a tuning gap. The **nulls carry the news**: degree 0.4278, crowding
  0.4815, **temporal support 0.4837/0.4838**. "A spurious detection has no continuation" — the strongest
  remaining intuition, and the one bd `kiw1` runs on — is worth nothing, because E46's confusor is a
  near-static lookalike sitting where the cell WAS and has the same t±1 neighbour a real cell has. The
  pairwise-unresolvable finding and the temporal-support null are one fact seen twice. bd `g89y` is NOT
  refuted by this: hierarchy selection scores nested contours the candidate set never contained, so it is
  outside the measurement's reach — but it must now supply information that is not a function of any of
  those eight, which is the sharpest form its precondition has taken.
  Detail: `interpretations/celltrack/2026-09-20_e49_offline_prune_axis_is_closed.md`.
- **The diagonal is now MEASURED, and no score we hold reaches the operating point (E48).** Center-voxel
  intensity: AUC 0.5221, and its **FP-recall equals its collateral to three decimals at every cut**
  (0.0222 at 0.02) — E47's q = r diagonal, literally, which upgrades the recovery-stack retro-explanation
  from argument to measurement. Best incident affinity is better and still 13× short: AUC 0.6380,
  FP-recall **0.0578 against the 0.75 required**. That failure was pre-registered with its reason — the
  E46 lookalike's signature IS high affinity to the source, so a want-to-link score cannot demote it.
  Read together: real-vs-FP status is not recoverable from anything DOWNSTREAM of the detector, which is
  the dense-model-gap conclusion reached from a new direction (convergence, not a new finding). `g89y` is
  sharpened, not closed — it must carry information neither score has, and the bar is a number, not a
  direction. Caveat: the cache has no per-detection score column, so the detector score itself is
  refuted only by analogy — measure it opportunistically on the next GPU pass.
- **The prune axis is priced, and COLLATERAL is what binds it (E47).** Oracle FP-deletion rescues 95.3 %
  of the band (0.1195 → 0.0056, net **+0.1139** offline) — the ceiling is not the problem. But deleting a
  real cell costs ~2× its own size (both its edges), so at 5 % collateral even a PERFECT pruner nets
  +0.021, while at 2 % collateral FP-recall 0.75 earns **+0.035**. The requirement is ASYMMETRIC, not
  aggressive. This retro-explains the LB-refuted recovery stack by mechanism: a threshold is
  intensity-ordered, so it buys recall and collateral on the SAME axis ~1:1, and E41 puts the at-risk
  real cells exactly there — it lands on the losing diagonal by construction, not by mistuning.
  **Precondition before building `g89y`:** check whether hierarchy selection's survival decision
  correlates with intensity; if it does, it inherits the same coupling.
- **Runtime is solved and is no longer a reason to avoid anything (E31/E32).** `-fast` = 1590 s vs 6286 s,
  byte-identical submission; hidden test is 4 videos at 9.93 predict-minutes, ~20x headroom in a 9 h kernel.
- **The ensemble idea is dead for the donors we hold (E33).** v1329f loses 98.9 % of the 961 genuinely
  contested parents; v50's inner `v1329_submission.csv` is byte-identical to v1329f so it is not an
  independent third voter. Final-2 = 0947 + v1329f as private-LB VARIANCE hedging only, never a merge.
- **The one live, un-run lever:** each fork carries ~3–4 k nodes the other has no node for within 5 µm
  (0947-only 3118, v1329f-only 3806), and **both sets are fully track-embedded — mean degree 1.73/1.76,
  ~0 % isolated — so neither is FP noise.** Which set is real is undecidable from submissions alone and
  needs GT: run v1329f's detector on holdout20 and score the unique nodes against labels. That is the next
  GPU run, and it is the only thing left that could plausibly be worth ≥1.5 %.

## NEWS 2026-09-18 — frontier went public; the tree's "no public >0.927" premise is OBSOLETE

- Public kernels now plateau at **0.947**; alfonso1799 V50 claims **0.9605** (pilkwang UNet+node-transformer
  primary, temporal-unet3d secondary, DeepCenter veto, v1327 w3 real-model, tracksdata ILP + heavy post-proc).
  Its guard report admits `leaderboard_feedback_used_for_configuration: True` → overfit-risk on private.
  Forked unchanged as `celltrack-public-v50` / `celltrack-public-0947`; LB pending (watcher auto-submits).
- **Consequence:** every LEVEL 3–4 verdict below was measured on OUR model family. The frontier model is a
  different (stronger) substrate — our replicate-first rule now points at *its* weights + trainer
  (`external/frontier_ds/…support-pack…/repo`), not our pmkf/HOCT stack. Build-past levers go on top of it.
- **Open build-past (bd, P1):** synthetic CC0 pretrain → real fine-tune of the frontier model (forum +0.012–0.018).
  Not a repeat: prior synth nulls were our generator / frozen UNet / from-scratch joint. Synth frames carry
  ~300–560 labelled nodes vs ~17 sparse on real → supplies the complete-candidate supervision
  ([[celltrack-pmkf-trains-gt-sparse-not-detected-crowd]]) our real labels lack. Held-out: 20 movies
  (7×44b6, 13×6bba) — the public weights trained on all 199, so both arms must retrain.
- **Ruler (09-19):** the public weights, LEAKED on holdout20, with default predict (`--use-ilp`, thr 0.99) score
  **0.8842** (edgeJ 0.889, divJ 0 of 17, nodeR 0.992). They clear the 20 movies' labels but score ~0.06 below
  the LB plateau, so the local number is a relative ruler only: compare the A/B arms with each other, never
  with the LB. Divisions score zero at default settings — the post-proc stack carries the LB gap.
- **V50's last layer is public-LB surgery (09-19).** After the V1329 foundation it (a) linearizes EVERY fork
  (keeps the nearer child — a general "predict no divisions" policy) and (b) re-adds ≤2 divisions per ≥30k-node
  movie through ~10 thresholds fit to one public event (`6bba_05db0fb1`, P=20025→D=20865). (b) is overfit but
  tiny; (a) is the real private-LB bet. Hedge fork `celltrack-public-v1329f` = foundation only (after
  ozermehmet's candidate pair); the V50−v1329f LB delta prices the surgery. Final-2 pick = one of each.
- **V50 fork scores 0.907 on OUR LB (09-19), below the banked 0.924.** The claimed 0.947/0.9605 did not reproduce.
  The visible-test run completed cleanly (V1329 took 25 min, all hashes matched, no traceback). But the V1327
  adapted-detector guard fell back to the untouched V1290 primary on 98/100 frames of `44b6_0b24845f` (median
  retention 0.76): per its own guard, the headline detector is unfit on a dense 44b6 movie. The hidden-run log is
  not downloadable, so the cause of the drop is not established.
- **v1329f (foundation, no surgery) = 0.939 on LB, the NEW CHAMPION (+0.015 over 0.924) (09-19).** Same notebook
  as V50 minus the surgery cell, so the surgery's measured price is **−0.032**. Fork linearization (drop every
  division) plus the 2-division rescue loses 0.032: the LB rewards the foundation's divisions, and "predict no
  divisions" is refuted on the LB, not just overfit. Final-2 = v1329f + one decorrelated candidate (not V50).
- **Why the surgery looked good upstream: it is fitted to the 4 visible test movies (09-19).** Those movies ship
  with GT under `train/`, so `tools/score_submission.py` scores kernel CSVs locally. V50 = **0.9605**, v1329f =
  0.9263: adjJ is equal (0.927 vs 0.926), and the whole +0.034 is division_jaccard 0.33 vs 0.0. That is one
  division TP out of **3 GT divisions** on the visible set. So V50's "0.9605" claim is a 3-division overfit that
  reverses on the hidden movies (−0.032). The visible-4 score cannot rank division policies. It also cannot tune
  the family's other knobs, because the pack's weights trained on all199, which includes these movies. The LB
  is the only clean ruler for this family.
- **thtennant forked 0947 three ways (09-19, unscored).** All three are post-process edits on the same "0.939 base +
  holdout-selected post-process" notebook (`research/frontier_kernels/thtennant_*`):
  - `gapfill` bridges gaps ≤3 frames with the detector's sub-threshold peaks (score ≥0.5).
  - `readmit` re-adds discarded peaks (score ≥0.965) within 4 µm of an open track end.
  - `divprec` tightens the sister symmetry τ from 0.6 to 0.4.
  The first two add recall, but node recall is already about 0.98 on the visible movies. So each is priced at
  ≤0.01, below the floor. Probe on the LB only once 0947's own score lands; without the base score, a variant's
  score can't be read.
- **0947 fork = 0.947 on LB, the NEW CHAMPION (+0.008 over v1329f, +0.023 over our own 0.924) (09-19).** It is
  reyhanksatria's notebook with three input paths changed; the model is Pilkwang's pack. So the gain is upstream's,
  not ours. Now that the base has a score, thtennant's `gapfill` and `readmit` variants are pushed as LB probes
  (`kaggle/kernels/celltrack-public-0947-*`). Final-2 default = 0947 + v1329f.
- **Forum 741749 (hikaggler, own 0.939): node-count calibration predicts LB, model ranking doesn't; synth pretrain is
  the one lever (09-19).**
  - The N_pred ratio term moved their LB every time. A one-point recall change did not.
  - Post-proc settings chosen locally carried over to the LB. The local ranking of two trained models did not.
  - Their synth recipe: CC0 José Freitas sequences, only 497 of 2174, pretrain detector + linker 80 epochs, then
    60 epochs on real. 24-video hold-out: 0.9146 → 0.9269 (+0.012).
    - Using all 2174 sequences at equal gradient steps scored −0.005.
    - Pretraining pushes N_pred up and doesn't help divisions.
  - Our synth A/B (from_synth 0.9063 < scratch 0.9127, trainer acc·recall) is **INCONCLUSIVE, not refuted**. It used a
    different recipe: 20 pretrain + 10 real epochs vs their 80 + 60, and 0.006 on the trainer metric is noise.
  - Re-run at their recipe: `synth_pre80` launched 09:03Z (580 seqs, ≈3.5 h). The 60-epoch real fine-tune (≈20 h) is
    gated on the pretrain converging.
- **New public forks (09-19), unscored:**
  - thtennant's det096 line, vs 0947:
    - DET_THRESHOLD 0.965 → 0.96, which raises N_pred, the term hikaggler says moves the LB.
    - frame cache 48, ILP timeout 1200 s, repair deadline 27000 s. "fast-tight60" is runtime engineering, not a model change.
    - gapfill-det096 adds a flow-seeded motion relink (`MOTION_RELINK_FLOW_*`).
  - noisyislands trains a TabPFN division classifier (divisions ≤0.002 LB per hikaggler).
  - newwang12 `biohub-v1-grouped` (10 votes), arnav170 `biohub-reid3`, ghazarosbarseghyan91 `biohub-dae-alpha-0-17`:
    V9-lineage hosts self-citing 0.939 — host LB is irrelevant, they are ELEMENT DONORS (inventory below).
  - beraterolelk `0-947-lb-biohub-deepcenter-ilp-tracker`: same 3 pinned weights; DET 0.96 + relink TIGHT 5.5 +
    PPSWEEP margin 0.0005; titled 0.947 = TIE with 0947 → det096 knob alone likely sub-floor.
- **ELEMENT INVENTORY vs 0947 (harvest elements, not notebooks; 09-19).** Diff = `BIOHUB_*` keys + new `def`s per
  public kernel (scratch `elements.py`). Each row = transplant candidate into the 0947 base:
  | # | element | donor | port | status |
  |---|---|---|---|---|
  | E1 | gap-fill from low-threshold detection pool (`fill_gaps_from_low_detections`, LOWDET 0.3, ≤3% added) | thtennant gapfill | code | **LB 0.947 = TIE with 0947** (probe 56354350) → sub-floor, not a lever alone |
  | E2 | flow-seeded motion relink (`MOTION_RELINK_FLOW_*`, K 12, radius 40 µm) | thtennant gapfill/readmit/divprec | code | inside E1/E3 probes |
  | E3 | readmit discarded detections (score ≥0.965, r 4 µm) | thtennant readmit | code | **LB 0.946 = TIE/−0.001** (probe 56354352) → sub-floor |
  | E4 | division precision: sister symmetry τ 0.6→0.4, sister-min 0 | thtennant divprec | knob | untested |
  | E5 | learned re-ID appearance descriptors as pair feature (sweep picks REID_WEIGHT 4.0) | arnav170 reid3 | code | **DONOR-NULL** (its own `reid_report.json`, 5656 held-out sources): GBM top-1 0.98568 = transformer top-1 0.98568 exactly; perm-importance tf_prob 0.202 AUC, every descriptor block ≤0.0014 ⇒ appearance adds nothing past tf_prob. Sweep adjJ +0.004 (8 vids) sub-floor; LOCAL re-run reproduces it (base 0.9260 → best reid8_tight55_relaxed9 0.9312, +0.005; proxy spread driven by 8-video divJ noise). Don't port. |
  | E6 | packet grouping post-process (`grouped_postprocess`, xy radial shell profile) | newwang12 grouped | code | **DONOR-UNMEASURED**: its `submission_audit.json` = `graph_schema_pass`, `quality_validated: false`; `v9_division_gate.json` only checks division-COUNT retention (101/102 = 0.99) on 4 test clips — no score evidence either way. Would need a slot. |
  | E7 | test-time denoising-AE prefilter (30 steps, α 0.17) | ghazarosbarseghyan91 dae | code | **DONOR-INCONCLUSIVE** (its validator vs reid3 `base` on the 4 shared stems): adjJ 0.8811 vs 0.8858 (−0.005), per-stem ±0.06, spurious −15 %, missed +1.25 ⇒ no carry evidence; low priority |
  | E8 | sub-voxel centroid refinement (`refine_all_centroids`) | evgendvorkin 0.927 | code | **MECHANISM-NULL, not submitted**: metric `DistanceMatching(max_distance=7.0 µm)` vs refine shift ≤ sub-voxel (<1 µm), and both donor + 0947 round output to int voxels. Ported anyway (`kaggle/element_transplant.py`, variant `celltrack-public-0947-refine`, smoke-tested) — the CLI is the reusable transplant harness. |
  | E9 | DET 0.965→0.96 | thtennant det096 / beraterolelk | knob | beraterolelk tie ⇒ sub-floor |
  | E10 | runtime budget: frame cache 48, ILP timeout 1200, deadline degrade | thtennant fast | code | ENABLER — headroom for stacking E5–E8 in 9 h |
  | E11 | TabPFN division classifier | noisyislands | code | low ceiling (divisions ≤0.002); donor log shows only `LightGBM outer accuracy: 1.000` on 1082 division rows (saturated ⇒ likely leaky/trivial labels), no track score ⇒ UNMEASURED, don't port |
  | E13 | XGBoost division classifier (13 GT-only features) | noisyislands xgboost-division-events | code | same family as E11; divisions ≤0.002 LB → skip |
  | E14 | linker 'association MLP' on candidate distance only | noisyislands linker-association-mlp | code | pilot, weaker than 0947's tf_prob (E5 showed even rich descriptors add 0 past tf_prob) → skip |
  | X1 | SCORER EXPLOIT, not an element: fake hub node t=−1000 → roots of top-1400 components + 5 chained off-image (−10000) fake divisions appended to submission | codezzzsleep 095-owned-validation (09-19) | — | **WON'T PORT**: games the metric, no tracking change; same family as the kirneo off-image hack (0 divJ for us); a host fix would void it on private |
  | E15 | none — verbatim fork | pawanmali zhincez947-fork-v1 | — | code + env + data sources byte-identical to our 0947 base → nothing to harvest |
  | E16 | none — strict subset of 0947 (single seed, no TTA, no DeepCenter veto; thr 0.985, tighter repair caps) | binasalama learned-unet-transformer-ilp-gap-recovery (09-19 re-pull, 0-line diff vs stored) | none stated | inference-only "closing note" of its line → nothing to harvest |
  | E17 | none — param deltas only: GAP_CLOSE 5.8 (0947 5.0), TIGHT 5.5 (6.0), DeepCenter div veto OFF (0947 ON) | gautiermarti deepcenter-unet3d (v29, same 3 mounts as 0947) | self-reported val n=4 0.943; table stops at v8 LB 0.934 | same lineage as 0947; its v30 "TTA link-logit fusion" is a note, not code; tight55 already in detthr probe |
  | E18 | 402-epoch support-pack edge predictor (vs pilkwang 50ep), thr 0.99, ILP app 1.0 / disapp 2.0 | yongjilyu ct-sp402 | none | the one WEIGHTS donor seen (longer-trained model = our model gap) but `yongjilyu/biohub-ct-inference-pack` is PRIVATE (API 403) → unharvestable; watch for it going public |
  | E19 | none — older 0947 ancestor (det 0.960, ILP div 1.2, safe-div thr 0.25) | chukkkk lb-942 | 0.942 | superseded by 0947 |
  | E20 | none — no model mounts, kinematic heuristic tracker | avikdas567 3d-kinematic | none | below learned base → skip |
  | E21 | = E2 (flow-seeded relink) + tight55 + safe-div 0.25 + validator off | thtennant flow2-v1 (09-19 re-list) | env | already inside gapfill 0.947 tie → skip |
  | E23 | leonixis v1/v2/v3 pipelines: organizers' unet_transformer 40-ep from scratch (local 0.780, LB strict 0.871–0.913); p400/p314159 = pilkwang's exact SHAs | leonixis v1-infer (09-19 re-list) | weights | same recipe, far weaker than the pilkwang pair → no seat → skip |
  | E22 | none — pseudocode scaffold (ResUNet3D + MCMF linker + motion GPT), driver commented out, no weights/score | umarshad notebook82c6959503 | none | from-scratch, unrunnable → skip |
  E12 probe (14:33Z): reid3 own divdiag — `retro6_nnk2` leaves OWNED at 6/12 (retro does not reach owned cases); steal never
  run by donor. Sized: owned 6/12 FN, divJ 0.23 → ≤~0.6 ⇒ ≤ +0.037 proxy IF steal is clean (adj cost unknown). Probe =
  `kaggle/donor_probe.py` (re-runs donor's OWN validator sweep with our configs, no slot) → kernel
  `celltrack-reid3-steal-probe` v1: steal ratio 1.5/2/3, owner-um 3/4.5, no-reattach, composed w/ reid3 selected.
  Gate: div_tp up, div_fp flat, adj loss ≤0.0005.
  LOCAL VALIDATOR (15:00Z): `kaggle/local_kernel.py` runs any kernel on the 5090 (remaps /kaggle paths; `--spec` =
  donor_probe candidates; `--env KEY=VAL` pins env). CORRECTION: kernel re-materializes its repo + predict wipes output
  ⇒ resume states did NOT cache GPU predictions; `kaggle/local_predict_cache.py` now caches per-video geffs keyed by
  argv+BIOHUB_* env+weights fingerprint (`--candidate-shard I/K` splits the CPU PP sweep across processes on one cache).
  Test predict 4 videos = 2.8 min locally. **FIDELITY PASS (15:20Z):** local base over the donor's 8 stems = donor
  `validator_results.csv` (division counts identical, |Δadj| ≤ 5e-5) ⇒ local PP ranking is trustworthy.
  GPU profile: predict = U-Net bs=1 → transformer bs=1 → CPU ILP serial per video ⇒ GPU idles during ILP (bursty
  util). Fix = in-process pipelining (ILP of video k on CPU thread while GPU runs k+1) + batched windows — also Kaggle
  runtime headroom for stacking. **PIPELINED PREDICT (16:00Z, `local_predict_patch.py`):** worker-thread ILP +
  frame prefetch + numpy edge candidates → geffs BIT-IDENTICAL to cache (4/4); wall 3.01 vs 2.94 min = no win
  because GPU side (~120 s/4 videos, bs=1 encodes) now dominates. Batched 7-view dihedral TTA is NOT exact (node
  counts differ 0.01–0.1%/video — cuDNN algo per batch size flips threshold-edge detections). **Timing 16:32Z (4 test videos, CPU sweeps running
  alongside — same load for both arms):** fp32 bs=1 120.2 GPU-s · batched-TTA fp32 151.5 (+26%, SLOWER — bs=7 at 64³
  loses to bs=1; keep off) · bf16 autocast + batched 97.2 (−36% vs batched, −19% vs baseline). bf16 geff node counts
  differ −0.03…+0.15% vs fp32 — same order as batched noise; score effect unmeasured. Speed only pays if it buys an
  ensemble seat inside the Kaggle runtime. **bf16 unbatched, N=20 (19:45Z): adj 0.9025 vs fp32 0.9022, 23.8 vs 29.2
  GPU-s/video (−18.5%) → accuracy-neutral.** But the Kaggle kernel runs on T4 (sm75, no native bf16), so the shippable
  form is fp16 autocast. **fp16, N=20 (20:44Z): adj 0.9018 (−0.0004), missed GT 446, 16.8 GPU-s/video (−42%), no
  NaN → accuracy-neutral.** Shipped as `celltrack-public-0947-fp16` (`kaggle/amp_variant.py`) — the Kaggle runtime vs
  0947's 6286 s says how much room it frees for a third member. **Kaggle fp16 run (22:16Z, T4, submitted):
  test predict 9.68→5.15 min (−47 %), validator predict 16.4→7.6 min (−54 %), wall 6286→5590 s (−11 % — the visible
  run is install + CPU post-proc bound). The GPU half is what a third member would buy time from, and fp16 nearly
  halves it; the hidden run has many more videos, so its GPU share (and the saving) is larger.**

  **E24 — the kernel is 77 % SELF-VALIDATOR, not model. Stage profile of the fp16 log (5590 s total):
  install+weights 313 · test predict 301 · test PP + write submission 639 · validator predict 499 ·
  validator base PP 440 · 7 candidate PP sweeps 3034 · validator rewrite 352. The shipped submission
  costs ~1250 s; the held-out self-validation block costs 4325 s and its only product is choosing one
  PP candidate — `tight55` (`MOTION_RELINK_TIGHT_UM 5.5`) in BOTH the fp32 and fp16 runs, with every
  candidate inside ±0.002 adj (noise). Pinning that override and setting `BIOHUB_VALIDATOR_ENABLE=0`
  reproduces the same config by code (the rewrite applies exactly that one key over the env base, and
  the drift guard does not cover it), so `celltrack-public-0947-fast` (`kaggle/env_variant.py`) should
  emit a byte-identical submission in ~1600 s. A kernel RUN costs no submission slot, so this is a free
  probe: pushed 22:36Z, verify by diffing its `submission.csv` against 0947's.

  **E48 — THE DIAGONAL, MEASURED: NEITHER SCORE WE HOLD REACHES E47'S OPERATING POINT.**
  E47 argued that a threshold fails because its decision variable is intensity-ordered.
  `kaggle/prune_score_quality.py` measures it on n=723133 detections (real 12482, FP 710651), scoring the
  only two rankings the cache and volumes provide, at the collateral E47 requires (the cut is placed on
  the real detections' own quantile, so collateral is exact by construction). **Center-voxel intensity:
  AUC(real>FP) 0.5221**, and FP-recall **equals collateral at every cut** — 0.0222 at 0.02, 0.0476 at
  0.05, 0.1011 at 0.10. That is the q = r diagonal exactly, which is what a coin-flip AUC means in
  operating terms, and every point on it is a loss by E47's grid: the recovery stack sat there by
  construction. **Best incident affinity: AUC 0.6380**, FP-recall **0.0578 against the 0.75 required —
  13× short.** Pre-registered as a failure, with the reason stated first: the E46 lookalike's signature
  is HIGH affinity to the source (it sits where the cell was), so a score that ranks by how much the
  tracker wants to link cannot demote the one detection it most wants to link. Strong reading:
  real-vs-FP is not recoverable from anything downstream of the detector — the dense-model-gap
  conclusion from a new direction, a convergence to cite rather than a discovery to claim. For `g89y`
  this raises the bar from "off the intensity axis" to a number: FP-recall 0.75 at 2 % collateral, which
  every future prune proposal should be asked for FIRST — it is cheap and it killed two scores in an
  afternoon. **Caveats:** the cache carries no per-detection score column (`coords` = `[t,z,y,x]`), so the
  DETECTOR score the recovery stack actually thresholded is refuted only by analogy — get it on the next
  GPU pass, do not wake the GPU for it; center-voxel is not integrated intensity (a better feature moves
  0.5221 somewhat, not to the 0.9-plus the operating point implies); our tunet substrate, FP field 0.972.
  Detail: `interpretations/celltrack/2026-09-20_e48_no_score_we_have_can_prune.md`.

  **E47 — THE PRUNE AXIS IS PRICED: THE CEILING IS BIG (+0.1139) AND COLLATERAL IS WHAT SPENDS IT.**
  E46 pointed at bd `g89y` because the confusor is an FP. "Prunable in principle" is not a licence to
  build a GPU detector pass, so `kaggle/prune_operating_point.py` prices it on CPU from the cache first.
  ORACLE (delete every unmatched detection): band 0.1195 → 0.0056, **95.3 % rescued, net +0.1139** offline
  mutual-best — but it deletes 97.2 % of detections and keeps 0.37 % of candidate edges, so it licenses
  nothing alone. The grid is the decision: **collateral enters at ~2× its own size** (a deleted cell
  takes the true edge on BOTH sides), so `q=0.05` nets only **+0.021 even at perfect FP-recall**, while
  `q=0.02` needs FP-recall ~0.5 to break even and earns **+0.035 at 0.75**. The spec is ASYMMETRIC — an
  order of magnitude more careful with real cells than thorough with FPs. **This retro-explains the
  LB-refuted recovery stack (≤0.900) by mechanism:** a threshold is intensity-ordered, so it buys recall
  and collateral on the SAME axis at ~1:1, and E41 puts the at-risk real cells precisely there (2.4×
  fainter, 46.9 % bottom decile) — it lands on the losing diagonal by construction, not by mistuning.
  `g89y` survives ONLY because hierarchy selection picks among nested contours by consistency rather than
  a global intensity cut; that is now the **cheap precondition to test before the build** — does its
  survival decision correlate with intensity? +0.035 clears the 1.5 % bar with room and the oracle leaves
  +0.11 of headroom, but this is the offline association proxy on OUR tunet caches (FP field 0.972),
  not the LB. Detail: `interpretations/celltrack/2026-09-20_e47_what_a_pruner_must_achieve.md`.

  **E46 — THE BAND IS NOT APPEARANCE-BOUND, IT IS PAIRWISE-BOUND; THE CONFUSOR IS A NEAR-STATIC FP
  LOOKALIKE.** *Replication, not discovery:* `celltrack-confusor-pairwise-unresolvable-needs-multiframe`
  (2026-08-23) already had this — appearance top-1 0.115 vs 0.260 chance, source correlating better with
  the rival (+0.808) than the true successor (+0.540), "only global multi-frame breaks it". It was
  MISSING FROM THE MEMORY INDEX, so the prior-art check did not surface it; index repaired, and the cost
  was re-deriving a known result. E46 adds power (n=37 one movie → n=1226 across 20), a
  displacement-matched control, and one correction: the rival is an FP, not a cell.
  Re-tested at n=1226 with the E43 denominator, CPU-only, raw NCC on cell-sized cubes
  (`kaggle/outranked_band_appearance.py`): band picks true **0.4853**, mutual-best control **0.8230** —
  so appearance works, and fails exactly where it is needed. Not fast-cell decorrelation either: at
  MATCHED displacement ≥3.46, mutual-best holds **0.7182** (n=330) while the band falls to **0.4345**
  (n=695), *below chance*. Below chance is the mechanism — the rival looks MORE like the source than the
  true target does, which composes with E45's "true edge farther in 0.7984" into one picture: the
  confusor is a near-static lookalike near where the cell was, while the true cell moved far and changed.
  **Every local pairwise cue selects the same wrong target; three of four now measure below chance.** So
  a learned embedding on the same pairwise view inherits the geometry — what separates true from
  lookalike is that the lookalike has NO CONTINUATION, a tracklet-level fact. **And the lookalike is an
  unmatched FP in 0.9698 of contests (n=992, only 30 are real GT-matched cells)** — so it is PRUNABLE,
  correcting the 2026-08-23 read that it was a distinct slower cell. This closes the pairwise axis on
  mechanism and points at bd `g89y` (segment-SELECTION ILP — structural pruning, since the threshold
  route is LB-refuted at ≤0.900) with the two-pass tracklet ILP (bd `kiw1`) as the other half of the same
  fix; both are jointly-necessary 0.945 levers. Detail: `interpretations/celltrack/2026-09-20_e46_appearance_on_the_outranked_band.md`.

  **E45 — THE OUTRANKED BAND IS A FAST-CELL POPULATION (5.7× enriched), AND MOTION IS DEAD ON IT AT THE
  ORACLE.** E44 left the 11.95 % outranked band as the one live reading of the fragmentation block. This
  characterises it and then prices the cue it appears to ask for. All CPU, on the same cache.

  *It is a displacement population.* Against a marginal outranked rate of 0.1195:

  | | n | mean disp | q50 | q90 |
  |---|---|---|---|---|
  | ALL true candidate edges | 11879 | 1.726 | 1.414 | 3.464 |
  | MUTUAL-BEST | 10459 | 1.441 | 1.414 | 2.828 |
  | **OUTRANKED** | **1420** | **3.831** | **3.742** | **5.831** |

  **P(outranked | displacement > the all-true q90 of 3.46) = 0.6769 against a 0.1195 marginal — 5.7×.**
  The affinity head's ranking failure IS a fast-cell failure: where the cell moves far, a nearer rival
  takes the link.

  *So a distance prior is worse than nothing.* In the outranked band the true edge is **farther** than
  its rival in 0.7984 of cases (true median 4.12 vs rival 2.24 voxels); among non-tied pairs distance
  picks the true edge **0.1767 of the time against 0.5 chance**. Adding a distance cost to the linker
  would actively select the confusor. This independently reproduces the below-chance signature already
  recorded for motion-in-ILP, now localised to the band that actually matters.

  *And velocity does not rescue it — tested at the ORACLE.* If the cell merely moved, the cue should be
  distance from `x_t + (x_t - x_{t-1})`, built here from **GT** positions, so it upper-bounds any learned
  motion feature. On the 950 band edges whose source has a GT predecessor: oracle velocity picks the true
  target **0.6400**, static distance **0.5747** in the same frame — velocity buys **+0.065** — with a
  median margin improvement of **+0.041 µm** and `frac>0 = 0.5053`, i.e. a coin flip on WHICH edges it
  helps (the mean +0.645 is a tail artifact). **Caveat, and it runs in my favour:** this frame measures
  from the GT source position to detections, and the true detection was itself selected as the one
  nearest the GT tail, so it is biased TOWARD the true edge — which is why static reads 0.5747 here and
  0.1767 on the unbiased cache column. The bias being optimistic is what makes the result usable: the
  true ceiling is **at most** 0.64. An oracle that barely separates needs no learned version.

  **Conclusion: the outranked band is not reachable by any geometric cue — not distance, not velocity,
  not at the oracle.** It is an appearance/affinity-quality problem, consistent with
  `celltrack-four-cues-fail-confusor-detection-side` and with the confusor sitting at the voxel
  resolution limit. Do not re-file a motion or distance term for it. Note this also explains the older
  "motion_distance lever aggregate-small" reading without contradicting it: aggregate-small is exactly
  what a real effect concentrated on 11.95 % of edges looks like — here the concentration is real and
  the cue is still dead, so the axis closes on mechanism rather than on dilution.

  **E44 — THE TERMINATION KNOB POINTS THE WRONG WAY. Raising disappearance cost past ~1.5 LOSES GT
  edges, so bd `d5er`'s proposed 3/4/6 direction is refuted on its own mechanism.** E43 put the
  fragmentation block on a solver knob; this prices the knob. The cached candidate graph is pure dt=1,
  so the faithful model is a per-frame-pair global assignment with a constant termination cost per
  unmatched node (costs `-log(affinity)`, termination as a square augmentation, decomposed over the
  connected components *of each frame pair* — components over the whole movie chain all 100 frames into
  one blob and buy nothing). Sweeping `term` over 8 values, CPU-only, no detector pass:

  | term | 0.25 | 0.5 | **1.0** | **1.5** | **2.0** | 3.0 | 4.0 | 6.0 |
  |---|---|---|---|---|---|---|---|---|
  | GT-edge recall | 0.7643 | 0.8455 | **0.8582** | **0.8589** | **0.8581** | 0.8532 | 0.8454 | 0.8293 |

  The curve is **non-monotone with a broad shallow plateau at 1.0–2.0**, and every step above it costs
  recall — 6.0 gives up 0.0296 against the peak, an order of magnitude over the 0.01–0.02 noise floor.
  d5er's hypothesis was that cheap termination (`BIOHUB_ILP_DISAPPEARANCE_WEIGHT=2`) lets the solver
  abandon tracks it should continue, so 3/4/6 would buy edges back. The mechanism **backfires**: with a
  global assignment, expensive termination forces every node to link to *something*, and in a field
  carrying ~980 FP per frame against ~17 GT cells the something it grabs is an FP rival that steals the
  true target. The shipped weight of 2 already sits on the plateau — **this knob is spent, not mis-set.**

  *Both axes degrade together, now measured rather than inferred.* Counting mislinks over the honest
  denominator — links whose two endpoints are BOTH GT-matched detections — gives 14 (0.0013) at term
  1.0, 18 (0.0017) at 1.5, 20 (0.0019) at 2.0, and 43 (0.0042) at 6.0. So climbing from the plateau to
  6.0 costs 0.0296 recall AND multiplies mislinks by 2.4. There is no trade to exploit in either
  direction. (Precision among real cells is near-perfect throughout — ~10.6 k GT-matched pairs linked
  against tens of mislinks — which independently reproduces d5er's original "mislinks only 74".)

  *Two denominators, because the naive ones lie.* The sweep's first `mislinks` column (taken links that
  are not GT edges) sat at 0.98 for every value of `term` and measured nothing — almost every link is
  FP-to-FP and was never a mistake about a real cell; the honest version is the one above
  (`kaggle/termination_cost_sweep.py`). And the recall LEVEL
  (0.859) is not comparable to the real tracker's 0.9475: this model has no distance gate and no
  division term. **The shape across `term` is the result; the level is not.** Caveat as E43: this is our
  tunet cache (node recall 0.697–0.750), not 0947's 0.824.

  *What survives.* E43's 11.95 % outranked band is untouched by this — those true edges lose to a
  **rival**, not to termination, and a termination cost cannot flip a swap. That band, with its median
  margin of 0.0985, remains the one live reading of the fragmentation block, and it is an
  affinity-*ranking* problem (global solver, or a better head) rather than a cost knob.

  **E43 — FRAGMENTATION IS A *SELECTION* FAILURE, NOT A CANDIDATE FAILURE. The 4.46 % block is on the
  table and the solver declines it — so it is solver-reachable, and reachable WITHOUT A DETECTOR PASS.**
  E42 closed the detection axis on the 0947 base, which left bd `d5er`'s fragmentation block as the
  largest addressable population still standing: an edge whose **both endpoints are detected AND matched**
  yet carries no predicted link. That is an omission, not a confusion — but "omission" hides two bugs
  with opposite fixes, and nobody had separated them:

  - **CANDIDATE** — the pair was never proposed. Then no solver knob reaches it; the fix is a wider gate.
  - **SELECTION** — the pair was proposed, with an affinity, and the solver preferred to terminate. Then
    the disappearance/termination cost reaches it, and the affinity head's *ranking* bounds how much.

  The `real_scratch` cache settles this for free: it stores **487238 proposed edges with their
  affinities**, so the candidate graph the linker saw is on disk. No GPU, no detector re-run.
  Instrument: `kaggle/fragment_selection_gate.py` (CPU, ~1 min, 20 movies, n=12346 GT edges).

  1. **The gate passes decisively — it is SELECTION.** Of GT edges whose endpoints both match at the
     official 7 µm ruler (11938 = 96.70 %), **99.51 % already have a candidate edge with an affinity**.
     Only **59 (0.48 % of all GT edges)** have none, and those are median **11.72 µm** displacement —
     outside the 10 µm gate, i.e. gate-bound and far too small to matter. Candidate generation is NOT
     the bug. Every fragmented edge was on the table.
  2. **The affinity head is mostly right, and wrong by a thin margin.** The true edge is **mutual-best**
     (top-ranked out of its source AND into its target) in **88.05 %** of cases. The **11.95 %** that are
     outranked lose by a **median margin of only 0.0985** (true 0.285 vs rival 0.510). A thin margin is
     precisely what a *global* solver flips and a greedy one cannot — which is the mechanism behind
     `celltrack-linker-is-the-gap` (global ILP broke the 0.902 wall). It also means these are not
     hopeless: they are near-misses, not confident errors.
  3. **The weak tail is real and it is ranked-down, not just low.** **12.48 %** of true candidate edges
     carry affinity **below coin-flip** (q10 = 0.433), and of those only **23.01 %** are mutual-best.
     So a true edge scored under 0.5 usually *also* loses its ranking — low affinity and bad ranking are
     the same population, not two independent taxes.

  **What this licenses and what it does not.** The 11.95 % outranked band is larger than the 4.99 %
  fragmentation rate, so fragmentation sits *inside* it — consistent, and it means the termination cost
  is not the whole story. Raising it recovers a true edge that lost **to nothing**; it cannot fix one that
  lost **to a rival** (that is a swap, and swaps show up as mislinks, of which d5er counts only 74).
  **Caveats that must travel:** measured on OUR `real_scratch` cache (node recall 0.697–0.750), not the
  0947 base (0.824); and the 7 µm matching here is nearest-detection, not the official metric's global
  assignment.

  **E40 — THE DoG UNION IS DEAD ON THE CONDITIONAL, AND E27'S KILL WAS SCALE-BOUND. Five CPU-only
  instruments, no GPU.** E39 left detection as the only axis with a ≥1.5 % mechanism, priced as
  GPU-blocked. It is not: a union with an *independent* detector needs no local card (a Kaggle fork run
  is free), and the gate — does the second detector find what the first missed — runs CPU-only against
  the cached `real_scratch` coords and the train `.geff`s.

  1. **E27's DoG kill was a SCALE artifact, and correcting it does not save the axis.** E27 refuted DoG
     as *preprocessing* at the default 4 µm band; the band is the whole result. GT recall at cell
     separation (2.87 µm): default `(2.83, 4.0, 5.66)` = **0.16–0.30** — E27's number — against a 2 µm
     band `(1.4, 2.0, 2.83)` = **0.62–0.74**, versus the learned detector's 0.75 on the same frames. So
     the learning-free detector is *competitive*, which is exactly what makes the next line decisive.
  2. **But the errors are DEPENDENT, so the union pays nothing.** n=256 GT over 8 movies: cache 0.7500,
     DoG 0.4141, union 0.7891 (+0.0391 marginal). The conditional kills it: **P(DoG finds it | cache
     missed) = 0.1562** against DoG's own 0.4141 marginal — DoG is **2.7× LESS** likely to find a cell
     the learned detector missed than to find an average cell. Shared blind spot, not complementary
     coverage. A union member must beat its own marginal on the conditional; this one is a third of it.
     **Widened 12.5× and it gets stronger, not weaker:** n=3206 over 20 movies, cache 0.6974, DoG 0.4498,
     union 0.7396 (+0.0421), and the conditional **0.1392 against a 0.4498 marginal = 3.2× less likely**.
     Not a small-sample verdict.
  3. **My own explanation for that was then falsified. The missed cells are VISIBLE.** I expected
     signal-absent (both detectors blind to dim cells), which would have closed the single-frame axis
     entirely. Measured local background-subtracted contrast at every GT centre: only **3.0 %** of
     missed cells have no local intensity excess, median contrast 1.29 vs 2.00 for found, and ~64 % are
     as bright as routinely-found cells. Bright, not blind.
  4. **Nor is it a merge with a neighbour.** n=3864 missed: **72.5 % have a detection within 5 µm**,
     median distance **3.83 µm** — but nearest *GT* neighbour is 24.8 µm for missed cells vs 23.2 µm for
     found. Identical. There is no second cell nearby to merge with.
  5. **The matched null says the nearby detection is the cell's OWN.** With ~854 detections in a 104 µm
     cube, 3.83 µm could be chance, so: score GT against detections from frame `(t+50) % 100` — same
     count, same spatial distribution, same density, identity destroyed. REAL nearest median **3.83 µm**
     vs NULL **7.53 µm**; own-frame closer in **0.807** of cases (0.5 = chance); REAL median dz −3.25
     vs NULL +0.00. The association is real.

  **E41 — WHAT THAT MEANS FOR E36, INCLUDING THE PART OF E40 THAT DOES NOT SURVIVE ITS OWN AUDIT.**

  - **Self-correction first.** I read the missed-subset signature (median dz = −3.25 µm = exactly −2
    z-voxels) as a discrete offset. It is **partly tautological**: one z-voxel is 1.625 µm, so
    conditioning on `distance > 2.87 µm` mechanically selects |dz| ≥ 2 voxels whenever the error is
    z-dominated. The per-axis voxel histogram over **all** GT (not the missed subset) is **unimodal at
    0** — dz `−2:10.9 % −1:28.4 % 0:35.0 % +1:15.0 % +2:3.9 %` — so there is **no −2-voxel mode and no
    plumbing bug**. I looked for one and it is not there.
  - **What survives unconditioned is a systematic NEGATIVE-Z BIAS, ~2.5:1** (−1 vs +1: 28.4/15.0; −2 vs
    +2: 10.9/3.9), mean dz −0.57 µm over all GT. dy/dx are near-symmetric by comparison. This is real
    and not a selection artifact — but it is **0.35 of a z-voxel**, and the official matcher is at
    7 µm, so a constant z correction is free to apply and there is **no mechanism by which it delivers
    1.5 %**. Directional at best; recorded, not planned around.
  - **The load-bearing consequence is for E36's MECHANISM, not its arithmetic.** E36 explained the
    0.824-vs-0.998 recall gap as a GT node *borrowing a neighbouring cell's detection* and still
    scoring found at 7 µm. On this cache that explanation **does not hold**: the nearest GT neighbour is
    24.8 µm away, so there is no neighbour to borrow from, and the null shows the matched detection is
    the cell's own, displaced. The 2.87 µm "recall" number is therefore measuring **localization
    scatter, not missing cells** — it is a ruler-choice, and recall at the official 7 µm really is
    ~1.0. **E36's 574-edge (2.5 %) detector-caused edge loss was derived from that mechanism and now
    needs re-derivation before any GPU is spent on it.** bd `lwrr` is re-priced on that basis; bd
    `1ocj` closes negative.
  - **Caveats that must travel with this.** (a) The cache is `real_scratch` = OUR tunet detector, not
    0947; this cache reads 0.697–0.750 where E36 reads 0.824, so the config differs and extending to
    0947 is an inference, not a measurement. (b) REAL and NULL **mean** dz are near-identical (−1.19 vs
    −1.27) while only the medians separate — the mean does not discriminate here and I have not
    explained why. (c) Every displacement discussed is 3–4 µm, i.e. **inside** the official 7 µm
    matcher, so nothing here claims a score effect on its own.

  **E41(b) — SELF-CORRECTION WITHIN THE HOUR: E36's SIZE SURVIVES. I OVERCLAIMED. The mechanism was
  wrong; the number was not.** E41 said the 574-edge (2.5 %) lever "needs re-deriving", implying it would
  shrink. Re-derived (bd `og4x`, CPU-only, same cache): a GT edge counted dead when either endpoint has
  nothing within the ruler —

  | ruler | GT nodes with nothing in reach | GT edges killed |
  |---|---|---|
  | 2.87 µm | 3864 / 12777 = 30.2 % | 4976 / 12346 = **40.3 %** |
  | 5.00 µm | 1062 = 8.3 % | 1502 = 12.2 % |
  | 7.00 µm (official) | 288 = **2.25 %** | 408 = **3.30 %** |

  - **40.3 % edge loss is impossible against a 0.92 LB**, which is the cleanest possible proof that 2.87 µm
    is the wrong ruler — exactly what E41 argued.
  - **But at the ruler that scores, the loss is 3.30 %**, on a cache reading recall 0.697–0.750 against
    E36's 0.824. A better detector lands lower, i.e. **right on E36's 2.5 %.** So E36's *size* is
    independently reproduced; only its *story* (borrowing a neighbour's detection) was wrong.
  - **And this SHARPENS the lever rather than closing it.** The 2.25 % of GT nodes with nothing within
    7 µm are genuinely, officially undetected — no displacement explanation available, since the ruler is
    the official one. That is a real recall gap with real material behind it, and it is the same
    population E40 measured as only 2.1 % blind at 10 µm. Detection stays the live axis, still
    GPU-priced, still parked. bd `lwrr` is therefore **re-instated, not retired** — with the correct
    population (nodes blind at 7 µm, n=288 here) rather than the 2.87 µm population (n=3864) that is
    13× larger and mostly localization scatter. Targeting the wrong one of those two is the actual
    mistake E36 would have caused.

  **E42 — THE 3.30 % LEVER IS REAL, AND IT LIVES IN THE FAINTEST DECILE — WHICH IS THE ONE LEVER
  ALREADY LB-REFUTED. The detection axis closes on an ARGUMENT, not on "the GPU is parked".** E41(b)
  localised the officially-costly population to the 288 GT nodes blind at 7 µm. Contrast measured at
  those centres against an equal-sized found sample **from the same frames** (so illumination, depth and
  movie are controlled):

  | | blind at 7 µm (n=288) | found, same frames (n=288) |
  |---|---|---|
  | contrast q10/25/50/75/90 | −0.02 / 0.12 / **0.29** / 0.66 / 1.18 | 0.27 / 0.42 / **0.70** / 1.27 / 1.91 |
  | no local intensity excess | **10.8 %** | 0.7 % |
  | at/below found's 10th pct | **46.9 %** | — |
  | touching the volume border | 9.7 % | — |

  1. **These are a different population from E40's.** The 2.87 µm "missed" cells were only 3.0 %
     excess-free at median contrast 1.29 — bright, and displaced. The 7 µm blind cells are 10.8 %
     excess-free at median **0.29**, i.e. **2.4× fainter than what the detector finds in the very same
     frame**. So the two rulers are not two readings of one gap; they name two unrelated failures, and
     only this one costs score.
  2. **Material exists — 89 % do have a local excess — so this is a detector SENSITIVITY gap, not
     blindness.** But 46.9 % sit in the bottom decile of what the detector already accepts, which means
     reaching them is a **threshold** move, not an architecture move.
  3. **And the threshold move is the one thing already refuted on the real board.** The recovery-stack
     arm is LB-refuted at ≤0.900 precisely because a threshold low enough to recover faint cells starves
     precision, and the candidate distribution already carries ~980 FP/frame against ~17 GT/frame with a
     +26 % surplus sizing a 5.5 % swap. Recovering 3.30 % of edges by admitting the faintest decile buys
     them at an FP rate that the same measurements say costs more than it pays.
  4. **What this does NOT close.** A detector that separates these cells *without* lowering the
     threshold — better sensitivity at equal precision, i.e. a genuinely better dense-regime model — is
     untouched by this and remains the standing frontier-replication bet. What closes is the cheap
     version: no threshold sweep, no recovery stack, no post-hoc recall pass on the 0947 base is owed a
     card. bd `lwrr` closes on that argument rather than on availability.
  5. **Honest gap in the evidence:** the cached `.npz` carries coords and edges but **no confidence**, so
     I could not draw the actual precision/recall trade at the threshold that would admit these 288 —
     the FP cost above is carried over from the prior FP-competition and recovery-stack measurements,
     not measured here. That is the one thing that would make this quantitative rather than argued.

  **E37(d) — THE REJECT HISTOGRAM, MEASURED. Pair existence is the binder, the peak-side gates are
  second, and loosening them takes the filler to its ceiling — where it is still ~1 %.** Both free
  instrumented kernels returned (00:54Z). `gapdiag` (stock gates) emits a submission with node and
  edge counts **identical** to `-gapfill` (123232 / 119036), so the instrumentation is score-neutral
  and the fork is faithful. Summed over the four test movies, at g=3 (the loosest gate):

  | | gapdiag (stock) | gaploose (loosened) |
  |---|---|---|
  | dangling ends considered | 5030 | 4873 |
  | **ends with NO start inside the gate** | **2740 (54 %)** | **2703 (55 %)** |
  | in-gate pairs (raw, many-to-many) | 1322 | 1130 |
  | rejected by the context filter | 717 | 619 |
  | rejected: no peak chain exists | 577 | 333 |
  | **pairs accepted at g=3** | **27** | **148** |
  | filler total: nodes / edges | 167 / 240 | **814 / 1165** |

  1. **E37(b)'s offline ceiling is confirmed by the kernel's own counters: 54 % of dangling ends
     have no candidate restart inside the euclidean gate at all**, three frames out. No knob reaches
     those; the partner is not there to be found.
  2. **Among the pairs that DO exist, the peak-side gates were genuinely binding** — loosening score
     (0.5→0.35), radius (3.5→5.0 µm) and synthetic tolerance (0→1) cut `chain_none` 577→333 and took
     accepted g=3 pairs 27→148, a **4.9× filler yield** (240→1165 edges). So the named suspect was
     real; it was just never the ceiling.
  3. **And it lands exactly on E37(b)'s predicted ceiling, which is the point.** gaploose's total
     change over base is +1693 nodes / +1894 edges = **1.60 % of predicted edges**, against the
     predicted perfect-filler cap of ~1870 edges / 1.6 %. The mechanism is now exhausted by
     construction: there is no third arm here, because there is nothing left to collect.
  4. **Corrections to E37 as first written:** the shipped `-gapfill` delta is +424 nodes / **+488**
     edges, not +534; and only **167 / 240** of that is the low-detection filler — the rest comes
     from the single-frame and gap-2 closers, which run in the same stage and which I had folded in.
     The instrumented number is the attributable one.
  5. **Submitted anyway, and deliberately** (slot spent 00:57Z, 4 remaining): the arm sits at its own
     measured ceiling, no competing use existed for the slot, and it settles the one thing the
     artefacts cannot — whether repair yield converts to score at all. Expect a tie or a small loss:
     193 of its nodes are SYNTHETIC straight-line interpolations with no detector evidence behind
     them. A tie would say the filler's edges are right and the axis is simply too small; a loss
     would say the fabricated nodes cost more than the true ones earn.

  **E39 — THE "ONLY LEVER LEFT WORTH ≥1.5 %" PRICES OUT AT ~0.9 %, AND IT CAN BE PRICED WITHOUT
  THE GPU. The fork-unique nodes are 73–87 % LOCALIZATION BLIPS, not missed cells.** The 09-19 NEWS
  block called the fork-unique node sets "the only thing left that could plausibly be worth ≥1.5 %"
  and sent it to GT on holdout20 (bd `d8rs`, a GPU run). It can be decomposed GT-free first: assign
  every node to its own fork's track (connected component of that fork's edges), match nodes to the
  other fork per frame, and report each track's *unmatched fraction*
  (`scratchpad/unique_tracks.py`, CPU, seconds). At MATCH_UM = 2.0 µm:

  | | v1329f-only | 0947-only |
  |---|---|---|
  | unmatched nodes | 7987 | 7311 |
  | **whole track invisible to the other fork** | **162 tracks / 1194 nodes (14.9 %)** | **54 / 415 (5.7 %)** |
  | majority-unmatched | 115 / 980 (12.3 %) | 80 / 560 (7.7 %) |
  | minority (blips inside a track both forks have) | 1992 / 5813 (**72.8 %**) | 1934 / 6336 (**86.7 %**) |

  1. **Most fork-unique nodes are not missed cells — they are the same cell localized differently.**
     A "blip" sits inside a track the other fork also has, and since BOTH forks' edges are all dt=1
     with zero gaps, the other fork did not miss that frame: it placed the node >2 µm away. That is
     a position disagreement, and it costs nothing at the official 7 µm matcher.
  2. **The tolerance sweep proves the split rather than assuming it.** Re-run at the GT NN floor
     (2.87 µm): the blip population collapses 7987 → 5947 and 7311 → 5275, while the whole-invisible
     track count barely moves, 162 → 156 and 54 → 53. A real recall difference is tolerance-
     insensitive; a localization difference is not. The two populations behave as claimed.
  3. **So the union's ceiling is the whole-invisible tracks, and they are small.** 1194 nodes in 162
     tracks is ~1 % of the node table and, at ~(len − 1) edges per track, **~1032 edges = 0.87 % of
     the 118548 predicted** — *if every one of them is a true cell and the union costs nothing
     elsewhere*. The reverse direction is 415 nodes / ~361 edges = 0.30 %. **Both are under the
     1.5 % bar before any GT is consulted**, so bd `d8rs` cannot license a ≥1.5 % move and drops off
     the critical path. (It stays worth running as an *instrument* if the card is ever free: which
     fork's unique tracks are real is the cleanest read we have on which detector to build on.)
  4. **The asymmetry is the interesting residue.** v1329f finds 3× as many whole cells 0947 misses
     as the other way round (162 vs 54), yet scores 0.008 LOWER on the LB. Either those tracks are
     false positives, or v1329f pays for them elsewhere — the same tension the E33 ensemble read
     left open, now with a size on it.

  Rule: **price a GPU run's ceiling from the artefacts it would be scoring before spending the
  card on it.** Recorded 09-20.

  **E38 — fp16 IS PROVEN PRECISION-NEUTRAL WITHOUT ITS LB SCORE, AND THE FINAL-2 HEDGE IS REAL
  AFTER ALL — BUT ONLY ON THE HONEST DENOMINATOR.** Pairwise disagreement of every banked
  submission against 0947, via the instrument that already exists for exactly this
  (`kaggle/tracker_agreement.py`, CPU, seconds — I wrote a duplicate in scratch before finding it):

  | vs 0947 | LB | shared edges / union | a_only | b_only |
  |---|---|---|---|---|
  | `fp16` | pending | **0.9956** | 257 | 268 |
  | `0947-gapfill` | 0.947 | 0.8792 | 7389 | 7877 |
  | `0947-readmit` | 0.946 | 0.8690 | 7618 | 9108 |
  | `v1329f` | 0.939 | **0.8371** | 10356 | 10700 |
  | `v50` | 0.907 | 0.8368 | 10417 | 10671 |

  1. **fp16 shares 99.56 % of the union with 0947 — 257 disputed edges out of 118548.**
     Half-precision is neutral *as a mechanism*, established offline for zero slots. Its pending
     LB score is a formality: anything but 0.947 would indict the ruler, not the model.
  2. **v1329f disputes ~10.4 k of 0947's edges — 16 % of the union.** The final-2 pair really is
     two different graphs, and the hedge is worth its slot (E33's pick stands, unchanged).
  3. **My first pass at this said 1.6 %, and it was wrong in the familiar way.** I measured edge
     agreement *conditioned on node pairs that matched*, which throws away precisely the region
     where the two forks disagree — the nodes one has and the other does not. The conditioned
     number (0.984) and the union number (0.837) differ by **10×**, and the conditioned one is
     the flattering one. Fifth occurrence of the denominator error in this campaign; the tell was
     that I reported a disagreement rate whose denominator was itself selected for agreement.
  4. **Where the forks separate is detection.** v50 and v1329f have identical node counts (123485)
     and agree with 0947 to within 0.0003 of each other, consistent with v50 being v1329f plus the
     surgery cell. Across the table the disagreement scales with the node-set difference, not with
     any association setting — the same conclusion E36 and E37 reach from the other side.

  Rule: **a disagreement rate measured only where two outputs already agree is not a disagreement
  rate.** Compare on the union, and check whether an instrument for it is already in the repo
  before writing a second one. Recorded 09-20.

  **E37 — THE E36-MATCHED REPAIR IS ALREADY BUILT AND ALREADY SHIPPING; IT JUST YIELDS 10 %. The
  0.947 tie of `celltrack-public-0947-gapfill` was a YIELD result, not a neutrality result, and we
  had read it as the latter.** Its `fill_gaps_from_low_detections` is exactly what E36 asks for: it
  bridges a dangling track end at `t` to a dangling start at `t+g+1` through the detector's
  SUB-THRESHOLD peaks, inserting real nodes where a cell was missed. Offline on the two cached
  submissions, CPU-only: the base 0947 graph has **122808 nodes / 118548 edges, every edge dt=1, zero
  gap edges, zero isolated nodes, and 4260 no-parent + 4384 no-child dangling ends**. The filler's
  pool is ample — **39523 free sub-threshold peaks ≥ 0.5** across the four test videos, after
  excluding those already on a node. Yet the fork's submission differs from the base by **+424 nodes
  (+0.35 %) and +534 edges (+0.45 %)** — an order of magnitude below the 1.5 % bar, which is precisely
  why the board returned the same 0.947 to four decimal places. The 3 % add cap (`~3700` nodes) is
  NOT what binds; something upstream of it rejects ~90 % of the dangling pairs. The gate chain, in
  order: the euclidean gate `GAPFILL_STEP_UM × (g+1)` (10 µm at g=1); `context_ok`'s direction test;
  and `chain_for`, which needs a free peak within **`GAPFILL_PEAK_RADIUS_UM = 3.5 µm`** of the
  straight-line sample at **EVERY** missing frame, since `GAPFILL_ALLOW_SYNTHETIC = 0`. E36 measured
  the loose endpoints' nearest prediction at **4.892 µm median — past that 3.5 µm radius**, which
  makes the radius the named suspect rather than a knob picked off a list. Two FREE kernel runs
  (no submission slot, no local GPU) settle it: `celltrack-public-0947-gapdiag` adds a reject counter
  at each gate at stock values, and `celltrack-public-0947-gaploose` runs the same counters with
  `MIN_SCORE 0.5→0.35`, `PEAK_RADIUS 3.5→5.0 µm`, `ALLOW_SYNTHETIC 0→1`. **Rule: a tie against a
  base is only evidence about a mechanism once you have measured how much of the mechanism reached
  the output — diff the artefacts, don't read the score.** Pushed 00:36Z 09-20; bd E37.

  **E37(b) — SELF-CORRECTION, SAME HOUR, AND IT CAPS THE ARM I JUST BUILT. The peak radius is not
  what binds; the PAIRS do not exist.** Asking the base graph directly how many dangling ends have
  ANY dangling start within the euclidean gate `5 µm × (g+1)`, by optimal assignment per frame:
  **g=1 → 50 pairs, g=2 → 316, g=3 → 569, total 935**, against **3469 interior dangling ends and
  3176 interior starts**. So a PERFECT filler at `MAX_GAP=3` could add at most ~935 bridges ≈ 1870
  edges = **1.6 % of predicted edges**, and the shipped fork already banked 534 of them. The whole
  remaining headroom on this mechanism is **~+1.1 %** — below the 1.5 % bar — and loosening the peak
  radius can only ever collect part of it. My "yield is 10 %" reading used the dangling-end count as
  the denominator when the reachable denominator is the in-gate pair count; against 935 the fork is
  already running at roughly half. **The real shape of the loss is worse for post-processing and
  better-aligned with E36: 2500 of the 3469 dangling ends have no plausible restart within three
  frames at all.** Those tracks do not resume — the cell stops being detected for a long stretch, not
  for a frame — so no bridging rule reaches them and no gate widening is licensed (5 µm/frame already
  exceeds the GT median step of 1.82 µm and the 10 µm at g=1 is the max observed GT step). **This is
  the same detector-weights conclusion E36 reached, now with a post-processing ceiling attached to
  it: ~1.1 % is ALL that repair-side work can buy on the 0947 base.** The two free runs still finish
  (they cost nothing and give the exact reject histogram), but `gaploose` is now priced as a
  sub-bar directional arm, not a submission candidate on its own.

  **E37(c) — AND A THIRD OF THOSE ENDS ARE NOT LOSSES AT ALL: CELLS LEAVE THE FIELD.** Dangling ends
  are strongly enriched at the volume boundary against the node population they come from (within 6
  voxels in y/x or 2 in z, on a ~63×254×254 volume): **0.791 vs 0.161** on `44b6_0113de3b`, **0.402
  vs 0.119** on `6bba_05db0fb1`, **0.364 vs 0.201** on `6bba_05b6850b` — 1363 of the 3469 ends, ~39 %,
  are cells exiting the imaged volume, which no tracker should link and no repair should invent.
  The one exception is **`44b6_0b24845f`, flat at 0.21 vs 0.176**: its 1217 ends are genuinely
  INTERIOR terminations, it holds the largest sub-threshold pool (26152 free peaks of the 39523) and
  it took the largest share of the filler's gain (+287 of the +424 nodes). **So the repair headroom
  is not spread over the test set; it is concentrated in one of the four videos**, which also caps
  how much any post-processing change can move a four-video LB score. Correcting E37(b) downward
  with this: the addressable interior ends are ~2100, of which only the in-gate subset is reachable
  at all. Cheap rule earned here: **before sizing a repair, subtract the losses that are physically
  correct — a track that walks out of the volume is not a broken track.**

  **E36 — THE OMISSION RESIDUE IS DETECTION, NOT ASSOCIATION, AND NODE RECALL AT 7 µm HIDES IT.
  Reverses the tree's standing "the lever is association, not detection" reading — for the 0947
  champion, measured, CPU-only.** E35 counted 713 in-gate fragmented GT edges and read 638 of them as
  slot competition. That reading was still too generous to the association axis: it never checked
  whether the GT endpoints were matched to the RIGHT prediction. They mostly are not. Match residual
  (GT node → its assigned prediction) is **3.362 µm median / 6.148 p90 on fragmented edges vs 1.724 /
  3.35 on linked ones** — the fragmented endpoints sit past the **2.87 µm GT nearest-neighbour floor**,
  so the 7 µm matcher is pairing them with a NEIGHBOURING CELL's detection. Gating on a confident match
  (both endpoints inside the floor) leaves **139 of 713**: `disappearance` 33 · `source_stole` 57 ·
  `target_taken` 24 · `both_reassigned` 25. The other **574 (81 %)** are matcher stretch.
  `_nearest_prediction` then settles what the stretch means, ignoring the assignment entirely: of the
  **827 loose endpoints, only 7 have ANY prediction within 2.87 µm**, median nearest **4.892 µm**. So
  they are **MISSING DETECTIONS (99.2 %), not contested ones** — no assignment or cost change can reach
  them. **Sizing: detection-caused edge loss 574/23080 = 2.5 % of edges, which CLEARS the 1.5 % bar;
  genuine association residue 106/23080 = 0.46 % and disappearance 33/23080 = 0.14 %, both far under
  it.** The mechanism that hid this: the official node metric also matches at 7 µm, so a GT node whose
  own cell was never detected still scores as a detected node by borrowing a neighbour's detection —
  **node recall stays ~1.0 while every edge through that node dies.** That is exactly how
  [[celltrack-local-official-metric-harness-reproduces-LB]]'s "recall 0.998 DEAD" survived: recall was
  never measured at a tolerance that could see a cell-swap. **Measured directly: recall at cell
  separation is 0.824** — 4292 of 24399 GT nodes (17.6 %) have NO prediction within 2.87 µm — against
  the ~0.998 the 7 µm ruler reports. **It is FLAT across all nine full 40-video caches this project has
  ever produced: 0.8173 – 0.8304, a 1.3-point spread**, so every post-processing and ILP knob we have
  swept leaves it untouched; it is a detector-weights property, GPU-priced. Note the honest bound: the
  metric LAUNDERS most of those misses, because the borrowed neighbour detection is itself correctly
  linked in its own track, so the edge maps onto a linked pair and scores as present. **The realised
  cost is the 574 edges (2.5 %), not the 4292 nodes (17.6 %)** — the node figure is the size of the
  detector's real blindness, the edge figure is what the board can see. Also finishes the distance-knob question
  (bd P1, filed this session): on the confident 139 the winner is shorter in 72 % of cases, true step
  5.139 µm vs winner 3.25 µm — but the population it governs is 0.46 % of edges, so an ILP distance
  reweight joins the disappearance weight as sub-bar. **Every remaining ILP cost knob is retired by
  size.** The live lever on the 0947 base is recall of *undetected cells in crowds* — which is the same
  bottleneck A ([[celltrack-two-bottlenecks-detector-separation-head]], finer decode) the tree already
  names, now sized on the CHAMPION rather than on our own detector, and now with an edge-loss price
  attached. **Rule earned: when the metric's matcher is looser than the object's own separation, a
  high recall is not evidence the object was found.**

  **E35 — THE DISAPPEARANCE SWEEP IS A 0.32 % EXPERIMENT; DON'T PAY THE GPU ARM. And the omission story
  IS the confusor story: 86 % of recoverable omissions are slot COMPETITION. CPU instrument, run before
  the run.** `fragment_audit.py` decomposes each in-gate fragmented GT edge by what the PREDICTED graph
  did with its two endpoints (`_verdict`): if the source ended its track and the target took no parent,
  the solver paid a disappearance and the disappearance weight is the binding knob; if either slot went
  to some other node, it is a competition failure no cost on disappearance can repair. Over the 40
  cached validator videos, **23080 matched GT edges, 739 fragmented (3.2 %), 713 in gate:
  `disappearance` 75 · `source_stole` 195 · `target_taken` 172 · `both_reassigned` 271.** Only **11 %**
  of the recoverable omissions are disappearances — **75/23080 = 0.32 % of edges is the ENTIRE ceiling of
  `logs/run_ilpdisapp.sh`**, five times under the 1.5 % bar, so the sweep is retired before it cost the
  ~23 min GPU arm E34 priced. The other **638 = 2.8 % of edges** are slot competition — a node other than
  the true one won the successor or the parent — which lands back on the confusor/association axis the
  whole tree keeps converging to; `both_reassigned` (271, the largest single class) means BOTH endpoints
  were re-used elsewhere. `_rival_gap` then sizes the winner: median **3.25 µm**, only **55/638 (8.6 %)
  within 1.6 µm** (a decode duplicate of the true cell) and **330/638 (52 %) beyond 3 µm** — past the
  2.87 µm GT nearest-neighbour floor, so **the slot-winner is usually a GENUINELY DIFFERENT CELL, not a
  duplicate detection.** Finer decode / NMS is therefore NOT the repair for this residue; edge-affinity
  discrimination is. Supersedes the E26/E28 reading of omission as absence — it is mostly DISPLACEMENT.
  **Methodology correction (self-inflicted, third denominator slip this session):** the first pass used
  greedy `cKDTree` NN at 5 µm, which lets several GT nodes claim one prediction and INVENTS fragments;
  fragmentation swung 0.62 % → 0.92 % → 1.91 % across 2/3/5 µm. `match_nodes` now mirrors the official
  ruler — `core/metrics/matching.py` `DistanceMatcher`, an OPTIMAL per-frame `linear_sum_assignment`
  under `_MAX_DISTANCE_UM = 7.0`. All numbers above are at that faithful setting. The RETIREMENT was
  stable across every tolerance tested (disappearance 21/85, 35/166, 64/397, 75/713 — always a small
  minority); the magnitudes were not. **Rule earned: match with the metric's own matcher, or the rate
  you report is a property of your tolerance, not of the tracker.**

  **E34 — PLUMBING CORRECTION: E29's candidate cache has NEVER ONCE FIRED, so the "ILP arms are free"
  saving is projected, not banked. `find runs -name '*.candidates.npz'` returns **0 files** across all
  21 prediction-cache dirs (which hold 40 `.geff` + 40 `.retention.jsonl` each and no candidates). Cause
  is ordering, not a bug: the `_save_candidates` code landed in `kaggle/local_predict_patch.py` at commit
  `6c1424a` 23:06Z, while the only run since (`secedge_0.30`) started 22:41Z and finished predicting
  ~23:04Z — it ran the older patch. Confirmed by grepping its restored `tracking_repo` predict script:
  0 hits for `_save_candidates`, but the CANDIDATES→vectorized rewrite IS present, so it was patched, by
  the previous revision. The patch itself is sound: applying it to a pristine
  `external/frontier_ds/biohub-tracking-support-pack-50ep-v1` script parses (`ast.parse` OK) and puts
  `_save_candidates(_candidates, coords, edges)` in the `else` branch directly after `predict_video`,
  exactly at the GPU/CPU seam. **Consequence for `logs/run_ilpdisapp.sh`: arm 1 pays the FULL GPU
  prediction (~23 min at N=20, per the secedge timings), and only arms 2-4 are CPU.** The sweep is
  ~1 GPU-arm + 3 CPU-arms, not 4 CPU-arms. Also corrects the kill accounting: `secedge_0.30` DID reach
  the end — `submission.csv` 241937 rows, `validator_results.csv` written, `run.log` ends at
  `FINAL: 22002 nodes, 21298 edges` — the only missing artifact is `ppsweep_results.csv`.
  **Rule earned: a cache is not a saving until a file exists on disk; assert the artifact, not the code.**

  **E33 — REFUTED, and it cost no slot and no GPU: v1329f has NO ensemble seat, because where the two
  trackers actually disagree the champion is right. `tracker_agreement.py --disputes` isolates the only
  edges a combination rule could ever swap — nodes BOTH trackers gave a parent to, but a different one —
  and there are **961 of them, not the 6.1 k E32 estimated** (the rest of the "a_only/b_only" counts were
  unmatched-node artifacts, the same denominator error E32 already had to correct once). On those 961:
  **0947 median step 2.071 µm vs v1329f 8.286 µm, and 0947 picks the shorter step in 98.86 %**; only 24
  disputes are within 1 µm of each other. GT median step is 1.82 µm with p99 7.2 µm and max 9.96
  ([[celltrack-gate-um-is-derived-from-displacement]]), so v1329f's contested picks sit at the gate and are
  physically implausible — this is not a tie to break, it is v1329f being wrong, and it is presumably WHY it
  scores 0.939 against 0947's 0.947. **Net for the whole donor: 428 orphan-fill edges (0.2 %) plus 961
  disputes it loses 99 % of. There is no headroom to combine.** Consequence for the plan: the final-2
  default of 0947 + v1329f stands only as SUBMISSION diversification against private-LB variance, never as
  a merge; and the 0.947→0.974 gap is NOT reachable by combining the public forks we hold — the remaining
  levers stay detection-side, where E32(i) put ~3–7 k nodes each fork finds and the other does not.**

  **E32 — THE ENSEMBLE SEAT IS OPEN, AND RUNTIME WAS NEVER THE THING BLOCKING IT. Two corrections in
  one measurement. (a) SIZE: the hidden test is **4 videos**, `predict_minutes_total` 9.93, and the
  validator-free kernel finishes in 1590 s of a 9 h budget — **~20x headroom**. "The models are too slow
  to ensemble" is false; we can afford several full passes in one kernel. (b) DIVERSITY: `kaggle/tracker_agreement.py`
  matches nodes between two submitted trackers within 2 µm per frame and compares edge sets. 0947 (LB 0.947)
  vs v1329f (LB 0.939) share **108192 edges, with 10356 0947-only and 10700 v1329f-only — shared_fraction_of_union
  0.8371**. Two trackers 0.008 apart on the board disagree about **one edge in six**. That is not the
  saturated pool of [[ensemble-diversity-has-a-quality-floor]] — that verdict was measured inside OUR
  single recipe; these are two independent public forks. The seat is open; what is unproven is the
  COMBINATION RULE. Note the merge needs no re-run: both submissions already exist as kernel outputs, so a
  merge kernel can attach them as inputs at ~zero GPU.
  **CORRECTION, same sitting — I first called the ~21 k disputed edges the headroom, and that is the wrong
  denominator.** `kaggle/merge_submissions.py` (constraint-safe union in the champion's frame) admits only
  **263 of 10700** donor edges at 2 µm and **428** at 5 µm, and the rejection histogram says why: **9492
  fail `unmatched_endpoint`** and 956 fail `target_has_parent`. Two facts follow. (i) Most of the
  0947/v1329f disagreement is **DETECTION-side** — each finds ~3–7 k nodes the other has no node for —
  which is the same detection-side verdict the rest of this tree keeps landing on, now measured between two
  frontier forks instead of inside our own. (ii) **Edge UNION is structurally dead as a rule**: the champion
  already assigns a parent to nearly every node, so a union can only fill orphans, and orphans are ~0.2 % of
  edges. The live rule is a **SWAP**, not a union — at 5 µm the two trackers assign *different* parents on
  **6141 / 6485** matched-node edges (~5.3 % of the graph), and a 2-member vote cannot break that tie. So
  the next step is a THIRD independent voter (v50, LB 0.907, the only non-0947-family fork we hold;
  gapfill/readmit are 0947 variants and therefore correlated voters), with the rule "flip 0947's parent only
  where the third member agrees with v1329f against it". 5.3 % contested is above the bar; the flip count is
  the number to measure before spending a slot.**

  **E31 — BANK: the validator is 74 % of the kernel's wall-clock, and removing it is OUTPUT-NEUTRAL.
  `celltrack-public-0947-fast` (= 0947 with `BIOHUB_VALIDATOR_ENABLE=0`, staged by `kaggle/env_variant.py`)
  finished in **1590 s vs the base kernel's 6286 s** and its `submission.csv` is **byte-identical**
  (`cmp` clean, 241357 lines both). Not an approximation that needs an LB confirm — the same bytes score
  the same. E30 independently found the field running `VALIDATOR_ENABLE=0` too. **Consequence: ~4700 s
  of the kernel budget is now free at zero cost to the shipped score**, which is what makes anything
  second-pass affordable on Kaggle, where E29's local candidate cache does NOT reach. Spend it on a
  SECOND ILP pass at a different disappearance weight with edge union (the two-pass tracklet ILP that
  [[celltrack-refuted-axis-is-the-long-run-lever]] flags as a jointly-necessary 0.945 lever) — not on a
  third detection vote, which E25 measured inert, and not on a base-family ensemble member, which
  [[celltrack-base-decorr-has-no-finer-member-redundant]] already called NO-GO.**

  **E30 — donor triage round 2 (23:12Z), three kernels, zero new arms — but one numeric.
  `newwang12/biohub-v1-grouped` (11 votes) and `leonixis/biohub-v1-infer` are both 0947-family
  (177 / 77 hits on `harmonic_association|DeepCenter|ILPSolver|BIOHUB_`); an env-block diff against
  `logs/0947_src.py` returns exactly ONE substantive numeric difference —
  `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` **0.25 vs our 0.20** — plus `BIOHUB_VALIDATOR_ENABLE=0`, which
  independently confirms the validator-free `-fast` variant is what the field runs. Neither posts a score,
  so 0.25 is a hint, not evidence; it is a DETECTION-side (DeepCenter division veto) knob, hence upstream
  of the E29 seam and still GPU-priced — queue it BEHIND the disappearance sweep, not ahead of it.
  `noisyislands/biohub-linker-association-mlp` is a 6-feature `EdgeMLP(6→16→16→1)` over GT-node geometry
  with injected synthetic dups/noise: that is simultaneously
  [[celltrack-four-cues-fail-confusor-detection-side]] (geometry-only cues rank the true successor at or
  below chance) and [[celltrack-pmkf-trains-gt-sparse-not-detected-crowd]] (trains on the GT-sparse pool,
  not the ~980-FP/frame detected pool). Refuted family, no score posted, NO arm. Einstein criterion held
  three times in one pass.**

  **E29 — the ILP-cost axis is no longer GPU-priced. `predict_video` returns `(coords, edges)` — the
  candidate graph with `edge_prob` — and everything under it (`build_graph` + `td.solvers.ILPSolver`) is
  pure CPU, so the GPU half knows nothing about the solver weights. `local_predict_patch.py` now caches
  the candidate graph on a key built WITHOUT the `--ilp-*` args (and without `BIOHUB_ILP_*`), so the
  first arm pays ~55 min of prediction and every later cost arm re-solves in seconds plus the PP sweep.
  A 4-arm disappearance sweep drops from ~3.7 h to ~1 h, and — the part that matters for rigor — the
  matched control at the shipped weight 2.0 is now FREE, off the identical candidate set, instead of
  being compared against a differently-seeded baseline. Staged as `logs/run_ilpdisapp.sh` (3.5 GPU,
  then 2.0 / 1.4 / 6.0 on CPU). Note this does NOT reach the Kaggle side, which always predicts fresh;
  it makes the local search cheap enough to spend one submission on a weight we actually chose.**
  **E28 — THE FRAGMENTED EDGES ARE REACHABLE: the ILP had the candidate and declined it. E26 said 1460
  GT edges have both endpoints detected and matched with no predicted link, but "matched" does not imply
  "on the solver's table" — an edge longer than the candidate gate was never offered. `kaggle/fragment_audit.py`
  walks the cached validator geffs against GT (per-frame nearest match ≤5 µm, µm from voxel × (1.625,
  0.40625, 0.40625)): of 22042 matched-endpoint GT edges, 417 are fragmented and **393 of them (94 %)
  sit inside the 10 µm gate**; only 24 are out of reach. So the loss is a SOLVER-COST decision, not a
  candidate-generation miss — which is what licenses the disappearance-weight arm rather than a gate
  widening. And the shape names the direction: fragmented edges have median step **2.73 µm** against
  **1.82 µm** for the edges that did link. The ILP is dropping the LONGER true steps, i.e. termination
  is out-competing continuation exactly where continuation costs most. Raising `ILP_DISAPPEARANCE_WEIGHT`
  above 2 is the mechanism-matched move; 1.4 (the hack kernel's value, E27) predicts MORE fragmentation
  and is the control, not the candidate.**
  **E27 — donor triage 2026-09-19 22:50Z. `codezzzsleep/biohub-095-owned-validation` (claims 0.95) is a
  METRIC HACK, not a method: it appends a hub node at `t=-1000, z=y=x=-10000`, links the 1400 largest
  components' roots to it, then chains 5 synthetic fork triples (`MAX_COMPONENTS=1400`, `FORKS=5`) —
  same family as `kirneo_metric-hack-last-call-update`, already recorded as farming zero divJ. NOT
  submittable, and its "0.95" is not evidence about any mechanism. One legit datum survives it: its ILP
  weights are edge −1.0 / appearance 0.0 / **disappearance 1.4**, against our 2. Someone else tuned this
  axis and landed BELOW us — the opposite direction from E26's prediction — so the arm is two-sided
  (3.5 and 1.4), not one. `binasalama/…-ilp-gap-recovery` is our own env family at stock defaults
  (disapp 0.1) = no new information. `fabriciodasilva/biohub-dodecatiad` is a from-scratch DoG-blob
  detector + greedy motion linker, no learned model, no posted score: structurally out-of-recipe (an
  ensemble-seat shape) but built on the DoG filter our own normalization axis already refuted as LOWERING
  centre detectability — not worth GPU ahead of the disappearance arms.**
  **E26 — THE MISSING EDGES ARE OMISSIONS, NOT CONFUSIONS. `validator_results.csv` already carried the
  decomposition and we had never summed it. Over the 40 held-out videos (base config, fp16 run):
  GT edges 47170 · edge recall 0.9475 · edge_fn 2476 = 1016 lost to detection (41 %) + 1460 FRAGMENTED
  (59 %) · `wrong_association_edges` = **74**. Mislinks are ~nil. The standing story — "association
  under crowding / confusor disambiguation" — is not what the remaining loss is made of: 1460 GT edges
  have BOTH endpoints detected and matched, and simply no predicted edge. That is 3.1 % of all edges,
  the largest single addressable block we have measured, and it is a RECALL problem at the linker.
  Mechanism candidate, never swept: `ILP_DISAPPEARANCE_WEIGHT = 2` (with appearance 0) prices ending a
  track cheaply, so the ILP is free to drop a marginal link; with only 74 wrong edges there is enormous
  headroom to pay more for continuation. It lives inside `predict_unet_transformer.py`, so it needs a
  GPU re-run (~55 min/arm), queued behind the edge-weight arms. Division in the same table: recall
  0.217 (26/120), precision 0.361 — measured at n=120 GT divisions, not the n=7 that the old
  "division postproc dead" verdict rested on. Gate knobs are already swept flat (bead yesg, 9 knobs
  ±0.003), so any division re-entry must be CANDIDACY, not gating.
  CAUTION: the `spurious_pred_nodes` column sums to 1.72 M against 1.77 M predicted nodes and 836
  missed GT nodes — it is not a per-node FP count. Do not build on that column without deriving it.**

  **E25 — `SECONDARY_DETECTION_WEIGHT` is INERT (local, fp16, N=20, 44 videos). 0.6 → adj 0.9021 /
  missed GT 418 · 0.8 (shipped) → 0.9018 / 446 · 0.95 → 0.9021 / 454. Spread 0.0003 = noise across a
  36-cell missed-GT swing: the blend moves detection RECALL and the score does not follow. Two readings,
  same direction — the two detectors agree wherever it matters, and the score is association-bound, not
  detection-bound (see "score lever is association not detection"). So the freed runtime above must NOT
  buy a third DETECTION vote. Follow-up arm running 22:41Z: `SECONDARY_EDGE_WEIGHT` 0.30 / 0.05 vs the
  shipped 0.15 — the same fusion knob on the association side, never swept.**

  This retires the "6-config ensemble infeasible, ~18-24 h" bound — it was measured with the validator
  block in every member. Marginal cost of an extra member is ~940 s (predict 301 + PP 639), so 3-4
  members fit inside one 9 h kernel even before fp16. "The models are slow" was never the models.** Cache-key bug found: env paths under the work dir made every shard miss → keys now content-fingerprinted
  (f4d499b). New donors 15:25Z: noisyislands linker-MLP (2 epochs, 4 videos, ~0 negatives =
  toy) · xgboost-division (13 GT-only fork feats, negatives = random non-forks, in-sample acc 0.997 ⇒ trivial
  labels, no track score) · thtennant divprec-v1 (ppsweep selected=base, no held-out) ⇒ none carries evidence. The
  IDEA they gesture at is real but untried by us: a fork classifier trained on PREDICTED-graph candidate forks
  (labels = GT match), replacing reid3's hand SAFE_DIV_* thresholds. DENOMINATOR is the point: donor validator = 8 stems, 12 GT
  divisions ⇒ a division knob moves 1 event = ±0.08 divJ = noise. Local lets N_PER_TYPE grow (pool 71 44b6 + 128
  6bba). Caveat: primary weights' train split undocumented (deepcenter: 71/128 split) ⇒ validator may be in-sample;
  use it to RANK PP configs, not as an LB estimate. First run = reid3 unchanged, fidelity check vs donor csv.
  Plan (rev 14:05Z): donor OUTPUTS are free evidence — `kaggle kernels output <owner>/<slug>` gives each donor's own
  validator/ppsweep/reid_report; read them BEFORE porting. E5/E8 retired by that read (above); E7 inconclusive.
  Remaining: E1–E3 (pending probes) → composite best-of(E1/E3) + E10 + E4. Donor divdiag (reid3, 8 held-out stems):
  divJ 0.23, 9/12 FN, **6/12 = "2nd daughter OWNED by another track"** — the steal/reattach axis (SAFE_DIV_STEAL_*, off by
  default in reid3) is the division-side element to look at next.
  **Division axis n=40 (18:12Z, reid3, 40 held-out / 60 GT div, `probes/reid3_divaxis.json`): FLAT.** base proxy 0.9161
  (adj 0.9022, divJ 0.140, 12 TP / 26 FP / 48 FN). All 9 knobs within ±0.003: steal_r20 0.9160, steal_own45 0.9161,
  rawsucc 0.9163, tight55 0.9163, dcdiv032 0.9153 (10 TP), diverge175 0.9146 (+10 FP), nnk2 0.9182, dcdiv018 0.9178,
  reid8+relaxed9 0.9177; combo nnk2+dcdiv018+reid8 0.9193 (+0.0032 directional, +3 TP / +9 FP). 48/60 GT divisions
  missed by EVERY knob → the n=8 steal hypothesis does not hold at n=40; division recall is upstream (detection /
  association), not a postproc knob. Division term ceiling = 0.1·divJ ≈ 0.014 → axis CLOSED for PP.
  **Detection threshold (same 40):** 0.955 adj 0.9008 vs 0.965 0.9022 (−0.0014): 17 fewer missed GT nodes, +20
  fragmented edges, +12.9k pred nodes — extra nodes do not become links. 0.975: adj 0.9020 (−0.0002), missed GT 461 vs 440 @0.955 — threshold axis FLAT over 0.955–0.975 (span 0.0014); only the LB probe det096 can still move it.
  **Kaggle steal probe (reid3 own 8-video validator, T4):** base 0.9490, all 7 steal/tight knobs 0.9477–0.9511
  (adj ±0.003, 3 TP / 9 FN divisions everywhere) — same FLAT verdict as n=40 on an independent run.
- **Cross-family decorrelation (18:20Z, GT-free, 4 visible test movies):** 0947 vs v1329f submission edges agree
  81–93 % (44b6_0b24845f 0.81, 6bba_05db0fb1 0.85; node match ≤2 vox). But the v1327-w3 "real model" is the
  0947 PRIMARY + 0.09 % weight delta (unet 0.0009, transformer 0.0003 rel-L2; vs seed314159 secondary ≈1.0) —
  the disagreement is PIPELINE (post-proc/decode), not model diversity. w3-as-secondary killed on mechanism
  (would replace the one decorrelated seed with a primary clone); no run.
- **LB instruments pushed 18:24Z (expiring slots, Kaggle GPU):** `0947-det096` (DET 0.965→0.96 only — tests
  hikaggler's "N_pred moves LB" on our base) and `0947-synthsec` (secondary seed314159 → our all199_from_synth80
  ep2 ckpt, private dataset `dimiturnt/celltrack-synth80-ft2` — first structurally-different member; an ep2
  under-trained ckpt, so a loss = "under-trained seat", not "synth refuted"). Watcher `logs/kaggle/watch_0947_probes2.log`.
  Both submitted 20:16/20:26Z. NB the kernel blends DETECTION as `0.2·primary + 0.8·secondary`
  (`BIOHUB_SECONDARY_DETECTION_WEIGHT` 0.80), so synthsec swaps 80 % of the detector, not just an edge vote.
  Local instrument `secdet_{0.95,0.6}` (fp16, N=20, vs fp16 base 0.9018) running 20:48Z — never swept before.
- **synth_pre80 done 13:01Z**: synth val best 0.9783 (80 ep). Chained all199_from_synth80 started; ep1 val 0.9201 on
  the 4-movie split (not comparable to the 20-holdout A/B arms' 0.89–0.91). ~25 min/epoch.
- **Forum 740145 (hengck23):** Kaggle GT sometimes sits on cell "corners" (Ultrack-derived); many FPs lie next to a GT
  node. Their trick: at a 99% edge-recall cutoff, re-rank only the surviving candidates with a heavier module.

## ROOT

- **Goal:** WIN (1st). Winning cluster 0.945–0.962. **Banked best = 0.924 LB** (sub 55779061 = faithful
  evgendvorkin champion replica + global tracksdata ILPSolver linker).
- **The gap = 0.924 → 0.945 = ASSOCIATION-under-crowding.** NOT detection: `node_recall ≈ 0.998` is DEAD.
- **Ruler discipline:** CTC/local proxy INVERTS above 0.90 (dw0 +0.0267 held-out → −0.028 LB). The only
  non-inverting local ruler = the 6-movie official-metric harness (`scratchpad/off_eval6.py`,
  `metrics.evaluate()`, tracksdata 0.1.0rc8, max_distance 7.0µm); gate ≈ 0.94 local-official. A slot is
  spent ONLY on a named mechanism clearing 0.910, gated on this ruler or on a card-free decorrelation gate.

## LEVEL 1 — WHERE is the gap? (three branches)

- **Detection recall** → **REFUTED as a lever.** node_recall 0.998; detector over-detects ~40× (~1005
  det/fr vs GT ~6.6 annotated/fr). Recall is not missing.
- **Division** → **REFUTED as a lever.** Champion's own real-LB sweep: div_weight 0.3–3.0 all ~0.915;
  7.0 → offline divJ first-nonzero but LB DROPPED 0.914 (proxy-inversion caught in the act). TEST has ~3 GT
  divisions → 0.1·divJ ≈ 0. Division postproc / FP-fork precision does not convert.
- **Association under crowding** → **THE gap.** edge_jaccard collapses monotone with density: sparse movie
  (4.3k nodes) 1.000 → dense movie (98k nodes) 0.598. Even the champion ILP only lifts the densest
  0.598→0.6495. Everything below hangs here.

## LEVEL 2 — WHY association fails (mechanism, banked)

1. **Sparse-train / dense-infer.** Champion edge head (`SimpleNodeTransformer`) trains on GT-touching pairs
   (~17/fr) but infers on the ~1005-FP detected crowd. NUANCE (density-collapse audit): the champion's
   *external* trainer ALREADY does complete-candidate source competition (`detect_and_match` ~258/fr,
   focal-BCE `softmax(dim=0)` masked to GT-touching rows∪cols) → the naive "it never saw the crowd" is only
   half true. The dense collapse saturates on the source-competition axis that IS trained → **representation/
   capacity limit, not purely a data limit.**
2. **The confusor is DETECTION-side.** Four cues — distance, incoming-direction, raw full-res appearance,
   learned features — ALL rank the true successor BELOW CHANCE on the mislinks, unanimously. The
   discriminating info is not in the detected representation; a fraction of labels may be wrong. → any
   link-time cost that encodes motion/direction is ≤ null, plausibly negative.

## LEVEL 3 — the config surface (every axis we support → folded verdict)

### A. Link-loss FORM (per-source target competition) — `--symmetric-links` / `--slack-links` / Sinkhorn/OT / parental-softmax
**AUDIT-FLAT.** Symmetric's headline 0.9278 was a dead-constant bug (constant-removed = 0.9060 = baseline).
Where it genuinely improves association (mislinks 46→43), the **shared-trunk detection-coupling tax** charges
it back (both heads read one backbone; harder assoc gradients over-fire the detector). SlackRow faithful
re-test −0.0107 ≤ noise. Sinkhorn couples the SAME two constraints harder → same tax. Only un-flat sub-angle
= doubly-stochastic mutual-exclusion vs FP-forks, but divJ tiny. Bounded as a family.

### B. Link-time COST (motion / appearance / context added to the objective)
**REFUTED.** motion-in-ILP: motion cue ranks true successor below chance → actively misleads solver
(solver-agnostic). PoE cost-shape refuted (confusor is per-frame-greedy STRUCTURE). appearance cosine =
coin-flip on confusor (n=37). pair_context −0.011. edge_options/view_tta −0.0049 (dense needs sharper, not
averaged). The confusor info isn't in the representation → no link-time recombination recovers it.

### C. Global topology — linker name: `greedy` → `flow` → `ilp`
**BANK = ilp.** Global min-cost ILP defeats per-frame-greedy confusor STRUCTURE: 0.902 → 0.924 (the bank).
Card-free alternates below champ (assign 0.900, flow 0.695). Topology axis closed AT ilp — but see Level 4:
the winner's ilp is **two-pass tracklet**, ours is single-pass.

### D. Edge-head ARCHITECTURE — `--head {pack, hoct}`, `edge_hidden_dim`
**HARNESS on prior kills; but CEILING-BOUNDED as a solo lever.** HOCT (edge-to-edge global attn + 3D RoPE +
σ line-bias) is fully BUILT (`hoct_edge_transformer.py`), drop-in for SimpleNodeTransformer. Prior kills
(597c0f2 "REFUTED", proxy 0.7234) were epoch 0.5/3 + (1,4,4) + from-scratch = **harness**, explicitly
reclassified. **The decisive result:** frozen-detector HOCT on GT-sparse gave a *conclusive confusor
instrument-lift* (top1 0.012→0.214 = +0.202, inverted 0.988→0.583) — head class carries bottleneck B for the
first time on a non-inverting ruler. BUT faithful DROPPED 0.6877→0.6432: HOCT's sharper affinity makes the
flow linker drop marginal-but-true edges calibrated for the standard head's softer scores. **B solved on the
instrument; recall-cost erases it at the ship point.** AND the winner's own Table 2: edge-stage-alone caps at
**CLB 0.926** ≈ our 0.924. **So a perfect edge head ≈ +0.002 = noise. The edge head is not the 0.945 lever.**

- **Width trap:** every HOCT ckpt is `hidden_dim=128`. "C=256" the human approved = HEAD WIDTH; `K=256` in
  the pmkf scripts = candidates/frame (a different knob). A width-256/288 head has NEVER been trained — but
  width is unlikely to be the lever when the ceiling itself is 0.926.

### E. Detection RESOLUTION — `--downsample 1,2,2` (finer122) vs 1,4,4
**HARNESS-then-BOUNDED.** finer(1,2,2) separates to 1.0µm vs (1,4,4) merges <2.0µm (oracle). Frozen-graft
REFUTED (off-grid, +60.9% peak inflation = fabricated fragments = harness). Co-adapt un-merges +2625
sub-1.6µm cells, physics-validated GT-free (bottleneck A HOLDS). BUT **B FLAT** (from-scratch/co-adapt head
too weak at 8× density) AND **GT nearest-neighbor floor = 2.87µm, ZERO cells <2µm across 131k frames** → sub-
1.6µm un-merges are FP splits, not recall. A-axis real but bounded; needs B solved AT finer to matter.
> **FOLD 2026-08-28 — the "B FLAT" finer-head verdicts are HARNESS, not capacity.** Every finer122
> edge-head checkpoint (coadapt_long standard, hoct_converge, AND the 3.D-CHEAP-CUT rzvw head-only run)
> trained with contrastive `nce=0 hn=0` — LossCfg defaults both to 0.0 (joint_config.py:288,298) and the
> launch scripts omit the flags. All three plateau proxy ~0.61, inv 0.97 = the degenerate untrained-affinity
> signature. avl8 (same joint arch, nce ON): proxy 0.79 / node R 0.92 / inv 0.54. So B-FLAT-at-finer was
> never tested with the association objective's key term ON. **Un-refutes** finer-head B AND the HOCT×slack
> coupling (3.D cheap-cut). The never-run cell = finer122 detector + nce ON → staged `launch_hoct_finer122_nce.sh`
> (contrastive 1.5, hard-neg 0.015 — recovered EXACT from avl8 loss decomposition, == nwfi record). CAVEAT:
> this is an INSTRUMENT to close the axis honestly (does nce flip finer inv 0.97→~0.5?), NOT a >0.945 bet —
> the 3.D ceiling (0.926 ≈ 0.924) still bounds any solo edge head. If the instrument clears the axis, the win
> is still the Terminal-#2 JOINT stack, not this arm alone.

### F. Data DISTRIBUTION — train on detected crowd (`--detected-videos` / dw0)
**REFUTED on the board.** Feeding the SimpleNodeTransformer head the complete detected crowd: +0.0267
held-out → **LB 0.872** (proxy anti-correlated at top). The data-half individually failed. pmkf's open bet
= does swapping to HOCT arch + freezing the detector flip this — untested, but capped by D's 0.926 ceiling.

### G. ENSEMBLE / decorrelation
**Common-mode → bounded.** All candidates are evgendvorkin variants making the SAME confusor errors.
Decorrelation instrument: unionJ 0.9093 < 0.910 (leans NO-GO). One-recipe pool saturates within noise. An
ensemble seat requires a member from OUTSIDE the recipe (a structurally-different detector), not another
copy. A genuine 2-model seat (champion SimpleNodeTransformer + a working HOCT) only pays off IF HOCT first
becomes a pipeline win — which D says it can't, alone.

## LEVEL 4 — the winner's THREE 0.945-levers, and why each looks flat SOLO on our data

The winner (Table 1/3) reaches 0.945+ via three **jointly-necessary** pieces, none of them the edge head.
We have tested each **in isolation** and banked a partial-refutation for each — which is exactly the trap,
because the winner says they are jointly necessary (→ 0.920 only together):

| winner lever | our banked solo result | tag |
|---|---|---|
| **Two-pass tracklet ILP** (Δt=1 tracklets → meta-node re-solve) | our GT: 128883 edges 100% dt=1, ZERO gap-edges → two-pass fires only on rare detector-miss + confusor-FP | HARNESS/UNTRIED (single-pass now) |
| **Variable-appearance** (parental softmax `1+Σexp` = SlackRow) | SlackRow faithful −0.0107 ≤ noise | REFUTED SOLO |
| **Finer decode** (0.926→0.950 bulk, detector-side) | un-merges physics-valid but B FLAT, GT NN floor 2.87µm | HARNESS/BOUNDED SOLO |

**THE conclusion of the tree:** we have been mining the EDGE-HEAD axis (HOCT, target-competition, detected-
crowd, confusor), which the winner's own ablation caps at 0.926 ≈ our bank. Every lever that actually reaches
0.945 is in the SOLVER + variable-appearance + finer-decode STACK, and we have only ever tested those **one
at a time and killed each in isolation** — against a source (Table 1) that says they only work together.
This matches the standing meta-conclusion: no TRIED mechanism has been DEMONSTRATED-BY-US >0.910, and every
kill on the association axis was HARNESS-bound. The path is **replicate the winner's FULL stack jointly**
(two-pass tracklet ILP + variable-appearance + finer decode, co-trained), not another solo edge-head arm.

## TERMINAL — remaining levers, ranked by plausible size × un-refutedness

1. **Two-pass tracklet ILP** (solver-side). Untried; winner marks it necessary. **NOT card-free** (bd wfh5
   audit 2026-08-26): shippable path WRAPS tracksdata.predict() INSIDE the kernel — our local motile
   `ILPLinker` is 7apb-unshippable and tracksdata doesn't import locally → build+validate are kernel-bound.
   wfh5 also imposes a GPU referee-gate BEFORE the build (produce hoct affinity → confusor referee under
   flow+ilp → does a global solver convert the hoct inv-gain to fewer mislinks vs pilkwang 143?). Risk: jm68
   sizing = our GT has zero gap-edges → two-pass may barely fire on the metric. GPU + wheel-gated, not a
   card-free advance.
2. **Full-stack joint replication** (finer decode + HOCT edge head + two-pass ILP + variable-appearance,
   co-trained). The only path the winner's ablation actually supports to 0.945. Expensive, GPU + wheel-gated,
   multi-arm. This is the real bet, not any single piece.
3. **pmkf / HOCT-on-detected-crowd frozen** (bd `biohub_kaggle-pmkf`, OPEN, trained today, UN-SCORED on
   off_eval6). Worth SCORING because the confusor instrument-lift is real and the 6-movie number is cheap —
   but D's 0.926 ceiling means best-case ≈ noise over 0.924. Score it to CLOSE the axis, do not build on it.
4. Everything in Level 3.A/B/F/G — refuted or bounded; do not re-open without a new mechanism.

## JOINT ARM — the staged plan for Terminal #2 (card-free design; launch is wheel+card-gated)

The winner's three levers are jointly necessary (Table 1 → 0.920 together); we killed each SOLO. The arm
that could actually reach 0.945 co-trains them AND co-designs the linker recalibration that erased the HOCT
gain last time. Spec (so the launch is designed, not improvised):

- **Substrate:** finer122 (1,2,2). Detector = warm from `finer122_coadapt_long.pt` (already un-merges +2625
  physics-valid), `--freeze-backbone-norm` (BN-pollution guard; joint_cli:152). Bottleneck A already held here.
- **Head:** `--head hoct` (edge-to-edge attn + RoPE; built, `hoct_edge_transformer.py`). Read width off ckpt
  (9f33205); train to CONVERGENCE (prior kills were epoch 0.5/3 = harness). This attacks B (confusor).
- **Variable-appearance = the recall-cost FIX, not a separate topper (KEY mechanism, 2026-08-27):**
  `--slack-links` (SlackRow = parental softmax `p=exp(ℓ)/(1+Σexp)`). The `1` is the NO-PARENT escape valve.
  HOCT-solo dropped recall (0.6877→0.6432) because a SHARP head with NO escape FORCES every cell to link →
  over-commits → drops marginal-true edges when uncertain. The `1+` lets low-confidence cells stay unlinked
  gracefully. So HOCT (sharp) and variable-appearance (escape) are **COUPLED, not additive** — the sharp
  head NEEDS the valve to not over-commit; the valve with no sharp head to regulate is inert (explains
  slack-solo −0.0107). This is the precise mechanism of "jointly necessary." Predicts: the joint arm's
  success hinges on the HOCT×slack coupling, testable by ablating the `1+` term within the joint arm.
- **Residual linker recalibration (secondary):** after the escape valve restores graceful non-linking, any
  remaining distribution mismatch (flow thr/disappearance-cost tuned for the softer standard head) is a
  smaller sweep, not the primary fix. A joint arm that lifts faithful past 0.6877 WHILE holding top1 >> 0.012
  is the first real B-driven pipeline win.
- **Solver:** two-pass tracklet ILP (bd wfh5), kernel-side (wraps tracksdata.predict()). Referee-gated FIRST
  (produce hoct affinity → confusor referee under flow+ilp → mislinks < pilkwang 143?). Build only if it
  converts.
- **Oracle (non-inverting gate, in order):** (1) confusor referee mislinks < 143 = affinity real; (2)
  off_eval6 6-movie micro ≥ 0.94 local-official; (3) only then a slot. Proxy above 0.90 is BLIND — never gate
  on it.
- **CHEAP FIRST-CUT (flags audited 2026-08-27, all compose — no launch crash):** the coupling hypothesis is
  testable WITHOUT the expensive co-adapt. `--detector-from finer122_coadapt_long.pt --head hoct
  --edge-hidden-dim 256 --slack-links --downsample 1 2 2` = warm+FREEZE the finer122 detector, fresh-init the
  HOCT head, train HEAD-ONLY (no detector backprop → fast, ~head-only budget not full-train). This is the
  exact 0825 frozen-detector design PLUS the slack valve — one variable added. Gate: does faithful hold/beat
  0.6877 (vs solo-HOCT's 0.6432) while top1 stays >> 0.012? If YES → the valve fixes the recall-cost, coupling
  confirmed, escalate to co-adapt + two-pass. If NO → coupling refuted cheaply, saved the multi-hour arm.
  (256/4=64 = RoPE-valid; config default-comments winner=288, also valid.) Guard: `--detector-from` loads BN
  finer122 → keep `--norm batch` (warm guard rejects `group`).
- **Launch order:** (0) CHEAP head-only frozen arm above — coupling gate. → if converts, (1) referee-gate
  hoct affinity (wfh5, GPU inference) → (2) GPU co-adapt joint train (finer+hoct+slack, converged,
  recalibrated linker) → (3) off_eval6 → (4) slot. Each step gates the next; do not skip.
- **Cost:** cheap arm = head-only GPU (short); full bet = multi-hour GPU. Both wheel+card-gated. The cheap
  arm is the highest-ratio next GPU spend — price it FIRST on wheel-grant.
- **CHEAP-CUT INVALIDATED 2026-08-28:** the rzvw head-only run above executed with `nce=hn=0` (flags
  omitted) → trained FLAT (proxy 0.6187, top1 0.20) = the nce-off plateau, NOT a coupling refutation. The
  coupling gate must be RE-RUN with `--contrastive-weight 1.5 --hard-negative-weight 0.015`. Staged as
  `launch_hoct_finer122_nce.sh` (frozen coadapt_long detector + hoct head + nce ON). Card-gated behind knee
  raddino (~0900z). Gate unchanged: faithful holds/beats 0.6877 AND top1 >> 0.012 = coupling confirmed.

## What is DEAD — do not re-suggest
motion-in-ILP · division postproc/FP-fork · directional-PE (detection-side) · consensus copy-ensemble ·
link-loss form family solo ·
appearance/pair_context/view_tta · finer sub-1.6µm as recall.
