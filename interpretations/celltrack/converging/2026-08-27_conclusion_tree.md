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
  | E1 | gap-fill from low-threshold detection pool (`fill_gaps_from_low_detections`, LOWDET 0.3, ≤3% added) | thtennant gapfill | code | on LB: probe 56354350 |
  | E2 | flow-seeded motion relink (`MOTION_RELINK_FLOW_*`, K 12, radius 40 µm) | thtennant gapfill/readmit/divprec | code | inside E1/E3 probes |
  | E3 | readmit discarded detections (score ≥0.965, r 4 µm) | thtennant readmit | code | on LB: probe 56354352 |
  | E4 | division precision: sister symmetry τ 0.6→0.4, sister-min 0 | thtennant divprec | knob | untested |
  | E5 | learned re-ID appearance descriptors as pair feature (sweep picks REID_WEIGHT 4.0) | arnav170 reid3 | code | **DONOR-NULL** (its own `reid_report.json`, 5656 held-out sources): GBM top-1 0.98568 = transformer top-1 0.98568 exactly; perm-importance tf_prob 0.202 AUC, every descriptor block ≤0.0014 ⇒ appearance adds nothing past tf_prob. Sweep adjJ +0.004 (8 vids) sub-floor. Don't port. |
  | E6 | packet grouping post-process (`grouped_postprocess`, xy radial shell profile) | newwang12 grouped | code | **DONOR-UNMEASURED**: its `submission_audit.json` = `graph_schema_pass`, `quality_validated: false`; `v9_division_gate.json` only checks division-COUNT retention (101/102 = 0.99) on 4 test clips — no score evidence either way. Would need a slot. |
  | E7 | test-time denoising-AE prefilter (30 steps, α 0.17) | ghazarosbarseghyan91 dae | code | **DONOR-INCONCLUSIVE** (its validator vs reid3 `base` on the 4 shared stems): adjJ 0.8811 vs 0.8858 (−0.005), per-stem ±0.06, spurious −15 %, missed +1.25 ⇒ no carry evidence; low priority |
  | E8 | sub-voxel centroid refinement (`refine_all_centroids`) | evgendvorkin 0.927 | code | **MECHANISM-NULL, not submitted**: metric `DistanceMatching(max_distance=7.0 µm)` vs refine shift ≤ sub-voxel (<1 µm), and both donor + 0947 round output to int voxels. Ported anyway (`kaggle/element_transplant.py`, variant `celltrack-public-0947-refine`, smoke-tested) — the CLI is the reusable transplant harness. |
  | E9 | DET 0.965→0.96 | thtennant det096 / beraterolelk | knob | beraterolelk tie ⇒ sub-floor |
  | E10 | runtime budget: frame cache 48, ILP timeout 1200, deadline degrade | thtennant fast | code | ENABLER — headroom for stacking E5–E8 in 9 h |
  | E11 | TabPFN division classifier | noisyislands | code | low ceiling (divisions ≤0.002); donor log shows only `LightGBM outer accuracy: 1.000` on 1082 division rows (saturated ⇒ likely leaky/trivial labels), no track score ⇒ UNMEASURED, don't port |
  Plan (rev 14:05Z): donor OUTPUTS are free evidence — `kaggle kernels output <owner>/<slug>` gives each donor's own
  validator/ppsweep/reid_report; read them BEFORE porting. E5/E8 retired by that read (above); E7 inconclusive.
  Remaining: E1–E3 (pending probes) → composite best-of(E1/E3) + E10 + E4. Donor divdiag (reid3, 8 held-out stems):
  divJ 0.23, 9/12 FN, **6/12 = "2nd daughter OWNED by another track"** — the steal/reattach axis (SAFE_DIV_STEAL_*, off by
  default in reid3) is the division-side element to look at next.
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
