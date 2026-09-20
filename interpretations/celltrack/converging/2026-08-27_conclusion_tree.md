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

## NEWS 2026-09-19 23:35Z — where the campaign stands

- **Board:** ours 0.947 (`celltrack-public-0947`, tied by `-gapfill`), then readmit 0.946, v1329f 0.939,
  v50 0.907, our own banked 0.924. LB top 0.974. Deadline 09-29.
- **Pending LB:** det096 (56370794), synthsec (56370924), fp16 (56372434). 5 fresh slots at 00:00 UTC.
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
