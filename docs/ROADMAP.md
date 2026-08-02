# celltrack — ROADMAP

Ordered execution status. Substance for each item lives in `PLAN.md`; this file is the order + state.
The build log itself is git history.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Where we are (2026-08-02)

**LB 0.859** (rank 1122; public top 0.943, public-notebook frontier ~0.913). Local fold-0 0.877 → LB
0.859, a **−0.018** offset, so local measurement is a trustworthy proxy and iteration does not need to
spend submission slots. Shipping pipeline is `pilktunet-motion-stlf`: pilkwang's public TemporalUNet3D
(CC0, grafted) → our motion Hungarian linker → short-track(3) + linefit(0.8).

## Done

- [x] Scaffold generated (sdlc-scaffold v1.26.0); data downloaded to the `paths.yaml` root.
- [x] `core/` data layer — OME-Zarr + GEFF readers, voxel↔µm scaling, submission CSV round-trip.
- [x] Local evaluator — **exact** agreement with the organizers on all six counts across 158 frozen cases.
- [x] Dataset statistics settling the PLAN open questions (`2026-07-23_dataset-survey.md`).
- [x] Score bracket: NN linking of GT nodes = 0.998, classical floor = 0.55 (`2026-07-23_bracket.md`).
- [x] Operating-point sweep; under-detection refuted (`2026-07-23_operating-point-sweep.md` — but see the
      reconciliation task below; its conclusion holds only for a jittery detector).
- [x] Fast eval loop — fp16 + channels-last inference 148s → 15s/video, `ResponseCache` replay,
      extract-once threshold sweep.
- [x] Plateau-NMS fix (`qc5`) — one centre per connected plateau; poisoned every number measured before it.
- [x] Motion Hungarian linker; short-track filter; linefit smoother.
- [x] pilkwang detector graft → fold-0 0.877, LB 0.859.
- [x] End-to-end code-competition submission path (kernel push → submit), verified against the LB.
- [x] Public-notebook frontier survey + `kaggle/survey.py` (`2026-08-02_public-notebook-frontier.md`).
- [x] Rules settled: external pretrained models permitted, the packs are CC0, no from-scratch requirement.

## Now — closing the 0.054 gap to the public frontier

Ranked by expected gain / cost.

- [ ] **Divisions.** Our division Jaccard is *structurally* zero — `linear_sum_assignment` is one-to-one,
      so no node can ever have two children. The term is worth 0.1 of the metric; a division-aware pass on
      GT nodes already measured 0.333 (`2026-07-23_division-linking.md`) ≈ **+0.033**. `division_linking.py`
      exists but is not in the shipping pipeline. **Largest identifiable single chunk.**
- [ ] **Detection threshold.** We run 0.5; the frontier runs 0.99. Our own full-41 sweep was still climbing
      at 0.95 (0.8742 → 0.8861). Cheapest possible experiment — cached responses, no retraining. Needs the
      data root.
- [ ] **Gap-close with synthetic node insertion** (`1ro`, S3 token `stlfg`). In *both* the classical 0.857
      recipe and the learned frontier; the one post-proc stage everyone has and we do not.
- [ ] **Topology repairs** — enforce-next-frame, single-parent repair, prune-isolated, edge ≤ 14 µm.
- [ ] **Dual-seed logit blend** — mount `biohub-temporal-unet3d-seed314159-v1` beside the 50ep pack.
- [ ] Learned edge bonus in the linker (needs `local-association-ranker-unet300`); DeepCenter veto model.

## Next — make the detector ours (`80e` / `cdg`)

Not on the critical path to a better score, but a frozen borrowed detector cannot be improved, and the
recipe is now fully specified rather than guesswork.

- [~] From-scratch TemporalUNet3D. Trajectory: init 0.049 → 2k **0.4227** → 4k **0.5633** (subset-8). The
      120k-step run died at 4k when the machine shut down. **Resume needs the GPU box.**
- [ ] Fine-tune the public weights on our fold-train — the cheaper half of the same lever.
- [ ] Augmentation (ours has none; pilkwang uses brightness + flip), LR schedule, early stopping.

## Housekeeping — the doc layer drifted

- [ ] **Reconcile the operating-point contradiction.** `2026-07-23_operating-point-sweep.md` concludes "do
      not chase the node-count bonus, operate at the estimate". The 2026-08-01 full-41 sweep on a *strong*
      detector is monotone the other way. Both are right in their own regime; only a beads memory records
      that, and the doc reads as settled. The one stale conclusion that could actively mislead.
- [ ] **Promote the beads memories into deep dives.** 21 `memory` records hold the real corpus (qc5, the
      graft, the motion A/B, calibration refuted, width-64 refuted *and its retraction*). `research/` looks
      thin because the practice drifted, not because the thinking didn't happen.
- [ ] Mark pre-`qc5` numbers in `interpretations/` as plateau-poisoned; several are unflagged.
- [ ] `interpretations/` stops at 2026-07-23 while the work ran to 08-02.

## Later

- [ ] Grouped CV by acquisition prefix; multi-seed once a number moves.
- [ ] Trackastra (`1f0`) — the 2026-07-31 SoTA dive's own recommended path, never tried.
- [ ] `pilktunet` × `ilp` — untried cell in the coverage grid.
