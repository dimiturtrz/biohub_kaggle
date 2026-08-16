# The from-scratch ceiling is resolved: every knob caps, the gap is supervised training budget on all data

<!-- Filename says "pretraining"; the original conclusion was WRONG. See the CORRECTION section below:
     the frontier uses no pretraining — it is supervised joint training for 50 epochs on all 199 movies. -->


*2026-08-16 — closes the fork the two 2026-08-15 docs left open. Reconciles their disagreement and records
the outcome of the augmentation / group-norm / longer-training arms they proposed.*

## The two 2026-08-15 docs disagreed; the later one wins, and then its own lever capped too

- **`2026-08-15_from-scratch-ceiling-is-representation-not-loss.md`** (earlier): per-axis histogram of the
  193 missed GT nodes → missed cells are a **systematic dim class** (dimmest-fifth over-missed 1.68×),
  proposed a targeted dim-cell synth lever.
- **`2026-08-15_from-scratch-gap-is-detection-generalization.md`** (later): the decisive metric-free test
  (`differential_missed.py` runs BOTH detectors through the *same shipped tracker* and intersects their
  per-GT missed masks) → the 178 recoverable cells (from-scratch-missed & warm-caught) are
  **normal-contrast** (local-contrast median 81.5 vs both-caught 89.0, far above the both-missed floor 75.5).
  Not a dim tail. This **refutes the dim-class finding**: a per-axis histogram over the missed set is
  confounded (the missed set skews dim because dim cells are slightly harder *everywhere*), while the
  intersection test asks the load-bearing question — *of the cells one detector misses, are they a dim
  class or ordinary cells the other reads from the same pixels?* — and answers "ordinary."

**Reconciliation:** the intersection test is the stronger ruler (it controls for "dim is harder for
everyone" by conditioning on what warm recovers). The dim-synth line is **dead**. Doc B's proposed lever was
augmentation.

## The augmentation / norm / length arms all capped (the experiments Doc B's fork asked for)

Run on the detector-only trainer (`tunet_detector.py`), eval = shipped-tracker proxy on the dense val-four,
eval-threshold 0.97 (the deployed op-point):

| run | recipe | best proxy | node R | verdict |
|---|---|---|---|---|
| baseline (`detector_tunet_ours.pt`) | no aug, flat LR, 1500 | ~0.757 (LB 0.757) | ~0.90 | deployed |
| `det_scratch_aug` | flip + brightness .3 + offset .1, cosine, 10k | **0.7649** | 0.93 | +0.01 = noise floor |
| `det_scratch_aug_long` | aug, 3k | 0.7505 | 0.957 | node R up but over-detects (ratio +0.16), proxy flat |
| `det_scratch_pilk_gn` | pilkwang-aug + **GroupNorm**, 8k | 0.7522 | 0.94 | **group-norm refuted** (< plain aug) |

Augmentation buys ~+0.01 proxy — at the score noise floor (~0.01–0.02), directional at best, not a result.
Group-norm (per-sample stats, the domain-gap-at-inference hypothesis) scored *below* plain aug: the
BatchNorm→GroupNorm swap does not close the microscope-domain gap here. Longer training raises node recall
into over-detection (ratio +0.16) without raising the clamped proxy. **The ceiling ~0.76 proxy / ~0.93 node R
is loss-invariant, norm-invariant, aug-invariant, threshold-invariant, and step-budget-invariant
(4k→40k identical).**

## What is actually left: training budget on all data, not "pretraining" (CORRECTION 2026-08-16)

**An earlier version of this section claimed the frontier's edge is reconstructive self-supervised
backbone pretraining. That is false and is retracted.** Reading the downloaded frontier source settles it:
`external/kaggle-cell-tracking-competition/` (README + `scripts/train_unet_transformer.py`) trains the
released public-baseline weights with **plain supervised joint end-to-end** learning — focal-BCE edge loss
+ BCE detection loss, the edge loss flowing back into the UNet — for `--epochs 50` on **all 199 movies**.
No masked-voxel, no denoising, no contrastive, no SSL of any kind. Every `mask` reference in that repo is a
padding / active-node mask. There is no pretraining to replicate.

The real difference is **training budget × data**, on the same supervised objective:

| axis | frontier (warm weights) | our from-scratch joint |
|---|---|---|
| objective | supervised joint (edge+det) | same |
| epochs | 50, to usable convergence | **`joint_pt_finetune`: 0.44 epoch, early-stopped** |
| data | all 199 movies | our train subset |
| best proxy | 0.847 (warm eval) | **0.7121** |

We *did* run joint-from-scratch — it is not an untried fork. `joint_pt_finetune` (the model the differential
test used as "from-scratch") early-stopped at **0.44 epochs / best proxy 0.7121** — undertrained by ~100×
against the frontier's 50. The two contrastive joint arms confirm the objective must stay plain supervised:
`joint_pt_contrastive` (NCE) capped ~0.69, and `joint_dim_ctrl` (NCE+HN) **collapsed to proxy 0.000 / node R
0.000** — representation shift into the shared trunk corrupts detection. So the SOTA replicate-then-build move
is not "pretrain the backbone" — it is **train the same supervised joint model to the frontier's budget
(50 epochs × all199)** and measure whether that alone closes the ~0.93→0.986 node-R gap. That is the honest,
non-warm-start lever, and it is untested at budget (we only ever ran fractions of an epoch).

Warm-start is the **same TemporalUNet3D architecture** (layers 32/64/128) reading the recoverable cells
correctly from the same pixels — so the gap is **not capacity, not the objective, not calibration.** It is
what 50 epochs on all data buys the representation that <0.5 epoch does not.

## Standing state

- **Submitted today (2026-08-16):** the self-estimated per-gap drift linker on the deployed detector
  (`celltrack-ourdrift-start`, thr 0.97) — the one conservative postproc lever mechanism-aligned with a noisy
  detector (nudges only still-unmoved track-starts by the frame's own common-mode of confident tight matches;
  no field, no tuned constant, no dual-seed). Score pending (LB lag).
- **From-scratch deliverable ceiling:** ~0.757 LB, an honest portfolio artifact. The frontier warm pipeline
  (LB 0.892–0.899) remains the reference ceiling; its edge is supervised training budget (50 epochs × all199),
  not pretraining.
- **Next lever (not yet built):** train the same supervised joint model (edge+det, no contrastive) to the
  frontier's budget — 50 epochs on all 199 movies — from scratch, and measure the node-R gap. Every arm so
  far ran <0.5 epoch; the budget itself is the untested variable. Multi-hour build, not a knob.

## Reproduce

The decisive test ran two detectors through the *same* shipped tracker and intersected their per-GT
missed masks (the from-scratch-missed ∩ warm-caught set is the recoverable one; its local contrast is the
detectability axis). The one-shot analysis scripts (`differential_missed`, `missed_cells`,
`intensity_calibration`) were not landed — they are evidence scaffolding, not reusable package code, and
carrying five test-mirrors for finished one-shot CLIs is tax the finding does not need. The load-bearing
numbers (recoverable local-contrast 81.5 vs both-caught 89.0 vs both-missed floor 75.5; the aug/norm/length
table above) are recorded here; the checkpoints they ran on live at
`D:/data/.../processed/biohub_cell_tracking/{det_scratch_aug,det_scratch_pilk_gn}.pt` with their `.log`s.

The one piece that *did* land is the mechanism-aligned linker change: the self-estimated per-gap drift
prior (`MotionHungarianLinker(use_drift=True)`), tested in `tests/unit/celltrack/linkers/motion_linking.py`.
