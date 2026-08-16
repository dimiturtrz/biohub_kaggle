# The from-scratch ceiling is resolved: every objective/data/norm lever caps, the gap is pretraining

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

## What is actually left: pretraining, the warm-start's only real edge

Both docs converge on the mechanism: warm-start is the **same TemporalUNet3D architecture** (layers
32/64/128) and reads the recoverable cells correctly from the same pixels — so the gap is **not capacity, not
loss, not calibration, not augmentable diversity over 12 movies.** It is the *representation* the pilkwang
weights carry from **broader pretraining**. Warm reads 0.986 node R / 0.847 proxy; from-scratch same-arch
caps 0.93 / 0.76.

Bigger architecture will not help (the existing arch already suffices when pretrained). The one honest
from-scratch lever — the SOTA replicate-then-build move, not borrowing pilkwang's weights — is to
**replicate the pretraining**: self-supervised pretraining of the backbone on the unlabeled microscopy
frames (masked-voxel reconstruction / denoising), then fine-tune the detection head on the 191 annotated
train movies. Contrastive pretraining is already refuted (Doc A's *contrastive* arm was actively harmful —
representation shift into the shared trunk corrupts detection), so the objective must be reconstructive, not
contrastive.

## Standing state

- **Submitted today (2026-08-16):** the self-estimated per-gap drift linker on the deployed detector
  (`celltrack-ourdrift-start`, thr 0.97) — the one conservative postproc lever mechanism-aligned with a noisy
  detector (nudges only still-unmoved track-starts by the frame's own common-mode of confident tight matches;
  no field, no tuned constant, no dual-seed). Score pending (LB lag).
- **From-scratch deliverable ceiling:** ~0.757 LB, an honest portfolio artifact. The frontier warm pipeline
  (LB 0.892–0.899) remains the reference ceiling; its edge is external pretraining, legitimately labelled.
- **Next lever (not yet built):** reconstructive self-supervised backbone pretraining on unlabeled frames.
  This is the only remaining mechanism that could move the representation floor; it is a multi-hour build,
  not a knob, and is the correct next investment if the from-scratch score is to be pushed further.

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
