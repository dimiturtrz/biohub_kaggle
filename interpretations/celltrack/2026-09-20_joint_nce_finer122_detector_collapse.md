# The joint finer122 + nce arm collapsed the detector, so its gate never got a reading

**Date:** 2026-09-20 · **Arm:** `launch_finer122_joint_nce.sh` · **Log:** `logs/finer122_joint_nce.log`

## What the arm was for

It was the correctly-specified never-run cell: finer122 substrate, trunk TRAINABLE, contrastive term ON,
trained on the detected corpus. Every earlier "nce ON vs OFF" comparison froze the detector, where the term
is structurally inert, so the flag had never actually been the live variable. The registered gate:

> proxy must clear ~0.70 to show the objective moved anything (plateau 0.6150); ~0.85 reopens the
> finer-decode leg; below 0.70 the substrate is the bound and the finer122 axis closes for good.

## What happened

| step | proxy | node R | ratio | det | nce | sanity AUC |
|-----:|------:|-------:|------:|----:|----:|-----------:|
| init (0) | 0.6150 | 0.885 | +0.89 | — | — | 1.0000 |
| 947 | **0.0000** | 0.000 | −0.95 | 0.1460 | 1.2991 | 0.9972 |
| 1894 | **0.0000** | 0.000 | −0.92 | 0.1574 | 0.8633 | 0.9951 |

Node recall goes to *exactly* zero while the emitted node count roughly DOUBLES (53k → 97k on
`44b6_e57ff5c6`). Those two facts only sit together one way: the probability map saturates, NMS returns a
near-regular lattice of peaks spread through the volume, and essentially none of that lattice lands within
the 7µm match radius of a real cell. Many detections, none of them on anything.

The detection loss does not recover — it RISES between the two windows (0.1460 → 0.1574) while the total
falls, so the optimizer is buying its progress on the contrastive term with the detector. `sanity AUC`
stays ~0.995 throughout, which is the tell that this is a *detector* failure and not a broken harness: the
affinity head is still discriminating fine on the same forward pass.

## The verdict the gate did NOT get

**The gate is unreadable and the finer122 axis does NOT close.** Below-0.70 was pre-registered to mean "the
substrate is the bound, not the objective". It cannot mean that here, because the arm never tested the
objective on a working detector — it tested whether `--contrastive-weight 1.5` at `--lr 5e-5` with the trunk
unfrozen destroys a finer122 detector. It does, within 947 steps.

This is the faulty-TRAINING branch, not a refutation. Reading 0.0000 as a substrate verdict would retire a
live axis on a recipe bug.

## Why 1.5 was not an unreasonable number, and what actually differs

`--contrastive-weight 1.5` is not arbitrary: it was recovered exactly from the joint family where nce lifted
0.61 → 0.87 (`elegant_H` 0.8716). The weight is the one that worked. What is new here is the pair of things
underneath it — the finer122 (1,2,2) decode, which quadruples the candidate count the contrastive term sums
over, and a trunk that is trainable while carrying a detector head. A term whose magnitude scales with the
candidate count, held at a weight calibrated on a 4× smaller candidate set, is a plausible mechanism for the
imbalance; the observed nce ≈ 1.30 against det ≈ 0.15 at step 947 is consistent with it.

That is a hypothesis about the mechanism, NOT a measured cause. Nothing here priced it.

## What a re-test would have to change

Do not relaunch this script as written. A re-test needs the contrastive term to stop dominating — the
candidate-count argument above implies scaling the weight by the grid ratio (≈1.5/4) rather than picking a
new number by feel — and, independently, an assertion that node recall has not collapsed, checked at the
FIRST eval window so an arm like this dies in 4 minutes instead of 18.

Not scheduled. It is a ~1.5h arm that by its own registration is not a submission arm, and the ruler work
(bd `brod`) outranks it: a faithful top-end ruler is what makes the remaining open levers priceable at all.

## Not affected

The 120× flow-linker swap measured on this same run stands — it is a post-processing timing result and is
independent of detector quality. It is what let this arm reach two eval windows in 18 minutes instead of
roughly two hours, which is the only reason the collapse was caught the same afternoon.
