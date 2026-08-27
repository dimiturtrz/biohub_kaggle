# Edge-jaccard collapses with detection density (recall stays ~1.0)

2026-08-27 · celltrack · own-results interpretation

## What unblocked this

Proxy inverts/blinds above 0.90 (`celltrack-proxy-saturated-affinity-quality-bound`), so no
offline number could gate a submission. Built the missing **non-inverting local ruler**: ran the
champion detector (`predict_unet_transformer.py`, thr 0.99) locally on the 5090 to produce per-movie
champion geffs, then scored them with the **official** metric (`src/biohub_tracking/metrics.py`
`evaluate()`, tracksdata 0.1.0rc8, `max_distance=7.0µm`, `scale=ds.scale`). Runners:
`scratchpad/{off_eval6,fork_sweep,cmp_ilp}.py`.

3-movie micro = 0.9213 (≈ LB champion 0.924, but subset-lucky). 6-movie micro = 0.846 (greedy linker).

## The finding

edge_jaccard collapses **monotonically** with detection density; node_recall stays ~1.0 throughout →
pure association-under-density, NOT detection/recall.

| movie | pred_nodes | edge_jac | node_recall |
|---|---|---|---|
| 6bba_718b21f9 | 4.3k | **1.000** | 1.000 |
| 6bba_784a78c9 | 13.5k | 0.978 | 1.000 |
| 6bba_bb9f20c3 | 24.6k | 0.898 | 0.998 |
| 6bba_09961292 | 30.8k | 0.881 | 0.998 |
| 44b6_ddf577ad | 40.7k | 0.808 | 0.998 |
| 6bba_57b7cc1e | **98k** | **0.598** | 0.990 |

eFP ≈ eFN (balanced mislinks). The confusors are the ~40× **unannotated** cells (detector finds
~258/frame, GT annotates ~6.6/frame) — invisible on GT-only close-pair analysis because annotated
cells sit 15–40µm apart. That is why the earlier per-frame-nearest instrument
(`scratchpad/champ_local_edge_acc.py`) read edge_acc 0.98 (optimistic): its greedy per-frame match
doesn't expose the linker's mis-assignment under crowding; the official DistanceMatching does.

## node_recall is DEAD as a lever

0.998 under 7µm matching — the detector finds ~every annotated cell. The earlier "20% miss @2µm" was
sub-voxel centroid noise (z-voxel = 1.625µm), not missed cells. Recall / finer-decode retired as win
levers (`celltrack-gt-nn-floor-kills-finer-inject`).

## Linker selection (greedy → ILP) is a PARTIAL fix, already banked

Ran the champion's real `--use-ilp` (tracksdata ILPSolver, ilpy 0.6.0 + motile) on the dense movie
57b7cc1e: edge_jac 0.598 → **0.6495** (+0.051), eFP 407 → 288 (global flow kills spurious links),
recall 0.990 → 0.975. This global-linker gain is the mechanism already banked in the 0.924 champion
(`celltrack-linker-is-the-gap`). Even with real ILP the densest movie = 0.65 vs sparse 1.0.

## Division precision (un-refuted, but small)

divJ = 0.0495, crushed by **93 FP forks** vs 5 TP. A FP fork = pred node forks (2+ successors) and
matches an annotated GT node that does NOT divide → spurious 2nd successor grabbed from an unannotated
neighbor (the same event that makes a FP edge). Prior division kills were recall/recovery
(`celltrack-global-ilp-division-refuted`, `celltrack-division-postproc-dead-root-is-detector-temporal`);
this precision angle is un-refuted. Tested a post-proc fork-gate (`scratchpad/fork_sweep.py`: keep top
edge_prob successor, keep 2nd only if prob ≥ 0.7) → 6-movie score 0.8463 → 0.8516 (**+0.005**,
sub-floor). Keep as a free composing default; not a finding, not a submit.

## The sized lever — and what the trainer audit ruled OUT

The remaining ~0.35 edge_jac on the densest movie = **affinity QUALITY**: `edge_prob` cannot separate
the true successor from an unannotated confusor at 98k-node density.

The obvious framing — "the champion edge head trains on sparse GT pairs, so add complete-candidate
negatives (the dw0 direction)" — is **refuted by auditing the external champion trainer**
(`train_unet_transformer.py`). `detect_and_match` (L620) detects all ~258/frame peaks including
unannotated; `build_matched_edge_targets` (L747) builds the full pairwise matrix over all detections;
`compute_loss` (L55) is focal-BCE with `softmax(logits, dim=0)` over sources, masked to GT-touching
rows∪cols. So **the champion edge head already trains complete-candidate source competition** — each
annotated target discriminates its true source among all detected confusors. dw0 (+0.0267 held-out,
LB 0.872) helped OUR trainer, which lacked this; it does not stack on the champion (redundant with the
2-seed ensemble's conservatism). Data/negative-sampling is banked, not a lever.

Two structural facts fall out of the loss: (1) softmax over dim=0 gives per-target source competition
but no per-source target competition, and "divisions allowed" is by design → structural
fork-permissiveness, the mechanism behind the 93 FP forks. (2) The dense collapse is on the
source-competition axis that is trained correctly → it **saturates at 98k density = a
representation/capacity limit, not a data limit.**

So the genuine remaining lever is a structurally-different edge head — Sinkhorn/OT one-to-one
assignment, graph-net mutual-exclusion, or HOCT edge-to-edge attention + RoPE
(`celltrack-hoct-arch-already-built-kill-was-harness`,
`celltrack-frontier-gather-names-rope-and-parental-softmax`). These were killed before on the inverting
proxy; they are now measurable on this ruler (`scratchpad/off_eval6.py`). Multi-hour GPU, from-scratch;
gate on local official ~0.94 over the 6-movie set before spending a Kaggle slot. bd = biohub_kaggle-pmkf.

## Status

Bank 0.924. Slot HELD until a mechanism clears local official ~0.94 on the 6-movie set (now
measurable). No submission from this diagnosis — it built the ruler and sized the lever; it did not
move a number past the bar.
