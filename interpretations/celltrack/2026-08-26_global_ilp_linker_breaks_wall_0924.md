# Global ILP linker breaks the 0.902 wall → 0.924 (+0.022) — confusor was assignment STRUCTURE

2026-08-26. The banked-best jumped **0.902 → 0.924** on the public LB. Verified against primary source
(`kaggle competitions submissions -c biohub-cell-tracking-during-development`), independent of the arm
that produced it.

## The clean single-variable A/B (LB truth, not proxy)
| sub | artifacts | linker | publicScore |
|---|---|---|---|
| 55774589 / 55774416 | dual-seed fusion + DeepCenter veto + divsub, thr0.96875 | per-frame greedy motion Hungarian | **0.902** |
| **55779061** | *same three artifacts* | **GLOBAL tracksdata `ILPSolver`** (min-cost, whole-track) | **0.924** |

The ONLY difference is the linker: everything upstream (detector, fusion, veto, threshold) is identical.
Greedy → global min-cost ILP = **+0.022**. This is an A/B on the board, not a proxy number, so it is not
subject to the proxy-inversion caveat above 0.89.

## Mechanism — the confusor was per-frame-greedy assignment structure
Our greedy Hungarian is byte-exact to evgendvorkin (oracle 1.0), so the long-standing "linker-is-not-the-gap"
finding only ever held **Hungarian-vs-Hungarian**. It said nothing about greedy-vs-global. The +0.022 lever
is **global temporal coupling**: the ILP solves assignment jointly across the whole track instead of
committing per frame. The dense confusor — the 85%-of-gap axis that defeated every attack — was never a
detection or a loss problem. It was the greedy linker locking in a locally-cheap wrong parent that a
global objective avoids.

This **vindicates** [[celltrack-poe-refuted-confusor-is-structural]] (confusor = per-frame-greedy STRUCTURE)
and **explains the whole run of nulls**: HOCT association head (detection-side), appearance InfoNCE
(coin-flip), SlackRow parental-softmax (faithful HARD FLAT, n=122337), directional-PE (flat) — all attacked
the confusor on the wrong axis (detection / loss). The axis was assignment structure, reachable only by the
linker.

Note: an earlier attempt at global coupling, sub 55677362 "global-flow coupling", scored **0.695** — a
BROKEN flow implementation, not a refutation of the idea. The working global coupling is the tracksdata
`ILPSolver`. Global-coupling-as-idea is therefore NOT pre-refuted by that null (kill the *use*, not the
*asset* — the asset was mis-implemented).

## Forward — the winning stack (untried, one config flip)
`LinkerConfig(name="ilp")` is selectable in the full tracker (`linkers.py:538`). So
`kernel_faithful_replica` (fusion thr0.96875) + `linker name="ilp"` runs the full tracker WITH the
auto-mounted reuse + max_gap2 (fire-check GO: 6bba dense +3550 reuse / +81 max_gap2 edges) AND the global
ILP linker = **three confirmed positive levers stacked** → plausible 0.926–0.93. Validate proxy/GT-free fire
vs the bespoke 0.924 kernel before spending a slot; needs GPU. This is the live path past 0.924 toward the
winning cluster (0.945–0.953).

## Status
Banked best = **0.924**; >0.910 milestone cleared. WIN goal (first place, 0.945–0.953) still open — bar to
advance is now 0.924. [[celltrack-linker-is-the-gap]] [[celltrack-0027-carrier-is-deepcenter-recovery-stack]]
