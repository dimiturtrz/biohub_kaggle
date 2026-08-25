# Idea board & trajectory map — organizing current knowledge before the next experiment trajectory

**Date:** 2026-08-25. **Purpose:** single home for the whole method board (external SOTA + internal
graveyard), indexed onto the 50 open beads and ~50 memory entries so nothing falls off by proximity-bias.
This is the map that picks the trajectory. No new experiments proposed here — organization only.

**Standing status phrasing:** *No mechanism we TRIED has DEMONSTRATED-BY-US performance indicating a
public score above 0.910. That does not mean we tried everything, or tried it well.* Banked LB **0.900**.
Bar to beat **0.910**. Exactly ONE held submission. Slot spent only on a mechanism that offline-clears
0.910 via a **non-inverting** instrument.

---

## 1. Where the bottleneck IS (settled)

Not annotation recall (champion node R **0.98**). Not linker structure (flow optimal, 0 merges at coarse).
Precision is mostly **uncounted** by the metric (ignores unlabeled preds) — the only precision that bites is
**identity-precision** = the confusor wearing a precision hat. The 0.900→0.950 gap lives in the **dense
regime the sparse annotation metric barely samples** (local GT 0.2–13% dense), split two ways:

- **A — detection MERGE.** At (1,4,4) decode grid two cells <2.0µm fuse to one centroid. Physics-locked:
  confusor sep 1.6µm ≈ one (1,4,4) voxel (1.625µm). merge_referee: +60.9% peak inflation at matched grid on
  dense e57. Fix = finer (1,2,2) decode (separates to 1.0µm). **LB-visible / proxy-blind.**
- **B — affinity CONFUSOR.** Correct detections, WRONG same-frame partner under crowding. n=37 mislink set =
  **85% of the measurable gap.** Directional (rival-nearer 97.3%, fast-movers changing direction).
  Detection-side representation gap, not architecture. Live keystone: P_true 0.06 << P_chosen 0.32.

**Unknown we never split:** the A-vs-B weight in the *LB* gap — both are proxy-blind, so only the board can
weigh them. finer122 tests A cleanly (node R faithful); B's LB weight is unmeasured.
**Blind axis:** divisions (CV8 7 GT divs, divJ ~0.1 weight, unmeasurable locally).

## 2. The two laws that govern every local decision

- **Proxy-LB transfer law** (`2026-08-09_proxy-lb-transfer-law.md`): proxy moves LB only when it changes the
  linking DECISION on contested cases. FAITHFUL below ~0.89, BLINDS/INVERTS above. dw0 +0.0267 proxy →
  −0.028 LB. → **aggregate proxy above 0.89 is BANNED as a go-signal** (bead 5e3u). Local validation must be
  mechanism-level: node R (grid-invariant), merge_referee (GT-free un-merge), net_new_over_champion (GT-free
  decorrelation), n=37 confusor set, oracle positive controls.
- **4-check discriminator (method-bad vs test-bad):** a kill counts ONLY if it passed (1) Fire — flag landed;
  (2) Positive control — oracle input the harness can see; (3) Clean recipe — schedule/lr/norm not broken;
  (4) Right ruler — metric that matters at a visible denominator. HARD kills had oracle controls; SOFT kills
  never did → SOFT kills are confounded, not refutations.

## 3. The board — every idea placed, indexed to beads

### NEW (untried, or tried-WRONG → effectively untried) — the ceiling levers
| Idea | Hits | Bead(s) | Note |
|---|---|---|---|
| **CELLECT** (Nat Methods 2025, PMC12615263) | **A+B** | **54qc** | SOTA dense-3D C.elegans = our regime. Contrastive on CENTER-POINTS + intraframe de-merge MLP (A, no 4× grid) + interframe assoc/div MLP over 5-NN (B) + public pretrain. Our contrastive kill used POOLED features → different method. Decorrelated seat. **Highest ceiling.** |
| **Ultrack multi-hyp ILP** | A+B | **g89y**, **we0z** | Global track-level opt, defers hard decisions vs greedy. CPU / CBC (Kaggle-legal). Prior run killed use-A (own watershed on our heatmap = region-too-small) — the SEAT (feed our fg+contours, use only ILP) UNTESTED. |
| **Complementary detector** (StarDist-3D / Cellpose-3D) | A+seat | *(none — ranked-open-levers lever-3)* | Decorrelated detection vote. Untried. **FILE.** |
| **Time-symmetric** (yeast PMC25) | B | *(none)* | Forward+backward *training*. Cheap. Distinct from our bidirectional-fusion (mutual-consistency gate only). **FILE.** |
| Trackastra parental-softmax `1+Σexp` | div | *(none)* | Cheap division-norm knob. **FILE (P3).** |
| SAM2 zero-shot / Cell-TRACTR | seat | *(none)* | Weak free seats; low priority. |

### HALF-TRIED (confounded kill → re-testable, NOT dead)
| Idea | Bead(s) | Why not dead |
|---|---|---|
| **finer122 co-adapt** | cvsb, bvol, yaj9, sdoo, h1mg, h3zi, wcwq | RUNNING. A. node R peaked 0.885 (v3 0.826). Lands as A-fix + seat, NOT B (P_true flat). |
| **directional-PE rider** | j7ig | Engaged but under-powered (dir/radial 0.19). Folded into keystone. |
| **HOCT directional edge-bias** (3D RoPE + line-to-line + parental softmax) | 1blg | #1 CTC linking. From-scratch head refuted but confusor detection-side + 2 plumbing bugs → re-test on finer122's better repr. |
| contrastive/appearance | sja3, urti, fapu, 5sap | pooled-feature version refuted; **CELLECT is the correct form.** |
| synth | eqdx, pcv3, 5c2j, w1vl, wmc8, q2bf | generator-audit-first; BN-pollution → GN un-kills. |
| edge-retrain / hardneg / LoRA | rp4e, 0hb8, p5zr, g8j8, 0i3n | died on from-scratch-undertraining OR frozen-backbone de-calib. No oracle control. |
| ranker minority tie-breaker | 2h3s, 2x7r | majority-use regressed; minority re-testable. |
| motion/drift priors | mpwq, 5fvt, grtd, gse5 | drift-subtract (mpwq) untried; PoE (gse5) cost-shape refuted. |

### TRIED-HARD (oracle control → real refutation of THAT USE, not the asset)
velocity-ref bias · turn-angle (jf5g) · fork-cap/min_track · global-ILP division (fabricates, −0.055; z3uw-adjacent) ·
PoE cost-shape (gse5) · Kalman-prior · **radial** distance-bias (confusor directional not radial) ·
**dw0 conservatism** (z3uw, 3q4i, bumo — proxy +0.0267 → LB −0.028) · recovery-stack (LB 0.898).

### DISCOUNTED (physics-gated, not killed)
Tube / 4D-PE model (ikxs, pknq, xqjk) — motion_stats: drift-free persistence τ = 1.2–2.4 frames; long
persistence = tissue DRIFT (common-mode, disambiguates nothing per-cell). Sequence model inherits the
eight-angle wall. 4D-PE validates the *encoder design* if ever built.

## 4. The meta-insight already filed (anchors the trajectory)

**Bead hmlo (P0):** *"REPLICATE the public 0.915 end-to-end before porting any more parts — we have been
merging without replicating (1 for 7)."* Combined with owner SOTA rule (gather→replicate→**merge**→build):
the disciplined path is to **replicate a frontier dense-regime model faithfully**, not port more isolated
parts (7 ported, 1 landed). Public-0.915 replicate stalled at 1/7 parts → **CELLECT is the cleaner replicate
target** (published, self-contained, our exact regime, both bottlenecks, decorrelated).

## 5. Ranked untried levers (by plausible gain; data/arch > knobs)
1. **CELLECT replicate** (54qc) — SOTA our regime, A+B, decorrelated, fixes contrastive-done-wrong. Top ceiling.
2. **finer122** (cvsb) — running, A, cheap-to-finish, seat.
3. **Ultrack ILP seat** (g89y/we0z) — B+A, CPU free-lane, decorrelated, untested.
4. **HOCT directional edge-bias on finer122 repr** (1blg) — B, re-test on better representation.
5. **Complementary detector seat** — decorrelated detection (unfiled).

CELLECT + Ultrack + HOCT-on-finer-repr = the three most-plausibly clearing 0.910, all decorrelated seats.

## 6. Bead-hygiene notes (for the trajectory-pick, not done here)
- **Close-candidates** (hard-refuted, keep as knowledge but off the ready-list): z3uw, 3q4i, bumo (dw0);
  gse5 (PoE); bvol/l178 detector-only re-pool refuted variants.
- **Gaps to FILE:** complementary-detector seat; time-symmetric link scoring; Trackastra parental-softmax.
- 50 open beads is a pile, not a plan — the ranking in §5 is the ready-list; the rest are backlog/knowledge.
