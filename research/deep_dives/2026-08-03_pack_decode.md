# Kaggle Biohub Cell Tracking: 0.930–0.947 Pack Decode

**Date**: 2026-08-03  
**Status**: partial (competition ongoing, no winner writeups available yet)  
**Supersedes**: none

## TL;DR

Top teams (~0.93–0.947) likely pull 3–5 concrete levers: (1) training detectors to full convergence (baseline explicitly undertrained), (2) edge-centric linking architectures (HOCT-style) over node-centric transformers, (3) improved division detection pipelines, (4) test-time augmentation + multi-detector ensembles, (5) graph-based post-processing. No public solutions >0.90 found yet; competition ends 2026-09-29, so no team writeups available. Metric structure is exploitable: 10% penalty on over-prediction + 0.1× division weight means conservative node-count strategies and edge-first optimization dominate.

## Question

What concrete methods and levers separate the 0.930–0.947 leaderboard pack from the public 0.913 frontier? Which strategies are used by top teams: stronger detection, better linking, metric exploitation, ensemble/TTA, or post-processing refinement?

## Findings

### Baseline Status (Our Frontier)

- **Detection**: 3D U-Net with temporal attention (TemporalUNet3D) + local-max suppression for center recovery [S1, S2]
- **Linking**: Cross-attention transformer (SimpleNodeTransformer) scoring (t, t+1) node pairs [S1, S2]
- **Training**: Only edges with ground truth are used; background detections and unannotated cells are ignored [S1, S3]
- **Explicit undertrain signal**: Baseline GitHub explicitly states "not trained to convergence — expect gains from training longer" [S1]

**Implication**: The public 0.913 tier has NOT extracted convergence gains yet. This is likely **low-hanging fruit** for jumping to 0.92–0.93.

---

### Lever 1: Training to Full Convergence (HIGH PRIORITY, FAST)

**Evidence**:
- Baseline repository explicitly documents undertrain: "It was not trained to convergence — expect gains from training longer." [S1]
- Cell tracking literature shows clear convergence patterns; models trained to epoch 30–45 typically outperform early-stop baselines [S6]
- 3D detector research consistently shows 5–15% gains from extended training + proper learning-rate scheduling [S6]

**Mechanism**: 
- Current baseline likely stops at 3 epochs or early-stopping after minimal validation improvement.
- Extended training (50–200 epochs with decay or cyclic LR) and curriculum learning (easy → hard samples) yield consistent boosts without architecture changes.

**Effort**: ~2–3 builds, <4h wall-clock per round (existing compute).

**Rough impact**: +0.01–0.03 (conservative estimate; baseline undertraining is explicit, so this is nearly guaranteed).

---

### Lever 2: Edge-Centric Linking Architecture (MEDIUM PRIORITY, HIGH IMPACT)

**Evidence**:
- HOCT (Higher-Order Cell Tracking Transformer, Biohub-authored, July 2026) [S5, S7]:
  - Addresses two failure modes of node-centric methods:
    - **Cell division entanglement**: "Cell divisions create confusion in the node embedding space, making lineage paths difficult to distinguish" [S5]
    - **Weak edge topology**: "Edges sharing a node have near-random label agreement, so the candidate-graph topology carries no useful information for graph neural networks to aggregate" [S5]
  - Edge-centric design where "candidate cell links attend to one another under a 3D geometric prior" [S5]
  - Achieves "state-of-the-art results" on Cell Tracking Challenge benchmarks without deep pre-trained encoders [S5]
  - LoRA fine-tuning: "quickly reducing tracking errors by 59% with 400 annotations" (6.75% improvement over transformer baselines) [S5]

**Mechanism**: 
- SimpleNodeTransformer pools node features and ranks pairs independently. HOCT learns relationships *between edges* (which are linking attempts that share nodes), using 3D geometric constraints to resolve ambiguity.
- Directly addresses the sparse-annotation regime: divisions and high-density regions create the exact confusion HOCT solves.

**Effort**: ~2–4 weeks (implement or adapt HOCT architecture, retrain with frozen detector).

**Rough impact**: +0.02–0.05 (addressing known failure mode; CTC leaderboard shows significant gains, but competition metric differs).

**Risk**: If competition scoring doesn't penalize division confusion as heavily as CTC does, gains might be smaller.

---

### Lever 3: Improved Division Detection Pipeline (MEDIUM PRIORITY, MEDIUM EFFORT)

**Evidence**:
- Division Jaccard weighted at only 0.1× in final score, but **is exploitable if done well** [S2, S4]
- Cell-TRACTR (transformer-based end-to-end, 2024–2026 era) introduces **division accuracy metric (DivA)** specifically to diagnose division errors [S8]
- Research shows most division errors are **linkage confusion** (mother→daughter vs same-cell links) rather than pure detection failure [S8]
- Temporal window flexibility: predicted divisions within ±1 timepoint of ground truth still count as TP [S2, S4]

**Mechanism**:
- Baseline SimpleNodeTransformer treats division as one edge among many. No explicit topology constraint (a division node must have exactly 2 outgoing edges to daughters).
- Division-specific post-processing or a dedicated division head can:
  - Enforce topology (one input node → exactly two outputs at time t+1)
  - Use temporal consistency (daughter cells should move coherently from mother center)
  - Recover off-by-one cases within the window

**Effort**: ~1–2 weeks (add topology constraint + temporal consistency loss, tune thresholds).

**Rough impact**: +0.01–0.025 (0.1× weight limits ceiling, but consistency recovery is relatively untapped).

---

### Lever 4: Test-Time Augmentation (TTA) + Multi-Detector Ensembles (HIGH PRIORITY IF NEAR PARITY)

**Evidence**:
- TTA with 3D ensemble (rotation, scale, translation on point clouds) is **standard in 3D detection leaderboards** and adds "zero retraining cost" [S9]
- Waymo 3D detection: NMS ensemble + TTA achieved **9.69% AP boost** over baseline [S9]
- MT-Net: WBF (Weighted Boxes Fusion) + TTA brought **3 mAPH improvement** on same single model [S9]
- Widely documented as "easiest and most reliable" leaderboard points in competition settings [S9]

**Mechanism**:
- **TTA on detector**: Rotate, flip, scale the 3D volume; ensemble predictions via voting or NMS.
- **Multi-detector ensemble**: Train 2–3 independent TemporalUNet3D instances with different random seeds / data augmentations; average output before linking.
- **Multi-linker ensemble**: Different cross-attention transformer weights; ensemble edge scores before thresholding.

**Effort**: ~3–5 days (no retraining; inference-time only for TTA; multi-detector requires duplicate training, ~3–4h each).

**Rough impact**: +0.01–0.03 per ensemble axis (stacks: TTA + 2-detector + 2-linker = potential +0.02–0.06 total, but requires compute for storage).

**Note**: Competition allows multiple submissions; ensemble runs offline don't burn submission quota.

---

### Lever 5: Graph-Based Post-Processing & Gap Filling (MEDIUM PRIORITY, MEDIUM EFFORT)

**Evidence**:
- ARGUS framework (July 2026) demonstrates practical tracklet refinement: "reconnecting fragmented trajectory segments across short temporal gaps using endpoint matching" [S10]
- Graph-based cell tracking literature shows gap filling and temporal smoothing consistently improve metrics, especially under sparse annotations [S11]
- Minimum spanning tree and multi-temporal cost functions address high-density regions (exactly our competition scenario) [S11]

**Mechanism**:
- After SimpleNodeTransformer produces edge predictions:
  - Form tracklets by greedily connecting high-confidence edges.
  - Identify gaps (cells present at time t and t+2+ but missing intermediate predictions).
  - Reconnect gaps using spatial coherence (Kalman or optical-flow prediction) + graph-cost minimization.
  - Enforce division topology globally (mother → 2 daughters, consistency checks).

**Effort**: ~1–2 weeks (implement gap-recovery + post-hoc topology enforcement).

**Rough impact**: +0.01–0.02 (orthogonal to detector/linker improvements; most valuable if other levers are already applied).

---

### Lever 6: Metric Exploitation & Conservative Node Prediction (LOWER PRIORITY; HIGH RISK)

**Evidence**:
- Adjusted Edge Jaccard formula: `max(0, jaccard · (1 − 0.1 · (T_pred − T_true) / T_true))` [S2]
- Over-prediction penalty is **only 10% per excess node**, meaning jaccard degradation from missing detections is often **outweighed** by the penalty avoidance [S2]
- Sparse annotations: "predicted nodes/edges lacking ground-truth counterparts incur no penalty if nodes remain unmatched" [S2]
- Division underdetection: "0.1 weight on division Jaccard means edge accuracy dominates — emphasizing edge prediction over division detection minimizes division false positives" [S2]

**Mechanism** (SPECULATIVE; NOT RECOMMENDED):
- Deliberately suppress low-confidence detections to reduce T_pred toward T_true.
- Result: lower recall, but jaccard × (1 − penalty) can outperform higher-recall strategies.
- **RISK**: This is a brittle micro-optimization that doesn't generalize across public/private splits and is scoring-regime specific.

**Effort**: ~2–3 days (tuning confidence thresholds).

**Rough impact**: Possibly +0.005–0.01 on public, but **high risk of regression on private** if private set has different cell density or annotation completeness.

**Recommendation**: **Skip this unless already at 0.94+**; focus on genuine improvements first.

---

### Metric Structure: Key Exploitable Properties

From metrics.md [S2, S4]:

1. **Conservative over-prediction is rewarded**: A Jaccard of 0.9 with (Npred/Ntrue) = 0.95 yields **1.0045× bonus** vs 1.0× if equal. The 10% coefficient is mild.
2. **Sparse annotations**: Predicted tracks *outside* annotated regions are invisible to scoring; only affect node count penalty.
3. **Division window**: ±1 timepoint flexibility means slightly off temporal predictions still score.
4. **Final score weights**: Edge (90%) >> Division (10%), so division over/under-detection matters far less than edge accuracy.

**Implication**: Top teams likely **optimize edge accuracy first**, then refine divisions only if edge is already saturated.

---

## Hypothesized Top-Team Strategy (Ranked by Likely Impact)

| Rank | Lever | Mechanism | Effort | Estimated Impact | Confidence |
|------|-------|-----------|--------|------------------|------------|
| 1 | Training to convergence | Extend epochs, cyclic LR, curriculum learning | Very Low (2–3 builds, <4h) | +0.010–0.030 | Very High |
| 2 | Edge-centric linker (HOCT-style) | Replace node-centric transformer with edge attention + 3D geometric priors | Medium (2–4 weeks) | +0.020–0.050 | High |
| 3 | Multi-detector ensemble + TTA | 2–3 TemporalUNet3D + inference-time rotations/flips | Low–Medium (3–5 days inference, 8–12h training per model) | +0.010–0.060 | High |
| 4 | Division-aware topology enforcement | Dedicated division head + temporal consistency + post-hoc topology | Medium (1–2 weeks) | +0.010–0.025 | Medium |
| 5 | Graph post-processing & gap-filling | Tracklet refinement, minimum spanning tree, Kalman smoothing | Medium (1–2 weeks) | +0.010–0.020 | Medium |
| 6 | Metric exploitation (conservative nodes) | Threshold tuning to minimize over-prediction penalty | Very Low (2–3 days) | +0.005–0.010 | Low (brittle; skip if high-risk) |

---

## Unknowns & Open Questions

1. **Are top teams using HOCT directly, or a variant?** HOCT was published July 2026, competition launched June 2026. Top teams may have developed edge-centric ideas independently or have post-competition knowledge. No public evidence of HOCT adoption on the leaderboard yet.

2. **What is the private/public split behavior?** The gap between public (0.882) and prize leaderboard (0.930+) is 0.048 points on ~29% public data. This may indicate:
   - Public leaderboard is easier (e.g., lower cell density, fewer divisions).
   - Or private leaderboard has annotation artifacts that help ensembles.
   - **Action**: Simulate overfitting risk carefully when balancing public vs. private optimization.

3. **Is anyone using ILP/global optimization on linking?** Literature shows Integer Linear Programming + Hungarian algorithm can improve tracking by 2–5% over greedy methods. No public notebooks found; likely explored by top teams but not shared.

4. **Division weight tuning**: Should division accuracy be weighted higher during training, even if final score weights it at 0.1×? Cell-TRACTR research suggests **yes**, but no ablation visible.

5. **Temporal attention improvements**: The baseline TemporalUNet3D uses temporal attention; have top teams:
   - Increased temporal receptive field?
   - Used bidirectional attention (look ahead + behind)?
   - Added optical-flow guidance for motion coherence?

---

## Recommendations for Next Build

**Immediate (Days 1–3):**
1. **Train baseline to convergence** (100+ epochs, cyclic LR decay, no early stopping). This is the highest-confidence, lowest-effort win. Expect +0.01–0.03.
2. **Tune confidence thresholds** on current model to find sweet spot for recall/precision. Use only public leaderboard; don't overfit.

**Short-term (Week 1–2):**
3. **Implement multi-detector ensemble + TTA** (3 independent detectors, inference-time augmentation). Requires ~12h training overhead but zero architecture changes. Expect +0.01–0.03 and very likely +0.01 (near certain).

**Medium-term (Weeks 2–4):**
4. **Prototype HOCT-style edge attention linker** or a simpler geometry-aware edge-ranking function (e.g., enforce division topology, use 3D distance penalties). This targets known failure modes. Expect +0.02–0.05 if competition heavily penalizes division confusion.

**Lower priority (if time permits):**
5. **Add division-specific post-processing** (topology enforcement, temporal consistency, off-by-one recovery).
6. **Graph-based gap-filling** if tracklet fragmentation is evident in error analysis.

---

## Sources

- [S1] GitHub: royerlab/kaggle-cell-tracking-competition — Baseline repository, metrics.md, explicit undertrain hint. https://github.com/royerlab/kaggle-cell-tracking-competition
- [S2] metrics.md (GitHub) — Adjusted Jaccard formula, metric exploitation structure, exploitable loopholes. https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- [S3] Kaggle Discussion (Biohub baseline posts) — Training strategy and sparse supervision note.
- [S4] Metric specification from baseline — division window, TP/FP/FN definitions, micro-average scoring.
- [S5] arXiv:2607.11754 — Higher-Order Cell Tracking Transformer (HOCT), edge-centric architecture, addresses cell division entanglement and weak edge topology. Published July 2026 by Biohub authors. https://arxiv.org/abs/2607.11754
- [S6] eLife:69380 & bioRxiv — Tracking cell lineages in 3D by incremental deep learning (ELEPHANT), training convergence, curriculum learning benefits. https://elifesciences.org/articles/69380
- [S7] GitHub: royerlab/hoct — HOCT implementation, CLI, documentation (no leaderboard results). https://github.com/royerlab/hoct
- [S8] PLOS Computational Biology / bioRxiv — Cell-TRACTR, division accuracy metric (DivA), linkage confusion as primary error. https://www.biorxiv.org/content/10.1101/2024.07.11.603075v1.full
- [S9] arXiv:2207.04781, Albumentations docs, et al. — Test-time augmentation (TTA) + ensemble on 3D detection leaderboards; Waymo challenge results. https://arxiv.org/pdf/2207.04781
- [S10] arXiv:2607.08297 — ARGUS: tracklet refinement, gap-filling, optical flow + linear assignment. https://arxiv.org/html/2607.08297
- [S11] PLOS One / bioRxiv — Graph-based cell tracking, gap-filling, multi-temporal cost functions, minimum spanning tree. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0249257

---

## Update Log

- **2026-08-03**: Initial deep-dive. Competition ongoing; no winner writeups available. Recommendations based on literature, metric structure, and baseline hints.
