# Association & Motion SOTA for Dense 3D Cell Tracking — Corrective Pass

**Date**: 2026-08-10
**Status**: settled (verification pass; every claim carries a primary source or is marked UNVERIFIED)
**Supersedes**: corrects specific claims in
[`2026-08-03_dense-association-training.md`](2026-08-03_dense-association-training.md) and
[`2026-08-03_mountable-detectors-sota.md`](2026-08-03_mountable-detectors-sota.md)
(neither is retired wholesale — see "Corrections to prior docs")

## TL;DR

The single most load-bearing thing current SOTA does that we do not: **Trackastra injects relative
spatio-temporal position into every attention layer via RoPE, plus a distance-based attention mask** [V1] —
i.e. the association transformer natively sees *relative displacement and time gap*, which is exactly the
representation our permutation-invariant temporal attention lacks. Second: **learned motion beats
constant-velocity Kalman specifically in the dense/non-linear regime** (DanceTrack: Mamba 52.1 vs KF 47.8
HOTA [V6]), while in the near-linear regime the gap collapses — so a learned relative-motion encoding is the
right bet for us, an explicit Kalman prior is not. Third: **hard-negative mining is real but narrower than our
prior doc claimed** — verified in CellTracker-GNN (multi-similarity loss + hard mining [V9]); Trackastra does
*not* mine hard negatives, it gets them structurally by masking attention to a spatial neighbourhood [V1].
Fourth: **Trackastra has no 3D pretrained weights** — the released models are `general_2d`, `ctc`,
`general_2d_w_SAM2_features` only [V2], and all paper experiments are 2D [V1]. Fifth: **StarDist ships no
pretrained 3D model at all** [V13], so the prior doc's `StarDist3D.from_pretrained('3D_demo')` ensemble recipe
does not do what it claims.

## Question

Which association techniques (hard negatives, metric learning, bidirectional consistency, neighbourhood
attention) are actually load-bearing in 2024–2026 SOTA; how do current trackers encode **motion/kinematics**
for association, and does motion help where the nearest neighbour is usually wrong; and which concrete
software is realistically mountable in an offline Kaggle kernel for 3D+t light-sheet data?

Our situation: detection saturated (recall 0.992–0.993); 100% of dense mislinks are the model scoring a wrong
near neighbour above the true farther successor; our backbone's temporal attention has **no positional
encoding over time**, so it cannot represent velocity.

---

## Findings

### 1. Association under crowding — what is load-bearing

#### 1a. Relative spatio-temporal positional encoding (VERIFIED, and the biggest gap for us)

Trackastra encodes each detection as coordinates + timepoint + shallow image/morphology features
(mean intensity, object area, inertia tensor), applies **learned Fourier spatial positional encodings** to
positions, and then injects **rotary positional embeddings (RoPE) in every attention layer** to
"directly and efficiently inject relative spatial and temporal information", additionally masking attention
between tokens further apart than `d_max` [V1].

Why this matters for us: RoPE over (space, time) makes attention a function of the *difference* of positions
and timestamps. A model with that can represent "this candidate is displaced by Δx over Δt consistent with the
cell's prior displacement" — i.e. velocity — without any explicit motion model. Our temporal attention with no
PE is permutation-invariant over time and provably cannot.

TWiX independently arrives at the same ingredient from the general-MOT side: coordinates min-max normalised to
[-1,1], linearly projected, with **fixed sinusoidal positional encoding carrying the timestamps** [V4].
CAMELTrack likewise tokenises each cue and adds "sinusoidal positional encoding based on temporal distance
from the current frame" [V7].

**Three independent SOTA lines all encode time explicitly into the association tokens.** This is the
standard we are missing.

#### 1b. Neighbourhood restriction — real, but implemented as masking/kNN, not "Neighborhood Attention"

- Trackastra: attention mask disabling attention beyond distance `d_max`; temporal window size s=6 [V1].
- Ultrack: candidate links restricted by KDTree — "query the 2k-nearest neighbors" then "select the k with the
  highest IoU", explicitly to avoid exhaustive pairwise comparison [V5].
- CellTracker-GNN: edges only within a dynamic radius
  `N_R = α · max(max_i size_BB_i, max_j size_move_j)`, α ∈ {2,4} by sequence density [V9]. Note the radius is
  scaled by the **maximum observed movement** — a motion-scale prior baked into graph construction.
- Cell-TRACTR: **deformable attention** following Deformable DETR — "In the encoder, deformable self-attention
  is used to further enhance the multi-scale features" [V3]. This is *not* NATTEN-style Neighborhood Attention
  (see Corrections).

Load-bearing verdict: restricting the candidate set is universal, but it is an efficiency/precision device on
the *candidate graph*, not a special attention operator. We already do this via our linker's candidate set.

#### 1c. Hard-negative mining / metric learning — verified in one cell-tracking method, not universal

- **CellTracker-GNN (VERIFIED)**: node features = 128-d deep-metric-learning embedding (ResNet18) trained with
  **multi-similarity loss using a hard mining strategy**, concatenated with spatio-temporal features (cell
  centre coordinates, frame number, intensity min/max/mean, and when masks exist, area and bounding-ellipse
  axes). Edge classifier trained with weighted cross-entropy with adaptive weights for the inactive-edge class
  imbalance [V9].
- **Trackastra (VERIFIED NEGATIVE)**: the extraction of the method section shows **no hard-negative mining**;
  negatives are all non-matching detections inside the temporal window, with the `d_max` attention mask
  concentrating capacity on the near set [V1]. Its loss is
  `L = L_BCE(A, Φ(Â), W) + λ·L_BCE(A, σ(Â), W)`, λ=1e-2, where Φ is the **parental softmax**
  `Ã_ij = exp(Â_ij) / (1 + Σ_{i'∈P_j} exp(Â_i'j))` over the previous frame's detections `P_j` (≤1 parent), and
  W upweights dividing cells (λ_div=10) vs continuing tracks (λ_cont=1) [V1].
- **CELLECT (VERIFIED to exist, Nature Methods, Oct 2025)**: contrastive learning of latent embeddings of
  cellular structures; a model pretrained on a single public dataset transfers across modalities and species
  including **light-sheet**, and reports faster processing than prior deep methods [V10]. We did **not** verify
  its negative-sampling scheme or any head-to-head numbers.

Verdict: the *structural* insight from our prior doc survives — the competitive negatives are the near
neighbours — but the dominant implementation is **normalisation over the near candidate set** (parental
softmax / contrastive denominator), not an explicit mining loop. Our observed failure (0.686 on a wrong near
neighbour vs 0.134 on the true one) is exactly what a softmax-over-competing-parents is designed to fix, and
we should try that before building a mining pipeline.

#### 1d. Bidirectional consistency — verified with an ablation, but from general MOT

TWiX defines `L_bdrC = L_fwdC + L_bwdC` (forward = negatives in the same row of the affinity matrix, backward
= same column) and **ablates it**: the bidirectional form beats forward-only, backward-only, binary
cross-entropy, focal loss, and triplet variants on KITTIMOT-car, KITTIMOT-pedestrian and MOT17 [V4]. That is
genuine, ablated evidence that the bidirectional contrastive objective is the better loss for coordinate-only
association.

Caveat: TWiX is a **pedestrian/dance/KITTI MOT paper**, not cell tracking [V4]. Transfer is plausible
(coordinates-only association is the same problem shape) but unproven on microscopy.

#### 1e. Mitosis/lineage-aware global assignment

"Cell Tracking according to Biological Needs" (IEEE TMI 2025) formulates a **non-bijective, mitosis-aware
assignment** inside multi-hypothesis tracking: one parent may take two daughters, mitosis costs derived from an
**Erlang distribution over cell age / expected lifetime**, combined with global proliferation constraints;
reported ~6× improvement on biologically-inspired metrics over nine CTC datasets [V8]. Motion representations
are lifted into probabilistic spatial densities via **test-time augmentation with spatial shifts (0,1,4,8 px)**,
whose variance is the uncertainty estimate [V8]. Isolated ablation numbers for the uncertainty component were
**not found** in the HTML we fetched — treat "how much the uncertainty is worth" as UNVERIFIED. Datasets named
in the fetched excerpt are 2D (BF-C2DL-HSC, BF-C2DL-MuSC, PhC-C2DH-U373); 3D support UNVERIFIED.

---

### 2. Motion / kinematics — the direct answer

**Taxonomy of what current methods actually do:**

| Encoding | Who | Verified evidence |
|---|---|---|
| Relative spatio-temporal PE inside attention (RoPE) | Trackastra | [V1] |
| Sinusoidal temporal PE on coordinate tokens | TWiX, CAMELTrack | [V4], [V7] |
| Learned per-pixel displacement head (t → t−1 centre) | EmbedTrack | [V11] |
| Learned motion predictor over a trajectory window (MLP/RNN/LSTM/Transformer/Mamba) | MotionTrack, MambaTrack | [V6] |
| Explicit constant-velocity Kalman | SORT/ByteTrack family; OC-SORT is the correction to it | [V12] |
| Dense optical flow as motion prior | ARGUS (Farneback + LAP + tracklet refinement) | [V14] |
| No motion model at all, by design | TWiX; Ultrack (IoU-overlap objective) | [V4], [V5] |

**The key evidence on whether motion helps in the dense regime:**

- "Exploring Learning-based Motion Models in MOT" compares KF vs MLP/RNN/LSTM/Transformer/Mamba. On
  **DanceTrack (non-linear, dense)** Mamba reaches **52.1 HOTA vs KF 47.8**; MambaTrack beats ByteTrack (KF) by
  **8.2 HOTA**. On **MOT17 (near-linear)** the advantage collapses — MambaTrack 61.1 vs ByteTrack 63.1 [V6].
  Window ablation: 5 frames best on DanceTrack; up to 25 frames better on MOT17/SportsMOT [V6].
- OC-SORT's stated premise: high frame rate makes linear motion a good approximation but **amplifies
  sensitivity to state-estimation noise, "where the noise of displacement can be of the same magnitude as the
  actual object displacement"** [V12]. That is precisely our regime — small inter-frame displacement, dense
  neighbours.
- CAMELTrack replaces the Kalman filter entirely with a learned temporal encoder; ablation on SportsMOT val:
  EMA appearance-only baseline 76.0 HOTA → temporal encoder with bounding-box (spatial/motion) cue **79.2
  (+3.2)** → full CAMEL 81.9. Adding keypoints *hurt* on that setting (71.3) [V7].
- TWiX is the counter-example worth respecting: it deliberately uses **no motion prior, no IoU, no camera-motion
  compensation**, arguing that extrapolative motion models "provoke drifts due to the autoregressive nature",
  and still reaches SOTA on DanceTrack (62.1 HOTA), while only matching on MOT17 where camera-motion estimation
  helps others [V4].

**Synthesis for our case (this is inference, not a cited claim):** the evidence converges on *relative
displacement made visible to a learned discriminator*, and diverges on *extrapolative generative motion models*.
Kalman/constant-velocity is the weakest option in the dense non-linear regime (both OC-SORT [V12] and the motion
survey [V6] say so directly). The cheap, well-evidenced move is to give our attention the Δposition/Δt signal
(RoPE or sinusoidal temporal PE) and let it learn kinematics, rather than bolt on a Kalman prior. No source we
found quantifies motion's contribution specifically in **3D developmental light-sheet crowding** — that gap is
real and we should say so.

**Ultrack is the notable dissent for our exact data type**: on the 3.4 TB zebrafish light-sheet dataset
(791×448×2174×2423 voxels, 21.5 M cell instances) it uses **no optical flow** — the objective is maximum total
IoU overlap between adjacent-frame segmentation hierarchies [V5]. The shipped software later added *optional*
flow-field alignment (`ultrack.imgproc.flow`, a `flow_field_3d` example on Tribolium), and the docs recommend it
only "if the movement is more complex, with cells moving in different directions" [V15][V16]. So in the one
method with published whole-embryo light-sheet results, motion is an optional pre-alignment, not the core signal.

---

### 3. Software candidates

| Software | URL | Licence | 3D pretrained weights, offline? | Replication effort | Kaggle-offline fit |
|---|---|---|---|---|---|
| **Trackastra** | github.com/weigertlab/trackastra | BSD-3-Clause [V2] | **No 3D weights.** Released models: `general_2d`, `ctc`, `general_2d_w_SAM2_features`, as GitHub-release zips [V2] — downloadable, so mountable as a dataset, but 2D-trained. Paper is 2D-only [V1] | `pip install trackastra` / conda-forge. Greedy/LAP linking need no solver; ILP mode uses `motile` with free **SCIP** or Gurobi, auto-falls-back to SCIP [V2] | **Good as an architecture to copy, poor as a drop-in.** Input shape is `time,(z),y,x` so the code path admits z [V2], but you would train 3D yourself |
| **Ultrack** | github.com/royerlab/ultrack | BSD-3-Clause [V17] | N/A (not a weights-based model — segmentation-hypothesis selection) | `pip install ultrack`. Gurobi "optional but recommended"; docs do not document a free-solver path on the repo page [V17] | Plausible; **Gurobi licence is the risk** in an offline kernel. This is the method behind the competition's own lineage [V5] |
| **Cellpose / Cellpose-SAM (`cpsam`, `cpsam_v2`) & CellposeDINO (`cpdino`, `cpdino-vitb`)** | github.com/MouseLand/cellpose | Code BSD-3; **"All Cellpose models are trained on data that is licensed under CC-BY-NC"** [V18] | Yes — auto-download to `models.MODELS_DIR` on first use, or manual HuggingFace download (`huggingface.co/mouseland/cellpose-sam`) [V19]. Model names `cpsam_v2` and `cpdino-vitb` are real and documented [V19] | `pip install cellpose[gui]`; PyTorch [V18] | Works offline once weights are mounted. **Non-commercial weight licence is a real flag** for competition use. Already tested by us: zero complementary recall |
| **StarDist / StarDist3D** | github.com/stardist/stardist | BSD-3-Clause [V13] | **No pretrained 3D model exists** — only `2D_versatile_fluo`, `2D_paper_dsb2018`, `2D_versatile_he`; 3D must be trained by the user [V13] | `pip install stardist`; **TensorFlow** backend [V13] | Poor. TF in a Torch kernel + no 3D weights = train-it-yourself |
| **Cell-TRACTR** | checkpoints at zenodo.org/records/14509424 [V3] | Paper CC-BY; repo licence not stated in the article [V3] | 2D only (E. coli mother machine, DynamicNuclearNet) [V3] | Deformable-DETR stack (deformable attention CUDA ops) — heavy | Poor for us. 2D, bacteria/2D-culture domain |
| **CellTracker-GNN** | github.com/talbenha/cell-tracker-gnn | **CC-BY-NC 4.0** [V20] | Pretrained CTC models in GitHub **Releases** [V20]; validated on 2D **and 3D** (Fluo-N3DH-SIM+) [V9] | conda + PyTorch 1.8 + PyTorch-Lightning 1.4.9 + PyG + Hydra + **faiss-gpu** [V20] — pinned old stack, painful | Mediocre: dependency archaeology + NC licence. **But its feature/loss design (DML + multi-similarity + hard mining, kNN-radius graph) is the most directly copyable to our edge scorer** |
| **EmbedTrack** | git.scc.kit.edu/kit-loe-ge/embedtrack ; github.com/kaloeffler/EmbedTrack | not stated in the paper excerpt [V11] | 2D CTC only [V11] | PyTorch; single network | Poor as a drop-in (2D); **the tracking head idea is the takeaway** — offsets from pixels at t to the cell centre at t−1, i.e. a learned dense displacement, trained with a Lovász hinge tracking loss [V11] |
| **BiologicalNeeds (mitosis-aware MHT)** | github.com/TimoK93/BiologicalNeeds | not verified | 2D datasets named; 3D UNVERIFIED [V8] | builds on EmbedTrack motion + MHT | Low priority; **borrow the Erlang cell-cycle mitosis cost** [V8] |
| **ELEPHANT** | github.com/elephant-track (org) — the prior doc's `github.com/phot-lab/elephant` **404s** [V21] | not verified | Interactive client-server (Mastodon/Fiji + Python server) [V21] | GUI/annotation loop | **Not viable headless.** 3D+4D nuclei detection & linking (eLife 2021) [V21] |
| **CELLECT** | Nature Methods s41592-025-02886-x [V10] | not verified | Claims cross-modality/species transfer incl. light-sheet from a single-dataset pretrain [V10] | code availability not verified | **Worth one look** — closest published claim to our modality |
| **ARGUS** | arXiv:2607.08297 [V14] | not verified | Unsupervised: adaptive detection + dense Farneback flow + frame-to-frame LAP + tracklet refinement; CTC DET 0.905–0.971, TRA 0.897–0.964 [V14] | classical CV, light deps | Cheap to reimplement; **the flow-prior baseline to beat**, though CTC-2D-scale numbers do not transfer to 66k detections/frame |

---

## What we are missing (versus what current methods treat as standard)

1. **Explicit relative time/space in the association attention.** Trackastra RoPE-per-layer [V1]; TWiX and
   CAMELTrack sinusoidal temporal PE [V4][V7]. We have none. This is the #1 gap and it is exactly the
   mechanism that would let the scorer prefer a farther-but-kinematically-consistent successor over a static
   near neighbour.
2. **A normalisation over the competing-parent set, not independent pairwise sigmoids.** Trackastra's parental
   softmax Φ with the ≤1-parent structure and division-upweighted W [V1]. Our symptom (wrong neighbour 0.686 >
   true 0.134) is a competition-among-candidates failure; a per-pair BCE never forces the comparison.
3. **Bidirectional (row+column) contrastive objective**, ablated to beat BCE/focal/triplet [V4]. We score
   source→target only.
4. **A learned dense displacement / motion representation over a short window.** EmbedTrack's t→t−1 offset head
   [V11]; learned motion predictors beating KF by 4.3 HOTA in the non-linear dense regime [V6]. Note the window
   ablation: **5 frames was optimal for the dense non-linear dataset** [V6] — short, not long.
5. **A biologically-parameterised mitosis cost** (Erlang over cell age) rather than a flat division term [V8],
   which matters because our score has a division component.
6. **Metric-learning appearance embedding with hard mining** as an *additional feature channel* alongside
   position — the CellTracker-GNN recipe [V9], which is validated in 3D (Fluo-N3DH-SIM+).
7. **Global, mitosis-aware, multi-hypothesis assignment** rather than a single min-cost-flow pass [V8]; and, in
   the whole-embryo light-sheet regime specifically, **segmentation-hypothesis selection** (Ultrack) where
   linking and segmentation are decided jointly [V5]. We fix detections first, which forecloses that.

**Honest gap in the literature:** no source we found reports a controlled measurement of motion-feature
contribution in **3D developmental light-sheet crowding**. Every motion ablation we verified is 2D
pedestrian/dance/CTC-2D. Ultrack, the one method with published whole-embryo light-sheet results, uses no flow
in its core objective [V5].

---

## Corrections to prior docs

| # | Prior claim | Verdict | Evidence |
|---|---|---|---|
| C1 | Cell-TRACTR uses "**Neighborhood Attention** localised to nearest neighbours" (dense-association doc, §Spatial Context, [S4]) | **WRONG.** It uses **deformable attention** per Deformable DETR: "In the encoder, deformable self-attention is used to further enhance the multi-scale features" | [V3] |
| C2 | Cell-TRACTR is a model for scenes "crowded with cells" relevant to us | **MISLEADING.** 2D only: E. coli in a mother-machine microfluidic device and DynamicNuclearNet 2D culture | [V3] |
| C3 | "Hard negative mining is **the dominant strategy** … used across cell detection **and tracking**" | **OVERSTATED / MIS-SOURCED.** Its only cited support [S2] is PseudoCell, a *detection* paper. Trackastra does **not** mine hard negatives. Verified hard mining in cell *tracking* exists only in CellTracker-GNN (multi-similarity loss + hard mining) | [V1], [V9] |
| C4 | TWiX presented alongside cell-tracking methods | **DOMAIN CONFLATION** (the loss claim itself is correct and ablated). TWiX is pedestrian/DanceTrack/KITTI MOT and explicitly uses **no motion prior and no IoU** — so citing it as support for "motion priors help" would be backwards | [V4] |
| C5 | Motion priors: "Kalman filtering with non-constant velocity models integrate velocity priors … cells move with consistent velocity" (implied as a recommended low-effort/high-ROI lever) | **NOT SUPPORTED as stated for the dense regime.** OC-SORT: at high frame rate displacement noise can equal true displacement; learned motion beats KF by 4.3 HOTA on the dense non-linear benchmark, while the advantage vanishes on near-linear data | [V12], [V6] |
| C6 | Detector doc: StarDist3D ensemble via `StarDist3D.from_pretrained('3D_demo')` | **NOT VIABLE.** StarDist ships **no pretrained 3D model**; 3D models must be trained by the user | [V13] |
| C7 | Detector doc table: StarDist3D precision 0.81 / recall 0.63; 3D U-Net recall 0.67–0.77; "Cellpose3 ~0.70" | **UNVERIFIABLE — treat as confabulated.** Source [S5] (ScienceDirect S2667290122000420) returns HTTP 403; we could not confirm any of these figures. They also contradict our own measured detector recall of 0.992 | — |
| C8 | Detector doc: "Expected Gain: +0.01–0.03 edge Jaccard (empirically observed in CTC ensembles) [S6]" | **UNVERIFIABLE.** [S6] is a YOLOv11+StarDist+SAM2 hybrid paper; CTC does not score edge Jaccard. Drop the number | — |
| C9 | Detector doc: ELEPHANT at `github.com/phot-lab/elephant` | **WRONG URL — 404.** Correct org: `github.com/elephant-track`; method published in eLife 2021 | [V21] |
| C10 | Detector doc omits licences | **GAP, and it matters.** Cellpose *weights* are CC-BY-NC (code BSD-3); CellTracker-GNN is CC-BY-NC. Trackastra, Ultrack, StarDist are BSD-3 | [V18], [V20], [V2], [V17], [V13] |
| C11 | Implicit assumption that Trackastra is a usable 3D drop-in | **WRONG.** Paper experiments are 2D and the released weights are 2D-only; the paper says only that it is "expected to scale well to 3D" | [V1], [V2] |
| C12 | ARGUS (arXiv:2607.08297) and CELLECT (Nat Methods 2025) suspected fabricated | **BOTH REAL.** ARGUS posted 2026-07-09; CELLECT published 2025-10-20 | [V14], [V10] |
| C13 | Trackastra loss formula and parental-softmax description | **CORRECT**, and now with the explicit Φ definition and λ_div=10 / λ_cont=1 weights | [V1] |
| C14 | TWiX bidirectional contrastive loss beats alternatives | **CORRECT and ablated** (beats forward-only, backward-only, BCE, focal, triplet) | [V4] |

---

## Sources (verification tier)

Fetched and read this pass:

- [V1] Gallusser & Weigert, *Trackastra: Transformer-based cell tracking for live-cell microscopy*, ECCV 2024 — arXiv:2405.15700 (HTML v2). RoPE per attention layer, learned Fourier spatial PE, `d_max` attention mask, s=6 window, parental-softmax loss, greedy/LAP/ILP(motile) linking, 2D-only experiments. https://arxiv.org/html/2405.15700v2
- [V2] Trackastra repo + `trackastra/model/pretrained.json` — BSD-3, pip/conda install, SCIP fallback for ILP, model list `general_2d` / `ctc` / `general_2d_w_SAM2_features` with GitHub-release URLs, input shape `time,(z),y,x`. https://github.com/weigertlab/trackastra
- [V3] O'Connor & Dunlop, *Cell-TRACTR*, PLOS Comput Biol 21(5):e1013071 (PMC12101859) — deformable attention, track queries, 2D mother machine + DynamicNuclearNet, Zenodo checkpoints 14509424. https://pmc.ncbi.nlm.nih.gov/articles/PMC12101859/
- [V4] Miah, Bilodeau & Saunier, *Learning data association for MOT using only coordinates*, Pattern Recognition 2024 — arXiv:2403.08018. Bidirectional contrastive loss + ablation; sinusoidal temporal PE; no motion prior/IoU by design; DanceTrack 62.1 HOTA, MOT17 63.1. https://arxiv.org/html/2403.08018v1
- [V5] Bragantini et al., *Large-Scale Multi-Hypotheses Cell Tracking Using Ultrametric Contour Maps* — arXiv:2308.04526 (ar5iv). ILP variables/constraints, KDTree 2k→k candidate restriction, **no optical flow**, Gurobi, 3.4 TB zebrafish light-sheet / 21.5 M instances. https://ar5iv.labs.arxiv.org/html/2308.04526
- [V6] *Exploring Learning-based Motion Models in Multi-Object Tracking* — arXiv:2403.10826. KF vs MLP/RNN/LSTM/Transformer/Mamba; DanceTrack Mamba 52.1 vs KF 47.8 HOTA; MOT17 advantage collapses; 5-frame window best on DanceTrack. https://arxiv.org/html/2403.10826v1
- [V7] *CAMELTrack: Context-Aware Multi-cue ExpLoitation for Online MOT* — arXiv:2505.01257. Cues = bbox/confidence, ReID appearance, pose; sinusoidal PE on temporal distance; SportsMOT-val ablation 76.0 → 79.2 (+bbox) → 81.9 (full); no Kalman filter. https://arxiv.org/html/2505.01257v1
- [V8] Kaiser, Schier & Rosenhahn, *Cell Tracking according to Biological Needs* — arXiv:2403.15011v5, IEEE TMI 2025. TTA-based aleatoric uncertainty on EmbedTrack motion offsets; Erlang mitosis cost; non-bijective mitosis-aware MHT assignment; ~6× on biologically-inspired metrics over 9 CTC datasets. https://arxiv.org/html/2403.15011v5
- [V9] Ben-Haim & Riklin-Raviv, *Graph Neural Network for Cell Tracking in Microscopy Videos* — arXiv:2202.04731 (ar5iv). ResNet18 DML 128-d + multi-similarity loss **with hard mining**; node features incl. centre coords, frame number, intensity stats; dynamic-radius edges `α·max(...)`; 2D **and 3D** (Fluo-N3DH-SIM+). https://ar5iv.labs.arxiv.org/html/2202.04731
- [V10] *CELLECT: contrastive embedding learning for large-scale efficient cell tracking*, Nature Methods, 2025-10-20. https://www.nature.com/articles/s41592-025-02886-x
- [V11] Löffler & Mikut, *EmbedTrack* — arXiv:2204.10713 (ar5iv). "tracking offsets of pixels belonging to a cell at t to their cell center at t−1"; three decoder branches (segmentation offsets, seediness, tracking); Lovász hinge tracking loss; 9× 2D CTC datasets. https://ar5iv.labs.arxiv.org/html/2204.10713
- [V12] Cao et al., *Observation-Centric SORT*, CVPR 2023 — arXiv:2203.14360. High frame rate amplifies state-estimation noise "of the same magnitude as the actual object displacement"; constant-velocity valid only over small intervals. https://arxiv.org/abs/2203.14360
- [V13] StarDist repo — BSD-3, TensorFlow, pretrained models `2D_versatile_fluo`/`2D_paper_dsb2018`/`2D_versatile_he`; **no 3D pretrained model**. https://github.com/stardist/stardist
- [V14] *ARGUS: Accelerated, Robust, General, and Unsupervised Cell Tracking Solutions* — arXiv:2607.08297 (2026-07-09). Adaptive detection + dense Farneback flow + LAP + tracklet refinement; CTC DET 0.905–0.971, TRA 0.897–0.964. https://arxiv.org/abs/2607.08297
- [V15] Ultrack docs, *Tuning tracking performance* — "If the movement is more complex, with cells moving in different directions, we recommend using the `flow` functionalities to align individual segments with distinct transforms"; `max_distance` ≈ 1.5× expected movement. https://royerlab.github.io/ultrack/optimizing.html
- [V16] Ultrack `examples/flow_field_3d/tribolium_cartograph.ipynb` — existence of the 3D flow-field path. https://github.com/royerlab/ultrack/tree/main/examples
- [V17] Ultrack repo — BSD-3-Clause, `pip install ultrack`, "gurobi is optional but recommended for best performance". https://github.com/royerlab/ultrack
- [V18] Cellpose repo — code BSD-3-Clause; "All Cellpose models are trained on data that is licensed under CC-BY-NC. The Cellpose annotated dataset is also CC-BY-NC." https://github.com/MouseLand/cellpose
- [V19] Cellpose models docs — `cpsam`, `cpsam_v2`, `cpdino`, `cpdino-vitb` documented; weights auto-download to `models.MODELS_DIR` on first use, manual download from `huggingface.co/mouseland/cellpose-sam`. https://cellpose.readthedocs.io/en/latest/models.html
- [V20] CellTracker-GNN repo — **CC-BY-NC 4.0**; PyTorch 1.8 / Lightning 1.4.9 / PyG / Hydra / faiss-gpu; "The submitted software and pretrained models to the cell tracking challenge are available at the Releases." https://github.com/talbenha/cell-tracker-gnn
- [V21] ELEPHANT — Sugawara et al., eLife 2021 (10.7554/eLife.69380); org `github.com/elephant-track`; Mastodon/Fiji client + Python DL server; 3D nuclei detection + 4D linking. https://elifesciences.org/articles/69380
- [V22] Ultrack (journal version), *Ultrack: pushing the limits of cell tracking across biological scales*, Nature Methods 22:2423–2436, 2025 — **cited but NOT fetched** (nature.com redirected to an auth endpoint). All Ultrack method claims above come from the arXiv preprint [V5] and the docs [V15][V17]. https://doi.org/10.1038/s41592-025-02778-0

Explicitly NOT verified this pass (do not treat as fact): StarDist-vs-U-Net precision/recall figures (C7);
any ensemble edge-Jaccard gain (C8); BiologicalNeeds uncertainty ablation magnitude and 3D support; CELLECT's
negative sampling, licence and code availability; Cell-TRACTR / EmbedTrack repository licences; whether any
CTC leaderboard entry currently uses an explicit velocity feature in 3D.
