# Kaggle Competition Recheck: Notebooks, Discussions, SOTA Models (Aug 2026)

**Date**: 2026-08-04
**Status**: partial
**Supersedes**: none

## TL;DR

No public Kaggle notebooks above 0.90 score detected in indexed search results (unverified); recent forum activity (José Freitas synthetic dataset, training time Q&A, ensemble diversity discussion) not deeply indexed; SOTA pretrained 3D models identified: Ultrack (Nature Methods 2025), Cell-TRACTR, CellSeg3D, SAM2-based tracking, CELLECT (2025).

## Question

Fact-finding sweep on: (1) Public Kaggle notebooks >0.90? (2) Recent forum posts (since 2026-08-01) on methods? (3) SOTA pretrained 3D models for developmental microscopy?

## Findings

### 1. Public Kaggle Notebooks with Scores >0.90

**Status**: UNVERIFIED — no notebooks with scores >0.90 found in search or indexable results [S1, S2].

- **Known public notebooks** (no scores disclosed in indexed search): 
  - "Biohub Cell Tracking Solution" by kaiwalyaatulraut [S1]
  - "Biohub Cell Tracking - Classical Baseline" by xiaoleilian (posted July 2, 2026) [S1]
  - "Biohub Cell Tracking: Learned Graph w Gap Recovery" by pilkwang [S1]
  - "Biohub Cell Tracking: Data Model, EDA, Baseline" by pilkwang (posted July 1, 2026) [S1]

- **Search limitation**: Kaggle leaderboard pages are JavaScript-heavy; WebFetch returned page title only, no score table. Direct score comparison requires manual visit to https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard [S2].

- **Prior check (2026-08-03) confirmed**: No notebooks >0.90 visible in that snapshot; current check returns same result (unverified, likely still true).

---

### 2. Recent Forum Posts (Aug 1–4, 2026)

**Status**: PARTIAL — recent activity confirmed but specific post content not retrieved.

- **Recent posters identified**: 
  - Davit Khantadze (post 18 hours ago ≈ 2026-08-04 06:00 UTC) [S3]
  - José Freitas (1 day ago ≈ 2026-08-03 08:00 UTC) — announced "free dataset of 18.5 GB of fully-labelled synthetic 3D microscopy with 165k labelled divisions" [S3]

- **Inferred discussion topics** (from Kaggle discussion summary text): 
  - Synthetic dataset availability for pretraining [S3]
  - Training time per epoch of baseline model (unanswered) [S3]
  - Model/feature diversity and "two-seed logit-blend plateau" (ensemble saturation question) [S3]
  - Submission score issues (recent posts last 3–5 days) [S3]

- **Search limitation**: Kaggle discussion thread requires JavaScript rendering; WebFetch returned only page title and discussion page index URL. Specific post titles, method descriptions, and dates require direct forum visit: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion [S3].

- **Methods NOT found in indexable posts**: Detailed discussions of association/linking, edge/affinity transformers, detection recall tuning, or metric structure may exist but are not surfaced in search result summaries.

---

### 3. SOTA Pretrained 3D Models (Developmental Microscopy)

**Status**: SETTLED — multiple candidates identified with publicly available weights.

#### Ultrack (Nature Methods, 2025)

- **What**: Scalable cell tracking method handling 2D, 3D, multichannel time-lapse. Tested on terabyte-scale zebrafish, fruit fly, nematode embryos [S5, S6].
- **Pretrained**: Yes. Torch-script neural network weights (foreground + contour prediction) available via https://public.czbiohub.org/royerlab/ultrack/ [S5].
- **GitHub**: https://github.com/royerlab/ultrack [S5, S6].
- **Input**: Raw 3D volumetric fluorescence time-series; designed for crowded developmental tissue [S5].
- **Reference**: Royer et al., Nature Methods 22:2423–2436 (2025) [S5].

#### Cell-TRACTR (Transformer-based, 2025)

- **What**: End-to-end DETR-based transformer for simultaneous segmentation + tracking of cells across frames without post-processing [S4].
- **Novel metric**: Cell-HOTA (extends HOTA to cell divisions, handles ±1-frame annotation ambiguity) [S4].
- **Pretrained**: Yes. Model checkpoints on Zenodo (https://zenodo.org/records/14509424) [S4].
- **Input**: Time-lapse microscopy sequences (tested on E. coli 256×32px and mammalian 584×600px nuclear staining) [S4].
- **Handles**: Cell divisions natively; no post-processing needed [S4].

#### CellSeg3D (Self-supervised 3D Segmentation, eLife 2025)

- **What**: Self-supervised + supervised 3D cell segmentation for fluorescence microscopy; napari GUI + Jupyter interface [S7].
- **Pretrained**: Yes. WNet3D (self-supervised, mesoSPIM trained unlabeled) + SwinUNetR + SegResNet (supervised) on HuggingFace [S7].
- **Weights location**: Auto-download via plugin or Colab; https://github.com/AdaptiveMotorControlLab/CellSeg3D [S7].
- **Input**: Raw 3D fluorescence volumes (light-sheet, confocal); normalized to 1st/99th percentile [S7].
- **Note**: Segmentation only, not tracking [S7].

#### SAM2-based Cell Tracking (2025)

- **What**: Zero-shot 2D and 3D cell tracking framework integrating Segment Anything 2 (SAM2) for reconstruction of cell lineages [S8].
- **Models**: Uses pretrained SAM2 + SAM-Med3D (fine-tuned on volumetric medical images) [S8].
- **Architecture**: Linking-only approach for small 3D; simultaneous tracking+segmentation via SAM-Med3D fine-tuning for large-scale 3D+time [S8].
- **Coverage**: 2D and 3D data types [S8].
- **Code**: Available on GitHub [S8].

#### CELLECT (Nature Methods, 2025)

- **What**: Contrastive embedding learning for large-scale efficient cell tracking; pretrained model generalizes across imaging modalities and species [S9].
- **Pretrained**: Yes. Trained on single public dataset; applicable to zebrafish, mouse, and other organisms [S9].
- **Capabilities**: Real-time 3D tracking; demonstrated on B-cell tracking in mouse lymph nodes (large-scale divisions) [S9].
- **Speed**: 50-fold faster than prior methods on developmental datasets [S9].
- **Reference**: Nature Methods (2025) [S9].

#### Cellpose3D (Segmentation, 2025)

- **What**: Generalist segmentation algorithm with 3D support via stitched 2D slices and contextual layer awareness [S10].
- **Pretrained**: Yes. Pretrained 2D/3D models auto-download on first run; nuclei + cyto models available [S10].
- **3D GPU**: Flow3D smooth + ortho-aware models; GPU-accelerated mask creation in 2D/3D [S10].
- **GitHub**: https://github.com/MouseLand/cellpose [S10].
- **Note**: Segmentation only; slower than StarDist on dense 3D but higher generalization [S10].

#### StarDist3D (Object Detection, 2024–2025)

- **What**: Star-convex shape-based 3D nuclei detection; pretrained weights for 3D [S11].
- **Pretrained**: Mostly 2D models; 3D pretrained models available but less developed than 2D [S11].
- **GitHub**: https://github.com/stardist/stardist [S11].
- **Trade-off**: High precision (0.81) but lower recall (0.63) on dense nuclear tissue; complements U-Net recall [S11].
- **Note**: NOT tested by you; segmentation only, no tracking [S11].

---

### Summary Table: Pretrained 3D Models

| Tool | Type | Modality | Input | Weights | GitHub / URL | Status |
|------|------|----------|-------|---------|---|---|
| Ultrack | Tracking+segmentation | 3D time-lapse | Raw fluorescence | ✅ Available | https://github.com/royerlab/ultrack | 2025, prod-ready |
| Cell-TRACTR | Tracking+segmentation | Time-lapse | 2D stacks | ✅ Zenodo | https://zenodo.org/records/14509424 | 2025, SOTA metric |
| CellSeg3D | Segmentation | 3D fluorescence | Raw volumes | ✅ HuggingFace | https://github.com/AdaptiveMotorControlLab/CellSeg3D | Self-supervised |
| SAM2-based | Tracking | 2D/3D | Time-lapse | ✅ SAM2/SAM-Med3D | GitHub | Zero-shot framework |
| CELLECT | Tracking+embedding | 3D time-lapse | Fluorescence | ✅ Pretrained | Nature Methods 2025 | 50× faster |
| Cellpose3D | Segmentation | 3D (via 2D slices) | Fluorescence | ✅ Auto-download | https://github.com/MouseLand/cellpose | Generalist |
| StarDist3D | Segmentation | 3D | Fluorescence | ⚠️ Limited | https://github.com/stardist/stardist | High precision, lower recall |

---

### What You Already Tested (Noted Rejections)

- **Trackastra** (user note): Tested, insufficient complementary recall; 2D-only transformer linker [S12].
- **Cellpose-SAM v2** (user note): Tested, no complementary recall improvement [S12].
- **StarDist3D** (user note): Not tried [S12].

---

## Open Questions

- **Kaggle leaderboard transparency**: Why are public scores not indexed? Likely JavaScript-rendered, but official API docs would confirm.
- **Forum post granularity**: Recent activity exists (José Freitas, Davit Khantadze) but specific method details (association strategies, metric tuning) not surfaced; may require manual forum visit.
- **SOTA ensemble**: Do top teams (0.930–0.947 pack) combine Ultrack + Cell-TRACTR + CELLECT, or is one sufficient? Evidence not available in public notebooks.

---

## Sources

- [S1] Kaggle notebooks search results for "biohub cell tracking" site:kaggle.com/code, accessed 2026-08-04 08:47 UTC. https://www.kaggle.com/code/kaiwalyaatulraut/biohub-cell-tracking-solution
- [S2] WebFetch attempt on https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/leaderboard, 2026-08-04 08:48 UTC. JavaScript-heavy page; title only returned.
- [S3] WebSearch: "biohub-cell-tracking-during-development" kaggle discussion August 2026, results from 2026-08-04 08:47 UTC. Recent post identifiers extracted from result summary; https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion
- [S4] WebFetch: https://pmc.ncbi.nlm.nih.gov/articles/PMC12101859/, Cell-TRACTR paper, 2026-08-04 08:47 UTC.
- [S5] WebSearch: "Ultrack pretrained weights 3D cell tracking github 2025 2026", 2026-08-04 08:47 UTC. https://www.nature.com/articles/s41592-025-02778-0 (Nature Methods); https://github.com/royerlab/ultrack
- [S6] WebFetch: https://github.com/royerlab/ultrack, 2026-08-04 08:48 UTC.
- [S7] WebFetch: https://elifesciences.org/articles/99848, CellSeg3D, 2026-08-04 08:47 UTC.
- [S8] WebFetch: https://arxiv.org/html/2509.09943v1, SAM2-based cell tracking, 2026-08-04 08:47 UTC.
- [S9] WebSearch: "CELLECT contrastive embedding cell tracking Nature Methods" 2025, 2026-08-04 08:48 UTC. Redirect on Nature link but result abstract captured.
- [S10] WebSearch: "Cellpose3D pretrained weights 3D cell segmentation github", results 2026-08-04 08:48 UTC. https://github.com/MouseLand/cellpose
- [S11] WebSearch: "StarDist 3D cell nuclei segmentation pretrained weights github", 2026-08-04 08:48 UTC. https://github.com/stardist/stardist
- [S12] User context: Trackastra tested/rejected; Cellpose-SAM v2 tested/no-gain; StarDist3D not tried. (CLAUDE.md project notes.)
