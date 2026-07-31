# 3D Cell Detection & Tracking for Fluorescence Microscopy — SoTA Review

**Date**: 2026-07-31  
**Status**: settled  
**Scope**: Detection + linking methods for 3D zebrafish embryo tracking (Kaggle-viable, GPU, offline)

## TL;DR

For Kaggle competitions: **Detection: Cellpose3/cyto3 (simplest, pretrained, 3D + denoising) or Ultrack's ensemble (more robust but slower)**. **Tracking: Trackastra (division-aware transformer, greedy mode is fast) + motile ILP (CBC solver, offline, flexible costs)**. Temporal jitter is solved by centre-of-mass regression or Gaussian subpixel refinement; classical distance-transform + peak-finding won't hold centres stable across frames. Gurobi is off-limits in Kaggle kernels (needs WLS licence in cloud); use CBC or scipy-based solvers instead.

---

## Question

What are the state-of-the-art methods for 3D cell centre detection and linking across time in fluorescence microscopy, particularly for dense nuclei in zebrafish embryos, that:
1. Run offline in Kaggle kernels (no internet after setup)
2. Fit within GPU memory (~10 GB RTX T4 or P100)
3. Complete in ~hours for ~14k-cell volumes over 50+ frames
4. Handle cell divisions (mother → daughter tracking)
5. Address detector jitter (frame-to-frame centre instability)

---

## Findings

### **Detection Methods**

#### **Cellpose & Cellpose3 (cyto3 model)**

- **Architecture & Training**: Deep learning-based generalist; cyto3 model trained on Cellpose Cyto, TissueNet, LiveCell, Lucchi datasets (two-channel: segmentation channel + optional nuclear channel) [S1, S2].
- **3D Support**: Full 3D support; Cellpose3 (Feb 2024) adds learnable **axial restoration** to correct z-degradation in 3D stacks. Improves 3D segmentation performance substantially [S6].
- **Pretrained Weights**: Available out-of-the-box; cyto3 loaded via `models.Cellpose(model_type='cyto3')` in Python or command-line. Napari plugin ships weights; no internet needed after install [S1, S2].
- **Inference Cost**: Benchmark data available in docs, but specific 3D VRAM/time not listed in text; typically on RTX 4070S or A100 batch-size 32 for 2D. 3D images will be heavier; recommend testing on your volume size.
- **Image Restoration (v3)**: Optional restoration module learns to denoise/deblur/upsample images *before* segmentation, boosting robustness on degraded data [S6].
- **Kaggle Viability**: **YES**. Offline-capable, shipped weights, GPU-friendly. Simplest entry point.

**Sources for Cellpose:**
- [S1] Cellpose docs: https://cellpose.readthedocs.io/en/v3.1.1.1/models.html
- [S2] Cellpose training, cyto3 fine-tuning example: https://www.life-science-alliance.org/content/8/6/e202403067
- [S6] Cellpose3 restoration paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11903308/

---

#### **StarDist 3D**

- **Architecture**: Star-convex shape detection + TensorFlow backend. Predicts distance maps in multiple radial directions to localize cell centres and boundaries [S4].
- **3D Capability**: Full 3D support; trained on seeded nuclei datasets.
- **GPU Requirement**: **GPU strongly recommended** for 3D; inference time explodes on CPU. Requires TensorFlow + CUDA + CUDNN [S4].
- **Post-processing**: Non-maximum suppression (NMS) is CPU-only; can parallelize across cores but still slower than GPU on large volumes [S4].
- **Pretrained Weights**: Available; 3D model trained on nuclei. Open-source; downloadable.
- **Kaggle Viability**: **CONDITIONAL**. Works offline, but GPU is mandatory for 3D speed. No licensing issues.

**Sources for StarDist:**
- [S4] StarDist FAQ & docs: https://stardist.net/faq/

---

#### **3D U-Net Variants (NuSeT, NISNet3D)**

- **NuSeT (Nuclear Segmentation Tool)**: Hybrid U-Net + Region Proposal Network + watershed for nuclei separation in crowded 3D data [S7].
- **NISNet3D**: Modified 3D U-Net for instance segmentation of nuclei [S8].
- **Pretrained Models**: Available; trained on multi-modality fluorescence datasets.
- **Offline Capability**: YES; ImageJ plugin available for local analysis [S7].
- **Kaggle Viability**: YES. Fewer "surprises" than Cellpose; standard architecture, easier to fine-tune.

**Sources for U-Net variants:**
- [S7] NuSeT paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008193
- [S8] NISNet3D paper: https://www.nature.com/articles/s41598-023-36243-9

---

### **Temporal Stability of Detection (Solving Jitter)**

**Problem**: Detector outputs bounding boxes or segmentation masks; extracting centres (e.g., via centroid) gives noisy frame-to-frame positions, breaking linking [user report: "good recall, jittery centres"].

**Standard Solutions**:

1. **Soft-argmax / Differentiable Argmax** [S9]: Train detector to output **heatmap likelihood** over a spatial grid, then compute weighted centre via soft-argmax rather than hard max. Enables sub-pixel precision and smooths across frames. Used in pose estimation, landmark detection, stereo matching [S9].

2. **Gaussian Subpixel Refinement**: Post-process each detection's segmentation mask (or distance map) by fitting a Gaussian to the PSF, localizing centre to sub-pixel accuracy. Standard in SMLM (single-molecule localization microscopy) [S10].

3. **Distance Transform + Peak Finding**: Apply Euclidean distance transform to binary foreground, find peaks, or use morphological markers (h-dome, extended minima). Works for moderate density; fails if nuclei touch [S11].

4. **Center-of-Mass Regression (End-to-End)**: Some modern detectors (e.g., newer Cellpose variants, custom U-Nets) output **regression targets** (offset maps) from pixels to centre offsets, allowing learnable refinement [implicit in Cellpose3 + restoration approach].

**Recommendation for Kaggle**: If using Cellpose/StarDist, treat output segmentation as binary mask → extract centre-of-mass, then **optionally post-process with Gaussian fitting** if masks are clean, or **train a lightweight refiner network** (2-3 conv layers) to predict centre offset from feature maps. The Cellpose3 restoration + segmentation pipeline implicitly handles some jitter via learned features.

**Sources for centre stability:**
- [S9] Soft-argmax for localization: https://arxiv.org/pdf/2110.08825
- [S10] Gaussian subpixel fitting in SMLM: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10951805/
- [S11] Distance transform + peak detection: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004970

---

### **Tracking/Linking as Graph Optimization**

**Formulation**: Given detected cell centres at time $t$ and $t+1$, form a **bipartite graph** where:
- Nodes = detections (cells at each timestep)
- Edges = possible links (same cell across frames)
- Costs = distance + appearance similarity + motion priors
- Constraints = each cell has ≤2 children (division), ≥1 parent (except roots)

Solve via **Integer Linear Programming (ILP)** or **Linear Assignment Problem (LAP)**.

---

#### **Trackastra (Transformer-based, Division-Aware)**

- **Method**: Directly learns pairwise **association probabilities** between detected cells within a temporal window, using transformer architecture on cell coordinates + features. Greedy linking or ILP-based for optimality [S12, S13].
- **Division Handling**: Blockwise parental softmax normalization enables **one-to-many associations** (mother → 2 daughters). Tested on CTC datasets and various biological data [S12].
- **Pretrained Weights**: Available in GitHub; general 2D models (e.g., `general_2d`) trained on CTC datasets. **Offline-capable**: models ship with napari plugin, no internet needed after install [S13].
- **Linking Modes**:
  - `greedy`: Fast, no division → simple bipartite matching [~seconds]
  - `greedy_nodiv`: Greedy without division handling
  - `ilp`: Requires motile + solver (CBC/Gurobi) for graph-optimal solution [~seconds to minutes, depending on volume]
- **Inference Cost**: Operates on coordinates only (not images), so very fast after detection (~seconds).
- **Kaggle Viability**: **YES**. Division-aware, pretrained, offline, fast. Works with motile + CBC solver for ILP.

**Sources for Trackastra:**
- [S12] Trackastra paper: https://arxiv.org/abs/2405.15700
- [S13] Trackastra GitHub: https://github.com/weigertlab/trackastra

---

#### **Motile (ILP-based Linking)**

- **Method**: Python library for framing tracking as ILP; users specify linking costs, constraints (division, non-merging), and motion priors [S14].
- **Solvers**:
  - **CBC (Coin-or-Branch-and-Cut)**: Open-source, free. Fallback default [S14]. Slower than Gurobi, uses more memory, harder to install on Windows [ultrack docs].
  - **Gurobi**: Commercial, **free for academic use** but requires license activation. Much faster than CBC. **NOT suitable for Kaggle kernels** (requires WLS – Workgroup License Server – which isn't available in cloud) [S15].
  - **SCIP**: Free open-source solver; motile falls back to SCIP if Gurobi absent [S14].
- **Offline Capability**: **YES**. All solvers run offline after install. Gurobi license must be activated offline beforehand [S15].
- **Integration with Trackastra**: Trackastra's `ilp` mode uses motile internally; pass your cost matrix and constraints [S12].
- **Kaggle Viability**: **YES, with CBC**. Use `solver='CBC'` or `solver='SCIP'`.

**Sources for Motile:**
- [S14] Motile PyPI & docs: https://pypi.org/project/motile-tracker/ and https://funkelab.github.io/motile_tracker/
- [S15] Gurobi academic license + offline: https://support.gurobi.com/hc/en-us/articles/4534601245713-How-do-I-get-started-with-Gurobi-for-academic-users and https://support.gurobi.com/hc/en-us/articles/12249724751121-Support-for-users-with-free-trial-licenses

---

#### **Ultrack (Consensus + Temporal Consistency)**

- **Method**: Generates multiple candidate segmentations (from different algorithms + parameter sets), then solves a tracking problem that **selects optimal segments over time**, leveraging temporal consistency to suppress noise. Acts as both segmenter and tracker [S16].
- **Performance**: Top-scoring algorithm at Cell Tracking Challenge for 3D embryonic datasets (zebrafish, fruit fly, nematode). Halves manual correction time in dense tissue [S16].
- **Solver**: Uses **Gurobi (preferred) or CBC** for optimization [S17]. Same trade-offs as motile.
- **Scaling**: Demonstrated on terabyte-scale datasets on 64 GB RAM laptop (out-of-core intermediate storage) [S17].
- **Offline Capability**: YES, with CBC solver fallback.
- **Kaggle Viability**: **YES, but slower than Cellpose + linking**. Requires running multiple segmenters (expensive); best used if segmentation quality is the bottleneck, not linking.

**Sources for Ultrack:**
- [S16] Ultrack Nature Methods: https://www.nature.com/articles/s41592-025-02778-0 and PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12615266/
- [S17] Ultrack solver & scaling: https://royerlab.github.io/ultrack/

---

#### **LapTrack (Python LAP Solver)**

- **Method**: Solves Linear Assignment Problem (LAP) for frame-to-frame linking. Allows arbitrary cost functions (distance, appearance, motion).
- **Offline Capability**: YES. Pure Python, no internet dependency [S18].
- **Solver**: Scipy-based (Hungarian algorithm); no external solver license needed [S18].
- **Division Handling**: Not built-in; requires custom cost matrix construction.
- **Kaggle Viability**: **YES**. Lightweight, offline, no solver hassles.

**Sources for LapTrack:**
- [S18] LapTrack paper & GitHub: https://github.com/yfukai/laptrack and https://www.biorxiv.org/content/10.1101/2022.10.05.511038v1.full.pdf

---

### **Competition-Specific: Biohub Kaggle 2026**

**Dataset**: Zebrafish embryo high-resolution 3D time-lapse videos; largest publicly available cell-tracking annotation set [S19].

**Metrics** [S20]:
- **Edge Jaccard**: Pairs predicted nodes with ground-truth by centroid distance (≤ 7 µm). Computes `TP / (TP + FP + FN)` for trajectory links. Adjusted for over-prediction: `adjusted = max(0, jaccard · (1 − 0.1 · (T_pred − T_true) / T_true))` [S20].
- **Division Jaccard**: Evaluates predicted cell divisions. Requires strict topology (parent → 2 daughters, correct directed links, unmerged branches). Matched via bipartite matching [S20].
- **Final Score**: `adjusted_edge_jaccard + 0.1 · division_jaccard` (micro-averaged over all videos) [S20].

**Baseline**: Classical baseline available on Kaggle (Cellpose-like + ILP); Ultrack-based solutions are current SOTA for public CTC benchmarks [S16].

**Sources for competition:**
- [S19] Kaggle competition page: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- [S20] Competition metrics: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

---

## Recommended Implementation Paths

### **Path A: Fast, Simple (Cellpose3 + Trackastra)**
1. **Detection**: Cellpose3 `cyto3` model (or custom fine-tune on training set if available).
2. **Centre refinement**: Extract centres via centroid; optionally post-process with Gaussian fitting.
3. **Linking**: Trackastra `greedy` mode (fast, no division) or `ilp` (motile + CBC).
4. **Division**: If greedy, post-process tracks for divisions (split when velocity jumps); if ilp, it's automatic.
5. **Pros**: Pretrained, offline, simple pipeline, fast inference (~hours for 50 frames × 14k cells).
6. **Cons**: Cellpose jitter may hurt linking; needs manual tuning of cost functions.

### **Path B: Robust (Ultrack)**
1. **Run Ultrack end-to-end**: Generates candidates + solves combined segmentation + tracking.
2. **Pros**: State-of-the-art on CTC datasets; handles segmentation uncertainty; minimal tuning.
3. **Cons**: Slower (multiple segmenters run per frame); overkill if segmentation is already good.

### **Path C: Custom (U-Net + CBC Solver)**
1. **Detector**: Train 3D U-Net on annotated training frames.
2. **Centre**: Predict heatmap + offset regression for sub-pixel centres.
3. **Linker**: Motile + CBC with custom cost (IoU + centroid distance + motion prior).
4. **Pros**: Full control, can optimize for competition metric.
5. **Cons**: Requires training data; slower iteration.

---

## Open Questions & Caveats

- **Cellpose3 on zebrafish**: No published 3D benchmark on this exact dataset; recommend testing on a small sample.
- **Gurobi in Kaggle**: Academic license doesn't work in cloud kernels (needs WLS); must use CBC or SCIP.
- **Division annotation**: Depends on ground-truth labels; if not provided, linking-only methods are sufficient.
- **VRAM**: 3D networks can fit in 10 GB (T4) but batch-size will be 1; profiling needed.
- **Temporal smoothing**: Post-linking Kalman filtering or spline fitting may reduce jitter further but hasn't been benchmarked here.

---

## Sources

- [S1] Cellpose 3.1.1 documentation: https://cellpose.readthedocs.io/en/v3.1.1.1/models.html
- [S2] Cellpose on 3D autofluorescence: https://www.life-science-alliance.org/content/8/6/e202403067
- [S4] StarDist FAQ: https://stardist.net/faq/
- [S6] Cellpose3 restoration (Nature Methods, Feb 2024): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11903308/
- [S7] NuSeT nuclei segmentation: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008193
- [S8] NISNet3D instance segmentation: https://www.nature.com/articles/s41598-023-36243-9
- [S9] Soft-argmax localization: https://arxiv.org/pdf/2110.08825
- [S10] Gaussian subpixel fitting (SMLM): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10951805/
- [S11] Distance transform peak detection: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004970
- [S12] Trackastra paper (arXiv 2405.15700): https://arxiv.org/abs/2405.15700
- [S13] Trackastra GitHub: https://github.com/weigertlab/trackastra
- [S14] Motile PyPI & docs: https://pypi.org/project/motile-tracker/
- [S15] Gurobi academic + offline: https://support.gurobi.com/hc/en-us/articles/4534601245713-How-do-I-get-started-with-Gurobi-for-academic-users
- [S16] Ultrack Nature Methods (2025): https://www.nature.com/articles/s41592-025-02778-0
- [S17] Ultrack docs & scaling: https://royerlab.github.io/ultrack/
- [S18] LapTrack GitHub: https://github.com/yfukai/laptrack
- [S19] Biohub Kaggle 2026: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development
- [S20] Competition metrics: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
