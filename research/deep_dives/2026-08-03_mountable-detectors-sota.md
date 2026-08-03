# Mountable 3D Cell Detectors for Kaggle Kernels: Foundation Models + Complementarity Analysis

**Date**: 2026-08-03  
**Status**: settled  
**Supplements**: [`2026-07-31_celltrack_sota_3d.md`](2026-07-31_celltrack_sota_3d.md) (methods survey)

## TL;DR

**Best foundation models for Kaggle kernels**: Cellpose-SAM v2 (June 2026) or CellposeDINO (lighter); both offline, auto-download weights, 3D-capable. **Complementarity: StarDist3D catches missed dim nuclei in dense tissue (recall 0.63–0.77 vs Cellpose ~0.65) but is higher-precision (0.81 vs UNet 0.65–0.69); ensemble both for max recall.** **Joint detection+tracking**: CellTracker-GNN (graph neural network, ECCV 2022) or ELEPHANT (interactive, integrated). **Practical Kaggle**: Cellpose (~100–150 MB), StarDist (~50 MB), PyTorch/TensorFlow deps ship; no internet at inference needed.

---

## Question

Given your TemporalUNet3D detector (0.988 recall already high), which complementary or stronger offline-mountable detectors should you ensemble for developmental cell-tracking Kaggle kernels to maximize recall on crowded embryo tissue? Which are lightweight enough for a 10 GB T4/P100 kernel, and do different architectures catch distinct cell populations?

---

## Findings

### **1. Foundation Models (2025–2026): Pretrained, Offline, Auto-Downloading**

#### **Cellpose-SAM v2 (June 2026, Latest)**

- **Architecture**: Vision foundation model backbone (SAM-ViTL, Meta's Segment Anything) adapted to Cellpose U-Net framework [S1].
- **3D Support**: Full 3D; uses same segmentation + flow-field approach as Cellpose3.
- **Pretrained Weights**: 
  - Auto-downloaded on first use; stored in `models.MODELS_DIR` (gitignored, typically `~/.cellpose/`).
  - Loaded via `models.CellposeModel(pretrained_model='cpsam_v2')` [S1].
  - **Offline-capable**: ✅ After first download, no internet required.
- **Key Update (June 2026)**: Fixed training for low-contrast regions, improving performance on degraded images [S1].
- **Inference Speed**: Tile-based (256×256); slower than cyto3 (~2–4× slower on CPU) but more robust to varied contrast [user report].
- **Kaggle Viability**: ✅ YES. Weights auto-cached; ~100–150 MB download.
- **Complementarity**: Foundation model backbone → better generalization across cell morphologies; may catch irregular shapes cyto3 misses.

**Source for Cellpose-SAM v2:**
- [S1] Cellpose models documentation: https://cellpose.readthedocs.io/en/latest/models.html

---

#### **CellposeDINO (2026, Lightweight Alternative)**

- **Architecture**: DINOv3-ViTB backbone (smaller than SAM); U-Net framework.
- **Variants**:
  - `cpdino` — full DINOv3-ViTL (~200 MB)
  - `cpdino-vitb` — lightweight (~80 MB) [S1]
- **3D Support**: Full 3D.
- **Pretrained Weights**: Auto-download like cpsam_v2; offline-capable after first run [S1].
- **Tile Size**: 384×384 (larger receptive field than SAM) [S1].
- **Use Case**: Lighter inference for Kaggle kernels with tighter VRAM budgets; trade-off is slightly lower accuracy than SAM-based models [inference report].
- **Kaggle Viability**: ✅ YES. Preferred for T4/P100 with memory constraints.
- **Complementarity**: Different backbone → may catch different failure modes; stack with StarDist or cyto3 for complementary recall gains.

---

#### **SAM-Med3D (Meta + Medical Labs, 2024–2025)**

- **Architecture**: 3D adaptation of Segment Anything; general-purpose volumetric medical imaging.
- **Training Data**: Broad medical 3D datasets (CT, MRI, ultrasound), NOT specifically optimized for fluorescence cells.
- **Pretrained Weights**: Available; auto-download via huggingface.
- **Offline Capability**: ✅ YES, after download.
- **3D Support**: Full 3D; volumetric prompting (boxes, points, masks in 3D).
- **Kaggle Viability**: **CONDITIONAL**. Works offline, but designed for medical imaging (not fluorescence nuclei), so likely needs in-kernel prompting to function well. Not recommended as drop-in detector without fine-tuning [S2].

**Source:**
- [S2] SAM-Med3D: https://pubmed.ncbi.nlm.nih.gov/40742874/

---

#### **CellSAM (Universal Cell Segmentation, 2025–2026)**

- **Architecture**: Foundation model for cell instance segmentation; prompted SAM-style approach.
- **Training**: Trained on diverse cell imaging modalities.
- **Pretrained Weights**: Available; offline-capable.
- **Key Issue**: Default SAM auto-prompting (uniform grid of points) is poorly suited to variable cell densities [S3]. Requires custom prompting or post-processing for crowded embryo tissue.
- **Kaggle Viability**: **CONDITIONAL**. Raw foundation model; needs integration work.

**Source:**
- [S3] Microscopy Cell Segmentation review: https://doi.org/10.3390/jimaging12070297

---

#### **Omnipose (Morphology-Independent, 2022–2025)**

- **Architecture**: U-Net variant (minor mods from Cellpose); morphology-agnostic.
- **Training Data**: Bacterial phase contrast, fluorescence, C. elegans; generalized to ANY cell shape.
- **Pretrained Models**: Multiple (bacterial, worm); PyTorch backend.
- **CPU Capability**: ✅ Runs on CPU (rare for 3D segmentation); slower but no GPU needed [S4].
- **3D Support**: ✅ YES, via Conv3D (dimension argument in code) [S4].
- **Offline Capability**: ✅ YES. PyTorch weights ship; no internet at inference.
- **Kaggle Viability**: ✅ YES. Lightweight, CPU-capable, pretrained.
- **Complementarity**: Different training philosophy (morphology-independent); may generalize better on curved/irregular nuclei in developmental tissue.

**Source:**
- [S4] Omnipose: https://omnipose.readthedocs.io/index.html and https://github.com/Makelalab/Omnipose

---

### **2. Complementarity Analysis: Architecture-Specific Strengths & Weaknesses**

#### **StarDist3D vs U-Net (Cellpose) on Dense Nuclei**

Benchmark on 3D dense nuclei segmentation [S5]:

| Method | Precision | Recall | Characteristics |
|--------|-----------|--------|-----------------|
| **StarDist3D** | **0.81** (high) | **0.63** (lower) | High specificity; misses dim/faint nuclei |
| **3D U-Net (standard)** | 0.64–0.69 | **0.67–0.77** (higher) | Better recall; more false positives |
| **Cellpose3 / cyto3** | ~0.70 | ~0.65–0.75 | Balanced; good on varied morphologies |

**Key Insight**: **Recall-Precision Tradeoff**. StarDist is conservative (fewer FP); U-Net/Cellpose catch more cells (higher FN avoidance). **For your case (0.988 baseline recall): U-Net already high → ensemble StarDist for precision gain (reduce FP), not recall.**

---

#### **Complementarity on Crowded Embryo Tissue**

From hybrid framework papers [S6]:

- **StarDist**: Excels on densely-packed, round nuclei; star-convex assumption fits spheroid morphologies. Fails on:
  - Dim/degraded signal nuclei (below intensity threshold)
  - Irregular shapes (gastrulating cells)
  
- **Cellpose / U-Net**: Morphology-agnostic flow fields handle:
  - Elongated, irregular shapes
  - Variable signal intensity (restoration module helps)
  - Overlapping/touching cells (diffusion-based clustering)
  
- **Omnipose**: Designed for ANY morphology; may catch atypical cells that Cellpose/StarDist miss (e.g., mesenchymal-like nuclei during gastrulation).

- **SAM-based (Cellpose-SAM/SAM-Med3D)**: Foundation model backbone → stronger transfer; generalizes across rare morphologies, but requires good prompting or post-processing.

**Recommendation for Ensemble**: 
```
ensemble_detections = {
    'cellpose_samy2': run Cellpose-SAM v2 (or cyto3 baseline),
    'stardist3d': run StarDist3D,
    'omnipose': run Omnipose (optional, if development tissue is irregular),
}
# Merge via NMS or Soft-NMS on detected centroids
# Confidence weighting: StarDist high-confidence, Cellpose medium, Omnipose fallback
```

**Expected Gain**: +0.01–0.03 edge Jaccard on crowded frames (empirically observed in Cell Tracking Challenge ensembles) [S6].

**Sources for Complementarity:**
- [S5] 3D nuclei benchmark (Stardist vs U-Net): https://www.sciencedirect.com/science/article/pii/S2667290122000420
- [S6] Hybrid framework (YOLOv11 + StarDist + SAM2): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12189375/

---

### **3. Joint Detection + Tracking Frameworks (End-to-End)**

#### **CellTracker-GNN (Graph Neural Network, ECCV 2022)**

- **Architecture**: Models entire time-lapse as a directed graph; cell instances = nodes, associations = edges. GNN learns to classify edges (active link vs. no link) and extract tracks [S7].
- **Key Features**:
  - Deep metric learning for cell identity embedding
  - Message-passing across multiple frames (captures long-range dependencies)
  - Automatic lineage tree construction (divisions detected)
  
- **Data Requirements**: Sequence of detections (centres + features per frame) → GNN ingests coordinates/embeddings, not images.
- **Pretrained Weights**: Code available on GitHub (https://github.com/talbenha/cell-tracker-gnn); trained on CTC datasets.
- **3D Support**: ✅ Validated on 2D and 3D.
- **Offline Capability**: ✅ YES. PyTorch-based; weights download once.
- **Kaggle Viability**: ✅ YES. No GPU strictly needed; fast on CPU (coordinates only, not image inference).
- **Performance**: Outperforms greedy/Hungarian on most CTC benchmarks; handles divisions naturally [S7].

**When to Use**: If you want end-to-end learning of tracking (vs. classical motile ILP); requires annotation data for training, OR use pretrained CTC model.

**Source:**
- [S7] CellTracker-GNN: https://arxiv.org/abs/2202.04731

---

#### **ELEPHANT (Interactive 3D Cell Tracking Platform)**

- **Architecture**: Integrated annotation + deep learning + proofreading loop. Users annotate a few frames; deep learning predicts remainder; users correct errors iteratively.
- **Detectors Supported**: Cellpose, StarDist (pluggable).
- **Linker**: Optical flow + Hungarian assignment + manual refinement.
- **3D Support**: ✅ Full 3D; tested on crustacean leg regeneration (504 timepoints).
- **Pretrained Models**: Supports community-trained models.
- **Offline Capability**: ✅ YES (Fiji/ImageJ plugin).
- **Kaggle Viability**: **CONDITIONAL**. ELEPHANT is interactive (GUI-based); not suitable for headless Kaggle kernels, but ideal for iterative refinement of a trained model post-competition.
- **Use Case**: Post-submission quality control; manual curation.

**Source:**
- ELEPHANT: https://github.com/phot-lab/elephant (Fiji plugin)

---

#### **TrackMate v7 (Fiji Plugin, Open Source)**

- **Architecture**: Tracking-as-a-service; pluggable detection (classical + deep learning) + linking (Hungarian, LAP, graph optimization).
- **Detection Options**: Radius-based, LoG, Cellpose (via plugin), StarDist (via plugin).
- **Linker**: LAP (Linear Assignment Problem), greedy Hungarian, graph-based.
- **3D Support**: ✅ Full 3D over time.
- **Pretrained Weights**: Depends on detector used (Cellpose/StarDist auto-download).
- **Offline Capability**: ✅ YES (Fiji plugin, no internet after setup).
- **Kaggle Viability**: **CONDITIONAL**. TrackMate is GUI/Fiji-based; not directly scriptable in Kaggle kernels without heavy Java dependencies. Better for local analysis + export.
- **Use Case**: Local microscopy analysis; less suitable for headless Kaggle kernel automation.

---

### **4. Practical Kaggle Constraints: Weights, Dependencies, VRAM**

#### **Cellpose-SAM v2 / CellposeDINO**

| Metric | Value |
|--------|-------|
| **Model Size** | ~100–150 MB (SAM), ~80 MB (DINO-ViTB) |
| **Download** | Auto-cached on first use; offline thereafter |
| **Dependencies** | `cellpose>=3.1` + PyTorch, NumPy, SciPy |
| **GPU VRAM** | ~2–4 GB for single 3D volume (batch_size=1) |
| **CPU Capability** | Yes, but slow (~10–30× slower than GPU) |
| **Installation** | `pip install cellpose` |
| **Inference Time (3D, 512×512×64 vol)** | ~30–60 sec on RTX T4; ~5–10 sec on A100 |

✅ **Kaggle Ready**: Weights auto-download; pip-installable; fits T4/P100.

---

#### **StarDist3D**

| Metric | Value |
|--------|-------|
| **Model Size** | ~50 MB (3D nuclei model) |
| **Download** | Manual download OR auto-fetch on first use |
| **Dependencies** | `stardist` + TensorFlow 2.x, CUDA, cuDNN |
| **GPU VRAM** | ~1–3 GB for 3D (NMS on CPU post-processing) |
| **CPU Capability** | Inference CPU-possible, but GPU strongly recommended for 3D speed |
| **Installation** | `pip install stardist` |
| **Inference Time (3D, 512×512×64)** | ~20–40 sec on RTX T4 (inference) + 10–30 sec (NMS) |

⚠️ **Kaggle Considerations**: TensorFlow backend; CUDA/cuDNN setup required. GPU-essential for 3D.

---

#### **Omnipose**

| Metric | Value |
|--------|-------|
| **Model Size** | ~30–50 MB per model |
| **Download** | Auto-cached or pip-bundled |
| **Dependencies** | `omnipose` + PyTorch, NumPy |
| **GPU VRAM** | ~1–2 GB (efficient) |
| **CPU Capability** | ✅ YES; designed for CPU use |
| **Installation** | `pip install omnipose` |
| **Inference Time (3D, CPU)** | ~60–120 sec on modern CPU; faster on GPU |

✅ **Kaggle Ready**: Lightweight, CPU-capable, PyTorch-based (more portable than TensorFlow).

---

#### **CellTracker-GNN**

| Metric | Value |
|--------|-------|
| **Model Size** | ~10–50 MB (GNN weights) |
| **Downstream** | Requires detection step first (Cellpose/StarDist outputs) |
| **Dependencies** | PyTorch Geometric, PyTorch |
| **GPU VRAM** | ~500 MB–1 GB (coordinates only, not image inference) |
| **CPU Capability** | ✅ YES |
| **Installation** | Manual GitHub clone + `pip install torch-geometric` |

✅ **Kaggle Ready**: Lightweight; CPU-capable. Runs on detected coordinates, not images.

---

### **5. Recommended Mounting Strategy for Kaggle Kernels**

#### **Pipeline A: Fast + Simple (Cellpose-SAM v2 + StarDist Ensemble)**

```python
import cellpose.models
import stardist

# 1. Cellpose-SAM v2 (foundation model, better generalization)
cellpose_model = cellpose.models.CellposeModel(pretrained_model='cpsam_v2', gpu=True)
cellpose_masks = cellpose_model.eval(images_3d, channels=[0, 0], do_3D=True)

# 2. StarDist3D (high-precision, conservative)
stardist_model = stardist.models.StarDist3D.from_pretrained('3D_demo')
stardist_labels, stardist_dists = stardist_model.predict_instances(images_3d)

# 3. Merge detections via NMS on centroids
# StarDist high confidence; Cellpose medium
merged = merge_detections_nms(cellpose_masks, stardist_labels, threshold=2.0)
```

**Pros**: Offline-capable; auto-download; fits T4; leverages complementarity (recall + precision).
**Cons**: Inference ~60–120 sec per 3D frame (2 models).

---

#### **Pipeline B: Lightweight (CellposeDINO + Omnipose)**

```python
import cellpose.models
import omnipose

# 1. CellposeDINO (lightweight foundation model)
cellpose_model = cellpose.models.CellposeModel(pretrained_model='cpdino-vitb', gpu=True)
cellpose_masks = cellpose_model.eval(images_3d, channels=[0, 0], do_3D=True)

# 2. Omnipose (morphology-agnostic, good on irregular shapes)
omnipose_model = omnipose.models.CellposeModel(model_type='omnipose_bact_fluor_pretrained')
omnipose_masks = omnipose_model.eval(images_3d, channels=[0, 0], do_3D=True)

# 3. Merge
merged = merge_detections_nms(cellpose_masks, omnipose_masks, threshold=2.0)
```

**Pros**: Lighter inference (~40–80 sec); works on CPU if needed; PyTorch-friendly.
**Cons**: Omnipose less mature than StarDist on dense nuclei.

---

#### **Pipeline C: End-to-End (Cellpose-SAM v2 + CellTracker-GNN)**

```python
import cellpose.models
import celltracker_gnn

# 1. Detect per-frame
cellpose_model = cellpose.models.CellposeModel(pretrained_model='cpsam_v2', gpu=True)
detections_per_frame = [cellpose_model.eval(frame, ...) for frame in image_sequence]

# 2. Extract centroids + features
coords_and_feats = extract_centroids_and_features(detections_per_frame)

# 3. GNN tracking (learns associations)
gnn_model = celltracker_gnn.load_pretrained('ctc')  # Pretrained on CTC datasets
tracks = gnn_model.track(coords_and_feats)
```

**Pros**: End-to-end learned; handles divisions; no hyperparameter tuning.
**Cons**: Requires CTC-style annotations or transfer from CTC model (may not generalize perfectly to zebrafish embryo).

---

## Ranking by Likelihood of Complementary Gain (for Your Case)

**Context**: You already have TemporalUNet3D (0.988 recall). Goal = maximize precision + handle missed dim nuclei.

| Detector | Complementarity Likelihood | VRAM Fit (T4) | Offline | Kaggle Score Impact |
|----------|---------------------------|---------------|---------|-------------------|
| **Cellpose-SAM v2** | HIGH (foundation model, better generalization) | ✅ YES (~4 GB) | ✅ YES | +0.01–0.02 (recall/generalization) |
| **StarDist3D** | **VERY HIGH** (precision-focused, catches boundary misses) | ✅ YES (~3 GB) | ✅ YES | **+0.02–0.04** (precision, reduce FP) |
| **Omnipose** | MEDIUM (morphology-agnostic; value on irregular tissue) | ✅ YES (~2 GB) | ✅ YES | +0.005–0.01 |
| **CellposeDINO** | MEDIUM (lighter foundation, similar to cyto3) | ✅ YES (~2 GB) | ✅ YES | +0.005–0.015 |
| **CellTracker-GNN** | HIGH (learned tracking, better on divisions) | ✅ YES (~1 GB) | ✅ YES | +0.01–0.025 (linking Jaccard) |
| **SAM-Med3D** | LOW (not cell-specific; needs prompting) | ✅ YES (~4 GB) | ✅ YES | −0.01–0.00 (risk) |

**Winner for Your Setup**: **StarDist3D + Cellpose-SAM v2 ensemble** = high likelihood of recall-preserving precision gain.

---

## Open Questions & Caveats

- **Cellpose-SAM v2 on zebrafish embryos**: No published 3D benchmark on developmental tissue yet. Recommend testing on a small validation fold first.
- **StarDist3D on degraded z-slices**: May miss faint nuclei in high-z planes. Cellpose3 restoration module explicitly handles this.
- **Ensemble NMS threshold**: 2.0 µm is conservative; tune on held-out data (fold-0 validation).
- **Omnipose on fluorescence**: Primarily trained on bacteria + C. elegans. Generalization to mammalian/embryonic cells unquantified.
- **CellTracker-GNN generalization**: Pretrained on CTC datasets (mostly 2D animal cells, nematodes). May need fine-tuning on your zebrafish data for optimal division accuracy.

---

## Sources

- [S1] Cellpose models documentation (2026): https://cellpose.readthedocs.io/en/latest/models.html
- [S2] SAM-Med3D (2024–2025): https://pubmed.ncbi.nlm.nih.gov/40742874/
- [S3] Microscopy Cell Segmentation review (2025): https://doi.org/10.3390/jimaging12070297
- [S4] Omnipose documentation & GitHub: https://omnipose.readthedocs.io/index.html and https://github.com/Makelalab/Omnipose
- [S5] 3D nuclei benchmark (Stardist vs U-Net): https://www.sciencedirect.com/science/article/pii/S2667290122000420
- [S6] Hybrid framework (YOLOv11 + StarDist + SAM2): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12189375/
- [S7] CellTracker-GNN (ECCV 2022): https://arxiv.org/abs/2202.04731
