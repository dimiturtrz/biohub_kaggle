# Multi-Object Linking Disambiguation in 3D Cell Microscopy

**Date**: 2026-08-03  
**Status**: settled  
**Scope**: Concrete methods for improving Hungarian motion linker with appearance and learned association terms

## TL;DR

Five analytical features + three learned methods disambiguate crowded linking: intensity/texture (LBP, Hu moments, eccentricity, SIFT) are cost-free and work offline; transformer (Trackastra, HOCT) and contrastive embedding (CELLECT) linkers are pretrained and need no retraining. Global ILP solvers (CBC/Gurobi) beat greedy Hungarian by 5-15% on dense frames by permitting division constraints and backtracking; Trackastra's ILP mode is production-ready with motile + CBC.

---

## Question

What are concrete, implementable methods to add appearance and learned-association terms to a greedy Hungarian linker processing 60k detections/frame? Which require training, which ship with pretrained weights, and when does global (ILP) beat frame-by-frame greedy?

---

## Findings

### **Appearance Features (Analytical, No Training)**

**Intensity & Texture Moments** [S1, S2]  
- Mean, variance, skewness of pixel intensities within cell mask
- **Local Binary Patterns (LBP)**: Compare each pixel to neighbors, encode as binary; robust to illumination, fast; used in texture classification [S2]
- **Hu Moments**: Geometric + texture features computed on binary mask; detect shape + intensity changes (e.g., distinguish mitotic from interphase) [S1]
- **CellProfiler measurements** [S3]: Implemented in open-source CellProfiler 4.0+; includes intensity, morphology, texture modules

**Morphological Shape Features** [S1, S4]  
- **Eccentricity**: Ratio focal-distance / major-axis-length of fitted ellipse; aspect ratio proxy, rotation-invariant [S1]
- **Equivalent diameter, solidity, compactness**: Standard morphological descriptors in scikit-image

**Local Descriptors** [S5, S6]  
- **SIFT**: Scale-invariant feature transform; robust to rotation/scale/illumination. Historic gold standard, used in live-cell DIC tracking [S5]
- **BRIEF, ORB**: Binary descriptors, orders of magnitude faster than SIFT; ORB (Oriented FAST + Rotated BRIEF) available in scikit-image, OpenCV [S6]

**Implementation**: All features computable in scikit-image or CellProfiler; pass as cost term in Hungarian edge-cost matrix. Combine via learned or fixed weights (e.g., `cost = 0.8·distance + 0.1·intensity_diff + 0.1·texture_sim`).

---

### **Learned Association Scoring (Transformer, GNN, Embedding)**

**Trackastra (Transformer-Based, Division-Aware)** [S7, S8]  
- **Mechanism**: Encoder-decoder transformer predicts pairwise association matrix from cell coordinates alone (no images). Blockwise softmax enables 1-to-many (division) [S7]
- **Input**: Coordinates + timestep indices in local temporal window (e.g., 10 frames)
- **Pretrained**: YES — general 2D models trained on Cell Tracking Challenge; 3D models available [S8]
- **Training Required**: NO. Greedy mode (~seconds) or ILP mode (motile + CBC, ~minutes) for global optimal
- **Availability**: GitHub: https://github.com/weigertlab/trackastra [S8]

**CELLECT (Contrastive Embedding Learning)** [S9]  
- **Mechanism**: Deep learning via contrastive loss; minimizes feature distances for same cell, maximizes for different cells. Learns 3D cell identity embeddings
- **Pretrained**: YES — model trained on public mskcc-confocal dataset generalizes across imaging modalities + species without retraining [S9]
- **Training Required**: NO for most tasks
- **Code**: https://github.com/zzz333za/CELLECT [S9]

**Higher-Order Cell Tracking Transformer (HOCT)** [S10]  
- **Mechanism**: Edge-centric (not node-centric); classifies candidate links using transformer + line-to-line distance attention bias. Avoids embedding-space merging in divisions [S10]
- **Features**: 19D hand-crafted (coordinates, cell size, intensity, inertia tensor) — no pretrained encoder needed
- **Training Required**: YES, but no image encoder needed

**Graph Neural Networks (GNN)** [S11]  
- **Mechanism**: Model time-lapse as directed graph; cell nodes + association edges. GNN message passing extracts paths. Deep metric learning produces discriminative embeddings [S11]
- **Features**: CNN + LSTM extract appearance + motion, fused via GNN for association inference [S11]
- **Training Required**: YES
- **Status**: Published (2022), less common in practice than Trackastra

---

### **Global vs Greedy Linking**

**Hungarian Algorithm (Greedy)** [S12]  
- Bipartite matching, **locally optimal** per frame; commits early, blocking better global assignments later
- **Speed**: ~milliseconds per frame
- **Limitation**: Cannot backtrack; no division constraints; no long-term memory

**Network Flow / Min-Cost Flow (ILP)** [S13, S14]  
- Formulate entire time-lapse as flow graph: nodes = detections, edges = possible links, costs = distance + appearance
- **Coupled min-cost flow**: Model mitosis + merging via edge coupling; one mother → two daughters, or two cells → one merged [S13]
- **Solver**: Gurobi (commercial, fast, **not available in Kaggle kernels**) or CBC (open-source, free, slower) [S14]
- **Performance**: 5–15% improvement in edge Jaccard on crowded frames vs greedy [S12]
- **Production examples**: Mastodon/ilastik (Fiji), Ultrack (Python), Trackastra ILP mode [S7, S14]

**Why Global Wins on Dense Frames** [S12, S13]  
- Greedy picks locally-optimal neighbor A, but globally A+B would have lower total cost with C+D. ILP sees entire cost matrix at once
- Division constraints (1→2) + merging (2→1) require graph reasoning; greedy cannot enforce
- Long-range occlusions recovered via temporal consistency; greedy has no temporal window

**Production Implementation** [S8, S14]  
- **Trackastra + motile + CBC**: Greedy or ILP mode; CBC solver is Kaggle-compatible [S8, S14]
- **Ultrack**: Multiple segmentation hypotheses + ILP linking; slower but handles segmentation uncertainty [S14]
- **Mastodon**: Fiji plugin; trainable classifiers + graph optimization [S14]

---

## Open Questions

- **Cost matrix weighting**: How to balance distance, appearance, motion in combined edge costs? Greedy search or learned weights?
- **Temporal window size**: For Trackastra/HOCT, is 10 frames optimal, or does it vary by cell speed / density?
- **CBC tuning**: Time-to-optimal vs approximation; when to bail on ILP and fall back to greedy?
- **Division detection**: Do appearance features or transformer attention naturally identify mother-daughter pairs, or does ILP need explicit division constraints?

---

## Sources

- [S1] Feature extraction techniques: https://www.geeksforgeeks.org/computer-vision/feature-extraction-in-image-processing-techniques-and-applications/
- [S2] LBP in object tracking: https://www.researchgate.net/publication/278705841_Local_Binary_Pattern_as_a_Texture_Feature_Descriptor_in_Object_Tracking_Algorithm
- [S3] CellProfiler 4.0 measurements: https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.0.5/modules/measurement.html
- [S4] Hu moments + intensity for mitosis: https://arxiv.org/pdf/2102.03889
- [S5] SIFT in live-cell DIC tracking: https://www.researchgate.net/publication/224138086_Live-cell_tracking_using_SIFT_features_in_DIC_microscopic_videos
- [S6] BRIEF, ORB features, scikit-image: https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/index.html
- [S7] Trackastra paper + transformer mechanism: https://arxiv.org/abs/2405.15700
- [S8] Trackastra GitHub: https://github.com/weigertlab/trackastra
- [S9] CELLECT contrastive embedding + pretrained: https://pmc.ncbi.nlm.nih.gov/articles/PMC12615263/
- [S10] Higher-Order Cell Tracking Transformer: https://arxiv.org/html/2607.11754
- [S11] GNN for cell tracking: https://arxiv.org/abs/2202.04731
- [S12] Hungarian vs network flow performance: https://www.numberanalytics.com/blog/hungarian-algorithm-network-flow-ultimate-guide
- [S13] Coupled min-cost flow cell tracking: https://pubmed.ncbi.nlm.nih.gov/20864383/
- [S14] Ultrack + Mastodon + motile solvers: https://www.nature.com/articles/s41592-025-02778-0 and https://github.com/royerlab/ultrack
