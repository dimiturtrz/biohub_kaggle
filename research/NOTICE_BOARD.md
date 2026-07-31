# Research Notice Board

## Open Questions & Investigation Status

### Detection + Tracking SoTA for Zebrafish 3D Embryo Tracking

**Status**: ✅ **SETTLED** (2026-07-31)  
**Deep-dive**: [`2026-07-31_celltrack_sota_3d.md`](deep_dives/2026-07-31_celltrack_sota_3d.md)

**Question**: What are the state-of-the-art methods for 3D cell detection + linking in fluorescence microscopy that work offline in Kaggle kernels?

**Findings**:
- **Detection**: Cellpose3/cyto3 (simplest, pretrained, offline) or Ultrack (SOTA, slower)
- **Linking**: Trackastra (transformer, division-aware) + motile ILP (CBC solver, offline)
- **Jitter fix**: Soft-argmax or Gaussian subpixel refinement; distance-transform peak-finding insufficient
- **Solver**: CBC is Kaggle-compatible (Gurobi needs WLS licence in cloud)
- **Metrics**: Edge Jaccard + Division Jaccard (weighted 0.1) for competition

**Next**: Implement Path A (Cellpose3 + Trackastra) or Path B (Ultrack) depending on segmentation quality baseline.

---

## Legend

- ✅ **SETTLED**: Findings complete, actionable, moved to SUMMARY
- 🔶 **PARTIAL**: Some answers found, remaining questions documented in Open Questions
- ❓ **OPEN**: Not yet investigated
