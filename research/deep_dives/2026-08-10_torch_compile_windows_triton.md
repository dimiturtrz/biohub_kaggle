# torch.compile + Triton on Windows (RTX 5090, PyTorch 2.10) — State of Aug 2026

**Date**: 2026-08-10
**Status**: settled
**Supersedes**: —

## TL;DR

- **triton-windows is maintained** (`triton-lang/triton-windows`, v3.7.1 as of Jun 2026) and supports torch 2.10+ via Triton 3.6+. **However, torch 2.10 does NOT officially support sm_120 (RTX 5090)**; PyTorch 2.7 is the first stable with Blackwell support.
- **Windows-native torch.compile+Triton requires MSVC + CUDA 12.8+ + Python 3.10+**, bundled TinyCC since post11 wheels, no extra compiler install needed. Path limits (260 char) handled by triton-windows 3.5.1+.
- **WSL2 ext4 is 60× faster than /mnt/d NTFS for small files**; copy dataset to VHDX for hot I/O. May 2026 WSL2 update (PR #40654) improved DMA contention but 9P overhead remains.

## Question

Can torch.compile + Triton run on Windows 11 native with RTX 5090 (sm_120) under PyTorch 2.10 with practical speedup? What are the blockers, and how does WSL2+ext4 compare?

## Findings

### Q1: triton-windows Package Status

**Availability & Maintenance**
- `triton-windows` is officially maintained under `triton-lang` organization on GitHub [S1].
- Original `woct0rdho/triton-windows` was archived Feb 18, 2026 and is read-only; use `triton-lang/triton-windows` [S2].
- Latest version: **3.7.1.post27** (Jun 21, 2026) on PyPI [S1]. Earlier `woct0rdho` final release was v3.6.0-windows.post25 (Jan 26, 2026) [S3].

**PyTorch Compatibility**
- Triton 3.6+ requires PyTorch >= 2.10 [S3], [S4].
- Triton version mapping [S1]:
  - PyTorch 2.10+ → Triton 3.6+
  - PyTorch 2.8–2.9 → Triton 3.4–3.5
  - PyTorch 2.6 → Triton 3.2
  - PyTorch 2.4–2.5 → Triton 3.1

**GPU Architecture Support**
- **Critical limitation for RTX 5090 (sm_120)**: PyTorch 2.10 does NOT officially support Blackwell/sm_120. Stable PyTorch releases only support up to sm_90 (Hopper) as of early 2026 [S5], [S6].
- **PyTorch 2.7 is the first stable release with Blackwell (sm_120) support**, shipping pre-built wheels for CUDA 12.8 [S7]. Triton 3.3 (PyTorch 2.7) adds sm_120 support [S7].
- **Bottom line**: Running torch.compile+Triton on RTX 5090 requires either **WSL2 with PyTorch built from source** (with `TORCH_CUDA_ARCH_LIST="12.0"`) or **upgrading to PyTorch 2.7+** (currently available; see PyTorch release blog).

**Python Support**
- Triton 3.6+ requires Python 3.10–3.14 [S1].

### Q2: Windows Installation Requirements

**C++ Compiler & CUDA**
- **MSVC is mandatory** for TorchInductor optimization [S8]. No alternative suffices alone; Intel Compiler (icx-cl) and LLVM (clang-cl) both depend on MSVC runtime [S8].
- **CUDA 12+ is required**. CUDA 11.x and older not supported [S4].
  - Triton 3.1–3.2 bundles CUDA 12.4 [S4].
  - Triton 3.3–3.7 bundles CUDA 12.8 [S4], [S1].
- Minimal CUDA toolchain is bundled in triton-windows wheels since post11 releases; no manual CUDA install needed [S4].

**Python & Path Limits**
- Python 3.10+ required.
- **Windows 260-character path limit is a known gotcha**; triton-windows 3.5.1+ (Jan 4, 2026) includes workaround to shorten cache paths [S3].

**Windows 11 Gotchas**
- No specific WSL vs native gotchas documented beyond path limits and MSVC requirement.
- NVIDIA driver 525.85+ required for CUDA support [S9] (typical for modern RTX cards).

### Q3: Non-Triton torch.compile Backends on Windows

**Available Backends**
- torch.compile supports `backend="inductor"` (Triton-based, GPU), `backend="cudagraphs"`, `backend="aot_eager"`, and `backend="eager"` [S10].

**Speedup for GPU-Bound 3D-Conv**
- **cudagraphs**: Reduces CPU overhead via CUDA graph capture; only beneficial if model is **CPU-overhead-bound** (e.g., many small kernels, PyTorch overhead). For dense 3D-conv workloads, speedup is negligible [S10].
- **aot_eager**: Runs AOTAutograd without full compilation; used for backward-graph debugging, provides **minimal-to-zero speedup** [S10], [S11].
- **empirical verdict** (aligned with user hypothesis): For GPU-bound 3D-conv, cudagraphs and aot_eager offer little/no speedup vs eager. Inductor+Triton is the only backend delivering GPU kernel fusion benefits.

**Windows Default If Triton Unavailable**
- Official PyTorch inductor Windows tutorial [S8] makes MSVC mandatory and doesn't document automatic fallback. If Triton is unavailable and backend="inductor" is forced, behavior is undefined (likely error or reversion to eager).
- Recommended: Use backend="eager" first to verify codegen, then backend="aot_eager" to check backward, finally backend="inductor" if Triton is available [S10].

### Q4: WSL2 File I/O Penalty

**NTFS via /mnt/d Performance**
- WSL2 accesses host NTFS via 9P network-style protocol [S12]. Every file operation incurs translation tax.
- **Magnitude**: 9P overhead is severe for small files. Benchmark: sequential write + small-file metadata (200 files create/delete) shows **60× difference** between ext4.vhdx and /mnt/c (same protocol as /mnt/d) [S13].
- **Use case impact**: Thousands of small files (e.g., zarr chunks, git checkouts, node_modules) are "unusably slow" on /mnt/d [S12].

**ext4 VHDX vs /mnt/d**
- ext4 VHDX performance is close to native Linux [S12].
- Recommendation: Keep hot data (source, build outputs, datasets) on ext4; only touch /mnt/d for infrequent cross-boundary access [S12].

**2026 WSL2 Improvements**
- May 27, 2026 PR #40654 (Ben Hillis) gave each virtio device its own DMA pool, removing contention between virtiofs mounts and network adapter [S12]. This **reduces (not eliminates)** jitter for /mnt/d operations.
- 9P protocol overhead persists; ext4 remains ~60× faster for small-file workloads [S12].

**Zarr Chunk I/O**
- No specific zarr benchmarks found. Assume zarr chunk reads hit /mnt/d slowdown if data is on Windows NTFS.

### Q5: Expected torch.compile + Triton Speedup for 3D-Conv U-Net

**General Benchmarks**
- TorchInductor typical speedup: **1.30–1.46× on average** (training/inference) [S14], [S15].
- Best case (complex ConvNets): **up to 5.23× speedup** [S16].
- Simple models: possible **8.28% regression** (JIT overhead dominates) [S16].

**For Pre-Optimized Workload (cudnn.benchmark + bf16 autocast + TF32)**
- Your U-Net already extracts device-level speedup via cuDNN kernel tuning and TF32 compute.
- torch.compile+Inductor adds **kernel fusion** (combine multiple ops into single Triton kernel) and **reduced kernel dispatch overhead**.
- **Realistic expectation for 3D-conv U-Net with prior optimizations**: **1.2–1.5× speedup** (fusion gains are smaller once cuDNN is already fast).
- GPU memory overhead: Inductor increases demand by 31–51%; CPU memory decreases 71–73% [S14].

**Caveat**: Speedup is workload-dependent. Only way to know is benchmark on your specific model + hardware.

## Recommendation: triton-windows Native vs WSL2

### Option A: Windows-Native torch.compile+Triton (Requires PyTorch 2.7+)

**Pros:**
- Zero WSL2 filesystem tax; data reads from local SSD.
- Simpler setup if torch.compile works out of box.

**Cons:**
- Requires **PyTorch 2.7 or later** for sm_120 support (2.10 alone will not work).
- Still need MSVC + CUDA 12.8 installed.
- Path limits, compiler toolchain complexity.
- Expected speedup: 1.2–1.5× for pre-optimized U-Net.

### Option B: WSL2 with PyTorch Built from Source (torch 2.10)

**Pros:**
- Can stay on torch 2.10 if desired (build from source with `TORCH_CUDA_ARCH_LIST="12.0"`).
- Linux environment may be simpler for dev iteration.

**Cons:**
- **Dataset copy to ext4 VHDX is non-negotiable** (~60× slowdown on /mnt/d, measured).
- WSL2 initial setup + one-time PyTorch build (1–3 hours) [S9].
- Management burden: maintain ext4 VHDX, mirror code between Windows and WSL2.
- Same expected speedup as Option A: 1.2–1.5×.

### Recommendation

**For a portfolio ML project where Windows-native is preferred:**
- **Upgrade to PyTorch 2.7+** and use Option A (Windows-native + triton-windows from PyPI).
- triton-windows 3.7.1.post27 is stable, CUDA bundled, path limits handled.
- Install via: `pip install triton-windows` (MSVC must be pre-installed).
- Expected payoff: 1.2–1.5× speedup; worth the ~2 hours setup if iterating on U-Net architecture.

**If torch 2.10 is a hard constraint:**
- WSL2 + build-from-source is only path for sm_120 support [S9].
- **Mandatory**: copy dataset to WSL2 ext4 VHDX to avoid 60× slowdown.
- Setup cost: ~4 hours (WSL2 init, CUDA, build, dataset copy).
- Only justified if you plan extended development in WSL2 anyway.

## Open Questions

- Does triton-windows 3.7.1 officially support sm_120, or does it follow Triton 3.7's architecture list (unclear from sources)?
- Measured speedup on your specific U-Net + RTX 5090 combination (only benchmark will tell; 1.2–1.5× is a hypothesis).
- WSL2 DMA improvements (May 2026) — did they measurably reduce /mnt/d jitter for small-file workloads?

## Sources

- [S1] https://pypi.org/project/triton-windows/ — triton-windows PyPI page
- [S2] https://github.com/woct0rdho/triton-windows/releases — woct0rdho final releases (archived Feb 18, 2026)
- [S3] https://github.com/woct0rdho/triton-windows — woct0rdho repository (archived)
- [S4] https://github.com/triton-lang/triton-windows — official triton-lang/triton-windows repository
- [S5] https://github.com/pytorch/pytorch/issues/159207 — RTX 5090 sm_120 support issue (open, 2.10 not supported)
- [S6] https://discuss.pytorch.org/t/is-there-a-pytorch-build-that-supports-nvidia-rtx-5090-compute-capability-12-0-sm-120/223536 — PyTorch forum: no sm_120 in 2.10
- [S7] https://pytorch.org/blog/pytorch-2-7/ — PyTorch 2.7 release: Blackwell/sm_120 support, Triton 3.3
- [S8] https://docs.pytorch.org/tutorials/unstable/inductor_windows.html — PyTorch torch.compile Windows tutorial (MSVC required, CUDA 12+)
- [S9] https://medium.com/@getnetdemil/getting-pytorch-to-actually-use-your-rtx-5090-a-complete-wsl2-setup-guide-for-blackwell-sm-120-61f86f64abc4 — WSL2 RTX 5090 setup guide (build from source, TORCH_CUDA_ARCH_LIST="12.0")
- [S10] https://modal.com/blog/torch-compile-parameters — torch.compile backend parameters (cudagraphs, aot_eager)
- [S11] https://depyf.readthedocs.io/en/latest/walk_through.html — torch.compile backends walkthrough
- [S12] https://brainwagon.org/blog/2026_07_11_wsl2_filesystem_speed — WSL2 filesystem performance (ext4 vs NTFS 60× difference, May 2026 DMA improvements)
- [S13] https://www.ceos3c.com/linux/wsl2-performance-optimization-speed-up-your-linux-experience/ — WSL2 performance optimization guide (2026)
- [S14] https://markaicode.com/howto/how-to-use-torchcompile/ — torch.compile speedup (1.3–1.5× average, 2× best case)
- [S15] https://discuss.pytorch.org/t/torch-compile-w-torch-inductor-benchmarks-models-for-multi-gpu/222147 — PyTorch forum: inductor benchmarks
- [S16] https://ashishmalik.in/post/torch_compile/ — torch.compile benchmark analysis (ConvNets 5.23× speedup, simple models regress)
