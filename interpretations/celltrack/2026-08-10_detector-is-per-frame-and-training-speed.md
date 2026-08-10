# The detector is per-frame — the fake temporal pair, and a ~29× training-speed stack (2026-08-10)

Two connected findings about the temporal-U-Net **detector**: what its temporal window actually models in our
pipeline (nothing, at detection), and the training-throughput stack that observation unlocked.

## The fake temporal pair carries no temporal information

The grafted detector is a `TemporalUNet3D` that expects a window `(B, T, C, Z, Y, X)` — it was trained jointly
for detection **and** cross-frame association, so the backbone has temporal attention across `T`. But we use
**only its detection head**; the cross-frame relationships in our pipeline are modelled by a separate stage
(the `AssignmentLinker` over detected nodes, and the edge-affinity scorer). Detection itself is a per-frame
question: *where are the cells in this frame*.

The shipped inference path honours the `T=2` arch by **duplicating the same frame twice**
(`stack([frame, frame])`), running the backbone, and reading output frame 0. Both frames are identical, so the
temporal attention sees two copies of one frame and models **zero** genuine cross-frame relationship. It is a
fake pair — a duplicate that exists only to satisfy the tensor shape, and every spatial convolution runs on it
and throws the result away.

**So T=1 loses no temporal modelling we use.** Measured on 20 frames of the dense movie (`6bba_05db0fb1`, the
hardest), a genuine one-frame window vs the duplicated two-frame window gives:

- **identical node count** (16269 = 16269), and
- **99.98 % peak match** both ways (4 of 16269 centres differ).

The residual 0.02 % is the temporal attention over two identical tokens not being *exactly* a no-op; it is far
below detection sensitivity (the peaks that differ sit away from the 0.97 threshold). Motion-aware detection
(feeding real `t-1, t, t+1`) would be a *different* model — and would keep real, distinct frames, not the
duplicate. `single_frame` is simply the honest form of detection-only.

## The training-speed stack it unlocked

All measured on an RTX 5090, batch 8, warm-start, steady-state (warmup-amortised marginal it/s). Each lever is
an opt-in flag; correctness is preserved (compile eval score matched eager exactly; single-frame trajectory
matched; the batched loss is a batch-mean, unit-proven lr-invariant).

| lever | what | speedup |
|---|---|---|
| batched forward | all 158 train frames are one static 64³ shape → stack into `(B, z, y, x)`, one pass | 3.6× frames/s |
| `torch.compile` | inductor + `triton-windows`; `dynamic=False` collapses to the static shapes, `sdpa_kernel(MATH)` dodges a broken efficient-attention backward meta-kernel | 1.7× |
| single-frame (`g50`) | drop the duplicate temporal frame → ~half the convolutions | 1.6× |
| eval-trim | the in-loop eval is a checkpoint *selector* — skip its 4× flip-TTA | 2.8× (eval) |

**Combined: ~5 frames/s (original batch=1) → ~144 frames/s ≈ ~29× throughput.** A from-scratch run that was
~2 h is now well under 15 min.

### Two implementation notes worth keeping

- **`dynamic=False` is the key for compile on fixed-shape workloads.** The default assumes the batch dim can
  vary (a symbolic `s77`), and that dynamic path is what tripped the `_scaled_dot_product_efficient_attention_backward`
  meta-kernel stride assert. Specialising to the concrete sizes fixes the crash *and* enables full fusion.
- **`torch.compile` works Windows-native on Blackwell (sm_120)** via the maintained `triton-windows` wheel — the
  "torch 2.10 doesn't support sm_120" claim is false (our eager training already runs on the 5090). Shipped as an
  OS-markered opt-in `compile` extra so a bleeding-edge Triton can't break a plain `uv sync`.

## Scope — honesty

This speeds the **own-weights research arm** (training our own detector, epics `cdg`/`80e`), not the shipped
leaderboard number: our clean own-weights detector is ~0.798 vs the leak-inflated pilkwang graft's 0.873, so the
shipped pipeline still mounts the published weights. The value here is **iteration speed** — the constraint on
the research arm was always wall-clock per training run, and that is now ~29× cheaper. `single_frame` stays off
for the shipped inference (T=2) until a leaderboard A/B rules out even the 0.02 % drift.
