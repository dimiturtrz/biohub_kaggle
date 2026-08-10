# Where the 6-7 hour submission actually goes

The Kaggle submission kernel runs 6-7 hours on a T4. That number was one opaque wall-clock, so every
proposal for speeding it up — batch the frames, move the linker to the GPU, cut the TTA — was a guess about
which part was slow. This is the measurement that replaced the guessing, and the two proposals it killed.

## The measurement

`CellTracker.run` now logs its three stages per video (detect / affinity / link+post). On the RTX 5090, the
shipped dual-seed configuration at threshold 0.97, forwarding fresh (an `EphemeralResponseStore`, exactly what
the kernel does — no cache to replay):

| video | nodes | detect | affinity | link + post | total |
|---|---|---|---|---|---|
| `6bba_05db0fb1` (dense) | 74 340 | 11.4 s (55 %) | 6.9 s (33 %) | **2.3 s (11 %)** | 20.6 s |
| `44b6_0113de3b` (sparse) | 25 828 | 11.3 s (63 %) | 6.4 s (36 %) | **0.3 s (1.7 %)** | 18.0 s |

Detection is two seeds through four-fold flip TTA — eight forwards per frame — and the affinity is two more
transformer passes per frame gap. Together they are **~88 % of the run, and both are GPU work.** Everything
after detection — the assignment linker, the gap bridge, the short-track filter, the smoother — is the
remaining 11 % on the worst movie and under 2 % on a sparse one.

## What that refutes

**GPU-offloading the linker's cost matrix.** The intuition was reasonable: the linker builds an `(s+t)²` cost
matrix with `cdist` on the CPU, and a dense frame has ~650 nodes per side, so the matrix is real work. But the
measurement caps the prize: even if linking took *zero* time it would buy 11 % on the worst movie. The
Hungarian solve underneath (`scipy.linear_sum_assignment`) is sequential and a poor GPU fit anyway, so the
offload would only move the matrix build — a fraction of that 11 %. Not worth the machinery it would need
(free-VRAM probing, a CPU fallback, a budget parameter) for a sub-noise return.

**Batching the inference forward.** Tested directly rather than assumed: batching eight frames per forward
instead of one measured **0.70× — slower** — and the outputs were not bitwise identical (max deviation 3.3e-2,
cuDNN choosing different convolution algorithms for batch-8 than batch-1). The premise was wrong: a 64³ frame
does not starve a modern GPU, so batching adds overhead, and a non-identical output is a leaderboard risk for
no gain. Reverted.

Both are the same lesson in different clothes: *the bottleneck was not where the optimisation instinct pointed*,
and one measurement was cheaper than either build.

## What it found instead: the forward was fp32

The same audit turned up something nobody had proposed. Grepping the inference path for `autocast`, `half`, or
`channels_last` returned **nothing** — every forward in the submission ran in full fp32. On the hardware the
submission actually runs on that is the whole game: a T4 peaks near 8 TFLOPS in fp32 and ~65 in fp16, so the
tensor cores sat idle through the entire 88 %. The isolated forward measured **1.98× faster** under autocast.

The catch is that this one *can* change the answer, unlike video-level parallelism, so it had to be arbitrated
by score rather than by stopwatch. The read-out is an equality comparison between a volume and its max-pool,
and a threshold at 0.97 — both sensitive to small perturbations of the peak logits (a half-precision *cached*
volume once cost 0.909 → 0.816 by tying a nucleus to the valley beside it). Computing reduced and casting the
read-out back to fp32 sidesteps that, and the four-movie proxy confirms it: **0.9334 with autocast against
0.9334 without, identical to four decimals.**

**fp16, not bf16** — and that distinction was measured, not assumed. Choosing the dtype by hardware capability
(bf16 where supported) scored **0.9309**, a real −0.0025. The reason is the mirror image of why training uses
bf16: bf16 buys exponent range at the cost of three mantissa bits, and range is what keeps *gradients* off the
underflow floor. A forward carries no gradients, so the range buys nothing here while the lost mantissa moves
exactly the logits the read-out thresholds. Inference wants the mantissa; training wants the range. fp16 is
also the only half a Turing T4 accelerates, so the two arguments agree on the same answer and the
capability branch disappears.

## What it leaves

Since ~88 % of the time is GPU work and **videos are independent**, the lever is parallelism over videos, not
a faster kernel. Kaggle offers two T4s. One worker process per device, each rebuilding the identical tracker on
its own GPU and running its shard, merges to one submission CSV — an exact transformation (per-video results
are unchanged, the videos simply run concurrently) worth close to **2×: 6-7 h → ~3-3.5 h**. The device list is a
parameter, so the same code path serves one local GPU, two on Kaggle, or N anywhere else.

Stacking on top of that, still untested: compiling the inference forward on the T4. Kaggle is Linux, so stock
Triton is available; inference has no backward pass, so the SDPA-backward bug that complicated *training*
compilation cannot apply; and the frame shape is static. That targets the same 88 %, for an expected 1.3-1.8×.

Reducing the TTA from four-fold to two would halve the largest single cost outright — but TTA is a recall
lever, and every recall-side change in this campaign has had to be arbitrated by the leaderboard rather than a
local proxy. It stays an A/B, never a default.
