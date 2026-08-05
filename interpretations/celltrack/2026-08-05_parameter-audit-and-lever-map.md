# Parameter audit and lever map (2026-08-05)

A consolidated read of where the shipped `CellTracker` stands after a systematic sweep of every operating-point
parameter and every candidate score lever. The pipeline is: dual-seed logit-blended detector → blended
edge-transformer affinity → `AssignmentLinker` (global 1-to-1, `distance − bonus·P`) → short-track filter →
one-frame gap bridge → linefit smoother.

## Every parameter, audited

| parameter | value | verdict |
|---|---|---|
| detection threshold | 0.99 default / 0.97 shipped | **LB-only recall lever** — proxy is threshold-blind (rises with node-count bonus, opposite of LB). Public 0.99→0.97 = 0.887→0.892. |
| linker gate | 10 µm | physical: the max annotated single-frame displacement (9.96 µm). Proxy flat 10–15. |
| affinity bonus | 20 (= 2·gate) | swept peak (bead 88x); re-confirmed post-smooth-fix (15/20/25/30 = 0.9365/0.9412/0.9390/0.9397). Crowding makes the nearest-distance prior anti-informative, so the learned signal needs fixed over-weighting. |
| edge blend (seed1 fraction) | 0.8 | swept peak — seed 1's edge transformer is stronger. |
| **detector blend** (seed1 fraction) | **equal (None)** | **self-balancing at equal**: 0.5/0.6/0.7/0.8 = 0.9412/0.9421/0.9320/0.9311 (0.6 is +0.0009 noise, tilting past it drops). The two *detectors* are equal quality even though seed 1's *edge* head is better — the edge/detector asymmetry is explained, not a bug. |
| **smooth strength** | **0.8 (proxy wanted 0.3)** | proxy favours 0.3 (dense raw Jaccard 0.892→0.906) but a clean LB A/B **refuted it**: smooth0.8 = 0.892 vs smooth0.3 = 0.889. The dense gain is recall-side and anti-transfers, like min-track/node-count — ship the LB value 0.8. A cautionary case: a proxy dense-Jaccard win shipped as default was a 0.003 LB regression. |
| gap-bridge reach | 10 µm | a *prediction-error* gate (the bridge already carries `end + 2·velocity`), Voronoi-dominated in crowding — not a span bias (refuted by reading). |
| bridge added-fraction cap | 0.05 | the synthetic bridge helps (+0.0017), cap at its plateau. |
| min track length | 6 | **proxy-misleading**: monotonic-up (3→7 = 0.9302→0.9474) but LB-anti (the sparse proxy rewards removing short tracks; the denser hidden set annotates true ones). Keep 6; the real lever is rescue. |
| reuse-gap (GapCloser) | off | wired but proxy-inert at 0.97 and 0.99 (too few isolated nodes to reuse). Available, not shipped. |
| NMS suppression window | 3³ (was 5³) | **the session's other real win** — the reference window; un-merged crowded cells (dense raw Jaccard 0.849→0.892). |

The discipline paid off twice: `smooth_strength` was a real matching bias, and the NMS window was a real recall
bug. Everything else is at a swept peak, physically argued, self-balancing, or an irreducibly-LB-only knob.

## Every lever, with its verdict

- **Dense crowding mislink** — the pooled residual. Refuted across *all* axes with mechanism: per-frame cost
  knobs (e9b), global min-cost-flow linking (tf5, +0.0007 noise), complementary/retrained detectors
  (884/ksv/slw/fcd), and affinity retraining across three substrates (lna synthetic, lna real-positives, zni
  hard-negatives — all crush `P_true` along with `P_chosen`). The mislinks are 100 % affinity-inverted and
  irreducible with the current detector, edge model, and (1,4,4) resolution — cracking them needs assets we lack.
- **Detection recall** — the leaderboard rewards it, but the sparse proxy cannot see it, and every proxy-side
  proxy for it has failed to transfer at the 0.89 tier: the NMS un-merge was LB-flat (0.892 either way), the
  smooth-strength "fix" anti-transferred (0.889 vs 0.892), node-count anti-transferred (−0.007), the neural
  bonus bought +0.001. **Only the detection threshold itself moves the LB** (0.995→0.97 = 0.880→0.892). Every
  raw-Jaccard proxy gain on the dense movie is a false positive for LB at this tier — a hard, repeated lesson.
- **Synthetic detector-pretrain** — the last untried detector-side lever, refuted by its own pre-gate. The
  mounted detector already recalls synthetic dense cells 0.952/0.99/0.99 at thr 0.99/0.9/0.5, so synthetic
  cells are not the no-response kind the detector misses on real data — pretraining on them teaches nothing
  about the real misses. Closes every detector-recall lever (884/ksv/slw/fcd/tms + synthetic); the real
  no-response cells are crowd-buried at (1,4,4) resolution, an asset gap, not a training-data gap.
- **Short-track rescue** — the frontier's min7+rescue lever, built and Kaggle-de-risked this session, off by
  default. Proxy-anti-informative (it re-adds short tracks the sparse proxy penalizes) but high-precision
  (−0.0005 on the proxy = the re-added tracks are almost all free/true). A blind-but-high-prior LB bet.

## State

Proxy-measurable optimization is exhausted — every parameter validated, every dense lever refuted, and the
LB has now shown that even the proxy's *recall-side* wins (NMS, smooth) do not transfer. The shipped config is
dual-seed + AssignmentLinker + smooth**0.8** at thr0.97 = the confirmed best, 0.892. The remaining headroom to
the pack (0.93–0.947) is an asset gap (detector/edge-model/resolution), reachable only by the disclosed Yusuke
0.897 same-weights config bundle (bead 73g: 14 µm edge gate, appearance 0.1, ILP-global, output repairs, min7)
— LB-proven but proxy-blind, so it needs a recipe port and LB arbitration, not another proxy tweak.
