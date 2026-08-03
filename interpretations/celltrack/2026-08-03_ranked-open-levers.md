# Ranked open levers — where the next real gain lives (aet)

Local synthesis of everything measured this campaign. Purpose: rank the remaining levers by
**expected real gain** and **evidence strength**, so the next goal spends wall-time where signal is.

## Where we stand (2026-08-03)

- **Best public LB = 0.882** (dual-seed logit blend, sub 55212692), up from 0.873 single-seed.
- **Proxy→LB transfer confirmed tight** (~0.002–0.005): the 4 test movies *are* the public eval, so the
  4-movie proxy ranks configs trustworthily. fold-0 is leaked, abandoned.
- **In flight** (proxy → predicted public): Assignment SCIP-free 0.9227 → ~0.92; thr0.995 bonus 0.9287 →
  ~0.924; neural-greedy 0.8989 → ~0.895.
- **The pack is 0.930–0.947.** We expect ~0.92 from the pending champions — **~0.01–0.015 below the pack
  floor.** Closing that is a *jaccard* problem, not a bonus-knob problem.

## The core diagnosis (reconciles wor.7 and e9b — they are not in conflict)

Two different ceilings, two different movies:

- **On GT nodes**, the linker is near-perfect (motion edge_jac 0.9947, wor.7). So *given perfect
  detections*, linking is solved. The pooled ~0.09 gap to 1.0 is therefore "detection" **in the sense that
  it comes from working with real detections instead of GT nodes** — not that recall is missing.
- **On real detections**, recall is already 0.988 and the edge-linkability ceiling is 0.98 (e9b). We link at
  ~0.87. That ~0.11 gap is **crowding-mislinks**: both endpoints are detected and real, but among many
  nearby real candidates the per-frame `distance − bonus·P` cost picks the wrong successor.

So the gap is **not** missing detections and **not** solver quality on clean inputs. It is **association
under crowding on the true (dense) detection field** — concentrated almost entirely on the one dense movie
`6bba_05db0fb1` (0.87 vs 0.91–1.02 on the other three). That single movie drags the pooled score.

## Ranked levers

### 1. Better-trained association SIGNAL (P0 — the only measured route into the pack)
The dense 0.11 headroom is pure association error, and **every per-frame cost-function knob is refuted**:
velocity (hurts), affinity_bonus (flat 15–40), edge-blend weight, threshold, division (worse), NMS-prune
(collapses), affinity re-normalization (sources/targets/raw/bidir all within noise), P-floor. Generic
pretrained linkers lose hard (Trackastra 0.48 on dense). The signal `P(edge)` itself is the bottleneck —
the transformer can't separate the true successor from its crowded neighbors.

**Lever:** retrain the edge transformer with a crowding-aware objective — hard-negative mining (the nearby
wrong candidates as explicit negatives), neighborhood/competition context (score "which of N" not pairwise),
richer per-node features. **This is exactly what research bead 66g is scoping.** Expected: if it recovers
even half the dense 0.11, pooled ~0.92 → ~0.93 = into the pack. **Highest expected real gain, gated on 66g.**

### 2. Track-level global-over-time optimization (P1 — untested real-jaccard lever)
Current assignment is per-frame greedy-optimal. A trajectory-level optimizer (whole-track cost, not
independent gaps) is the classic crowding fix and is **untested** (e9b build, was gated on Trackastra which
failed → now unblocked). Could resolve mislinks a one-frame view can't. Lower confidence than #1 (may need
the same better signal to feed it), but it's a distinct paradigm we haven't tried. Needs a global solver
that runs offline on Kaggle (the SCIP-on-Kaggle block is why we went assignment-only — revisit with a
pure-Python/scipy global formulation, or the frontier tracksdata stack).

### 3. Precision on the dense field / complementary detector (P1 — gated on ceb)
Dense fires ~66k detections. Recall is saturated (0.988) so the lever is **not** recall — it's whether a
*different-architecture* detector (Stardist-3D, Cellpose-3D) or a precision filter reduces the crowd of
real-but-competing candidates, making association easier. **Research bead ceb scoping mountable detectors.**
Risk: our own precision attempts (NMS-prune) collapsed because the extra detections are load-bearing real
cells that annotated lineages link *through*. So this is speculative until ceb shows a genuinely
complementary model. Medium confidence.

### 4. Node-count bonus farming (P2 — safe, small, already in flight)
thr0.995 = +0.006 proxy, split-invariant (a0u: factor depends only on Npred + estimated_node_count, both
split-independent), so **not** public-overfit. Per-video oracle reaches 0.9423 proxy and **is**
productionizable (test movies are train movies; hardcode per-movie operating point). But it caps ~+0.02 and
does **nothing for jaccard** — a rank nudge, not a pack-crack. Pending LB confirm (55220635) before trusting
the aggressive per-video map. **Use for the safe 2nd final submission, not as the win path.**

### 5. More/better detector seeds (P3 — marginal, one direction refuted)
Dual-seed gave +0.0094 (two *good* seeds, logit blend). A 3rd blend seed from our own all-199 training
*hurt* (over-detection drags the logit blend, tms). A genuinely strong 3rd public seed *might* add, but
we have none. Low priority unless ceb surfaces one.

### 6. Division recovery (P3 — detection-limited, low ceiling)
Faithful frontier port (tight 4.7µm parent gate) nets ~+0.0002; div_jac caps 0.0114 vs 0.447 on GT nodes =
most true 2nd-daughters are undetected so the fork can't be placed. Gated on a DeepCenter recall model we
don't have. The 0.1× weight makes the ceiling tiny regardless. Deprioritize.

## The one assumption everything rests on
The proxy's trustworthiness and the bonus split-invariance both assume **test = these 4 movies, public/
private = annotation split**. If private is *other* movies, per-video bonus maps overfit and the proxy is
weaker. **Research bead 1st is resolving this** — it gates how aggressively we bonus-farm the 2nd submission.

## Verdict
The pack lives in **jaccard**, and the only measured jaccard headroom is the dense association error
(lever 1). Bonus-farming (lever 4) is banked and safe but can't reach the pack. **Next goal should be
built around lever 1 (edge-signal retrain, informed by 66g), with lever 2 as the fallback paradigm and
lever 3 gated on ceb.** Everything else is a rank nudge.
