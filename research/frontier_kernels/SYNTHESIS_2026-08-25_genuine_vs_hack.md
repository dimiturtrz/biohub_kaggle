# Frontier gather — what carries 0.900 → 0.926-0.927 (three top public kernels, read for mechanism)

2026-08-25. Three highest-scoring public biohub kernels pulled + read cell-by-cell (Opus agents, code
cited). Champion (ours) = 0.900. This settles whether the public frontier is real tracking or a metric
mirage, and names the exact levers.

## TL;DR

- **Two independent GENUINE kernels reproduce 0.926 (rockerritesh "divsub") and 0.923-0.927
  (evgendvorkin) via the SAME stack.** Not a hack — both self-attest `metric_hack_used: False` and their
  emitted graphs pass strict lineage invariants (in-deg ≤1, out-deg ≤2, consecutive-frame edges).
- **A THIRD kernel (kirneo, 126 votes) = ~0.90 legit + ~0.02-0.03 PURE METRIC EXPLOIT** bolted post-hoc.
- **All three weight packs are PUBLIC on Kaggle** (verified 2026-08-25) → the genuine stack is fully
  reproducible under our harness:
  - `pilkwang/biohub-tracking-support-pack-50ep-v1` (base, = our `reference/pilkwang`)
  - `pilkwang/biohub-temporal-unet3d-seed314159-v1` (secondary independent seed, 349MB)
  - `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` (center-prior UNet, 71MB)

## The GENUINE stack (the real 0.900→0.926 lever, portfolio-honest, survives private)

Ranked by plausible contribution (agents' judgment; only the harmonic +0.002 is isolated in-kernel):

1. **Dual-seed decorrelated ensemble — the structural half.** A SECOND independently-seeded
   detector+edge model (seed314159), blended:
   - detection maps at weight **0.475** with std-ratio alignment;
   - edge logits **z-score-calibrated** to primary scale, blended weight **0.15** ONLY where primary
     top-2 parent softmax margin < **0.35** AND both models agree same parent (`low_margin_consensus`);
   - candidate edge threshold dropped to **0.48**; a retention guard reverts a frame to primary if the
     blend drops candidate count below 90% (anti-collapse).
   This is exactly the "decorrelated-ensemble / different-recipe member" our own memory keeps naming as
   the best-justified shot. It is the published 0.913 base both genuine kernels build on.

2. **Motion-relink Hungarian re-association — REPLACES the linker's raw edges.** Per-consecutive-frame
   `linear_sum_assignment`, cost `motion + 0.05*raw - 1.0*prob`, motion = distance from velocity-predicted
   position `pos + 0.5*(pos - prev_pos)`, two gated passes (tight 6µm / relaxed 10µm). The final one-to-one
   association is a global optimal matching fusing a motion model with the learned probs — NOT the
   transformer's greedy/ILP output. Both genuine kernels discard raw edges at output time for this.

3. **DeepCenter center-prior UNet — add-only veto gating ALL recall recovery.** Separate small 3D-UNet
   (base 24ch, GN/SiLU). Not a submission detector — it confirms real intensity peaks before any
   gap-close synthetic node or safe-division fork is accepted. This is the "third CC0 pack / 0.027 recovery
   carrier" our notes already identified — implemented as a veto, not a primary detector.

4. **divsub — safe-division recovery (first non-zero division-J of the competition).** Adds a 2nd child
   edge to a mid-track single-successor node only if ALL hold: parent mid-track; two daughters mutual-
   nearest orphans; parent≤8µm / sister≤11µm / existing-child≤10µm caps; DeepCenter confirms; **C3
   divergence** — both daughters continue at t+2 and separate by ≥**2.25µm** MORE than at t+1 (post-mitotic
   signature). Global cap ~0.375% of edges. CV: +0.0046, div_J 0.0→0.0625. The C3 gate is what converts
   "hundreds of forks, zero TP" into real divisions.

5. Cheap real add-ons: **8-view D4 detection TTA** (flips+rot90+transpose), **centroid refinement** to
   local intensity peak (raises node-match rate under the 7µm gate — proxy-blind, LB-visible),
   bidirectional harmonic edge fusion (w=0.30, isolated +0.002), ILP disappearance=1.4-1.5 (keep tracks
   alive). Both winning configs run gap2 recovery OFF and division-geometry-filter OFF — MORE conservative
   than the shipped knobs.

Detector under all of it = pilkwang `TemporalUNet3D` + `SimpleNodeTransformer` edge head, det threshold
**0.96875** (HIGH, recall recovered downstream), downsample (1,4,4). Same stack our notes peg at ~0.90.

## The METRIC HACK (kirneo) — separable, ~0.02-0.03, DO NOT chase as modeling signal

`augment_dataset` (~30 lines, pure post-hoc on `submission.csv`): appends one hub node at t=-1000,
(-10000,-10000,-10000) linked to every track root, then a chain of **FORKS=32** fabricated divisions
(`divider→{child,continuation}`) at off-image coords, per movie. Farms the **0.1×division-Jaccard** term.
Free because (a) the edge tally ignores links where either endpoint is unmatched (no precision penalty),
(b) the node-count penalty `1-0.1*(pred-est)/est` is negligible for ~97 fake nodes vs thousands.

- **Our OWN faithful `core/metrics/divisions.py` is hardened against exactly this** — off-image forks are
  not considered/evaluable/invalid → contribute 0. So the exploit is INVISIBLE on our local metric; it
  only pays if the LIVE competition metric credits unmatched forks. The 126 votes + 0.926 LB are the
  empirical evidence that it does.
- Author's own comments show FORKS swept 20→30→40 and MAX_COMPONENTS 2600→3400 chasing LB — actively
  tuned exploit, not incidental.
- Survives public→private (same metric code) UNLESS organizers patch the division scorer.
- **Composable on ANY forest `submission.csv`** including ours — one probe submission would MEASURE live
  lift. But it is worthless tracking signal and un-validatable on our instrument.

## Strategic implication

- The winning cluster is 0.945-0.953 (#1 = 0.962). If the top LB is metric-hacked (+0.02-0.03 free,
  public), the genuine ceiling of the top methods is ~0.92-0.93 and the hack accounts for the rest. That
  makes the **honest-vs-hack decision an owner/values call** (portfolio artifact), not one to spend the
  held submission on autonomously. Flag, don't fire.
- The GENUINE stack (dual-seed calibrated fusion + motion-relink + DeepCenter veto + divsub-C3) is a
  reproducible, private-surviving +0.026 that clears 0.910 honestly. This is the "replicate frontier →
  merge → build past" mandate. **This is the lever.** Build it. Gate the submission on faithfully
  reproducing the published 0.926 (a known reproduced number, not our inverting proxy).
