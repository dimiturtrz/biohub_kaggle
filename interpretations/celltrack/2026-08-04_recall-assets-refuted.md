# The "better detection asset → recall" thesis, refuted three ways (epic xzf)

**Date:** 2026-08-04 · **Verdict:** detector-side recall is saturated on the four test movies; no detection
asset adds true recall. The recall the leaderboard rewards is reachable only by the inference threshold.

## The premise being tested

The one leaderboard-confirmed lever is *recall*: matched A/B, thr0.99 → LB 0.887 beats thr0.995 → LB 0.880
(−0.007 from trimming detections). Epic xzf proposed climbing by building **better detection assets** —
a second detector (DeepCenter), test-time augmentation, and a recall-weighted retrain — on the theory that
more/better detections mean more recall. Each was tested against the four test movies (fold-0 is debunked;
these movies with their sparse GEFF annotations are the tightest local proxy) with a **complementary-recall**
probe: how many ground-truth cells does the new asset catch that the champion dual-seed detector misses?

## Three refutations, each checkable

| lever | probe | result |
|-------|-------|--------|
| **slw** DeepCenter as a recall source | peak-extract its cached centre heatmaps, match GT vs blend@0.99 | complementary ≈ 0: three movies already 0.9988–1.0; dense adds **+6/1229** only by flooding 45k extra nodes. DeepCenter fires *fewer* nodes than the 66k-node main detector at every threshold — it is a subset, not a complement. |
| **ksv** recall-weighted retrain | `pos_gain` on the positive BCE mass, warm-start pilkwang, complementary recall of step-N weights | floods **false** peaks + erodes calibration. pos_gain=8/600st: dense +10 GT but 6bba_05b6850b 0.9988→0.9024, 44b6_0b24845f 1.0→0.9608, nodes 22.8k→39.4k (past the estimated count → node penalty). pos_gain=2/250st: same failure. |
| **fcd** 8-fold dihedral TTA | full-D4 vs the current 4-fold flip TTA on dense | **net worse**: recall 0.9894→0.9707. The network learned orientation-specific in-plane features (flip-aug only), so rotated-view responses misalign and the averaged logits over-smooth, merging more true peaks than the +6 rotations recover. |

## The mechanism they share

The champion detector's GT recall is already **0.988–1.0** on all four movies. The residual misses on the
dense movie are *crowd-buried no-response cells* — the network produces no peak there at all (the
"endpoint-missing" 2% of the [dense-ceiling diagnosis](2026-08-04_dense-ceiling-diagnosis.md)). None of the
three interventions surfaces them: DeepCenter (a second network) doesn't respond either; a recall-weighted
retrain just lowers the firing threshold *globally* (more false peaks, not the missing true ones); more TTA
views blur the peaks that exist. All three hit the same wall the capstone predicted.

Crucially, a recall-weighted retrain that lowers the global firing threshold is **strictly worse** than
lowering the *inference* threshold on the strong pilkwang detector: same "more peaks", but with calibration
damage and no cache reuse. The recall the LB rewards lives in the **hidden denser annotations** the sparse
proxy can't see, and it is reached by the inference threshold on the existing detector — not by any new asset.

## Where this leaves the climb

- **Detection is not the lever.** Recall is saturated; the gap is dense **association** (crowd mislinks), at
  the per-frame Hungarian ceiling.
- **Live lever:** the inference-threshold LB probe (26l) — thr0.98 and thr0.985 kernels submitted, mapping the
  hidden-recall curve directly, since the proxy is threshold-blind.
- **One un-refuted lever left:** a global-over-time linker (Ultrack, filed) whose hierarchical re-segmentation
  is a different mechanism than fixed-detection assignment. Capstone-quantified EV is small (~+0.005–0.008
  pooled) but it is the only untested paradigm.
- **Retired:** a new detector architecture (5yv) faces this same saturated-recall wall and the recurring
  finding that our own-trained weights lose to mounted pilkwang.
