# The public frontier decoded, and the hard limit of a sparse-annotation proxy

**Date:** 2026-08-04 · **Bottom line:** the public non-hack frontier is **LB 0.908 on the exact pilkwang
weights we already use** — the whole gap from our 0.887 is *linking/post-processing config*, and its one
transferable lever is the **detection threshold**, which is **structurally invisible to any local proxy built
on the competition's sparse annotations**. Beating ~0.908 needs a better/higher-resolution detector we lack.

## What the public kernels actually are

`yusuketogashi/lb897-baseline` and `.../clean-approach-lightweight-local-cv` both run the **same pilkwang
support-pack-50ep weights** (`unet_transformer`, `edge_predictor_best.pth`) we mount. Their clean baseline is
**LB 0.908**. So the pack of public solutions is not a better model — it is a tuned *recipe* on the shared
detector. The disclosed config differences from our champion, and their fate when tested:

| knob | their value | our test | verdict |
|------|-------------|----------|---------|
| detection threshold | **0.96875** (we use 0.99) | proxy-blind (below) | the real lever, LB-only |
| ILP appearance weight | 0.0 / 0.1 (contradict) | 4pz was +0.0075, subsumed | noise |
| edge/gate max µm | 9.5 / 14 (contradict) | proxy flat 10→16 | noise |
| velocity motion-relink | primary + small P bonus | proxy 0.9004 < 0.9227 | worse on proxy |
| disappearance cost | ~1.575 | proxy monotone ↓ 0.9227→0.9169 | refuted @0.99 |
| short-track rescue | len-5, high-P | proxy 0.9236→0.9223 | op-point-coupled |

Every knob is either proxy-refuted, proxy-blind, or coupled to their *lower threshold* — none is a free
structural win on our pipeline. Yusuke's own two notebooks give **contradictory** appearance and gate values,
confirming those are noise he varied, not levers.

## Why the proxy cannot see the threshold lever

The four test movies (and the broader fixed-8 CV) carry **sparse** annotations — a few hundred lineage nodes
out of ~30 k cells. Two measurements settle it:

1. **Threshold sweep, both proxies rise with threshold** (test-4 0.99→0.995 = 0.9227→0.9287; cv-8 =
   0.9074→0.9116) — the node-count *bonus*, the exact opposite of the LB (0.99 = 0.887 > 0.995 = 0.880).
2. **Detection recall is flat across threshold on every video** (e.g. 6bba_fc83837d holds 0.9739 from 0.95 to
   0.995 while node count climbs 12 492→14 176). Lowering the threshold adds only *false* peaks; the missing
   cells produce **no peak at any threshold** — crowd-buried at the `(1,4,4)` resolution.

So the LB's threshold benefit is **not** local recall recovery. It is that a high threshold trims true peaks
that *are* annotated in the hidden denser set (lowering hidden Jaccard) but are absent from our sparse
annotations, so locally the score only *rises*. The lever is real on the leaderboard and invisible here.

A useful by-product: the **8-video CV is better *calibrated*** — cv-8 @0.99 = 0.9074 sits closer to our LB
0.887 than the four test movies' inflated 0.9227 (the test four are the bonus-farmable easy ones). Use cv-8
for an absolute LB *estimate*, the test four for linker-structure *ranking*, and **neither** for the threshold.

## The detection ceiling

The missing cells are irreducible: flat across threshold, and unrecovered by every same-family detector we
tried (DeepCenter, a recall-weighted retrain, expanded TTA, Cellpose — all in prior interpretations). The one
*different-mechanism* detector (divaug's tophat 2-U-Net) has **no publicly downloadable weights**. Closing the
~1.3–2.6 % no-response gap on the harder videos therefore needs a higher-resolution or architecturally
different detector — the assets the 0.93–0.947 pack presumably has and we do not.

## Where this leaves the climb

- **Reachable now:** ~0.90–0.908 via the detection threshold alone (dropped toward 0.96875). Submitted probes
  `thr0.98`/`thr0.97` are pending Kaggle's scoring backlog; they test exactly this.
- **Infrastructure built this session:** `CellTracker` (the shipped recipe as one runnable, validated to
  reproduce 0.9227) and a generalized `proxy_eval` sweep — every future config is now evaluable in-repo,
  ending the scratchpad-eval era.
- **Beyond 0.908:** needs a better detector (higher resolution / no-response-cell recovery), not another
  recipe knob. That is an asset-acquisition problem, not a tuning one.
