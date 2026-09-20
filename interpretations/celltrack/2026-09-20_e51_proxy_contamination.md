# E51 — our proxy is in-sample, and the split cannot measure how much

**Verdict: the public stack's affinity model trained on ALL 199 videos, so every one of our eight proxy
movies is affinity-in-sample. The detector (DeepCenter) held out 128, and the four proxy movies it DID
train on are exactly the four `44b6` ones — which are also the four most sparsely annotated. The
detector-side confound is unbreakable on this split.**

## The manifests (primary source, not the forum)

`biohub-temporal-unet3d-seed314159-v1/weights/unet_transformer/split_0/split_manifest.json`:
method `unet_transformer_alltrain_seed314159_v1`, `train` n=**199**, `test` n=40 — and the 40 are a
**strict subset** of the 199 (40/40). The public secondary/affinity model has seen every video in the
competition's train set, and its own reported "test" number is an in-sample number.

`biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/split_manifest.json`:
`train` n=**71**, `val` n=**128**. The detector leaves 128 movies genuinely held out.

## Where our proxy movies land

`celltrack/eval/proxy.py:30-47` — `TEST_MOVIES` (4) + four dense additions = `CV_MOVIES` (8).

| proxy movie | role | DeepCenter | annotated frac | adj_jac |
|---|---|---|---|---|
| 44b6_0113de3b | TEST | **in-train** | 0.0020 | 0.9628 |
| 44b6_0b24845f | TEST | **in-train** | 0.0016 | 0.9767 |
| 6bba_05b6850b | TEST | held-out | 0.1353 | 0.9735 |
| 6bba_05db0fb1 | TEST | held-out | 0.0176 | 0.8854 |
| 44b6_341df25f | VALIDATION | **in-train** | 0.0290 | 0.9456 |
| 44b6_e57ff5c6 | VALIDATION | **in-train** | 0.0095 | 0.7903 |
| 6bba_969618f6 | VALIDATION | held-out | 0.0489 | 0.9303 |
| 6bba_fc83837d | VALIDATION | held-out | 0.0731 | 0.8090 |

(adj_jac from `logs/assign_ranker_cv8.log:18-25`; fracs from `logs/annotated_fraction.log`.)

**The split is exactly by prefix.** All four `44b6` are in DeepCenter's 71; all four `6bba` are in its 128.

## Why the obvious test fails

In-train mean adj_jac **0.9189** vs held-out **0.8996** — a 0.019 gap, barely at the noise floor with
n=4 per side. It is also **uninterpretable**, because on this split three properties are perfectly
collinear:

    in-train  ==  44b6 prefix  ==  sparsest annotation (frac ≤ 0.029 vs 0.018-0.135)

A sparse-GT movie scores over 51-216 loaded nodes; a memorised movie scores high because the detector saw
it; a `44b6` movie differs in lineage density. **All three predict the same four movies**, so the 0.019
attributes to nothing. No rearrangement of `CV_MOVIES` fixes this — the prefixes ARE the split.
Measuring detector memorisation needs held-out `44b6` movies from DeepCenter's 128, evaluated against
whatever annotation they carry; that is a new proxy set, not a re-read of this one.

## What DOES follow, unconditionally

**All eight proxy movies are affinity-in-sample.** The secondary model trained on 199/199, so there is no
held-out fraction to appeal to — every association-side number we have ever read locally is an
optimistic one. This is a *mechanism* for a rule we have so far held only empirically
(`celltrack-proxy-saturated-affinity-quality-bound`, and CLAUDE.md's "proxy inverts at the ceiling"):
the local affinity is better on these movies than it will be on the hidden set, so offline association
headroom reads smaller than it truly is, and an offline association win need not transfer. dw0
(+0.0267 held-out, −0.028 LB) and the public datapoint of +0.0078 local / −0.002 LB are the same shape.

**This strengthens E49 rather than weakening it.** E49's affinity feature reached AUC 0.6380 and the
8-feature joint 0.6622 — those are *in-sample* affinity numbers, i.e. an upper bound on what the feature
can do on hidden movies, and it still fell 9× short of E47's FP-recall 0.75 @ 2 % collateral. An
optimistic ceiling that fails is a stronger kill than a fair one that fails.

**E50 is unaffected.** Its three valid movies (`6bba_09961292`, `6bba_bb9f20c3`, `6bba_784a78c9`) are all
in DeepCenter's 128, and the hierarchy instrument uses no affinity score at all.

## What this does NOT say

Not that our detector-side results are wrong. E36's 0.824 @ 2.87 µm and the E40-E42 detection-axis
closure read on all eight movies, half in-sample — they are *optimistic by an unmeasured amount*, and the
amount is unmeasurable here. They are not refuted; they are ceilings. Where a detector-side conclusion is
a **kill** (E40-E42: the axis is closed), an optimistic ceiling that still fails makes the kill safer, the
same way it does for E49. Where one would be a **license to build**, it should not be trusted alone.
