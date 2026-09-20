# E54 — the error budget in metric units: there is ONE axis, and it is 105 broken GT edges

**Verdict: scored with the official rule, the champion has exactly `wrong-association = 0` on all four
movies — it never mislinks two annotated cells — and 99 % of its false-positive edges (139/141) are
incident to an endpoint of a GT edge it failed to recover. FP and FN are not independent axes; they are
two views of the same 105 broken GT edges, each costing ~2.3 metric units. Of those 105, 79 (75 %) are
fragmentations the solver could already reach. That is the entire remaining budget.**

## The measurement

`kaggle/error_budget.py` on `runs/kaggle_out/celltrack-public-0947/submission.csv` against the train
geffs. First error budget drawn in the units the leaderboard pays in: every earlier one counted
candidate-set quantities, which E53 showed the scorer cannot see.

| movie | jaccard | × node-count | = adjusted | tp | fp | fn | invisible edges |
|---|---|---|---|---|---|---|---|
| 44b6_0113de3b | 0.8679 | 1.0005 | 0.8683 | 46 | 3 | 4 | 24 884 (99.8 %) |
| 44b6_0b24845f | 0.9412 | 1.0368 | 0.9758 | 48 | 2 | 1 | 19 399 (99.7 %) |
| 6bba_05b6850b | 0.9594 | 1.0033 | 0.9626 | 828 | 18 | 17 | 5 112 (85.8 %) |
| 6bba_05db0fb1 | 0.8455 | 0.9993 | 0.8449 | 1 100 | 118 | 83 | 66 990 (98.2 %) |

**Pooled weighted adjusted jaccard 0.8932**, pooled raw 0.8915. (Not comparable to the 0.947 LB —
different movies, train-set geffs — but the *decomposition* is the point.) The invisible column is E53
restated as a measurement: **85.8-99.8 % of predicted edges enter the score nowhere at all.**

## `wrong-association = 0`, verified not assumed

`wrong_association` counts predicted edges where **both** endpoints matched annotated GT nodes but the
pair is not a GT edge — a genuine identity swap between two labelled cells. A direct recount of the
endpoint-match shape of every predicted edge:

| movie | both matched, is GT (tp) | both matched, not GT | source-only | target-only | neither |
|---|---|---|---|---|---|
| 44b6_0113de3b | 46 | **0** | 3 | 2 | 24 882 |
| 44b6_0b24845f | 48 | **0** | 4 | 2 | 19 395 |
| 6bba_05b6850b | 828 | **0** | 18 | 19 | 5 093 |
| 6bba_05db0fb1 | 1 100 | **0** | 98 | 99 | 66 911 |

Zero on every movie, and the FP counts reconcile exactly (dense: 58 source-only + 60 target-only = 118).
**Among cells the annotators labelled, the global ILP links them to each other correctly every time.**
Every FP is single-endpoint-matched: an annotated cell continued into a detection the GT does not contain.

This retires a framing. The confusor is **not** a competition between two tracked cells — the solver never
loses that. It is a competition between the true successor and an unannotated detection.

## The axes are not independent

| movie | broken GT edges | fp | fp incident to a broken endpoint |
|---|---|---|---|
| 44b6_0113de3b | 4 | 3 | 3 (100 %) |
| 44b6_0b24845f | 1 | 2 | 1 (50 %) |
| 6bba_05b6850b | 17 | 18 | 18 (100 %) |
| 6bba_05db0fb1 | 83 | 118 | 117 (99 %) |
| **total** | **105** | **141** | **139 (99 %)** |

A GT edge `a→b` the tracker misses does not merely go uncounted: the tracker still had to link `a`
somewhere, and it links `a→x` (source-orphan FP) while something links `y→b` (target-orphan FP). The
near-symmetry of the source-only/target-only columns above is that pairing.

So **each broken GT edge costs 105 FN + 139 FP over 105 edges ≈ 2.3 metric units.** E47 found this ~2×
asymmetry empirically for a deleted cell; here it is derived, and it applies to every missed link, not
just to prunes. It also means the three "perfect-fix" ceilings below **overlap almost completely** and
must not be added:

| if solved perfectly | pooled jaccard | gain |
|---|---|---|
| all 105 broken GT edges recovered, their incident FPs gone | 0.9991 | +0.108 |
| false-positive edges → 0 (alone) | 0.9506 | +0.059 |
| fragmentation → recovered (alone) | 0.9264 | +0.035 |
| detection loss → recovered (alone) | 0.9030 | +0.012 |

## What this licenses

- **One target: 105 broken GT edges, 79 of them fragmentations.** E43 already showed 99.51 % of
  fragmented GT edges are solver-reachable — the candidates exist and the solver ranks them below a
  rival. This is `kiw1`'s multi-frame identity problem, and it is now sized: recovering a fragmentation
  pays ~2.3 units, not 1, because it deletes the paired FP too.
- **Do not chase "FP suppression" as a separate programme.** 99 % of FPs are symptoms. Deleting an FP
  edge without supplying the right one converts a 2.3-unit error into a 1.3-unit error at best, and risks
  collateral; supplying the right link collects both.
- **Detection is genuinely the smallest axis (26 of 105).** Consistent with E40/E41.
- **Any future proposal must say how many of the 105 it moves.** That is now a checkable claim, and the
  1.5 % bar translates to roughly **35 recovered GT edges** pooled.

## Caveat

Four movies, and E51 applies: all four are affinity-in-sample, two are DeepCenter-in-sample. The *shape*
(zero wrong-association, FP as a symptom of FN) is structural and unlikely to be an in-sample artifact;
the magnitudes are optimistic ceilings. 83 of the 105 broken edges are on one movie, `6bba_05db0fb1` —
the dense one — so the budget is really a statement about dense-regime association, which is where every
other axis has pointed.
