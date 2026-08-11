# Right on average, confidently wrong where it counts

This file was first written under the title *"The head was never the problem — one scalar was."* That
conclusion was wrong, and the experiment it motivated refuted it within the hour. The measurement it was built
on is sound; the inference drawn from it was not. Both are kept below, because the gap between them is the
useful part.

## The question

`cost = distance − affinity_bonus · P`. The head is fed the candidate pair's relative displacement, and true
motion is small (median 1.7 µm), so *"near = successor"* is an excellent rule almost everywhere — and exactly
wrong for the crowded fast-movers where all our dense mislinks live. If the head leaned on that shortcut, the
learned term would be partly **redundant** with the distance term it exists to correct.

The test is AUC of `P` **within matched distance bands**. If `P` cannot rank inside a band, it carries nothing
beyond proximity.

## The probe had to be fixed twice

The first attempt scored the **GT graph**: annotated nodes are sparse, so within a 10 µm gate 446 of 457
candidate pairs were true (base rate 0.976) and both AUCs read ~1.0 by construction. That produced the standing
base-rate guard — implemented in the tool rather than left as a habit.

Moving to **real detections** was not enough. The guard refused again at base rate **0.9807** (1167 of 1190),
because judgedness required **both** endpoints matched to GT, which rebuilds the annotated graph inside the
detections. The fix follows the endpoint the annotation **reached** — the scorer's own `edge_auc` convention.
An in-gate target that matched nothing is not *unknown* truth; the annotation already named that source's
successor, and this is not it. Base rate **0.9807 → 0.1487**, n = 7846.

The population was never the artifact. The **truth rule** was.

## The measurement (this part stands)

| band | AUC(P) | AUC(−distance) |
|---|---|---|
| 0–2 µm | *refused* — 96% of in-gate candidates are the true successor | — |
| 2–4 µm | *refused* — 82% positive | — |
| 4–7 µm | **0.972** | 0.706 |
| 7–10 µm | **0.950** (25 pos, se ≈ 0.10) | 0.642 |
| 5–10 µm pooled | **0.970** | 0.823 |

Two solid results. **The head has learned appearance, not distance** — with distance held near-constant it
still ranks the true successor at 0.97. And **at short range there are no distractors**: within 2 µm of an
annotated source, 96% of everything in the gate *is* the true successor, which is why the shortcut reads 0.982
pooled and tells you nothing.

## The inference that was wrong

From "AUC(P) holds at range while AUC(−distance) collapses" I concluded the head's value *relative* to distance
rises with range, so a **constant** bonus under-weights it exactly where the mislinks are — and that the cost
function, not the signal, was the wall.

Implemented as a self-normalising ramp, `bonus · max(0, 1 + k·(d/mean_gated_d − 1))`, budget-preserving on
average, byte-identical at `k = 0`:

| k | clamped | Δ |
|---|---|---|
| 0 | 0.9353 | — (exact control) |
| 0.25 | 0.9326 | −0.0027 |
| 0.5 | 0.9339 | −0.0014 |
| 1.0 | 0.9113 | −0.0240 |
| 2.0 | 0.2632 | collapse |

Mislinks **38 → 39**. Unmoved. The mechanism check — demanded up front precisely so a score wobble could not be
mistaken for repair — says nothing was repaired.

## Why it could never have worked

On this config's own 38 dense mislinks: **97.4% affinity-inverted, mean P_true 0.079 against P_chosen 0.589.**
In 37 of 38 the head scores the *wrong* neighbour far above the true successor.

Those are **signal errors, not cost-weighting errors.** A ramp that spends more of the bonus at range amplifies
the wrong evidence. No dose fixes it, which is exactly the flat-then-falling curve. The `k = 2.0` collapse is
the other half: with the near weight clamped toward zero, short links stop beating the boundary cost and tracks
shatter.

## The reconciliation, and the lesson

An AUC of 0.97 over 7846 pairs with 1167 positives is entirely compatible with 38 confidently-inverted cases.
Thirty-eight is a tail far too thin to register. **Aggregate ranking quality says nothing about the tail, and
the tail is the whole score.**

The diagnosis said so explicitly — *"a ranking AUC of 0.97 permits a handful of confident inversions"* — and I
quoted that line in the same document where I concluded the head was fine. The warning was in front of me; the
appealing story won. That is the failure mode worth naming: not a missing measurement, but reading a strong
aggregate as though it described the cases that matter.

What survives: the head genuinely does carry appearance information at range, on the pairs where it is not
inverted. What is refuted: that this makes the cost function the bottleneck. The dense mislinks remain
**signal-limited**, which is now the seventh independent axis to reach that conclusion — and the first to reach
it from the cost-weighting side rather than by retraining something.
