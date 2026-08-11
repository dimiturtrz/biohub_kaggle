# The head was never the problem — one scalar was

For weeks the working theory was that our association wall is a **representation** wall: the features at
(1,4,4) do not encode what separates a true successor from a near decoy, so no linker can recover it. It was
supported by a direct measurement — raw cosine similarity picked the true successor over the decoy only
**0.229** of the time, and a bilinear probe on those features read held-out ≈ chance (0.574).

Measured on real detections, the shipped head ranks the true successor at **0.97 with distance held fixed**.
The signal was there the whole time. The wall is the cost function.

## The question

`cost = distance − affinity_bonus · P`. The head is fed the candidate pair's relative displacement, and true
motion is small (median 1.7 µm), so *"near = successor"* is an excellent rule almost everywhere — and exactly
wrong for the crowded fast-movers where all 35 of our dense mislinks live. If the head had leaned on that
shortcut, the learned term would be partly **redundant** with the distance term it exists to correct, which
would explain why sweeping the bonus finds a broad flat plateau (15/20/25/30 → 0.9365/0.9412/0.9390/0.9397)
rather than a peak.

The test is AUC of `P` **within matched distance bands**. If `P` cannot rank inside a band, it carries nothing
beyond proximity.

## The probe had to be fixed twice, and the second fix is the lesson

The first attempt scored the **GT graph**: annotated nodes are sparse, so within a 10 µm gate 446 of 457
candidate pairs were true (base rate 0.976) and both AUCs read ~1.0 by construction. That produced the
standing rule — *a probe reporting a separability number must print its base rate and refuse when it is near
0 or 1* — implemented as a guard in the tool rather than a habit.

Moving to **real detections** was not enough. The guard refused again, at base rate **0.9807** (1167 of 1190).
The cause was the truth rule: judgedness required **both** endpoints matched to GT, which rebuilds the
annotated graph inside the detections. A matched source sees, within its gate, essentially only its own
matched successor.

The fix follows the endpoint the annotation **reached** — active row *or* active column, which is the scorer's
own `edge_auc` convention. An in-gate target that matched nothing is not *unknown* truth: the annotation
already named that source's successor, and this is not it. Those unmatched neighbours are precisely the
distractors we lose score to, so they are the negatives. Base rate **0.9807 → 0.1487**, n = 7846.

The population was never the artifact. **The truth rule was.**

## The result

| band | AUC(P) | AUC(−distance) |
|---|---|---|
| 0–2 µm | *refused* — 96% of in-gate candidates are the true successor | — |
| 2–4 µm | *refused* — 82% positive | — |
| 4–7 µm | **0.972** | 0.706 |
| 7–10 µm | **0.950** (25 pos, se ≈ 0.10) | 0.642 |
| 5–10 µm pooled | **0.970** | 0.823 |
| overall | 0.995 | 0.982 |

The refusals are part of the finding: **at short range there are no distractors.** Within 2 µm of an annotated
source, 96% of everything in the gate *is* the true successor. That is why the shortcut looks so strong pooled
(0.982!) and why it tells you nothing.

## The asymmetry is the lever

As range grows, `AUC(P)` barely moves (0.972 → 0.950) while `AUC(−distance)` collapses (0.706 → 0.642). The
head's value **relative to distance rises with distance** — and our cost applies a **constant** bonus, so it
weights the learned evidence *least* exactly where proximity has stopped working and where every one of our
mislinks is.

One scalar cannot be right at 1 µm and at 8 µm at once. That explains the flat bonus plateau better than
redundancy ever did: the sweep is averaging two regimes that want opposite weights.

The arithmetic of a known mislink makes it concrete. True far edge collects `20 × 0.134 = 2.7` of bonus; the
near wrong edge collects `20 × 0.686 = 13.7`. The wrong edge wins on the bonus alone, before distance is
counted.

## Rank is strong; calibration at range is not

Both of our observations are true simultaneously, and it took separating them to see it. On genuinely-true
pairs at ≥ 5 µm the mean `P` is **0.519** (median 0.588) against **0.839** on true pairs overall. The ordering
survives at range; the absolute confidence decays to a coin flip. And 35 confident inversions is a tail far too
thin for an AUC over 83 far positives to register — a ranking AUC of 0.97 permits a handful of them.

So a cost built on **raw `P`** systematically under-rewards the true long steps the solver must accept.

## What this changes

The association gap is not waiting on a better representation, a finer resolution, or another model. It is
waiting on a cost function that spends its evidence budget where the evidence is worth something. The first
test of that is a self-normalising ramp, `bonus · (1 + k·(d/mean_gated_d − 1))`, which preserves the budget on
average and reduces to today's cost byte-identically at `k = 0` — a reweighting, not an increase, so it is
comparable to the control rather than confounded with "more bonus".

The mechanism check matters more than the score here: if the number moves but the **mislink count** does not,
it moved for some other reason.
