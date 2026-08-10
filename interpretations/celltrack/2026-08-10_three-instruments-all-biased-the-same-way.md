# Three instruments, all biased toward finding fewer cells

Today produced no new method. It produced three defects in how we *measure*, and they turned out to be the
same defect wearing different clothes: **every instrument in the loop paid us to detect less.** Each had
already been written up as a fact about the problem rather than a fact about our tooling.

## The three

**1. The checkpoint selector rewarded under-detection.** Both trainers selected on the competition metric,
`adjusted = jaccard × (1 − 0.1·ratio)`, which pays a *bonus* for undershooting the estimated node count. So a
model that simply found fewer cells could win. Not hypothetical — caught firing live: a run saved a checkpoint
whose recall had dropped 0.966 → 0.938, scored as an improvement purely by the bonus. Under a clamped score it
would have been rejected, and when the run was restarted with the fix, it *was* rejected.

**2. The proxy's worst blind spot was self-inflicted.** Detection threshold was recorded across two separate
investigations as *"irreducibly LB-only — no sparse local proxy captures it"*, because the proxy ranked
thresholds backwards against the leaderboard. It wasn't sparsity. It was the same bonus:

| threshold | faithful | clamped | bonus | LB |
|---|---|---|---|---|
| 0.995 | 0.9326 | 0.9203 | +0.0123 | 0.880 |
| 0.99 | **0.9348** | 0.9264 | +0.0083 | 0.887 |
| 0.98 | 0.9344 | 0.9291 | +0.0053 | 0.891 |
| 0.97 | 0.9334 | **0.9295** | +0.0039 | **0.892** |

The faithful score picks 0.99 — the config that lost 0.005 on the leaderboard. Clamped, the proxy reproduces
the leaderboard's order on all four points, and the bonus column is monotone in threshold, which is the
confound made visible.

**3. The detection loss labelled ~98% of real cells as background.** `BalancedBCE` built its target as
`zeros_like(logits)` with ones only at annotated centres, and our annotations cover 1-2% of the cells in a
frame. The objective literally taught the detector to stop firing on true cells.

## Why they compound rather than add

The loss pushed detection down; the selector rewarded that push; the proxy then confirmed it as an
improvement. A run could degrade, be saved as better, and be validated by a sweep — three independent-looking
instruments agreeing, because they shared one bias. That is the failure mode worth naming: **agreement between
instruments is not evidence when the instruments share an assumption.**

## What it costs, and what it explains

The loss defect has a mechanism that explains a campaign's worth of results. From scratch the objective works
— the published pack reached 0.993 recall with it — because false-negative pressure is diffuse and positives
teach "cellness" faster than mislabelled negatives unteach it. Fine-tuning a *saturated* detector inverts
that: the model already fires on thousands of unannotated true cells, and each one aims a confident
wrong-answer gradient at a **correct** detection. The better the detector, the harder the loss pushes it down.

Every warm-start refutation on record — `lna` on both substrates, `ksv`, `tms` all-199, the joint 1e-4 failure
— trained a saturated detector against that loss. None of them isolated the method from the defect. They are
not thereby wrong, but none is settled either.

The first measurement after the fix, step-matched against the unmasked run:

| | clamped | recall | det loss | edge loss |
|---|---|---|---|---|
| init | 0.8398 | 0.966 | | |
| unmasked, epoch 0.5 | 0.8376 | 0.936 | 0.0316 | 0.2468 |
| masked, epoch 0.5 | **0.8513** | 0.940 | 0.0108 | 0.2023 |

+0.0137 at the same step, and the first time warm-start joint training has improved a model on an unbiased
selector.

## Where the story corrects itself

I predicted masking would hold recall near 0.966. It didn't — 0.940 against the unmasked 0.936, essentially
unchanged. The gain arrives by another route: the detection term collapses, the detector stops being dragged,
and the **association** head sharing the trunk gets a cleaner signal (edge loss 0.2023 vs 0.2468). The fix is
real; my account of why was half wrong, and the residual slide says the ignore threshold (0.97, the shipped
operating point) is too permissive — a cell the detector reports at 0.6 is still supervised as background.

The same honesty applies to the clamped proxy. It is **not** a general corrector. Predicted and confirmed: it
does nothing for `smooth_strength`, because smoothing moves node *positions* at constant count, so the bonus
is identical (+0.0039) at both settings and the ordering cannot change. Predicted and **refuted**: it would
flip `min_track_length` toward the leaderboard's preference — it halved the bias and did not flip it. The
correction is scoped to knobs that change *counts*, and even there it is partial.

## The transferable part

Two rules come out of this that are not about cell tracking:

**A default you did not choose is not a decision you made.** `zeros_like` as a detection target, the
competition metric as a selection objective, `steps` as a schedule unit — each arrived as the obvious thing
and each silently defined what the experiment meant.

**When a metric can be satisfied two ways, check which one your model took.** The adjusted score can rise
because linking improved or because detection shrank. Reading node recall alongside it costs nothing and is
the difference between a result and an artefact.
