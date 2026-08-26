# SlackRow — confirmed component on bottleneck B, but set-limited and non-submittable alone

> **✅ FINAL FAITHFUL VERDICT (2026-08-26 04:48Z) — NO-GO on the CORRECT axis; slack retired.** The clean
> re-test (matched A/B, warm `joint_short_masked.pt`, LIVE inverted generator `--synthetic-scenes 2000
> --confusor-invert-velocity`, held-out synthetic HARD eval **n=122337**, scale-free HARD top1) gives
> ctrl top1 **0.5983** / inv 0.4017 vs slack top1 **0.5876** / inv 0.4124 → slack **−0.0107 top1,
> +0.0107 inverted = wrong direction, ≤ the 0.01–0.02 noise floor = FLAT**. Harness verified clean
> (recipe code-audited, live generator confirmed, both arms matched, n=122k, correct ruler). Loss-side
> axis-supervision does **not** move the faithful inverted-confusor axis — this is the first CLEAN test and
> it REFUTES the source-softmax hypothesis for the standalone loss. Converges with the detection-side
> evidence (four cues fail, appearance coin-flip): the confusor is DETECTION-side, not fixable by an
> association-loss denominator. Slack retired as a lever. Forward = LEVER B (frontier dense-regime
> association replicate) and b101ac's #1 GPU arm (DeepCenter reuse + max_gap 1→2). Everything below (easy-axis
> numbers + component framing) does NOT stand.
>
> **⛔ RETRACTED (2026-08-26 00:44Z) — the recorded direction was measured on the WRONG axis.** b101ac's
> data-path trace: the probe ran the STATIC-npz path (`--synthetic-sequences` / `SyntheticPairs.sequences`,
> `pair_split.py:232`, "complete by construction"), which does NOT honor `--confusor-invert-velocity` —
> only the LIVE generators (`--synthetic-scenes>0` Scene multi-frame, or GpuScenes) consume the invert
> flag. So the probe trained the velocity-CONTINUOUS EASY axis, not the faithful inverted one, and the
> −0.096 align diagnostic (from a separate `confusor_audit` live-generator run) never described the probe's
> training data. **SlackRow is UNCONFIRMED on the faithful axis; the numbers below are the easy axis and do
> not stand.** The composed finer+slack synergy is also refuted by audit — finer grid grows the confusor
> set on no code path (synthetic placement is grid-independent by rate; detected-corpus nodes come from the
> shipped detector, not the finer model). The source-softmax mechanism stays a valid UNTESTED hypothesis;
> the first correct test is SlackRow on the LIVE inverted generator (`--synthetic-scenes`) with a large
> held-out inverted-synthetic eval. The rest of this doc records the easy-axis run for history.

2026-08-26. The SlackRow parental-softmax loss (bd-4xse) is the **first mechanism to move the confusor
axis the right way** on the faithful inverted-velocity distribution. Two independent rulers agree on
direction. But magnitude sits at the noise floor and cannot be hardened in isolation — the limit is the
size of the faithful confusor set, not the training budget. Slack is a **confirmed component**, kept for a
composed build, not a standalone submission.

## What SlackRow fixes (mechanism)
The association loss (`SoftmaxFocalBCE`) normalizes over the SOURCE axis ("which parent for this target"),
but the linker chooses over TARGETS ("which successor"). Every dense mislink partner is an UNANNOTATED
detection with no competing supervised source → the confusor axis (85% of the gap) was **never directly
supervised**, a named TRAINING-harness reason B stayed flat at P_true ~0.078 across every prior attempt
(finer122, HOCT head, finer-repr). SlackRow adds one learned no-parent row (`1+Σexp`) that gives
unannotated-rival mass somewhere to go, so the axis gets supervised.

## The numbers (matched A/B, warm from `joint_short_masked.pt`, faithful inverted confusor dist)
Scale-free top1 (share of mislinks where affinity ranks the true target first); mean_p_true is
sharpness-confounded and NOT used for the cross-arm call.

| ruler | init | CTRL (no slack) | SLACK (`--slack-links`) |
|---|---|---|---|
| synth n~90, step-250 | top1 0.052 / inv 0.93 | top1 0.032 / inv 0.97 | top1 0.058 / inv 0.93 |
| CV n=48, `.resume.pt` trained | — | top1 0.021 / inv 0.979 / edgeJ 0.8974 | top1 0.042 / inv 0.958 / edgeJ 0.8959 |

Direction CONFIRMED on both: slack top1 > ctrl, inverted-fraction lower, on two independent sets.

## Why it can't be hardened in isolation — three kill-facts
1. **Magnitude = noise floor.** The CV top1 gap is literally one case (2/48 vs 1/48); the inverted gap one
   case (46 vs 47/48). Below the 0.01–0.02 floor → directional KEEP, not a result.
2. **n is set-limited, not training-limited.** The faithful confusor set is 48 cases on CV, ~90 on the
   synth batch. 1500 more steps converges the *head* but does not grow the *case count* — top1 stays a
   1–3/48 statistic. The pre-registered "converge → sharpen top1" run is therefore ~0-expected (its own
   prediction is still-noise), so it was not fired. This is the WALL-CLOCK ÷ GAIN rule refusing a
   zero-expected run, and it correctly killed the follow-up before it burned 40 min.
3. **edgeJ de-calibrated 0.9113 → 0.896 in both arms.** Edge-only retraining perturbs the shipped head's
   calibration against the fixed 0.97 threshold — the warm-finetune-degrades pattern. So the arm is
   non-submittable standalone regardless of slack.

## Decision — don't prove it isolated, ride it free into the composed build
A larger synth-confusor eval pool (hundreds of cases) would lift top1 out of 1-case noise and could prove
slack real. Skipped as low-ratio: even proven-real, the component is non-submittable alone (edgeJ de-cal)
and pays off only inside a composed build. Slack is a loss-denominator — mechanism-sound and FREE to
include — so by the KEEP rule it rides into the composed build at ~zero cost. Proving it in isolation does
not gate that build → the eval pool is a ~0-gating instrument.

## Forward
The real next lever is the **composed finer-grid + slack build** (GPU frontier): finer-grid decode fixes
bottleneck A (referee-PASS'd on `finer122_coadapt_long.pt`) and slack supervises the bottleneck-B confusor
axis its head never touched. Warm from `finer122_coadapt_long.pt` (co-adapt, not from-scratch). This is the
only live shot at the 85%-of-gap B wall and the >0.910 candidate — everything else (replication,
finer122 ensemble seat, no-train knobs) is refuted or dead. Gated on the card and b101ac's timing. Submit
only if the composed arm evals a real mechanism clearing 0.910 — never the 250-step probe. Banked LB stays
0.900; no TRIED mechanism demonstrated-by-us >0.910.
