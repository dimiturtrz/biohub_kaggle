# We picked our models with the wrong instrument

Reading source, I found that both trainers built their checkpoint scorer as
`TrackerConfig(threshold=config.eval.threshold)` — the shipped *threshold* grafted onto otherwise-default
everything. `TrackerConfig.linker` defaults to the per-frame `assignment` linker and `bidirectional_edges` to
`False`, so that object **is** the 0.892 pipeline, while the recipe we submit (global flow linker at boundary 3
plus bidirectional harmonic fusion, leaderboard 0.895) existed only as four literals inside the Kaggle kernel.

That made the bias *possible*. This is the measurement of whether it was *real*.

## The acceptance test

Score saved checkpoints under both pipelines and ask one question: **does the argmax move?** The distortion is
invisible by construction — a mis-ranked checkpoint is simply never saved, leaving no artefact — so the only
way to size it is to re-rank a set that already exists.

Same weights, same detections, identical everything except the linker and the fusion:

| checkpoint | default (assignment, no fusion) | **shipped** (flow b=3 + bidirectional) | move |
|---|---|---|---|
| `joint_warm_cos3k` | **0.9175** ① | 0.9153 ③ | −0.0022 |
| `joint_short_masked` | 0.9147 ② | 0.9192 ② | +0.0045 |
| `joint_hn_control` | 0.9141 ③ | **0.9206** ① | +0.0065 |
| `joint_velocity` | 0.9148 | 0.9133 ④ | −0.0015 |
| `joint_masked` | 0.8710 | 0.8942 | +0.0232 |

**The ranking is exactly inverted** across the three that the campaign actually compared. First becomes last,
last becomes first. Under the pipeline we selected with, `cos3k` leads by +0.0034 over `hn_control`; under the
pipeline we submit, `hn_control` leads by +0.0053. Note the arms move in *opposite directions* — that is what
makes this a re-ranking rather than a level shift.

## Why the numbers can be trusted

All five **default** figures reproduce their historical values to four decimals — 0.9175, 0.9147, 0.9141,
0.9148 and 0.8710 were recorded on 2026-08-10, before any of this work existed. The sweep is measuring the same thing the campaign
measured; the only new quantity is the shipped arm. Had the default column drifted, the right conclusion would
have been "the harness changed", not "the ranking inverts".

The margins (+0.0034, +0.0053) also clear the selector's repeatability floor, which was ~0.0017 when
`cudnn.benchmark` autotuning was live and is ~0 now that the scoring pass disables it.

## What was actually being rewarded — and the story that does not survive the fourth point

The tempting explanation is trimming. Under the default pipeline `cos3k` is the most trimmed of the compared
three (ratio −0.052 against −0.039 and −0.043), it is the one the default instrument crowns, and the faithful
metric pays a node-count bonus for undershooting. A global solver prices track *boundaries*, so an extra true
detection can become a kept continuation, where a per-frame matcher decides each gap alone and the same
detection is just another candidate to get wrong. Neat: the default instrument rewards the trimmed model twice,
the shipped one rewards the fuller model.

**Two points refute that as a monotone rule.** `joint_velocity` keeps the *fullest* graph of all five under the
shipped pipeline (ratio −0.000) and scores the *lowest* of the serious four (0.9133); if "fuller is better
under flow" were the mechanism it should lead. And `joint_masked`, by far the most trimmed (ratio −0.149
default, −0.072 shipped), gains the *most* from the shipped pipeline (+0.0232) rather than the least. Graph
fullness is at best one term among several — model quality obviously still matters, and velocity was
independently measured inert.

What survives is narrower and is the part that matters: **the two pipelines disagree about which checkpoint is
best, and they disagree by more than the instrument's noise.** The mechanism sketch above is a hypothesis that
fits three of four points; it is not established, and I am recording it as a lead rather than a finding.

That is the same coupling that has now appeared four times: an isolated threshold ladder picked 0.96 while the
shipped config prefers 0.97 with the order reversed; the CC0 ranker measured positive in one linker and
negative in another; bidirectional fusion measured −0.0027 under the assignment linker and **+0.004 on the
leaderboard** under flow. Here it decided which weights survive.

## Consequences

**`joint_hn_control` is our best own-weights checkpoint**, at 0.9206 faithful on the untouched test four — not
`joint_warm_cos3k`, which the campaign has called its best since 2026-08-10. Against the same-shaped
single-seed pilkwang at 0.9118 that is **+0.0088**, comfortably past the selector floor though still short of
the 0.01–0.02 proxy→leaderboard transfer floor, and still below the shipped dual-seed 0.9334.

There is an irony worth stating plainly: `hn_control` was the *control* arm of the hard-negative experiment —
the run that existed only to be compared against. It won on the instrument nobody was using.

**Every "best checkpoint" judgement of this campaign was made with the wrong instrument.** They need
re-reading, not re-labelling; the conclusions about *methods* (masking helps, longer training hurts, velocity
is inert) were mostly drawn from curves rather than single argmaxes, so they are weakened rather than
overturned — but each one that turned on a comparison between saved checkpoints is now open.

## The fix, and why it is not a default edit

Bumping `TrackerConfig`'s defaults would fix today's mismatch and leave the mechanism intact for the next knob
that moves. Instead:

- `TrackerConfig.shipped()` names the submitted recipe **once**, in the library, verified equal field-for-field
  to the config the leaderboard scored 0.895 and pinned by a test, since four modules now mount it;
- `EvalCfg.threshold: float` became `EvalCfg.tracker: TrackerConfig`. The old field derived **one of fourteen**
  values from the deployed config while two of the other thirteen are exactly what separates the tiers;
- the kernel **mounts** the recipe and adds one variable instead of restating it;
- `joint_eval`, `dense_diagnosis`, `localisation`, `mitotic_appearance`, `seed_dispersion` and `corpus_audit`
  all take the same base — `joint_eval` most of all, since it exists *specifically* to score our own heads.

The point of the hierarchy is not tidiness. It is that a partially-specified operating point stops being
constructible.

## The rule

> **Sweep, select, and diagnose in the configuration you ship.** If a component can be reinterpreted
> downstream, its measurement is a property of the pair, not of the component.

And the corollary this measurement adds: an instrument error that only shifts *levels* is survivable, because
comparisons still hold. One that shifts *rankings* is not — and the two are indistinguishable until you check.
The check cost ten evaluation runs against a question that had been open, unnoticed, for the whole campaign.
