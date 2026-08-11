# Three times we measured in a configuration we do not ship

Each of these looked like a different mistake. They are one mistake, and the third instance is the expensive
one because it selects **weights** rather than settling a knob.

## The three

**Threshold, swept alone.** An isolated ladder under the per-frame assignment linker read
0.96 = 0.9304 as its maximum against 0.97 = 0.9295. Under the configuration we actually ship — global flow
linker at boundary 3, bidirectional fusion — the order **reverses**: 0.97 = 0.9357 against 0.96 = 0.9340, and
the margin doubles. Had I trusted the isolated sweep I would have shipped 0.96 and lost.

**The CC0 ranker.** Measured +0.0032 inside `MotionHungarianLinker` and monotone-negative inside our flow
cost. I first attributed the swing to geometry; that hypothesis was refuted (the ranker computes its motion
features from the *context graph*, never from the cost geometry, so changing the geometry cannot move them).
What remains is the same shape: the component's value is a property of the consumer, and the consumer we
measured it in first was not the one we ship.

**The checkpoint selector.** Both trainers build their scoring pipeline as
`TrackerConfig(threshold=config.eval.threshold)` — the shipped *threshold* grafted onto otherwise-default
everything. `TrackerConfig.linker` defaults to `assignment` and `bidirectional_edges` defaults to `False`, so
that object **is** the 0.892 pipeline. Every checkpoint this campaign has selected was ranked under a linker we
do not deploy.

## Why the third one is not bookkeeping

We have a leaderboard-confirmed proof that the sign of an affinity-side change depends on its consumer:
bidirectional harmonic fusion measures **−0.0027** under the per-frame assignment linker and **+0.004 on the
leaderboard** under the global flow linker. The mechanism is understood — `EdgeTransformerScorer` softmaxes
over sources, encoding *one parent per target* into the probability itself; a per-frame matcher already imposes
that constraint, so a symmetric mutual-consistency affinity restates it while blunting the directional
sharpness the assignment cost depends on, whereas a global solver imposes it structurally and wants the
symmetric affinity.

Checkpoint selection is an affinity-side judgement: it ranks edge heads. So the proof applies directly. **A
head that is better under flow can lose selection under assignment**, and we would never see it — the loser is
simply not saved.

That makes this instance categorically worse than the other two. A knob measured in the wrong config gives a
wrong answer you can re-measure. A *selector* in the wrong config silently discards weights, and there is no
artefact left to re-examine.

## What the three have in common

Not carelessness about defaults — each call site had a defensible local reason. The threshold ladder was run to
isolate one variable. The ranker was first tried in the linker its authors used. `EvalCfg` derives its
threshold *from* `TrackerConfig().threshold` rather than restating it, with a comment explaining that a
permissive default "silently ranks by a pipeline we never ship."

That comment is the tell. The right instinct was applied **to one field out of fourteen**. The other thirteen
kept their defaults, and two of them are exactly the fields that define our best result.

## The fix is structural, not a default edit

Bumping the defaults would fix today's mismatch and leave the mechanism intact — the next knob that moves would
drift the same way.

The shipped recipe needs **one home**: a named operating point in `celltrack/operating_point.py` that the
Kaggle kernel *mounts* instead of restating in four literals, and that `EvalCfg` carries **whole** rather than
reducing to a threshold. Then the selector is structurally incapable of ranking under a pipeline we do not
ship, instead of depending on each call site to remember which four fields matter this month.

This is the hierarchical-configuration directive doing real work: the value of the hierarchy is not tidiness,
it is that a partially-specified operating point stops being expressible.

## The rule

> **Sweep, select, and diagnose in the configuration you ship. If a component can be reinterpreted downstream,
> its measurement is a property of the pair, not of the component.**

And the corollary that generalises past this repo: when you find yourself deriving *one* field of a config from
the deployed one, that is evidence the whole object should be carried, not that the field was special.

## Scope, honestly

This is read from source, not measured. I have not quantified how much selection was distorted — that needs a
matched re-run of a saved checkpoint set under both pipelines, which is worth doing precisely because the
distortion is invisible by construction.

Diagnostics that ran **with** explicit `--set linker.name=flow --set linker.disappearance_cost=3 --set
bidirectional_edges=true` are unaffected; the mislink and false-positive decompositions were run that way. Any
diagnostic run without them describes the 0.892 pipeline. The candidate-density audit is unaffected — it reads
`gate_um`, which is 10 either way.
