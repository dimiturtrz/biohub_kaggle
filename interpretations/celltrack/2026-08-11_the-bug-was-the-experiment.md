# Four arms, one objective, and a noise floor three times wider than I thought

*(This document was first written under the title "The bug was the experiment", claiming a defect had
accidentally implemented contest weighting and produced the campaign's best own-weights checkpoint. A gradient
check refuted that within the hour. The refutation is the finding; the original claim is kept below only where
it explains how the error was made.)*

## What I thought I had

Three arms on the detected-pair corpus, held out through the shipped pipeline:

| arm | faithful | mislinks |
|---|---|---|
| control — source axis only | 0.9060 | 56 |
| symmetric, averaging a dead constant | **0.9278** | **43** |
| symmetric, constant removed | 0.9060 | 56 |

0.9278 is well above the previous best own-weights checkpoint (0.9197) and its own initialisation (0.9060). I
read it as the defect being load-bearing: a one-source pair made the source softmax constant, so asking for
both axes computed `(66.67 + live)/2`, and since the constant has no gradient the whole effect was that
uncontested pairs pushed half as hard — contest weighting, arrived at by accident.

## Why that was wrong

Two of those rows are **byte-identical on every column**, which is the signature of a dead code path, not a
tie. Neither arm beat its own initialisation on the selector, so `save-best` kept the warm-start weights: I had
evaluated the untrained model twice and called it a comparison.

Then the confirmation arm — the "principled" rewrite, dividing by the axes asked for rather than the ones that
bind — scored 0.8413 against the original's 0.8474 and also failed to save. That should have reproduced the
result, so I checked whether the two forms actually differ:

```
one source (the dead-axis case): gradients identical = True  max|diff| = 0.000e+00
five sources:                    gradients identical = True  max|diff| = 0.000e+00
```

**Bitwise identical.** The "bug" and the "fix" compute the same gradients; the constant only ever polluted the
logged value. So the two arms were one objective run twice, and the 0.0061 between them is run-to-run
nondeterminism.

## What the four arms actually say

| arm | objective | selector |
|---|---|---|
| control | source axis only | 0.8426 |
| sym | both axes, `/len(axes)` | 0.8474 |
| sym2 | both axes, `/len(binding)` | 0.8432 |
| sym3 | **identical to sym** | 0.8413 |

Two runs of one objective span 0.0061, and every arm sits inside that band. **Nothing here is
distinguishable.** The 0.9278 came from the one run that happened to peak above its init and therefore saved a
checkpoint at all.

That number is still real — it is a genuine held-out score for a genuine checkpoint. What is not real is the
attribution. It is a draw from a distribution whose spread I had not measured, selected for by the very
threshold that makes it look special.

## Two instrument lessons, both cheap and both skipped

**The run-to-run floor for this recipe is ~0.006 on the selector, not ~0.002.** The recorded 0.0017 figure is
the *selector's repeatability on identical weights* — cudnn autotuning moving peak logits. Training
nondeterminism is a different and much larger quantity, and I had been comparing arms against the smaller one
all session. Every "+0.0048 over its control" I reported today is inside it.

**`save-best` silently substitutes the initialisation.** An arm that never beats its init produces a checkpoint
file that is the warm start, so evaluating it measures nothing about the arm. The trained weights are in
`.resume.pt` the whole time. Checking whether an arm SAVED must precede comparing checkpoints — and the
matched comparison should read `.resume.pt`, which makes every arm measurable whether or not it cleared its
own bar.

## What survives

The loss changes are still correct and stay: the degenerate axis genuinely carried no gradient on 28.8% of
pairs, and dropping its constant took the logged edge loss from 20.04 to 0.357, which is the difference between
a readable training curve and a meaningless one. Dividing by the axes asked for still has an argument that does
not depend on any arm — a target with one candidate parent carries less evidence than one with forty. It is
simply **not measured**, and it should not have been presented as measured.

What is genuinely open is whether the symmetric axis helps at all. Four arms could not tell, because the
instrument is wider than the effect. Answering it needs several seeds per arm and `.resume.pt` as the read-out,
not one run each.

## The pattern

Three times today I credited a variable I was testing with a change I had not attributed: an affinity gap the
control moved further, a difficulty sampler I had not read, and now a defect that computes the same gradients
as its own fix. Each time a cheap check — running the control, reading the function, comparing the gradients —
was available *before* I wrote the claim down, and each time I wrote first.

The corrective is not more caution in the prose. It is that a claim about a mechanism should be accompanied by
the measurement that isolates it, computed in the same breath. `torch.allclose` on two gradients took eleven
seconds.
