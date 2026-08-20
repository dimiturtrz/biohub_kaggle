# The dense confusor is genuine fast motion, not label noise

**2026-08-20 · dense movie `6bba_05db0fb1` · CPU, no GPU/Kaggle spend**

## The question

Three independent physical cues — inter-node distance, learned pilkwang features, and raw
full-resolution appearance — all rank the *near rival* above the *true far successor* on the
n≈37 confusor mislinks, unanimously (interp `2026-08-12_appearance-identity-at-full-resolution.md`).
When every cue agrees against the label, the parsimonious reading is that a fraction of the labels
are **wrong**. A wrong label carries a signature the cues do not: a genuine fast mover *continues*
in its direction of travel (incoming→outgoing turn cosine near +1), while a swapped or spurious
successor sits back toward where the cell came from (cosine near −1, an out-and-back).

If a chunk of the mislinks read cos < −0.5, the dense ceiling is *below* the computed 0.086 —
the wall is partly unreachable label noise, not a lever, and attacking it is wasted effort.

## The measurement

`AnnotatedTurnByFate` (`celltrack/analysis/annotated_motion.py`) — turn cosine at each edge's
source, off ANNOTATED centres only (no detection, no model), split by the fate the tracker gave
the edge. Sources with no annotated predecessor are dropped (turn undefined), and the covered `n`
is reported so a rate on the covered share is never read as a rate over all edges.

```
annotated turn (mislinked) n= 25  median cos +0.447  reversal(<-0.5)=16.0%  aligned(>+0.5)=48.0%
annotated turn (correct  ) n=812  median cos +0.316  reversal(<-0.5)=17.2%  aligned(>+0.5)=38.4%
```

## The verdict

**Label-noise hypothesis REFUTED.** The prediction was a reversal share *concentrated on the
mislinks*. The opposite holds: the mislinked reversal share (16.0%) is *below* the correct-edge
baseline (17.2%), and the mislinks are *more* aligned than correct edges on every statistic —
median cosine +0.447 vs +0.316, aligned share 48.0% vs 38.4%. The mislinked cells continue in
their direction of travel more smoothly than the average reproduced edge. They are genuine fast
movers, not swapped labels.

Which link broke: the **IDEA** (the data/label-noise suspect), not the harness. The annotations on
the confusor are clean. The dense ceiling of 0.086 stands as reachable — the tracker gets these
edges genuinely *wrong*, it is not chasing noise.

## What it does NOT rule out

- The turn test speaks only to the *tail* geometry (does the cell double back). It does not
  re-open appearance — that stays refuted across frozen+raw+full-res by the 4-measurement chain.
- It does not rule out isolated bad labels elsewhere; it rules out label noise as the *explanation
  of the confusor* on this movie.

## Consequence

Both forks now point the same way. The confusor is a real **model/architecture gap** in the dense
regime — a genuine fast mover whose slower near-rival the association head prefers. This converges
with `celltrack-TRUE-bottleneck-is-dense-regime-model-gap`: the lever is frontier dense-model
replication (LB-blind), not another knob on the local proxy. The local escape hatch — "the wall is
partly noise, stop pushing" — is closed. The wall is real and it is the model's.
