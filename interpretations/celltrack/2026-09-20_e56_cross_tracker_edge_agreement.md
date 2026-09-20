# E56 — comparing two trackers edge by edge, and what it says about the ruler

`kaggle/error_budget.py --rival` takes one or more other cached submissions and partitions the GT edges
the champion fails to recover into three buckets: a rival recovers this, a rival breaks this instead,
both break it. Four movies carry public ground truth and they are exactly the four the submission covers,
so every number below is over the same 2127 GT edges.

## The instrument works: fp16 is the control

| rival | LB | rival broken | champion-only | rival-only | both |
|---|---|---|---|---|---|
| fp16 | 0.945 | 104 | **1** | 0 | 104 |
| v1329f | 0.939 | 78 | 39 | 12 | 66 |
| v50 | 0.907 | 77 | 39 | 11 | 66 |

fp16 is a precision variant of the champion itself, and its broken set is **100% nested** inside the
champion's. That is the control behaving: a near-identical model has nothing to contribute, which
reconfirms E38's 0.9956 edge union at GT-edge resolution rather than at the level of a score.

The 39 is not a matching artifact. Sweeping the node-match ruler gives **44 / 38 / 39 / 37** edges at
2.87 / 5 / 7 / 10 µm — flat across a 3.5x change in the matching radius.

Of the 39: **17 swaps, 9 source-stolen, 7 target-stolen** — 33 of 39 are pure selection, only 5 detection
and 1 division. And **33 of 39 sit in the dense movie** `6bba_05db0fb1`. That is the regime that matters,
not the sparse-movie artifact E51 warns about.

## Correction: v1329f and v50 are not two witnesses

I first wrote this up as "two unrelated recipes rescue the identical 39 edges, so it is a champion defect
rather than one rival's luck", and committed that sentence. It is wrong.

`runs/kaggle_out/celltrack-public-v50/v1329_submission.csv` is **byte-identical** (MD5
`3D24DC60…`) to v1329f's own submission, and the two have **identical node counts** (123485). v50 is
v1329 plus one more stage. Two members of one family agreeing on 39 edges is close to trivial. The
honest count of independent witnesses against the champion is **one**, not two.

What survives: one decorrelated *family* pair, ruler-stable, concentrated in the dense movie, and a
control that proves the instrument can tell a nested variant from a decorrelated one.

## The deflation: broken-edge count is anti-correlated with the score

| submission | nodes | GT edges broken | LB |
|---|---|---|---|
| champion 0947 | 122808 | **105** | **0.947** |
| fp16 | 122826 | 104 | 0.945 |
| v1329f | 123485 | 78 | 0.939 |
| v50 | 123485 | 77 | 0.907 |

**The champion breaks the MOST ground-truth edges of the four and scores the BEST.** Ordered by broken
edges the ranking is exactly reversed against the leaderboard. My pooled harness agrees with the broken
count and not with the board: it scores the champion 0.8932 and v1329f 0.9263, a 0.033 gap in the
opposite direction from the board's 0.008.

This is not the E51 saturation story, where a proxy on *different, sparser* movies stops tracking the
board. These are the same four movies the board scores. Same data, opposite order.

The confound is uncontrolled and it is visible in the table: the rivals also carry **677 more nodes**.
The official metric's signed node-count term (E53) prices those, and v50 shows the sensitivity — it
differs from v1329f by 90 edges and no nodes at all, and drops **0.032**. So the correct reading is not
"recovering GT edges hurts". It is:

> **The official score is not dominated by GT-edge recovery, and this harness cannot separate the two
> effects.** Recovered edges and node count move together across these four submissions, and the board
> follows the node count.

## What this licenses and what it forbids

- **Do not price a cross-tracker change with the E54 per-edge value.** The ~2.3 metric units per GT edge
  was derived *within* one tracker, where node count is held fixed. Across trackers the node-count term
  swamps it, and this table shows the sign flipping.
- **E54's own bar survives**, because it compares the champion against itself: recover 35 GT edges
  *without adding nodes* and the 1.5% still follows. The new constraint is the clause "without adding
  nodes", which was implicit before and is now measured.
- **A merge of the champion with the v1329 family is not licensed by the 39.** Adopting a rival's edge
  means adopting endpoints the champion does not have, which adds nodes — the exact term the board
  punishes. A merge is only worth building if it can take the 33 selection fixes **without** taking the
  677 nodes, and nothing here shows that is possible.
- The instrument itself is kept: `--rival` is the cheapest way to ask whether a new candidate is a nested
  variant or a genuinely different tracker, and it answers in counts rather than in a score the board
  disagrees with.

## Addendum — the 33 selection fixes are node-free BY CONSTRUCTION

The clause above ("only worth building if it can take the selection fixes without the nodes") reads like
an open question. It is not: the classifier already answers it, and the answer is free.

`classify_broken` (`kaggle/error_budget.py:104-116`) tests its rules in order, and the three detection
rules fire first. So a `swap` / `source linked to a RIVAL` / `target taken by a RIVAL` label is only
reachable when **both** GT endpoints already matched a champion node. **All 33 selection failures connect
two nodes the champion already has.** Adding those edges imports **zero** nodes. The 677-node penalty
belongs to the rival's detector, not to the edges — the two were confounded in the table above, and they
separate cleanly here.

Better: a swap fix is edge-neutral too. The champion currently spends that slot on a wrong partner, so the
repair removes one edge and adds one. And E54c measured the collateral at zero — **0 of 126 slot-stealing
rivals is an annotated cell**, so the edge given up cannot be a GT edge.

That puts the 33 squarely back in the within-tracker regime where E54's ~2.3 units/edge is the valid
price: node count fixed, FP and FN moving together on one axis. **33 edges ≈ 1.4%** — just under the
35-edge / 1.5% bar, and only under a perfect oracle that fixes all 33 and breaks none. Directional, not a
result, and it composes with anything else on the same axis.

**So the axis is alive, and it is a re-ranking axis, not a merge axis.** What the champion lacks is the
discriminator (E54c), and the v1329 model is an existence proof that one exists — it makes the right call
on these 33. Transfer is geometrically plausible without GT to mediate: both trackers matched the same GT
node within the ruler, so their two nodes sit within 2x the ruler of each other, and a direct node-to-node
match at test time is the same bipartite step already implemented.

The cost is now affordable for the first time: `twrp` verified the fast kernel at **1580s vs 5590s**, so a
rival predict fits in the freed 4010s at ~930s. That is the one shape in which the 39 is worth spending
anything on. Filed as bd `x6bd`; it blocks `kiw1`, which must **not** be built as a plain edge union.

## Open

Whether the v1329 family trained on these four movies is **unresolved** — the kernel's split file is
inference-only (`n_train = 0`). One argument bounds it: the four scored movies have public ground truth,
so a tracker that had memorized them would not sit at 0.939 on the board. That makes wholesale
memorization unlikely, and says nothing about partial exposure.
