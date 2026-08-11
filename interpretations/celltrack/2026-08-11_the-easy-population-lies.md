# Every model-quality number we trusted was measured on the easy population

Four separate quantities told us the association model was fine, or that a fix was safe. Each was measured
on a population where the problem does not occur. The pattern only became visible because they surfaced
within hours of each other.

## The four

**1. Edge AUC reads 1.0000, always.** The joint trainer logged it every window for the whole campaign. It is
computed over **ground-truth node pairs**, where our linker already sits at 0.9947. A from-scratch model goes
0.6823 → 0.9987 in half an epoch and then flat. It cannot fail, so every association experiment ran blind on
its own target metric.

**2. The sparse proxy inverts the threshold ranking.** Detection threshold was recorded across two separate
investigations as *"irreducibly LB-only — no sparse local proxy captures it"*. The cause was not sparsity but
the **node-count bonus**: trimming true cells costs no jaccard locally (they are mostly unannotated) while
paying the bonus, whereas the hidden set annotates them densely and charges for them. Removing the bonus made
the proxy reproduce the leaderboard's ordering on all four points.

**3. Calibration looks excellent — where crowding is absent.** On GT node pairs the affinity is genuinely well
calibrated at the point it commits: at `P ≥ 0.6`, mean 0.987 against 0.985 actually true, a gap of +0.002. On
**real detections in crowding** the same model puts **0.686 on edges that are wrong essentially 100% of the
time**. Both measurements are correct. They describe different populations, and only one of them is where the
score is lost.

**4. My own probe, one hour after writing the rule down.** Testing whether the head had learned distance
rather than appearance, I scored the GT graph: 446 of 457 in-gate candidate pairs were true (base rate 0.976),
so both AUCs read ~1.0 by construction and every matched-distance band was all-positive and undefined. I had
just recorded "state the population before quoting any model-quality number" as a standing rule.

## Why this is not a discipline problem

The fourth instance is the informative one. I knew the rule, had written it down deliberately, and violated it
immediately — because the easy population is the *convenient* one. Ground-truth pairs need no detection run
and no matching; real detections need both. Every one of these measurements took the cheap path, and the cheap
path is exactly the one where the phenomenon is absent.

A rule you must remember is one you will skip. The fix is mechanical: **a probe that reports a separability
number must print its base rate first and refuse to report when that rate is near 0 or 1.** The guard belongs
in the tool, not in the analyst.

## What it cost, and what it saved

Cost: the campaign's association work was steered by a metric that could not fail, threshold was written off
as untestable for weeks, and one probe produced a worthless answer.

Saved: the calibration result **stopped a filed P1 from being built on a false premise**. The likelihood-correct
linker cost replaces `distance − bonus·P` with `−log P`, and the GT-pair calibration says that is safe. On real
detections it is the opposite of safe — a confidently-wrong 0.686 edge would take a *small* penalty while the
true 0.134 edge takes a *large* one, strictly worse than the linear cost we ship. That build was two days of
work away from starting.

## The finding hiding underneath

Separating the populations produced the first genuinely new observation about the association wall in weeks:

| | mislinks | inverted | P(true) | P(chosen) |
|---|---|---|---|---|
| shipped pilkwang | 35 | 1.00 | 0.134 | **0.686** |
| our from-scratch, half an epoch | 16 | 0.94 | 0.135 | **0.250** |

Identical confidence in the truth; wildly different confidence in the error. **The pretrained head is
confidently wrong where ours is merely uncertain.** That is not the same problem as "the head is not
discriminative enough", and it does not have the same fix. A sharp, systematic preference for the near
neighbour is what a head would learn if it leaned on displacement — an excellent rule at a median step of
1.7 µm, and exactly wrong for the crowded fast-movers where the score is lost.

If that is what happened, the learned term in `cost = distance − bonus·P` is partly **redundant** with the
distance term it exists to correct, which would explain why sweeping the bonus finds a broad flat plateau
rather than a peak. Testing it needs the hard population — real detections, matched distance bands, AUC of P
*within* a band. That tool now exists with the base-rate guard built in, because the first attempt at it is
instance four above.
