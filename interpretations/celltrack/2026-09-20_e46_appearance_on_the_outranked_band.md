# E46 — the outranked band is not appearance-bound, it is *pairwise*-bound

**Prior art first: this is a replication, not a discovery.** `celltrack-confusor-pairwise-unresolvable-needs-multiframe`
(2026-08-23) already reached this conclusion, and reached it with the same qualitative signature —
full-res appearance top-1 **0.115 against a 0.260 chance**, the source correlating *better* with the
rival (+0.808) than with the true successor (+0.540), "a fast mover fled; a slower neighbour sits
between", and "only GLOBAL MULTI-FRAME assignment breaks it". I did not surface it on the prior-art
check because **it was missing from the memory index**; the index has been repaired. The cost of that
miss was re-deriving a known result, and the process lesson is recorded with it: an unindexed memory is
an unfindable one.

What E46 legitimately adds over that result is power, breadth, a control, and one **correction**:

- **n=37 on one movie → n=1226 across 20 movies**, on the E43 denominator (the outranked band as
  defined by the fragmentation split), so the fact is corpus-level rather than one-movie.
- **A displacement-matched control**, which the original lacked. Without it, "the true target looks
  less like the source" is confounded — the band is a fast-cell population (E45), and appearance
  decorrelates with motion on its own. The matched stratum is what turns the observation into a verdict.
- **The rival's identity, measured**: the original assumed the confusor was a *distinct slower cell*
  ("NMS can't remove it without merging real cells"). At n=992 it is an unmatched **FP in 0.9698** of
  contests. That changes the licensed lever — see below.

Instrument: `kaggle/outranked_band_appearance.py`. Normalized cross-correlation between a cell-sized
cube (`±2, ±6, ±6` voxels, spacing `(1.625, 0.40625, 0.40625)` µm — the 2.87 µm GT nearest-neighbour
floor is ~1.8 z / ~7 xy voxels) around the source detection at *t*, and cubes around the TRUE target
and the RIVAL at *t+1*. Measured twice: on the band, and on the MUTUAL-BEST edges against their
runner-up rival, so a null can be read.

## The control is what makes the result sayable

```
band    n= 1226  NCC true +0.5648  rival +0.5733  picks true 0.4853
mutual  n= 8493  NCC true +0.7052  rival +0.4575  picks true 0.8230
```

Appearance is emphatically **not** dead: the identical measurement separates true from rival at 0.8230
on mutual-best edges. Patches carry variance (source/target median std 110.9 / 110.7, zero
zero-variance patches out of 19438), so the band's 0.4853 is a real coin flip, not a dead read.

## It is not fast-cell decorrelation either

The band is a fast-cell population (E45: mean displacement 3.831 vs 1.441 mutual-best), and appearance
decorrelates with motion on its own, so only a **matched** stratum licenses a verdict:

| displacement (substrate units, ×1.625 µm) | band | mutual-best |
|---|---|---|
| [0, 2) | 0.7009 (n=107) | 0.8436 (n=6164) |
| [2, 3.46) | 0.5142 (n=424) | 0.7769 (n=1999) |
| [3.46, ∞) | **0.4345** (n=695) | **0.7182** (n=330) |

At matched displacement, mutual-best loses only 0.125 to motion (0.8436 → 0.7182) while the band falls
to 0.4345 — **below chance**. The collapse is band-specific. It also survives the harness check in the
other direction: the band's own slow subset reads 0.7009, so the instrument works on band edges too.

## Below chance is the mechanism, not noise

`picks true 0.4345` says the rival looks **more like the source** than the true target does. Compose
with E45's finding that on the same band the true edge is FARTHER than its rival in 0.7984 of cases
(true median 4.12 vs rival 2.24 substrate units): the confusor is a **near-static lookalike** sitting
close to where the cell was, while the true cell has moved far and changed appearance. Every local
pairwise cue — affinity, distance, motion, appearance — therefore points at the same wrong target, and
three of the four have now been measured *below* chance on this band.

This retires the framing E45 left standing. The band is not waiting for a better appearance feature; a
learned embedding trained on the same pairwise view inherits the same anti-informative geometry. What
distinguishes the true target from the lookalike is not visible in the pair at all.

## The lookalike is an FP, not a rival cell — and that decides the lever

The 2026-08-23 result read the confusor as a real, slower neighbouring cell, which forecloses pruning:
you cannot delete a cell without losing it. Measured on the same contests (`outranked_band_geometry.py`,
question 4 — a detection counts as real if it matches a GT node within the official 7 µm):

```
RIVAL is an unmatched FP in 0.9698 of contests (n=992)   real GT-matched cell 30 (0.0302)
```

So the target that outranks the truth is, almost always, a **spurious duplicate** — consistent with
E45's "true edge farther than its rival in 0.7984 of cases" (true median 4.12 vs rival 2.24 substrate
units): it sits near where the cell *was*. The true cell moved far and changed appearance; the FP left
behind did neither. That is why every local pairwise cue — affinity, distance, motion, appearance —
selects the same wrong target, three of four now measured *below* chance.

The caveat is the FP field: these caches carry our tunet's node recall (0.697–0.750), not 0947's 0.824,
so duplicates are denser here than under the champion. 0.9698 is not a champion-calibrated number. The
*sign* is what carries — an FP-dominated confusor population is prunable in principle, a cell one is not.

## What this licenses

- **Closes** the pairwise-cue axis on the outranked band. Do not file another appearance, embedding,
  distance or velocity term for it — the mechanism says they all select the confusor.
- **Points at `g89y` (Ultrack multi-hypothesis contour hierarchy + segment-SELECTION ILP) first**, not
  at a better edge feature. The confusor is an intra-frame FP duplicate, and `g89y` is the only open
  item that *removes* such a candidate structurally rather than by a threshold. This matters because
  the threshold route is already refuted on the board: E36/E41 show raising the detector threshold to
  kill FPs starves recall (the recovery stack, LB ≤ 0.900). Selection, not thresholding.
- **Keeps the two-pass tracklet ILP (bd `kiw1`) as the second half of the same fix**: what marks an FP
  duplicate is that it has **no continuation**, a tracklet-level fact invisible in the pair. It is one
  of the jointly-necessary 0.945 levers in the conclusion tree, tested only solo and killed. The two
  items attack the same population from the two sides — prune the hypothesis, or out-compete it with a
  sequence — which is why the tree lists them as jointly necessary.
- Sizes the prize honestly: the band is 11.95 % of true candidate edges, with the affinity margin at a
  median 0.0985. Neither item automatically wins it — they are the only structures that *could*.

## Caveats

- Raw NCC, not a learned embedding. The mechanism argument (anti-informative pairwise geometry) is what
  generalizes to embeddings; the number is not a bound on a learned feature by itself.
- The FP-field caveat above applies to every number here, not just to 0.9698: these caches are our
  tunet's, so the lookalike population is denser than 0947's.
- The 0.0302 real-cell remainder is a floor, not zero. Those contests are genuine cell-vs-cell
  competition that no pruning reaches — they need the solver.
