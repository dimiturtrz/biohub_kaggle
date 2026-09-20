# E46 — the outranked band is not appearance-bound, it is *pairwise*-bound

E45 closed the geometric cues on the 11.95 % outranked band and concluded it was appearance-bound.
That conclusion rested on `celltrack-appearance-refuted-right-denominator`, a kill measured at **n=37**
that explicitly marks itself re-testable. E43 hands us the same band at **n=1226** with a correct
denominator, and the volumes are CPU-readable, so the re-test costs nothing but wall-clock.

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
distinguishes the true target from the lookalike is not visible in the pair at all: it is that the
lookalike **has no continuation** — it is an FP that does not extend into a track, whereas the true
target does. That is a tracklet-level fact, available only to a solver that competes *sequences*, not
edges.

## What this licenses

- **Closes** the pairwise-cue axis on the outranked band. Do not file another appearance, embedding,
  distance or velocity term for it — the mechanism says they all select the confusor.
- **Promotes** the two-pass tracklet ILP (bd `kiw1`) from "a lever on the frontier's list" to the lever
  this axis *argues for by mechanism*: it is the only open item that can use a candidate's own
  continuation to disqualify it. This is also one of the jointly-necessary 0.945 levers in the
  conclusion tree, tested only solo and killed.
- Sizes the prize honestly: the band is 11.95 % of true candidate edges, with the affinity margin at a
  median 0.0985. A tracklet pass does not automatically win it — it is the only structure that *could*.

## Caveats

- Raw NCC, not a learned embedding. The mechanism argument (anti-informative pairwise geometry) is what
  generalizes to embeddings; the number is not a bound on a learned feature by itself.
- Caches carry our tunet's node recall (0.697–0.750), not 0947's 0.824, so the FP field here is denser
  than the champion's. That makes the near-static-lookalike population, if anything, *more* prevalent
  than it would be under 0947.
