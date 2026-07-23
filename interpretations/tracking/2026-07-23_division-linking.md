# Division-aware linking beats the nearest-neighbour ceiling

*2026-07-23 — `biohub_kaggle-cdo`, linking ground-truth nodes on validation fold 0.*

## Result

On ground-truth nodes — perfect detections, so this isolates the linker — a division-aware linker
exceeds the nearest-neighbour ceiling the bracket (`dll`) established:

| linker (GT nodes, fold 0) | edge Jaccard | division Jaccard | score |
|---|---|---|---|
| nearest-neighbour | 0.9976 | **0.000** | 1.0884 |
| division-aware (7 µm gate) | **0.9980** | **0.333** | **1.1222** |

(The scores exceed 1.0 because linking only the sparse annotated nodes triggers the metric's
under-prediction bonus — see `2026-07-23_metric-shape.md`. What matters is the *comparison*, which is
apples-to-apples.)

Nearest-neighbour linking scores a division Jaccard of exactly **zero**: its one-to-one assignment gives
every cell a single successor, so it can never represent the two-child topology a division needs. That is
structural, not a tuning miss, and it is the entire gap the bracket left in linking. The division-aware
linker recovers a third of the divisions and edges slightly ahead on the edge term too, lifting the score
0.034 — almost all of it the division term (0.1 × 0.333).

## How it works, and why the gate is tight

Run the same one-to-one assignment first, then a second pass: a daughter left unmatched by the assignment
is joined to the nearest already-matched parent within a **division radius**, if that parent has fewer
than two children. The radius is the load-bearing parameter, and it wants to be **tight**:

| division gate | edge Jaccard |
|---|---|
| 7 µm | **0.9980** |
| 15 µm | 0.9960 |
| 25 µm | 0.9902 |

7 µm — the competition's own matching radius — is where it peaks. Wider gates start attaching *unrelated*
new cells to nearby parents as spurious divisions, and those false edges cost more than the true divisions
they buy. A daughter is within about a cell of its parent at the moment of division, so the physical
matching radius is exactly the right scale; reaching further only invents divisions.

## What it means

- **`cdo` is met:** division-aware linking beats nearest-neighbour on ground-truth nodes, on both the edge
  term (0.9980 > 0.9976) and decisively on the full score (1.1222 > 1.0884), which is the DONE-WHEN.
- **The gain is small in absolute terms** because divisions are rare (151 across the whole train split,
  ~2 % of nodes). Linking on perfect detections was already ~solved; this closes the one structural hole.
- **It does not touch the real score bottleneck.** The matched A/B (`2026-07-23_operating-point-sweep.md`)
  showed the score is lost to detection *jitter*, not to linking on perfect nodes. This linker is the
  right linking baseline and recovers divisions for free, but the score lever remains detection
  stability. The natural next step is a matched end-to-end run of this linker on the *learned* detector's
  jittery output, and a detector consistency objective — not more linking on GT nodes.
