# The proxy→LB transfer law (2026-08-09)

A consolidated read of every matched proxy↔leaderboard comparison this campaign. The 4-movie sparse proxy is
trustworthy for *some* changes and actively misleading for others; the split is not random — it follows the
**mechanism** of the change.

## The law

**A change transfers to the leaderboard only when it changes the linking *decision rule*. Changes that add
recall the sparse proxy can already see, or that merely widen candidate admission, do not transfer — and some
reverse.**

| change | class | proxy | LB | verdict |
|---|---|---|---|---|
| greedy → global `AssignmentLinker` | **decision rule** | +0.014 | 0.882 → 0.887 | **transfers (+0.005)** |
| detection threshold 0.995→0.97 | hidden-recall | flat/inverted | 0.880 → 0.892 | **transfers** (hidden-only) |
| NMS 3³ un-merge | recall | +0.012 | 0.892 → 0.892 | flat |
| smooth 0.8→0.3 | recall (match radius) | +0.008 | 0.892 → 0.889 | **anti** (−0.003) |
| node-count thr 0.995 | bonus-farm | +0.006 | 0.887 → 0.880 | **anti** (−0.007) |
| neural edge bonus (greedy) | cost tweak | +0.011 | 0.882 → 0.883 | +0.001 (noise) |
| edge gate 10→14 µm | admission | flat | 0.892 → 0.892 | flat |

## Why

The proxy is scored against **sparse** GEFF annotations (52–1229 nodes/movie of ~6–70k real cells). Three
consequences:

1. **Recall gains land on unannotated cells** — invisible to the LB's denser hidden annotation, or harmful when
   the extra smoothing/trimming perturbs the true tracks the hidden set *does* annotate.
2. **Admission widening (the gate) adds no true edges** — the annotated displacement maxes at 9.96 µm, and the
   hidden set matches it, so a 14 µm gate admits only candidates that are never the true successor. Flat both
   places.
3. **Only re-deciding which edges exist moves both** — the global linker re-solves the per-frame assignment
   (greedy → optimal bipartite), flipping genuinely-wrong choices the proxy and the LB both contain. That is the
   single mechanism that transferred a real gain.

Detection threshold is the lone LB mover the proxy can't see at all: it changes which true, hidden-annotated
peaks survive — a pure hidden-set phenomenon, structurally invisible to any sparse local proxy.

## Consequences (how to spend slots)

1. **Never ship a proxy recall-win as a default without an LB A/B.** The smooth-strength "fix" was a
   self-inflicted 0.003 LB regression — a beautiful dense raw-Jaccard gain that reversed on the board.
2. **Proxy ranks linker *structure*, nothing else.** Trust it to choose a linker; distrust it for recall,
   thresholds, node-count, and gate width.
3. **0.892 is an asset gap, not a tuning gap.** Proxy-measurable optimization is exhausted; the dense mislink is
   100 % affinity-inverted and irreducible at (1,4,4) resolution (e9b/lna/zni/tf5/gate14 all refuted). Climbing
   needs either a *different decision rule* (the disclosed Yusuke 0.897 ILP-global division-aware linker, bead
   73g — not its 14 µm gate, which we refuted) or a better association *asset* (a learned contrastive per-cell
   embedding that makes the crowded features separable, bead a1v — the untried train-the-features angle).

## Best confirmed public LB progression

0.873 (single-seed) → 0.882 (dual-seed) → 0.883 (neural bonus) → **0.887 (global AssignmentLinker)** →
**0.892 (threshold 0.97 + smooth0.8 + NMS-fix)**. Every step that stuck was a decision-rule change or the
threshold; every recall tweak and the gate widening stalled or reversed.
