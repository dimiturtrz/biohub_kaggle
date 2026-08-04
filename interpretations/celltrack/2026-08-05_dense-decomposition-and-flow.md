# Dense-movie loss after the NMS fix, and the flow-linker test (2026-08-05)

Sense-making of our own results on the dense test movie `6bba_05db0fb1`, which carries nearly all the
residual edge-Jaccard loss. Two questions: after the 3³ NMS window fix (bead 83r), *what* is the remaining
dense loss made of, and does a global min-cost-flow linker (bead tf5) recover the part a per-frame linker
cannot?

## The fate decomposition (bead 53h)

`python -m celltrack.dense_diagnosis` classifies every annotated edge on the dense movie by what the shipped
tracker did with it. Each ground-truth node is matched to a detection; the tracker's own edges then say
whether that detection linked to the truth's successor, to a wrong neighbour, or to nothing. Post-NMS-fix, at
threshold 0.99 (1183 annotated edges):

| fate | edges | % | what it means | lever |
|---|---|---|---|---|
| CORRECT | 1122 | 94.8 | reproduced | — |
| MISLINK_CONFLICT | 26 | 2.2 | linked elsewhere; true target taken by a competing source | tf5 (global flow) |
| SKIP | 14 | 1.2 | both endpoints detected, source links to nothing | cost / gap bridge |
| ENDPOINT_MISSING | 12 | 1.0 | an endpoint has no detection | detection (irreducible) |
| MISLINK_FREE | 9 | 0.8 | linked elsewhere; true target sat unclaimed | zni (signal) |

The NMS fix improved every bucket versus the pre-fix capstone (mislink 52→35, endpoint-missing 24→12, skip
26→14, correct 91.4→94.8 %): un-merging crowded cells recovered endpoints *and* removed mislinks, confirming
the fix was a genuine recall gain on the exact bottleneck, not a bonus-farm.

The decomposition isolates each lever's addressable set:

- **MISLINK_CONFLICT (26 edges)** is the only subset a re-assignment can free — the true target exists and is
  detected, but a competing source took it. This is tf5's ceiling. If a global optimiser freed all 26, the
  dense edge counts move from TP 1122 / FP 75 / FN 61 (Jaccard 0.892) to TP 1148 / FP 49 / FN 35 = **0.932**,
  a +0.04 dense best case. But `AssignmentLinker` is already *per-frame optimal*, so a conflict within a
  single frame gap is already resolved optimally there; a global linker can only improve the subset where
  *cross-frame* coupling flips the per-frame choice. Realistic gain is therefore below the 26-edge ceiling.
- **MISLINK_FREE (9)** is signal-limited: distance + learned P genuinely rank a wrong-but-detected neighbour
  above the truth even when the truth is free. No re-assignment fixes this — only a better-trained affinity
  (zni) can.
- **ENDPOINT_MISSING (12)** is detection-limited and irreducible with the current detector.

## The leaderboard recall lever (context)

Independently, lowering the detection threshold is a real leaderboard gain that the local proxy cannot see
(the proxy's sparse annotations are recall-saturated): public 0.99 → 0.887, 0.98 → 0.891, 0.97 → **0.892**
(new best public), matching the disclosed clean-baseline threshold ~0.96875. The NMS fix (un-merging crowded
cells = more recall) and the threshold (keeping more low-confidence true peaks) are independent recall levers
and should stack.

## The flow linker (bead tf5)

`FlowLinker` (`celltrack/flow_linking.py`) poses the whole video as one min-cost circulation: each detection
is a unit-capacity node (1-to-1, as the assignment linker), a source/sink feed every node at a **boundary
cost** (appearance + disappearance), and an in-gate transition costs `distance − bonus·P`, identical to
`AssignmentLinker`. Minimising over the whole graph couples the frame gaps through the boundary cost: a track
break costs an appearance plus a disappearance, so the optimiser keeps a true track intact across a crowded
frame where the per-frame solver would sever it — exactly the MISLINK_CONFLICT failure. At `boundary_cost = 0`
boundaries are free and the flow decomposes back to the per-frame assignment.

**Reduction verified.** `FlowLinker(boundary_cost=0)` reproduces `AssignmentLinker` edge for edge: per-movie
dense 0.8919 and pooled proxy 0.9344, both identical to the shipped assignment linker. The 65k-node dense
network solves in ~90 s (tractable for a submission).

### Boundary sweep verdict — refuted as a lever

Pooled proxy over `boundary_cost ∈ {0, 3, 6, 10}`: 0.9344 / **0.9351** / 0.9313 / 0.9282. The continuity
coupling is real — `boundary_cost = 3` beats the free-boundary baseline by +0.0007, i.e. the global optimiser
does keep some true tracks intact across a crowded frame — but the gain is noise-level, far below the
~0.016–0.036 the proxy overshoots the leaderboard at this tier, so it will not transfer. Past 3 the score
falls: a higher boundary penalises track *starts*, dropping the legitimate short and fast tracks the sparse
44b6 movies are full of, faster than it fixes dense conflicts.

This is exactly what the fate decomposition predicted. `AssignmentLinker` is already per-frame optimal, so
the flow can only improve the subset of the 26 conflict edges where cross-frame coupling flips the per-frame
choice — a fraction of an already-small set. The global-over-time paradigm is sound and now measured; it is
not the dense lever. `FlowLinker` stays wired as a selectable strategy (`LinkerConfig(name="flow")`), the
network-simplex solve on 65k nodes runs in ~90 s, but the shipped default remains the assignment linker.

## Conclusion: the dense mislink is exhausted; the lever is recall

Post-NMS-fix the dense movie is 94.8 % correct, and every residual bucket is either irreducible with current
assets (detection), signal-limited to ~9 edges (zni), or noise-level to a re-assignment (flow). No dense
post-processing lever cracks the pack. The measured, transferable lever is the detection **recall** the
leaderboard rewards and the sparse local proxy cannot see — the threshold ladder (0.99→0.97 = 0.887→0.892)
and the NMS un-merge, which are independent and should stack. Wall-time belongs there, not on the dense tail.
