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

## The mislink signal gate (bead zni)

Before retraining anything, `MislinkSignal` (in `dense_diagnosis`) asks whether the affinity even *can* be the
problem: for every mislinked edge it reads the edge transformer's probability of the true successor versus the
wrong neighbour the tracker chose. If the affinity ranks the true successor higher (`p_true ≥ p_chosen`) the
mislink is a *cost* error — the nearer wrong cell won on the distance term — which the bonus/gate sweep already
refuted (e9b). If it ranks the wrong one higher (`p_chosen > p_true`) the affinity itself is wrong, and only a
better-trained affinity can fix it.

Post-NMS-fix, on the dense movie's 35 mislinks: **100 % are affinity-inverted**, mean `p_true = 0.134` vs
`p_chosen = 0.686`. The edge transformer is not merely losing on distance — it is *confidently* scoring the
wrong near-neighbour far above the true far-successor on exactly the hard crowding cases (the 1.4 % AUC tail
the aggregate 0.986 hides). This is a signal error, and its shape — a confident wrong *near* neighbour — is
exactly what **hard-negative mining** targets: teach the transformer that the spatially-proximate competitor is
*not* the successor. The earlier edge-retrain refutations (bead lna) used synthetic pairs and real
positives-only; neither mined these near-neighbour hard negatives. So substrate C (real detections + mined
hard-negatives) is both untested and the correctly-aimed fix — the gate passes, and zni is worth building.

### The hard-negative finetune — refuted

`celltrack/training/edge_finetune.py` runs the probe the gate motivated: detections on three denser 6bba
movies, each annotated source's nearest wrong targets mined as hard negatives, the transformer finetuned
(UNet frozen) with the frontier focal-BCE plus a term pushing `P(source → near-decoy) → 0`, evaluated on the
held-out dense movie. Both weight settings degrade:

| arm | correct | mislink | inversion | P_true | P_chosen |
|---|---|---|---|---|---|
| before | 1122 | 35 | 100 % | 0.134 | 0.686 |
| hard-neg wt 1.0 | 948 | 29 | 96.6 % | 0.017 | 0.501 |
| hard-neg wt 0.3 | 1046 | 31 | 93.5 % | 0.032 | 0.606 |

The hard-negative term does deflate `P_chosen`, but it crushes `P_true` at least as hard — the finetune
*flattens* the column rather than re-ranking it, and every arm loses far more correct edges (76–174) than the
mislinks it fixes (4–6). The mechanism is clear: the wrong near-neighbour and the true far-successor share the
frozen UNet features, so suppressing the near one's probability drags the whole softmax column — including the
true edge — down with it. The features at (1,4,4) resolution genuinely favour the near cell; no re-weighting of
the head separates them. This is the third substrate to refute an edge-retrain (after `lna`'s synthetic and
real-positives-only), and it rules out the hard-negative shape the gate specifically pointed at.

## Conclusion

Every lever on the dense mislink is now exhausted with a mechanism: per-frame cost knobs (`e9b`), global-flow
linking (`tf5`, +0.0007), complementary/retrained detectors (`884`/`ksv`/`slw`/`fcd`), and affinity retraining
across three substrates (`lna`×2, `zni`). The dense crowding mislinks are irreducible with the current detector,
edge model, and (1,4,4) resolution — cracking them needs assets we lack (a stronger detector or edge model, or
finer resolution), not another post-processing or retraining pass. The one live, transferable lever is the
detection **recall** the leaderboard rewards and the sparse proxy cannot see: the threshold ladder
(0.99→0.97 = 0.887→0.892) stacked with the NMS un-merge. That is where the wall-time goes.
