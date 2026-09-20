# E57 — the broken edges are attached to badly-placed nodes, and the label said "selection"

E54 decomposed the champion's 105 broken ground-truth edges and found `wrong-association = 0`, with 76 of
the 105 carrying a **selection** label (swap, source-rival, target-rival). E54c followed the thieves and
concluded the ILP structure is right and the **discriminator is missing**. Both readings took the node
matching for granted. This asks what those matched nodes actually are.

## The measurement

`error_budget.py --shapes` already prints endpoint displacement per bucket. Read at the official ruler,
over the same 105 broken and 2022 recovered edges — one denominator, no ruler swap:

| bucket | median displacement | mean | share beyond 5 µm |
|---|---|---|---|
| **broken** GT edges | **4.08 µm** | 4.31 | **46 %** |
| **recovered** GT edges | 1.46 µm | 1.52 | 2 % |

**A broken edge's endpoints are about three times worse localized than a recovered edge's, and nearly
half of them sit more than a cell diameter from the cell they are supposed to be.** The official node
matcher forgives that: at 7 µm — roughly one cell width — a node 5 µm off still counts as detected, so its
dead edge gets blamed on selection.

That the label is mostly a function of the ruler is visible directly:

| ruler | recovered | detection-labelled | selection-labelled |
|---|---|---|---|
| 7.0 µm (official) | 2022 | 26 | **76** |
| 2.87 µm (cell separation) | 1464 | **641** | 20 |

These are different broken sets, not the same 105 relabelled, so the two rows do not subtract. What they
establish is weaker and still useful: **the detection/selection split in E54 is not a property of the
tracker, it is a property of the tolerance the split was read at.** Any claim resting on "only 5 of 39 are
detection" inherits that.

## What this does to E54c

E54c measured every thief as a real, unannotated, median-61-frame track sitting 6.92 µm from the GT cell,
and concluded the champion has the right structure and lacks a discriminator. The displacement table
supplies a second reading that E54c could not see: when the champion's own node for the GT cell is 4–5 µm
off, "the thief is nearer" is partly a statement about **where the champion put its node**, not about
which cell is really closer. The discriminator is not being fed bad features so much as bad coordinates.

This does not overturn E54c — a mis-localized node and a genuine look-alike are not exclusive, and nothing
here shows the displacement *causes* the break rather than both being symptoms of a crowded neighbourhood.
It does mean **localization is a live suspect that was never in the dock**, and it is the one E54's own
instrument was blind to by construction.

## A model-free detector localizes a slice the learned family cannot

E36 read node recall at cell separation as 0.824 and found it **flat across all nine cached detectors** —
the signature of a blind spot the whole training recipe shares, not a knob left unturned. The way to tell
a shared inductive bias from a property of the data is a detector with no weights at all.

`kaggle/classical_blindspot.py` harvests the multi-scale difference-of-Gaussians from
`fabriciodasilva/biohub-dodecatiad-cell-tracking` — a CPU-only, zero-weight tracker, and therefore the one
detector family that cannot share the pool's bias. Over every frame of the four public-GT movies:

| | |
|---|---|
| GT cells / frames | 2193 / 292 |
| champion node recall @ 2.87 µm | 0.7747 |
| champion node recall @ 7.0 µm | 0.9863 |
| GT cells the champion misses @ 2.87 µm | 494 |
| … a model-free detection within 2.87 µm | 0.1336 |
| … the same against a **frame-shifted null** | 0.0486 |
| **excess over the matched null** | **+0.0850 → 42 cells** |
| detections per frame | champion 338 · model-free **245 (0.7×)** |

Two things make this more than a coincidence. The null is the same detector's output from a frame 50 away,
which preserves its count, density and spatial distribution and destroys only cell-to-detection identity —
and the real rate nearly triples it. And the model-free detector emits **fewer** detections than the
champion, so the rescue is not bought by carpeting the volume with peaks. The stride-5 subsample gave
+0.0909 against the full run's +0.0850, so this is not a sampling artifact either.

## What this licenses and what it forbids

- **Node recall at the official ruler is CLOSED.** The champion recalls 0.9863 and misses six cells in
  2193; the model-free excess there is a single cell. Nobody should spend anything on detector *recall*.
  This is E40/E41/E42's conclusion, now with a number at the scorer's own tolerance.
- **The open channel is localization, not recall.** The same cells are found and placed badly. That is
  exactly the axis the official node matcher is designed not to see, and exactly the axis edges die on.
- **A detector swap is still forbidden.** finer122's frozen graft scored 0.7503 and the ILP swap 0.6555:
  the champion's affinities are keyed to its own detections. These 42 cells are evidence for a
  **center-offset / localization head** on the existing detector (bd Campaign A), not for importing
  dodecatiad's nodes.
- **Do not price this as 42 recovered edges.** 42 better-placed cells touch at most ~84 edge slots, which
  would clear E54's 35-edge bar, but nothing here shows that re-placing a node repairs its edge. The
  obvious oracle — snap the champion's matched nodes onto their GT positions and re-run `--shapes` — is a
  **no-op by construction**: the broken/recovered verdict depends only on the node matching, the ILP has
  already run, and moving a node cannot change which edges exist. Any test of causality has to re-solve.

## E54b already ran the oracle, and it says the discriminator is fine

The one measurement that does settle this was banked five hours earlier and under-read.
`kaggle/cue_oracle.py` ranked every in-gate candidate for each of E54's selection failures:

| cue | ranks the true successor first |
|---|---|
| plain distance from source | 10 / 61 |
| velocity from the predicted tracklet | 8 / 61 |
| oracle velocity from the TRUE GT trajectory | 10 / 61 |
| **distance to the true GT position of the target** | **61 / 61** |

E54b read the first three rows and closed the cue inventory: position, appearance and motion all fail.
That conclusion holds — **given the champion's coordinates**. The fourth row is the one this reframes.
**With correct coordinates, the simplest discriminator there is — plain distance — is perfect on every
case.** So E54c's "the structure is right and the discriminator is missing" is better stated as **the
discriminator is already there and is being fed nodes a cell diameter off**. E57's displacement table is
that same fact measured over the whole corpus instead of over 61 cases.

E54b also wrote down the mechanism and then left it: *15 more true successors sit OUTSIDE the 10 µm gate,
because matched predicted positions carry up to 7 µm of localization slack each* — noted as "unpriced and
cheap", never priced. `gate_um = 10` was derived as the maximum **ground-truth** step, and it is correct
for clean coordinates; it is applied to predicted endpoints that each carry several µm of error, so the
gate is systematically undersized for the data it actually sees. Widening it is a linker constant, needs
no training, and is the cheapest unspent item on the board.

## The chain

Bad localization (E57, corpus) → the true successor is either out-ranked or out-gated (E54b, 61 cases) →
the edge dies and the classifier, matching at a 7 µm tolerance, records it as a **selection** failure
(E54) → which sent the search after a discriminator that E54b's fourth row shows already works. A
model-free detector reaches part of the gap without any ground truth (42 cells, null-beating, at 0.7× the
detection count), which is what makes the localization channel a lever rather than an oracle.

**Next, in cost order:** widen `gate_um` and re-solve (a constant, no training) · price a center-offset
head against the 42 (bd Campaign A) · never swap the detector wholesale (finer122 graft 0.7503).

## Cost

Three CPU runs, about twenty-five minutes, zero GPU, zero submissions.
