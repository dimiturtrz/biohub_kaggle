# E58 — the champion already holds two thirds of its broken edges, one node to the side

E57 showed the node the scorer picks for a broken edge sits median 4.08 µm from the cell, three times
further than for a recovered edge. That leaves the question the displacement table cannot answer: is the
edge really absent from the champion's graph, or does it exist between a **different** pair of nodes that
are also inside the 7 µm ruler of the same two ground-truth cells, and merely lost the assignment?

`kaggle/representative_alternates.py` asks exactly that. For every GT edge it collects every predicted node
within the ruler of the GT source and of the GT target, and tests whether any predicted edge joins the two
sets. Recovered edges run through the same test as a control and must read 1.000, since their own matched
pair qualifies.

| bucket | n | an in-ruler pair IS already linked | candidate nodes near the source |
|---|---|---|---|
| recovered (control) | 2022 | **2022 — 1.000** | 1.23 |
| **broken** | 79 | **51 — 0.646** | **1.71** |

*(26 further GT edges have an endpoint that matched nothing at all and are out of scope here.)*

**The champion's tracker holds 51 of its 79 in-scope broken edges.** It linked a node inside the ruler of
the source cell to a node inside the ruler of the target cell; the scorer's Hungarian assignment simply
represents those two cells by a different pair, and the edge it looks for between *those* is absent. The
broken bucket is also duplicate-enriched — 1.71 candidate nodes near the source against 1.23 for recovered.

51 edges clears E54's 35-edge bar. So the obvious move is to remove the decoy that wins the assignment and
let the linked node be chosen. That is the move this probe then kills.

## The decoys are not spurious detections — they are a second real track

| | source | target |
|---|---|---|
| edges carried by the node that wins the assignment | **1.84** | **1.88** |
| of the 51, how many are isolated | **0** | **0** |

**Not one of the 102 decoy endpoints is isolated.** Each carries about 1.85 edges, which is the degree of a
node in the middle of a track, and the champion already runs `OUTPUT_PRUNE_ISOLATED`, so anything truly
stray is gone before the submission is written. Deleting a decoy destroys ~1.85 edges to gain one. The
arithmetic is negative and it does not depend on any threshold.

This is E54c measured at node resolution instead of track resolution. E54c found every thief to be a real,
unannotated, median-61-frame track sitting 6.92 µm from the GT cell. E58 says the same thing about the
*node* the scorer chose: two genuine tracks run within one ruler-width of each other, and the annotated
cell is represented by the unannotated track's node.

## What is left, and what is not

- **The CPU post-processing route is closed.** Node deletion is negative by the degree arithmetic, and a
  blanket near-duplicate merge is hopeless anyway: 28197 of 122808 nodes (**23 %**) have a same-frame
  neighbour inside the ruler, so 51 targets sit in a pool of 28197 — the E48/E49 base-rate trap, at a worse
  ratio than the one that closed the prune axis.
- **Perturbing the decoy's coordinates to lose the assignment is not on the table.** It would work, and it
  is gaming the matcher rather than tracking cells. Not submitted.
- **The honest version of that same effect is localization.** A decoy sits on the GT cell because it is
  mis-placed by ~4 µm (E57). Centre it on the cell it actually belongs to and it stops contesting the
  assignment on its own, both tracks stay intact, and the linked node wins. That is the center-offset head
  (bd Campaign A) and it needs training — the one axis on the board that a CPU cannot reach.

## The chain, now complete

The detector places a node ~4 µm off (E57) → that node falls inside the 7 µm ruler of a *neighbouring*
cell and wins its representative slot (E58) → the edge the champion genuinely has, between the correct
pair, is invisible to the scorer, and the GT edge reads as broken → E54 files it under **selection**, and
E54b finds no cue that fixes it, because with correct coordinates plain distance is already perfect 76/76.
Every step is now measured, and all of them point at localization.

## Cost

Two CPU runs, a few minutes, zero GPU, zero submissions.
