# E47 — pricing the prune axis before building it: collateral, not FP-recall, is the binding constraint

E46 ended by pointing at bd `g89y` (Ultrack multi-hypothesis contour hierarchy + segment-SELECTION ILP),
because the detection that outranks a true edge is an unmatched FP in 0.9698 of contests. "Prunable in
principle" is not a licence to build: `g89y` is a GPU build with a detector pass behind it. The cache
already holds everything needed to price it on CPU, so this run answers *what a pruner must achieve*
before it is worth anyone's wall-clock. Instrument: `kaggle/prune_operating_point.py`.

The measured quantity is the offline association proxy — the fraction of true edges that are mutual-best
(best proposal out of their source and into their target), whose complement is the E43 outranked band.

## The ceiling, and the thing that eats it

```
true edges n=11879   baseline mutual-best 0.8805 (outranked band 0.1195)
  FP-recall 1.00  collateral 0.00   mutual-best 0.9944  net +0.1139     <- ORACLE
  FP-recall 0.75  collateral 0.00   mutual-best 0.9494  net +0.0689
  FP-recall 0.75  collateral 0.02   mutual-best 0.9151  net +0.0347
  FP-recall 0.75  collateral 0.05   mutual-best 0.8648  net -0.0157
  FP-recall 0.50  collateral 0.02   mutual-best 0.8844  net +0.0040
  FP-recall 0.25  collateral 0.02   mutual-best 0.8605  net -0.0200
  FP-recall 1.00  collateral 0.05   mutual-best 0.9015  net +0.0210
```

Deleting every unmatched detection rescues **95.3 %** of the band (0.1195 → 0.0056). That is the ceiling
of the entire prune axis, and it is large. But the oracle deletes 97.2 % of all detections and keeps
0.37 % of candidate edges — it is "given the GT, keep only the GT", so the ceiling licenses nothing by
itself.

The operating grid is where the decision lives, and it has one dominant term. **Collateral enters at
roughly twice its own size**: deleting 2 % of real cells destroys 0.040 of true edges, 5 % destroys
0.095. The factor two is structural — a deleted cell takes the true edge on *both* sides of it, while
FP-recall only rescues inside a band 0.1195 wide. So:

- at **q = 0.05, even a perfect FP pruner nets +0.021**, and every realistic one is negative;
- at **q = 0.02** you need FP-recall ≈ 0.5 merely to break even, and 0.75 to earn +0.035;
- at **q = 0** the axis is worth up to +0.11.

The requirement is therefore not "aggressive" but **asymmetric**: a pruner must be roughly an order of
magnitude more careful with real cells than it is thorough with FPs.

## This retro-explains the LB-refuted recovery stack

The obvious pruner is a higher detector threshold, and it is already refuted on the board (recovery
stack, LB ≤ 0.900). E47 says *why*, mechanistically, rather than as a tuning failure. A threshold is an
intensity-ordered decision, so it buys FP-recall and collateral along the **same axis**, at roughly 1:1 —
and E41 established that the real cells at risk sit exactly there (the missed ones are 2.4× fainter than
same-frame found, 46.9 % in the bottom intensity decile). A threshold move therefore lands near the
q ≈ r diagonal of this grid, where every row is a loss. The recovery stack did not fail because it was
mistuned; it failed because its decision variable couples the two columns.

## What this licenses — and the cheap precondition to test first

`g89y` survives this pricing, but conditionally, and the condition is now nameable. Ultrack selects
among **nested contour hypotheses** by a consistency criterion, not by a global intensity cut, so its
errors are not forced to ride the intensity axis — that is the *only* reason it can sit off the diagonal
where a threshold cannot. That is a claim about the method, and it is testable far more cheaply than the
build:

> Before building `g89y`, measure whether hierarchy selection's survival decision correlates with
> detection intensity. If it does, it inherits the recovery stack's coupling and this axis is closed for
> it too; if it does not, the +0.035 operating point is reachable.

Sizing honestly against the 1.5 % bar: +0.035 offline at a plausible operating point (r = 0.75,
q = 0.02) clears it with room, and the oracle says the axis has +0.11 of headroom above that. This is
the first axis in some time whose *ceiling* is not the problem.

## Caveats

- **Offline association proxy, not the official metric, and not the LB.** The proxy is established to
  invert at the top of the range; this prices the AXIS, it does not promise a board movement.
- **Measured on our tunet caches** (node recall 0.697–0.750), whose FP field is 0.972 of all detections —
  far denser than 0947's. The band width, the oracle ceiling and the operating points are all relative
  to that substrate.
- **Prune errors are drawn uniformly at random**, which flatters a real pruner: its mistakes concentrate
  on faint and crowded cells, the same population the rescued edges live in. Every gain here is an upper
  bound at its own q.
