# 14th place — Vibes&Edges Trade-Off: regress correspondence, never classify a division

Source: topic **#744486** (Tom, 14th of 4,020, gold). 10 votes.

**This is the strongest published answer to the two axes we could not move.** We closed the confusor axis
as "pairwise-unresolvable, needs multi-frame" and the division axis as "model-bound". 14th solves the first
with a **dense per-voxel field** instead of a pairwise classifier, and the second by **not classifying at
all** — it regresses where each voxel's parent was and lets geometry declare the division.

Two independent pipelines run over the same volumes and are **grafted** per video, with a circuit breaker
that falls back to the stronger graph when their own detector collapses.

## 1. FlowSeg — separation carried by field direction, not by contrast

> "Two touching cells have no intensity valley between them, so they merge into one blob. Predicting
> boundaries directly is worse in 3D: a boundary is a thin, low-mass target, and one gap in it merges two
> cells."

Every voxel regresses a vector pointing at the centre of *its own* cell. Segmentation becomes dynamical:
release each foreground voxel and integrate along the field (200 steps of 0.5 µm, released where
`cellprob ≥ 0.935`); voxels of one cell converge on one attractor. At a shared surface between two touching
cells **the flows point in opposite directions** — a discontinuity no threshold can see but an integrator
resolves immediately. Instances under 20 voxels are discarded; four flip views decode independently and
their sinks merge by consensus.

**Why this matters to us.** Our confusor work (`E43`–`E46`, the four-cue study, the appearance-below-chance
result) framed the problem as *deciding between two candidates*, and we proved that framing exhausted at
voxel resolution. FlowSeg never decides between candidates: the separation is produced upstream, in a dense
field, before any candidate exists. Our conclusion "pairwise-unresolvable" was correct **about pairwise
methods** and says nothing about this one.

**One mask, three readings** — the detail Tom flags as easy to miss. The same FlowSeg mask supplies
(a) centroids as the node set, (b) per-instance voxel membership as the support divflow averages over,
(c) shape statistics (volume, equivalent radius, three covariance eigenvalues) as GNN node features. One
forward pass, three distinct downstream meanings.

## 2. Hungarian exclusive consensus, because the failure mode is over-detection

Two FlowSeg checkpoints differ only in temporal evidence: `sym` sees `[t−k, t, t+k]` with one central
difference; `stack4` sees `[t−2k … t+2k]` and stacks four differences (two symmetric spans, two asymmetric
cross-spans). Short symmetric = sharp on slow cells; long mixed = survives frames where the short
difference is noise-dominated.

Their sinks merge by **Hungarian assignment**, not an inverse-distance vote. Pairwise distances in µm
(voxel scale `(1.625, 0.40625, 0.40625)`), pairs beyond `r = 6.0 µm` priced out, one-to-one solve, forced
over-radius takes discarded.

> "The earlier merge was a soft inverse-distance vote, which is many-to-one: when one model splits a single
> cell into two fragments, both fragments can collect a vote from the same sink in the other model, and
> both survive. This pipeline's dominant failure mode is over-detection, so a rule that lets one piece of
> evidence back an unbounded number of detections is the wrong rule."

Run twice with each model as anchor (the assignment accumulates onto the first list, so it cannot create a
cluster the anchor never proposed), unioned, deduplicated at 4.0 µm, quorum `0.75 × n_entries` = unanimity
at two entries. The surviving position is the **mean of the matched pair** — the fusion also buys a little
localisation.

**`unettf` vetoes, never votes, never seeds.** A separate architecture's detections are queried from a
KD-tree; a FlowSeg node survives if any lies within 6.0 µm. The kept node **keeps FlowSeg's coordinates** —
"every attempt at mixing the two coordinate sources measured negative". A cell only `unettf` sees never
enters the graph: "its job is precision, not recall, and admitting its unique detections was measured as a
loss." Its folds fuse by **elementwise logit mean before peak-finding**, because a union of peaks would make
the corroboration gate strictly *easier* with every fold added — the opposite of what a gate should do.

They kept the many-to-one form here despite the exclusivity argument, and said why: mutual-NN scored
0.89730 vs 0.89980, cutting division FP 12→10 but costing a division TP (15→14) and dropping node recall
0.9652→0.9579, "because reciprocal matching throws away a perfectly good pair whenever the unettf node's own
nearest happens to be someone else."

## 3. divflow — the division model our post-mortem named as the gap

> "The training set contains 121 real division events. A classifier trained on 121 positives against a
> large negative pool learns the negatives."

Instead: for every voxel of a later frame, predict the displacement back to where its parent was. **Division
is never classified.** It is a coincidence of two pure distance tests on quantities the network produced:

1. **Convergence in parent space** — two instances at `k+1` whose predicted parent positions fall within
   **4 µm** claim a common origin.
2. **Separation in observed space** — those same instances must be **3–16 µm** apart now.

> "One cell over-segmented into two fragments converges — but the fragments are closer than 3 µm, so it is
> rejected. A flow error that maps two unrelated cells to the same place converges — but they are further
> apart than 16 µm, so it is rejected. Neither test alone would be trustworthy; together they are, because
> they fail in different directions."

**The reframing is the mechanism**: a rare-event classification becomes a dense regression, so every voxel
of every supervised cell contributes gradient rather than one label per event.

Training: frame pairs `(k, t)` with `|t − k| ≤ 7` in both directions; channel 0 is the division frame the
target is expressed in, channel 1 is where the output lives. Supervised only inside the daughters' instance
masks, target scaled down by 10. **Negatives are cells whose entire lineage contains no division, given a
zero target** — "a cell that predicts its own centroid converges with nothing, which is exactly what 'no
division' means."

**Balance by voxel mass, not cell count.** In a voxel-weighted loss, counting cells hands the negatives
**94% of the gradient**. "This is the difference between divflow training and divflow collapsing, and it is
invisible until you print the realised mass per epoch." The one augmentation that matters lifts a whole
division event and pastes it elsewhere — another region, another video's division-free pair, or several
stacked into one volume.

divflow's output is **a vote, not a division**: clustering, parent search and xgboost/catboost/verifier
filters all sit downstream.

## 4. The ILP never divides — a fourth independent confirmation

> "With edge = −1.0 and division = 1.2, a second outgoing edge gains 1.0 and costs 1.2. The break-even sits
> exactly at |edge| = 1.0, so at 1.2 a fork never pays and the ILP output contains **zero divisions by
> construction**. That is why every division in our final submission is created by a later stage rather
> than by the linker. Knowing this changed where we spent the rest of the competition."

This is the same fact that made our own `division_cost` sweeps plumbing with no consumer — now confirmed by
a fourth team, with the arithmetic spelled out. `appearance = 0.0`, `disappearance = 2`: ending a track is
expensive, starting one is free, biasing the solver toward continuing through weak evidence.

Their division supply chain, after the ILP contributes nothing:

- **Safe-division repair with a learned veto** — generous geometry proposes, a centre-prior network
  discriminates. A real log: 176 geometric candidates, 142 vetoed, 34 survive, 32 added. "Geometry is cheap
  and has nothing to overfit; the model is expensive and has very few events."
- **An RBR adder** — a dividing nucleus passes a characteristic red–blue–red intensity signature across
  three time points, computed as a fixed **39-weight oriented prior over Δcellprob at several lags**, no
  training, byte-identical offline and in-kernel.
- **A div-site ViT re-ranker, whose target had to be corrected.** Version one was trained on *"would
  inserting here earn a division true positive?"* — "which turned out to be a property of the graph rather
  than of the image — two of the four TP-causing insertions scored below the FP median." The shipped model
  asks a question about the image instead: does this candidate sit on a real division, mother within 5 µm of
  an annotated mother, children matching her daughters.
- **A mother model, to break a symmetry.** Every signal centred on the daughter pair is reflection-
  symmetric, so a mirrored candidate pair shares a midpoint and gets the same logit. A separate "is this
  node a division mother at t → t+1" model gives the layer a direction.
- At most one append per video, **append-only**: nothing deleted, no already-forking node touched, so every
  upstream division is untouchable *by construction rather than by convention*.

## 5. Localisation helps the scorer and hurts the graph — measured on both sides

Their coordinate correction is a **single constant vector, zero parameters**, the mean GT − prediction
residual fitted on folds 1–4 only: **(+0.6807, +0.2067, +0.1725) µm = (+0.419, +0.509, +0.425) voxels**.

> "Measured both ways, the shift is worth **+0.01116 at the scorer and −0.00573 to the graph** — moving
> detections before linking makes the linker draw **73 fewer true edges**. Applied last, only the good half
> is banked."

**This contradicts 213th and 303rd, and the contradiction is informative** — see
[the coordinate-head axis](2026-09-30_the_coordinate_head_axis.md). Short version: a *constant* shift moves
every detection equally, so it cannot improve relative geometry and can only disturb the linker's gates; a
*learned per-detection* shift changes relative geometry and therefore helps the matcher. 14th also reports
that "every learned correction head we tried bought its gain on the edge term by destroying a division."

## 6. The rest of the chain, priced

- **Linefit smoothing: turning it off costs −0.01914** — "the single largest mechanism in our chain."
  A node's position is measured independently per frame, so a track inherits per-frame jitter; refit each
  node to a straight line through its own neighbourhood.
- **Motion gates on the residual after whole-frame rigid motion is removed.** "A gate expressed as 'a cell
  cannot travel more than X µm between frames' is really measuring two things at once — the cell's own
  motion and the whole volume's drift — and the second term can dominate." Two gates: 5.5 µm for confident
  pairs, 10.0 µm otherwise, 14.0 µm hard ceiling; non-consecutive-frame edges removed outright.
- **Three gap closers, cheapest first**, then readmit (a discarded peak ≥ 0.965 within 4 µm of a node with a
  missing in- or out-edge), then prune (edgeless nodes; components under 6 frames). "None of these invent a
  cell. Every node added is a real peak the detector produced and the threshold rejected."
- **GNN linker**: GraphSAGE, hidden 128, 6 layers, `jk='cat'`, dropout 0.3, one graph per (video, frame),
  5-fold GroupKFold by video, node features = the FlowSeg mask's shape statistics.
- **Graft, do not average.** "Averaging two graphs is not defined." One graph is the frame; the other
  contributes named structures. Forks in the donor are classified near (same event, reduced to nearest child
  so the graft cannot duplicate) or far (transplanted), under three refusals: never create an accidental
  fork, never displace an in-edge of an original fork, never orphan a node.
- **A per-video circuit breaker with a measured trip point.** A donor node with nothing of theirs within
  7 µm in its frame is a *hole*; `hole_fraction = holes / donor nodes` measures their detector's deficit
  directly. Healthy fold-0 videos ≤ 0.125, collapsed ones 0.21–0.30, trip at ≥ 0.18 → discard the merge and
  ship the donor graph. "Out of sample on fold-0 it wins exactly on those videos (0.656 vs 0.601, +0.007
  overall) and never touches a healthy one."
- **A format guard, not an upstream fix.** The last thing before the writer removes self-loops, endpoints
  that are not declared nodes, duplicate edges, and any edge whose frame delta is not 1; it writes the dict
  key as `node_id` rather than the stored field because "edges reference keys and one drifted field is a
  KeyError inside the scorer's id map." "Why a guard rather than a fix upstream: it is a no-op on a valid
  graph. Its cost is zero, and its value is that a rare, data-dependent inconsistency cannot reach the
  scorer."

## 7. Five methodological findings, in their words

1. **"A zero median does not mean no effect."** The post-hoc shift had a per-video median of exactly zero
   locally and returned +0.002 on the board. "We now judge a candidate on its per-video win rate and its
   spread, not on whether the median moved."
2. **"Out-of-fold can understate a shipped model."** An OOF score vector is a mixture of k
   differently-calibrated models, and one absolute threshold cuts it at k operating points — "which smears
   exactly the head of the ranking, where a deletion or selection layer lives." Retrained on a disjoint set
   and applied once, the same layer measured materially better than its own OOF estimate.
3. **"A verifier only knows what you already find."** Their lineage verifier scores median **0.9393** on the
   divisions the chain finds and **0.0001** on the ones it misses. "It is a filter, not a discoverer, and
   used as a feature to propose divisions it actively suppresses them."
4. **GroupKFold by video is not a detail.** "Two candidates from the same video share a detector, a frame
   and an embryo, and a row-wise split reports a score the leaderboard never reproduces." The lineage
   verifier — a deliberately tiny GRU, a few thousand parameters, because the corpus holds ~151 annotated
   divisions — imposes daughter-swap symmetry **architecturally** (shared encoder, symmetric pooling) rather
   than by augmentation: "that halves the effective data requirement and removes a whole class of spurious
   learning."
5. **What they deliberately did not ship**: a steal layer and an insertion layer, "both of which moved the
   division term but delete or insert structure, making every number conditional on the rest of the chain
   being exactly as measured"; a second divflow offset (0.7–1.05 h of runtime, and it "changes the score
   denominator of every single-offset candidate"); and a learned appearance model for linking, which "did
   not transfer across the two embryo families in this data" — the same result we got from
   `contrastive appearance built, sites refuted`.

## 8. What this changes for us

| our verdict | status after 14th |
|---|---|
| confusor pairwise-unresolvable, needs multi-frame | **correct about pairwise; FlowSeg's dense field is a different method we never built** |
| division axis is MODEL-bound | **confirmed, and the model is published** — dense parent-correspondence regression, not classification |
| public ILP cannot fork at shipped weights | **fourth independent confirmation**, with the 1.0-vs-1.2 arithmetic |
| snap/localisation oracle could not price localisation (`E59`) | **both halves now measured**: +0.01116 scorer, −0.00573 graph. Our oracle measured the scorer half only |
| learned appearance for linking refuted | **independently reproduced** — does not transfer across the two embryo families |
