# The center-offset head learns, and the annotation cannot teach it the thing it is for

**E63.** Campaign A (the center-offset head) was the only lever memory still listed as open. It is now built,
trained, and measured. The head works. The supervision does not contain the configuration the head exists to
resolve, and that is a data fact, measured, not an opinion about the method.

## What was built

A second 1x1 conv beside `detect_head`, reading the same backbone features through one pass
(`TemporalUNetDetector.forward_heads` -> `DetectorHeads`), regressing each voxel's micron vector to the
annotated centre that owns it. Target is the Voronoi assignment of voxels to nearest annotated centre,
bounded to one cell radius (`CELL_SCALE_UM / voxel_um`, so `(1,4,4)` -> a `(2,2,2)` box), built with a
`scatter_reduce` amin rather than a per-centre loop. Masked L1, because the target is piecewise-constant
across a seam and a squared penalty would let seam voxels dominate the interiors a readout clusters on.
Default `offset_weight=0.0` builds no head at all, so no prior arm's objective moved.

## Stage 1: the term fires and falls

Null stated before the GPU ran: a constant-zero predictor scores mean |d| over the supervised box,
(2+1+0+1+2)/5 = 1.2 voxels = 1.95um per axis, x3 = **5.85um**. Random init measured 5.99.

| arm | steps | offset term | proxy |
|---|---|---|---|
| offsetA1 | 600 | 5.912 -> **5.274** | 0.8443 -> 0.8581 |
| offsetA2 | 6000 | -> **2.90**, still descending | 0.8443 -> 0.8205 (best 0.8434) |

Monotone, halves the null, no plateau at 6000 steps. Backbone grad norm 1.26 at launch, so the term reshapes
*shared* features rather than decorating its own conv. The head is real.

### The recipe-collapse attribution was wrong, and the control says so

The proxy decline was first attributed to the known warm-finetune recipe collapse (four prior arms collapsed
identically with no offset head at all). The matched `offset_weight=0.0` control — same warm start, same
6000 steps, same eval cadence, only the offset term removed — refutes that:

| step | ctrlNoOffset (w=0.0) | offsetA2 (w=0.01) |
|---|---|---|
| init | 0.8443 | 0.8443 |
| 1500 | **0.8546** | 0.8194 |
| 3000 | **0.8614** | 0.7951 |
| 4500 | **0.8704** | 0.8182 |
| 6000 | 0.8427 | 0.8205 |

The control *improves* monotonically to +0.026 over init before its last-window dip; the offset arm never
returns to init. The matched gap at 4500 is **−0.052**, far above the 0.01–0.02 noise floor. So the offset
term at weight 0.01 costs the proxy on its own, and the recipe is not the culprit here.

Two consequences. First, the auxiliary term competes with detection rather than regularising it — consistent
with a head that reshapes shared features (grad norm 1.26) toward a Voronoi field that ~98% of frames only
ever exercise on isolated cells. Second, **this same warm-finetune recipe works** (+0.026 from 0.8443),
which the "warm-finetune recipe degrades proxy" memory would not have predicted; that memory is about the
arms that carried a payload, not about the recipe alone. The recipe is exonerated; the payload is the cost.

## Stage 2 is refuted at the data, before it was built

A cluster-by-offset readout only earns its keep where two cells' responses have merged into one blob. For
the head to learn what to do there, two annotated centres must sit close enough that their supervision
windows meet — a Voronoi **seam**. Measured over 25 train videos, 2307 annotated frames, 6609 centres:

- centres per frame: mean **2.86**, median 2, max 11 (68.2% of frames have >=2)
- nearest-neighbour distance between annotated centres: p5 **9.33um**, median **24.54um**, p75 32.90um
- pairs within one supervision box-width (6.5um): **1.3%**
- multi-centre frames containing any seam at all: **38 of 1573 = 2.4%**

And the error is not flat across the box. Stratified by distance ring on the 600-step checkpoint, the head is
**0.364um at the centre voxel**, then 3.045 / 5.153 / 7.961um outward — sub-voxel where it is easy, and worst
in the outer shell. A seam between two touching cells lives at roughly one cell radius from each centre, i.e.
in that outer shell. So the head is least accurate exactly where a readout would have to trust it.

So ~98% of the head's supervision is isolated single cells. It learned exactly that, which is why the loss
falls cleanly. The merged-blob case it was built for is not in the training signal at any useful rate, and
no amount of further training reaches it.

**Denominator note.** The first read of this was wrong and caught by the repo's own rule. Two videos gave
"median 1 centre per frame"; 25 videos give 2.86. The conclusion survived the correction, the number did not.

## What this kills, and what it does not

Killed: training the seam behaviour from **annotated centroids alone**. That is the specific use, and it is
dead on measurement, not on a score.

Not killed, and explicitly re-testable: the head itself, its plumbing, and the offset field as a cue. The
remedy has the shape the `pmkf` finding already named — GT-sparse supervision cannot teach a detected-crowd
behaviour. The offset field would have to be supervised against **detected** cells (~980/frame) rather than
annotated ones, i.e. pseudo-labels, which is a new supervision source rather than a knob, and offline
unpriceable per E59. Filed rather than started.

## Instrument kept

`python -m celltrack.eval.offset_precision` reports the offset error stratified by distance ring and by the
detection head's own confidence, so the head is priced on the voxels a readout would use rather than on a
whole-box mean. It cross-checks against the training heartbeat (5.384um measured vs 5.274um last logged on
the same checkpoint). `frame_targets` was lifted out of the trainer's `main` so a diagnostic cannot index its
frames differently from the run it diagnoses.
