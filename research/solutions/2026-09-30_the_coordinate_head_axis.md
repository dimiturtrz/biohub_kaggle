# The coordinate axis — five teams, one component, and three real contradictions

**Every team that wrote up a mid-to-high finish names sub-voxel coordinate correction as its largest
non-detector lever, and they disagree about how to use it.** The disagreements are not noise: each pair
resolves on a mechanism, and the resolutions are the useful part.

Sources: **#744562** (Hammad Farooq / KR_no_1, 213th, private 0.923) · **#744548** (Korokke3, private
0.924) · **#744507** (Eesh saxena, 303rd, bronze) · **#744490** (Takahiro Katsumata, 89th, silver) ·
**#744486** (Tom, 14th, gold) · **#744093** (the pre-deadline float-centroid warning).

## 0. Why this axis pays on *this* metric

303rd states the scorer explicitly, which is what makes the mechanism legible:

```
score  = adjEJ + 0.1 * divJ
adjEJ  = Σ_i w_i · EJ_i · (1 − 0.1·ρ_i) / Σ_i w_i ,   w_i = TP_i + FP_i + FN_i
ρ_i    = (N_pred_i − N_est_i) / N_est_i               (node-count penalty, UNCLIPPED)
```

Edges match one-to-one per frame at a **7 µm** gate; only `dt == 1` edges count; an edge between two nodes
that were both left unmatched is **free**.

> "Adjusted edge Jaccard rewards edges whose two endpoints both fall within the 7 µm match gate, so a
> systematic sub-voxel correction pulls the borderline edges back inside the gate and converts near-misses
> into true positives." — 303rd

And the corollary we never stated so bluntly: **ρ is unclipped and the bonus for fewer nodes has no cap, so
"the metric actually rewards mild under-detection."** That is the same fact our `E47`/`E52`/`E53` prune work
circled — 89th monetised it directly (§4 below).

## 1. The measured prices

| team | change | public | private |
|---|---|---|---|
| Korokke3 | x138 as published (V1284 head, 20 movies) | 0.953 | 0.917 |
| Korokke3 | **head retrained on all 199 movies** (128,931 pairs, 3 seeds, 5 epochs) | 0.948 | **0.924** |
| Korokke3 | 0.5 original + 0.5 retrained | 0.954 | 0.923 |
| Korokke3 | 0.85 original + 0.15 retrained | 0.956 | 0.919 |
| 213th | own **B251** head replaces V1284 | 0.954 | 0.922 |
| 213th | 0.7·B251 + 0.3·V1284 | 0.956 | 0.921 |
| 213th | **same blend, shift × 1.15** (final) | 0.957 | **0.923** |
| 89th | chassis byte-identical (no head) | 0.946 | 0.913 |
| 89th | **+ own head, 48 movies** | 0.953 | **0.920** |
| 303rd | + sub-voxel soft-argmax, all axes, gain 2 | 0.951 | — |
| 303rd | + V1284 head for y/x, own refined z fed to the edge model | 0.953 | — |
| 14th | + one **constant** zero-parameter shift, applied last | +0.002 | — |

The axis is worth **+0.007 private** on the public chassis, three times independently (Korokke3 +0.007 over
x138; 89th +0.007 on both boards; 213th +0.005 over the no-head lineage). That is ~half the distance from
our 0.918 to the 0.924 ceiling of our own line.

## 2. B251 — the full architecture, so it is reproducible

- **Inputs, 251 dims**: 32 UNet feature channels at the peak, plus 6 directional differences to its
  ±z/±y/±x neighbours (**224**), plus a **3×3×3 patch of blended detector logits expressed relative to the
  centre logit** (**27**).
- **Model**: `Linear(251→32) → SiLU → Linear(32→3)`, bounded output `d = 2h / (1 + |h|)`, so `|d| ≤ 2 µm`.
- **Training**: detection-to-GT offsets from 45 movies.
- The public **V1284** head is the same shape **without the logit patch** (224 dims), trained on 20 movies.
- 89th's is V1284's shape trained on 48 movies with the x138 recipe: centre error fell **20.7%** in
  video-grouped CV, and **3.5% / 18.5%** leave-one-embryo-out — note the factor-of-five asymmetry between
  embryos.

Korokke3's capture recipe is the cheap part: run the x138 pipeline over all 199 training movies in a mode
that matches detections to the sparse GT → **128,931 pairs**, then retrain the same MLP (3 seeds, 5 epochs).
Weights and scripts are CC0: `korokke3/biohub-v1284-head-199-movies`.

## 3. Contradiction one — before linking, or after every graph decision?

| team | finding |
|---|---|
| 213th | head applied **before** linking: **0.951**; same head applied only to output centres, graph unchanged: **0.946** |
| 303rd | feeding the refined **z** back into the edge model, not just the submission, was worth **+0.002** |
| 89th | "through the edge-scorer features, it also changes what the ILP keeps" |
| 14th | shift is **+0.01116 at the scorer** and **−0.00573 to the graph** (73 fewer true edges) → **apply last** |

**Resolution: constant versus per-detection.** 14th's shift is a *single global vector* — it translates every
detection identically, so it cannot improve any pairwise geometry the linker gates on, and it can only
perturb those gates. A *learned per-detection* shift changes **relative** geometry, which is exactly what the
matcher and the edge scorer consume; hence 213th's +0.005 and 303rd's +0.002 for putting it upstream.

The same split explains why 14th says "every learned correction head we tried bought its gain on the edge
term by destroying a division" while four other teams shipped learned heads: 14th's divisions are all
created downstream by geometric rules with tight µm gates, so moving detections upstream moves those gates
out from under the division layer. On the public chassis the division layer is also downstream but far less
finely tuned.

**Bearing on our own work.** `E59` concluded a snap oracle "cannot price localization". It measured the
**scorer half only** — the half 14th prices at +0.01116. The graph half (−0.00573 for a constant shift,
**positive** for a learned one) was never in our oracle. The memory note "the real fix is a global affine
FIELD" was directionally right and one step short: the fix is a **per-detection** field entering the
**matcher**, and its value is mostly *not* in where the written centre lands.

## 4. Contradiction two — does the retrained/public head actually help?

303rd **falsifies** the very component Korokke3 measures at +0.007:

> "The public retrained localization head plus z-refiner (coord-inject). **Falsified**: it makes the > 3 µm
> displacement tail worse on both embryos and every axis, because it optimizes the board-flat 'node within
> 2 µm' quantity rather than the tail the metric actually pays for."

Both statements can hold. 303rd measured the **tail** on a replay harness running on **in-sample** movies
(their own bias one: "the harness runs on trained movies, so it literally cannot see the in-plane
localization effect that the hidden embryos would feel"). Korokke3 measured the **private board**, on unseen
embryos, and won +0.007. On an in-sample detector the head has little left to correct and its tail damage is
all that remains visible — which is the same trap as our `E51` "the proxy is IN-SAMPLE, offline numbers are
CEILINGS", arriving at a *negative* instead of an inflated positive.

**The tie-breaker is the split, and it points at Korokke3**: the quantity that pays is measured on embryos
the head never saw.

## 5. Contradiction three — scale the shift up, or down?

| team | optimal scalar | measured on |
|---|---|---|
| 213th | **×1.15** (the only change that also improved private, 0.923) | public + private |
| 303rd | **~0.6** — "the head under-corrects on one embryo and over-corrects on the other" | public board |

213th's mechanism is the sharper one, and it is a general lesson about ensembling *displacements*:

> "When two heads disagree in direction, their average is shorter than either shift. The blend therefore
> systematically under-corrects. Multiplying the blended shift by 1.15 restores the step length."
>
> "**Averaging vectors shrinks them.** If you ensemble regressors of a displacement, check the norm of the
> average and rescale it."

So ×1.15 corrects an artefact **of blending two heads**; 303rd's 0.6 scales a **single** head whose
per-embryo error changes sign. Not the same quantity. 303rd also found blending their two coordinate
estimates scored *below both parents* (0.953) because the estimates are correlated **0.8 in y, 0.7 in x** —
"averaging them buys almost nothing and just splits the difference." Consistent with 213th, whose two heads
used *different inputs and different training movies* precisely to decorrelate.

213th's second-order observation ties the axis back to the metric:

> "On film 44b6_0b24845f the node count after linking falls monotonically as the shift grows: **+4.6% at
> ×0.85, +1.8% at ×1.0, −0.2% at ×1.15**. The adjusted edge Jaccard penalises over-predicting nodes, which
> is consistent with the longer step helping."

The shift is not only a localisation knob — it is a **node-count** knob, acting through the uncapped ρ term.

## 6. The head learns an absolute per-embryo offset, not jitter

Korokke3's physics note, which decides how to train it:

> "The gain comes largely from absolute position accuracy, including per-embryo offsets (the mean
> GT-minus-detection z-offset is **+0.17 µm for 44b6 and +0.84 µm for 6bba**), not only from frame-to-frame
> jitter. A head trained on **de-meaned targets scored 0.001–0.003 worse** on both public and private."

14th measured the same thing as a constant: **(+0.419, +0.509, +0.425) voxels** of GT − prediction residual.
So the detector carries a systematic **positive half-voxel** offset on every axis relative to the annotation
convention.

This also explains hjyact's isolated negative — *"learned coordinate refiners: worked on one embryo, didn't
transfer to the other"*. If the head's payload is the per-embryo mean, a head trained on one embryo
**anti-transfers** to the other by construction. The cure is Korokke3's: train across both.

## 7. The float-centroid trap — measured three times, posted before the deadline

Two teams changed exactly one line, `round(v, 3)` instead of `int(round(v))`, verified the graph was
byte-identical, and lost score:

| team | int | float | delta |
|---|---|---|---|
| 213th / #744093 | 0.954 public | 0.946 public | **−0.008 public, −0.007 private** |
| hjyact | — | "floats looked +0.002 better locally" | **−0.007 LB** |

> "Our best guess is that the Kaggle-side scorer truncates non-integer input rather than rounding.
> Truncation would move every centre by about **−0.5 voxel on average, roughly 0.8 µm in z**. We have not
> confirmed the mechanism. Either way: keep the int write." — #744562

The reference metric (`royerlab/kaggle-cell-tracking-competition`, `csv_to_geffs.build_graph_from_rows`)
casts z/y/x to `Float64`, which is why writing floats looks free. **#744093 was posted 2026-09-28, a day
before the deadline, and we did not read it.**

Our own writer declared `pl.Int64` — so the +0.008 was already ours — but reached it with a **silent
truncating cast**, the same −0.5-voxel-per-axis bias, fixed in `c9e791a` with `np.rint`. See
`interpretations/celltrack/converging/2026-09-30_private_lb_post_mortem.md` §4a. Note the arithmetic
coincidence worth not over-reading: the detector's own residual is **+0.42…+0.51 voxels** (§6) and
truncation costs **−0.5** — a truncating writer on an uncorrected detector partially cancels, which is one
reason the bias may never have shown up as a visible score cliff in our own readings. Unverified; the scout
tracing whether sub-voxel floats ever reached our writer did not complete.

## 8. The z axis is where the remaining headroom is

303rd's ablation is the cleanest statement of it:

- Sub-voxel soft-argmax on **all** axes: 0.947 → **0.951**.
- Zeroing the in-plane correction and keeping **only sub-voxel z retained most of that gain** — "z was
  carrying it."
- And z is simultaneously the worst-estimated axis: "a 3-point logit soft-argmax overshoots on z (mean
  absolute z error is worse than the in-plane axes), and I tried a pile of alternatives (5 and 7 slice
  soft-argmax, Gaussian and parabolic fits, lower gains, robust variants) without finding one I trusted
  more. **A clean z estimator looks like the most obvious remaining gain.**"

Mechanically unsurprising: z voxels are **1.625 µm** against 0.40625 µm in plane — a 4× coarser grid against
a 7 µm gate. Their production setting is a board optimum (~15 sweep arms all lost): soft-argmax over
detector logits in a small window, **gain 2** on the logits, shift clipped inside the voxel (~0.5), computed
separately for in-plane and z.

## 9. Two data facts from 303rd that reframe every offline number

> "The four visible test movies are **placeholders copied from train**, so scoring the public split locally
> predicts nothing. And train and test are **embryo-disjoint**: the hidden movies come from embryos no one
> has seen. So leave-one-embryo-out is the only sensible proxy."

Both are in the data description. Their replay harness (real detector outputs, real ILP, the organisers'
scorer ported call for call, ~1 min per movie per config, paired bootstraps across the two embryos) produced
a **calibration** rather than a ranking:

- board delta ≈ **replay delta − 0.8 ticks**, leave-one-out RMSE ≈ 0.6 → "a change had to clear roughly
  +0.8 on replay before I'd expect it to even hold serve on the board."
- **Bias one**: in-sample detector → cannot see the localisation effect the hidden embryos feel.
- **Bias two**: over-trusts anything leaning harder on the learned edges (rawlink, learned bonus, image
  flow), because the edge model was trained on all 199 movies. Cross-body check: their sub-voxel body read
  **−10 ticks** on replay against the learned-head body and was only **−1** on the board.

> "An offline harness on an in-sample detector will call board-transferable localization changes wrong and
> over-rate anything that trusts the ILP more."

Their best offline lever all week — image-flow / phase-correlation link repair in high-motion windows, +2.9
on replay and positive on LOEO — "tied or lost on the board every single time."

## 10. Everything on this axis that lost

- **8-view or full-D4 detection TTA: −0.008** (213th). Detection TTA saturates on the public stack and then
  costs.
- Conditional two-model detection fusion near the threshold: **−0.013 to −0.016** (213th).
- **Disabling post-hoc divisions entirely: −0.035** (213th) — "divisions matter", the cheapest possible
  confirmation that the 10% term carries far more than 10% of the available headroom.
- Mitosis-CNN orphan forks: ±0.000 to −0.003 (213th). Motion-relink tight gates 6.0/6.5/7.0 µm: ±0.000.
- Coordinate blending of two correlated estimates: 0.953, below both parents (303rd).
- Learned-head gain above 1 on a single head (303rd); sub-voxel sweeps, ~15 arms, all lost.
- **Trackastra and DaXi as external models: neutral or negative** (303rd). Note this prices them as
  *models*, not as *features into a tabular rescorer* — which remains the never-tried item from the 18th
  place write-up.
- ILP division weight 0.4: a public post reported +0.001; 303rd's full-path replay showed the extra forks
  land on noise, net negative, not submitted. (Korokke3 submitted it: 0.958 public / **0.918** private, the
  worst private of their five.)

## 11. Hedging the final two, when one axis dominates

303rd's rule is the sharpest statement of the lesson our own final-2 note reached from the other side:

> "Kaggle keeps the better of your two selected submissions. On an embryo-disjoint hidden set the failure
> that scares me is one coordinate scheme mislocalizing on an unseen embryo, so I picked one submission from
> **each body**: the learned-head 0.955 and the sub-voxel 0.954. Those two differ from each other far more
> than any same-body pair (a same-body sibling differs on **under 1% of edges** and rises and falls with its
> twin)."

Hedge by **which component differs**, not by how much the outputs disagree — and the component to hedge is
the one carrying the most score.
