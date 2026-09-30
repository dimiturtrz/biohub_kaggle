# External data — what was available, and what each source was actually worth

Our post-mortem lists external data under "never evaluated". The write-ups and one forum thread price most
of it, and the answer is not a single number: **the value depends on which component you pretrain**, and the
largest free resource was posted to this competition's own forum two months before the deadline.

## 1. The ledger

| source | pretrain target | measured |
|---|---|---|
| **ZebraHub** (4 datasets) | the **linker** | **+0.014 public / +0.016 private** (5th place) |
| Zebrahub | the **detector** | **no gain** (hjyact, #744485) |
| **CC0 synthetic, 165k divisions** (#732103) | division supervision | **negative** for one team, **"0.960"** claimed by the author |
| self-generated synthetic, ~2,500 sequences | detector pretrain / finetune | **did not beat baseline** (hjyact) |
| **Trackastra**, **DaXi** | drop-in models | **neutral or negative** (303rd) |
| FOCUS-3D | second detector | helped coverage, **~112 s per frame** — unusable in a 9 h kernel (hjyact) |
| CZII competition (hengck23's 2.5D design) | **architecture**, not data | the decoder depth-pooling in a **private 0.939** detector (hjyact) |

Two structural readings:

- **The linker is the data-starved component, not the detector.** The detector sees every nucleus in every
  frame of 199 movies; the linker sees only the sparse annotated edges (~304 divisions, ~2.8% of nuclei
  annotated). Pretraining pays where the supervision is thin. Our own external-data entry should be split
  accordingly.
- **The highest-yield transfer was an architecture, not a dataset.** A private-0.939 detector borrowed its
  decoder from a solution to a different competition, for free.

## 2. The CC0 synthetic dataset — 165,267 labelled divisions, on our own forum, 2026-08-01

Source: topic **#732103** (José Freitas). **65 votes, 25 comments.** 18.5 GB, CC0, generator source open.

- **1,539** static volumes at native 256×256, centroids at sub-voxel precision.
- **2,174** time sequences with complete lineage graphs — nodes, edges, mitosis events.
- **165,267 labelled divisions across 4,056,226 nodes** ≈ **540×** the competition's mitosis supervision
  (~304 events across 199 videos).

> "I got tired of trying to train a division model on ~304 events, so I built a synthetic dataset and I'm
> releasing it free."

**It is a physical model, not a GAN**: dark medium, physically-sized ellipsoidal nuclei with a super-gaussian
profile ("the real radial profile is a **flat top**, not a gaussian — I measured it"), light emission that
scatters and accumulates where cells are dense, anisotropic PSF, then Poisson + read noise.

**Three calibration details, and the first one is a trap we should check in our own generator:**

1. **"The pooling matches the evaluator exactly. The official pipeline downsamples XY by 4 with a stride
   (`vol[:, ::4, ::4]`), not a block mean. A block mean averages noise away and would hand you data that is
   cleaner than what your detector really sees."** He generates at native resolution and applies the
   identical stride.
2. **Detectability is calibrated, not assumed.** A classical DoG recovers ~**0.89** of synthetic nuclei in a
   moderately dense field and ~**0.76** crowded, against **0.91–0.94** on the real annotated nuclei — "so if
   anything the synthetic volumes are slightly **harder**, not easier." Measured live at three densities in
   the notebook.
3. **Tissue geometry is fitted from the real detections** — surface, curvature, thickness — then resampled.
   "The geometry is learned from real data; the coordinates are generated. No real positions are copied."

**Motion calibrated on the real lineage edges**: **1.86 µm/frame median step**, **+0.30 lag-1 directional
persistence**, **7.24 µm sister separation**.

**Stated caveats**: the division rate is deliberately inflated to **4.07% of nodes vs ~0.26% real** ("a model
starved of examples never learns mitosis; re-weight your loss by the real rate if you need calibrated
priors"), and "**nucleus texture and contrast are the weakest axes.** The value here is the labels — density,
lineage and mitosis — much more than photorealism."

**What it was worth, honestly:** Juan Neira integrated it as extra division supervision and reported it got
worse (hengck23 diagnosed domain shift and pointed at a worked example). hjyact's own synthetic sequences
also failed to beat baseline. The author, asked on **2026-09-27** whether it works, replied *"it works! I'm
at 0.960 right now!"* and declined to say which component he trained with it until after the close. **That
number never appeared on a leaderboard row we can check, so treat it as unverified** — the same discipline as
the "0.965+" author-written string in the public-ceiling note.

**Two things to take from it regardless of the negative results:**

- **Audit our own generator against detail (1).** Our synth-null work concluded "flat `P_true` audits the
  GENERATOR first" and traced one null to `confusor_rate = 0.0`. If our generator block-means the XY
  downsample where the evaluator strides, our synthetic volumes were **cleaner than anything our detector
  sees at test time** — a domain shift in exactly the axis (noise at the crowded threshold) our confusor work
  depended on. One line to check, and the same class of bug as the truncating submission writer: a
  plausible-looking operation whose *statistics* differ from the one the pipeline actually uses.
- **The measured constants are free**, whatever the images are worth: flat-top (super-gaussian) radial
  profile, 1.86 µm/frame median step, +0.30 lag-1 persistence, 7.24 µm sister separation, DoG recall
  0.91–0.94 on real annotated nuclei. Ours were derived independently; these are a second opinion to check
  them against.

Also linked in that thread, by hengck23, and never evaluated by anyone who wrote up:

- **Biohub's own synthetic data** — `virtual-embryo-zoo.sf.czbiohub.org`
- **Keller fish-embryo trajectories** — `ssbd.riken.jp/database/project/5-Keller-FishEmbryo`

## 3. GT annotation quality — the reason some of this is hard to price

Topic **#742942** (Quantizr, 7 votes, 6 comments) — multiple teams independently report the sparse GT is
partly wrong: centroids further than 7 µm from the correctly-linked cell, wrong links, and annotated
"divisions" where the daughter was visibly present in earlier frames. Himanish Goel: "sometimes a split is
marked at the wrong coordinates and sometimes the daughter cell is dropped 1 frame later."

The measurement worth keeping is sghwr's:

> "We checked all FP originated from our model and did a blind test using our human eyes; the result found
> that **~70% of those FP can actually be TP**, since some of the [cells are unannotated]."

That is `E53` (unannotated FPs are metric-invisible) confirmed from the opposite direction — by eye, on a
different pipeline's false positives. And it is the mechanism behind 89th's PRN and 18th's `a`·`q`: if ~70%
of your "false" positives are real cells the annotation missed, then **precision measured against this GT is
not precision**, and the only thing the metric actually rewards is matching the *annotated* subset — which
is what makes deliberate mild under-detection pay.

Quantizr's closing guess is worth recording because the private board bore it out:

> "I just hope the test set isn't annotated this badly… but that might explain why some models which I think
> should do worse on manually annotated data seem to do better on the actual test set."
