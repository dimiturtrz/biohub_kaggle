# A from-scratch detector at private 0.939 — and it reproduced most of our findings independently

Source: topic **#744485** (hjyact). Public 0.929 / **private 0.939** ≈ 22nd–25th — **not their official
rank**: they selected two public-better versions instead, and both landed 0.917.

**This is the most useful single write-up for auditing our own conclusions**, because it was built from
scratch, validated on held-out embryos, and independently arrived at five of our findings — while
contradicting one of them. Their 0.939 is also **+0.015 above the ceiling of the entire public fork lineage**
(~0.924), which is the existence proof that the gap was a model, not a knob.

## 1. The detector — four channels, and the fourth one is the finding

- **2.5D UNet with a ConvNeXt-nano encoder** (timm). The **depth pooling in the decoder is borrowed from
  hengck23's CZII 2.5D design** — i.e. the architecture came off the shelf from a different competition.
- Outputs: a Gaussian centre heatmap **and a sub-voxel offset**.
- **Input is 4 channels: frames t−1, t, t+1, plus a local contrast channel.**
- Gaussian target **σ = 2.5 µm**; peaks at least **5 µm** apart.
- Main model "foldA": 100 epochs on all 128 `6bba` videos plus half the `44b6` videos; the rest of `44b6`
  never trained on and used to judge everything.
- A second detector with a **narrower target (σ 2.2 µm, `6bba` only)**. On its own, narrow σ was worse on
  both embryos — "it only worked as the second model." They tried ~ten variants; narrow-σ complemented best.

**Why the contrast channel exists** — the diagnosis is the valuable part:

> "Missed cells weren't dark. **They were in crowded areas with brighter background.** That's why I added
> [the contrast channel]; the missed cells' local maxima went from **4.2% to 14.2%**."

Our own dense-regime diagnosis said the bottleneck was a model gap in crowded regions and that the
confusor is a near-static lookalike at voxel resolution. hjyact's answer is that the crowded-region miss is
**not a resolution limit but a normalisation one** — the cell is visible, but its local contrast against a
bright neighbourhood is not in the input. **A one-channel input change, which we never tried.**

## 2. Averaging probability maps, not coordinates

> "Merging the two models' coordinates after peak picking **always** made things worse. **Averaging their
> probability maps before peak picking worked, and gave about +0.006** on the LB the first time."

Third independent statement of this rule (14th place fuses `unettf` folds by elementwise logit mean before
peak-finding for the same reason; 303rd's coordinate blend scored below both parents). Fuse *before* the
decision that discretises, never after.

Runtime shape: the second model runs **fp16 on the second T4** so both fit the 9 hours, with a time check
that drops it if the run is slow. "One of my earlier versions with more models in fp32 timed out."

## 3. Five of our findings, reproduced from scratch

| their sentence | ours |
|---|---|
| "**Almost all false-positive edges connect two real cells. Only one end is matched to the wrong GT cell. You can't fix that by looking at one edge at a time.**" | `E54c` — every thief is an unannotated real track; the error is a SPLICE |
| "Over half of the 'missed' cells weren't really missed. There was a node about **3.35 µm** off, just in the wrong spot." | `E36` (node recall at 7 µm hides edge-fatal misses; 0.824 at 2.87 µm) and `E57` (localization, not selection; 4.08 µm vs 1.46 µm) |
| "The labels are sparse, so cells look farther apart than they are. **The real nearest-neighbour distance is about 5.8 µm.** My first duplicate-removal radius of 7 µm was deleting real neighbours." | our GT-NN floor of 2.87 µm and the finer-inject kill; same trap, measured on a different quantity |
| "A classifier for extra nodes: **AUC 0.9999 on synthetic, 0.59 on real data.**" | `E48`/`E49` — no available score can prune; GBM FP-recall 0.0846 vs the 0.75 needed |
| "**The real limit was the division model missing parents.** Loosening gates didn't help." | `E61` — division recall unreachable from post-processing; the parent side is the bound |
| "A classifier for bad edges: AUC ~0.57." | `E56`/`E58` — the champion already holds the edge; no selector separates |

Six independent reproductions, from a different architecture, a different harness and a different validation
protocol. **Our diagnoses were right.** What we lacked was not insight into the failure modes.

## 4. The one that contradicts us — and how it resolves

> "**The score is mostly about the nodes.** With perfect detections, even a simple distance-based linker
> gets **98% of the edges** right. **None of my fancier linkers beat the simple one**" — about nine of them,
> flow-based and otherwise.

Against our own `celltrack-linker-is-the-gap`: a global ILP linker broke the 0.902 wall and took us to
0.924, the largest single jump we ever measured.

**Resolution by detection quality, not by discarding either.** Both are statements about *different
detectors*. Given near-perfect nodes there is nothing for a linker to arbitrate — the nearest neighbour is
the answer. Given a detector with systematic crowded-region misses and over-detection, the linker is doing
error correction, and a global objective beats a greedy one by a lot. **Linker sophistication is worth
exactly as much as the detector is wrong**, which means our +0.022 from the ILP was real *and* was a measure
of our detector's deficit. 14th place holds both facts at once: they built a 6-layer GraphSAGE linker *and*
a per-video circuit breaker keyed on `hole_fraction`, their detector's deficit.

## 5. The post-processing ceiling — the reconciliation, stated by both sides

> "**36 post-processing ideas on a fixed graph: zero wins. The ceiling I measured was about +0.003.**"

That is our `E37`/`E58` verdict, independently and with a bigger sample. And it sits beside 12th place's
**+0.013 private** from post-champion CPU stages — which is not a contradiction, because 12th's stages
(consolidation, a mitosis specialist, recentring) **change the node set**. The boundary is exact:

- **Fixed graph, edge surgery only: ceiling ≈ +0.003.** Our verdict, confirmed.
- **Stages that add, delete or move nodes: +0.013 measured.** Our verdict was wrong to generalise here, and
  `E58` "post-proc CLOSED" stays retracted.

## 6. Validation — the part they say is the most useful

> "The two embryos behave very differently. **One filter I had gave +0.004 on 44b6 and −0.036 on 6bba.**
> After that I judged everything on both embryos, using videos the model never trained on, and with the
> official scorer on the full pipeline output. **Quick proxy metrics fooled me more than once. I retracted
> two 'improvements' in one day** because of that."

- Noise floor on their 71-video set: **~0.004**. "I tried not to pick the best of many runs."
- **"Filtering dark detections turned out to be overfitting to one embryo. Removing the filter gave +0.028
  on the LB, my biggest single jump."** The largest single move in the whole write-up was **deleting** a
  component that a single-embryo measurement had licensed.
- Per-embryo routing by video name: **+0.008 locally, nothing on the LB**.
- "The division threshold had been tuned for an older detector, and **it quietly cut divJ in half** when I
  swapped detectors." Same class as our own re-fit-the-threshold-before-reading-any-grid-changing-arm rule.
- **Integer coordinates**: "Floats looked +0.002 better locally but were **−0.007 on the LB**."

## 7. The public/private table

| version | public | private |
|---|---|---|
| **foldA + σ2.2 (best)** | 0.929 | **0.939** |
| foldA + a1long + all-199 model | 0.932 | 0.938 |
| all-199 primary + a1long | 0.932 | 0.938 |
| foldA + anisoxy199 | 0.931 | 0.938 |
| foldA alone | 0.926 | 0.936 |
| foldA + zaniso | 0.938 | 0.928 |
| same, threshold 0.35 | 0.938 | **0.914** |
| routing by embryo | 0.938 | 0.928 |

> "**My highest public scores were my lowest private scores. Everything I had checked on held-out embryos
> moved up together.** Next time I'm keeping at least one honestly validated model in my final two."

Note that **even their worst arm (0.914) is within our own 0.916–0.918 private band**, and every arm they
validated on held-out embryos beat our best by 0.018–0.021. The separation is in the detector, and it shows
up as a *floor*, not as a lucky draw.

## 8. What didn't work, from a from-scratch build

**Detection**: a DETR-style transformer tracker (queries that detect and track jointly) — kept producing
duplicates, and tracking ate into detection. Centre-voting head −0.008 to −0.016. DSNT coordinate head: far
too many nodes. Full z resolution in the decoder: +0.0007, not worth it. Longer temporal context (±3 frames):
slightly better J but too many nodes. **Copy-paste augmentation with max blending: made ghost labels and
hurt.**

**Data**: ~2,500 self-generated fully-labelled synthetic sequences ("I had to make them denser to look like
the real data") — pretraining or fine-tuning on them **didn't beat the baseline**. **Zebrahub pretraining of
the detector: no gain** (cf. 5th place, where pretraining the *linker* on ZebraHub is +0.016 private).
Pseudo-labels: "the gains and the drift came from the same unlabelled areas, so I couldn't separate them."

**Recovering missed cells**: heatmap deconvolution net −3 peaks. CLEAN-style residual peeling found 33% of
missed cells but added 35% more nodes (**−0.036**). Splitting blobs by mass: AUC 0.31. **Learned coordinate
refiners: worked on one embryo, didn't transfer** — see the per-embryo-offset resolution in
[the coordinate-head axis](2026-09-30_the_coordinate_head_axis.md) §6. FOCUS-3D as a second detector: helped
coverage at ~112 s per frame.

**Divisions**: rules like "connect track starts to a nearby single-child parent" added hundreds of edges per
video and lost **0.034**. "Geometric filters on the daughters didn't work because real divisions look
asymmetric at the node level."

**Ensembling**: 3–5 detectors, no gain, and one version timed out. **"Training the second model longer made
it a worse partner."** — the same shape as our own finding that a pool from one recipe saturates, and a
direct warning against the instinct to improve each member.

**Node-count caps**: per node **−0.069**; per track, slightly positive.

## 9. What this changes for us

| our verdict | status |
|---|---|
| `E54c`, `E36`/`E57`, `E48`/`E49`, `E56`/`E58`, `E61`, GT-NN floor | **six independent reproductions** — our failure-mode diagnoses were correct |
| `E37`/`E58` post-processing ceiling | **confirmed at +0.003 for a fixed graph**; still retracted for node-set-changing stages |
| linker is the gap | **scoped**: true for *our* detector, false given good detections. Linker value = detector deficit |
| dense-regime crowding is a model gap | **confirmed, and the cheap fix named**: a local-contrast input channel, 4.2% → 14.2% local maxima on missed cells |
| our own synthetic generator work | a second team's 2,500 labelled sequences also failed to beat baseline — the negative is now replicated, not ours alone |
| ensemble diversity has a quality floor | **"training the second model longer made it a worse partner"** — a sharper statement of the same effect |
