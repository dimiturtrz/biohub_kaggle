# 12th place (Team Corwin) — private 0.946, gold, on tuned public models

Source: Kaggle discussion topic **#744501**, "12th Place Solution: a tuned public tracker, extended by
measured, fail-safe repair stages", posted 2026-09-30 03:24Z. Faithful synthesis; all numbers theirs.

| private | public | rank | submissions | kernel |
|---|---|---|---|---|
| **0.946** (best of their 110 private scores) | 0.970 | 12th of 4,017, gold (14th on public at close) | 112 in 52 days, 109 scored | 2 × T4, 6.8–8.4 h submit→score |

**They trained no detector.** They *tuned* `pilkwang`'s public detection-and-linking models and
replicated `prvsiyan`'s public chain (public 0.913 / private 0.897 on its own), then added four stages
and a CSV writer. This is the second existence proof (with [18th place](2026-09-30_ymg_aq_18th_place.md))
that ~0.03–0.05 of private score lived downstream of the public detector.

This is also the most forensically useful document in the field: they restored the host scorer, pinned it
by sha256, and **replay it to 1e-12**, which lets every claim below carry an exact price.

## Their own headline error

> Our largest error was validation: our full-chain bench was in-sample and over-read the private by about
> 0.03. Yet the private scores of all our submissions, read after the close, show that the stages we
> added after our base champion were worth **twice as much on the private as on the public (+0.013 vs
> +0.006)**.

## 1. The data, as they measured it

| | 44b6 | 6bba | all |
|---|---|---|---|
| training videos | 71 | 128 | 199 |
| annotated nodes / edges | 20,197 / 19,826 | 113,121 / 109,057 | 133,318 / 128,883 |
| annotated divisions | 26 | 125 | **151** |
| annotated share of host's nucleus estimate | **0.77 %** | **5.37 %** | **2.82 %** |
| videos with byte-identical copied frames | 0 | 114 | 114 (947 frames) |
| hazy videos (median contrast < 1.6) | 38 | 35 | 73 |

Videos are (T, Z, Y, X) = (100, 64, 256, 256), voxel 1.625 × 0.40625 × 0.40625 µm — a ~104 µm cube.

Their base chain predicted a median of **256** nuclei/frame on a 44b6 video and **97** on a 6bba video
against 2.82 % annotation. An edge touching one annotated nucleus is scored; an edge between two
unannotated nuclei is invisible; every predicted nucleus counts in the node-count factor; **an
unannotated division is not a false one**.

### Five dataset artefacts, none of which we found

1. **Copied frames.** 947 frames across 114 6bba videos are byte-identical to their predecessor — an
   acquisition artefact. **Those videos carry 100 of the 151 divisions.** A probe that stripped their
   copy handling read the same score on both boards, so neither hidden split holds a copy video. 14th
   place reports the same 947 frames.
2. **Whole-stack frame jumps.** The volume jumps for one frame, mostly **3.3–8.1 µm in z**, then returns:
   52 GT jump frames in 36 6bba videos, 11 in 44b6; consecutive large jumps are **54 or 55 frames apart
   in 12 of 23 cases**. There their nodes sit a median **2.63 µm** from GT against **1.86 µm** at t±1.
3. **Overlapping crops.** 80 pairs of 44b6 crops are byte-identical, in 6 voxel-sharing components; **2
   of the 26 44b6 divisions are one event counted twice.** *"Video-level folds can leak."*
4. **Annotation z convention differs per embryo.** 6bba GT sits **+0.675 plane** above each nucleus's z
   centre, 44b6 **+0.115**. A one-plane z lift of ~23 % of nodes, *rejected* on its public read (−0.001),
   scored **+0.004 on the private**. *"The convention depends on the embryo: hedge it, do not tune it on
   the public."*
5. **Integer coordinates.** GT coordinates are integer voxel indices. The same world scored **0.950 with
   integers vs 0.942 with floats** on public (private 0.911 vs 0.906) — a free +0.008/+0.005 from
   rounding at write time.

## 2. The metric, replayed exactly — and the price of every event

Score = node-adjusted edge Jaccard + 0.1 × division Jaccard (max 1.1). No term clipped on their
predictions, so the deficit splits exactly:

```
1.1 − score = FN_e/E_tot + FP_e/E_tot + node term + 0.1·FN_d/D_tot + 0.1·FP_d/D_tot
```

with `E_tot` = matched + missed + false edges (133,795 for their base champion) and `D_tot` = 181.

**In-sample, the base champion's 0.1505 deficit splits into edges 0.0759, divisions 0.0768 (51 %), and a
node-term bonus of −0.0022.** Three mechanisms make **83 % of the edge deficit**: a neighbour within 7 µm
takes the link (**0.0295**), a GT endpoint has no node within 7 µm (**0.0211**), link swaps (**0.0122**).

| event | value | ruler |
|---|---|---|
| one missed or false edge | **7.47e-6** (1 / 133,795) | exact split, in-sample |
| one missed division | **5.5e-4** (0.1 / 181) — **as much as 74 edges** | exact split, in-sample |
| division Jaccard J, public / private | ~0.47 / at most ~0.33 | every-fork-cut probe: 0.906 vs 0.954 public, 0.891 vs 0.924 private; the cut costs 0.1·J |
| one true division on the public | ~+0.002 | estimate, ~42 annotated divisions on ~58 public videos |
| **break-even precision of added divisions, J/(1+J)** | **32 % public, ~25 % private** | derived from the above |

14th place independently measured one division TP at **+0.000585**, the same order.

Two shipping rules they derived from these rates: no division the base champion got right may be lost (an
audit before every submission), and a division change ships only as a final pass on the base graph with
an expected public value ≥ +0.002 under a transport discount of 0.65. **None of 8 submissions that lost a
true base division rose on the public** (6 below, 2 equal).

## 3. Validation, and where it failed

The public networks they built on were trained on all 199 training videos, so **every full-chain read on
them was in-sample**. They knew this from 10 Sept.; their OOF world arrived 27 Sept.

| read of the final chain | A | B (final) | kind |
|---|---|---|---|
| 199 training videos | 0.975 | 0.979 | in-sample |
| 199 training videos | 0.881 | 0.888 | OOF (37–57 epochs vs 377, no final-repair rules) |
| leaderboards | 0.970 / 0.946 | 0.970 / 0.946 | public / private |

In-sample over-read the private by ~0.03; the OOF world under-read it by ~0.06; **3rd place's OOF CV
over-read theirs by 0.011.** Out of fold, position errors beyond 7 µm grow **×3.8** — *"the in-sample
bench repaired the errors of memorised videos"* — and it flipped sign: secondary TTA off read **+0.0143
in-sample, −0.012 public, −0.003 private**.

## 4. Detection and public post-processing (tuned, not retrained)

`pilkwang`'s temporal 3D U-Net (two input frames, (1,4,4) downsampling) + node transformer + ILP (edge
−1.0, appear 0.0, disappear 1.5, division 1.0). Four tuned knobs:

| knob | theirs | public preset | public (alt vs theirs) | private, same pair |
|---|---|---|---|---|
| detection threshold | 0.96 | 0.96875 | 0.965: 0.954 vs 0.958 | −0.002 |
| second-model fusion (det / links) | 0.475 / 0.15 | same | 0.70: 0.946 vs 0.957; 0.30: 0.953 vs 0.957; off: 0.948 vs 0.954 | −0.001; +0.001; −0.001 |
| second-model TTA | **16 views (D4 + z-flip)** | 8 views, yx only | 8 views: 0.958 vs 0.964 | 0.929 vs 0.933 |
| sub-voxel centre refinement | on (mean shift 0.28–0.74 µm) | off | off: 0.954 vs 0.955 | 0.926 vs 0.927 |

Only the z-flip view held on the private (+0.006 public, +0.004 private). The 0.475 fusion weight and the
lateral TTA phases were optima of the public embryo only.

The five public post-processing passes, priced by switching them off (all lost on both boards): DeepCenter
gate; motion relink (two-pass Hungarian at 5.5 then 10 µm); gap closing of 1–2 frames (5.8 then 4.4 µm per
step); safe divisions (geometric grafts, mother within 9 µm, sisters within 14 µm); short-component filter
(< 6 nodes, **after** gap closing); line-fit smoothing (0.2·p + 0.8 × 5-frame linear fit).

### Two properties of the public pipeline that bear directly on our record

> **With division cost 1.0 against appearance 0.0, the public ILP never forms a fork on the 199 videos:
> all 352 divisions of this stage on the visible videos are geometric grafts.** The 14th place proves the
> same property for their weights.

(288th place, topic #744512, found it a third time independently: *zero forks in all 36 raw ILP graphs*.)

> The filter has a cost: it erased **279 cells** whose tracks were missed in xy, and **9 of the 28 absent
> daughters of GT divisions**. Our re-stitching stages ran **after** it, too late.

## 5. Consolidation → the base champion (public 0.964, private 0.933, 19 Sept.)

| step | what it does | visible counts | evidence (public; private) |
|---|---|---|---|
| division then edges | rebuild divisions (mother within 10 µm), then drop links the relinker does not endorse, protecting forks | −6,121 / +1,899 links | 0.930 vs 0.913; 0.899 vs 0.897 |
| exact local re-solve | re-decide ambiguous links within 10 µm | +2,573 links | 0.933 vs 0.930; 0.904 vs 0.899 |
| fork certifier | certify or cut forks | 292 seen, 124 cut | these three steps off: 0.942 vs 0.957; 0.921 vs 0.929 |
| safe adds | orphan divisions, top-K | +22 links, +12 forks | |
| smooth, purge, insert | **integer coordinates from here on** | 29,881 nodes moved | integers vs floats: 0.950 vs 0.942; 0.911 vs 0.906 |
| drop unlikely links | | −46 links | |
| two-column learned fork rule | cut a branch when link probability **and** mother score agree | 62 links cut | +0.001; **+0.005** over its parent |
| re-aim | gated link repair | 873 of 4,602 proposals applied | 0.957 vs 0.955; 0.929 vs 0.927 |

Base champion: 122,824 nodes, 117,704 links, 73 divisions; in-sample 0.9495, crediting **42 of 151** GT
divisions (109 missed, 30 false).

Division count along the chain, visible videos: **352 (public chain) → 73 (base champion) → 81
(specialist) → 94 (final file)**.

## 6. Mitosis specialist (+0.0066 in-sample; public +0.001, private +0.003)

A final pass that never removes or moves an existing division. Candidates: every mother with a single
child and an **orphan** (a node without a parent) within 15 µm at t+1. Action: link the mother to the
nearest orphan unless it is taken or the mother already has two children.

Funnel on the visible videos: 19,733 candidates → 2,324 pass a cheap booster bound → **332 kept (top 2 %
by a label-free chooser score) and read at image level → 8 fire.** Divisions 73 → 81, 0 links lost, 308 s.
Every head is 5-fold grouped by video:

- **Division reader CNN** (2.8 M params) on spatio-temporal crops at two scales, initialised on
  **Zebrahub** (host-authorised), fine-tuned on 150 positives: OOF AUC 0.9125 (6bba), 0.844 (44b6), 0.829
  on external **linajea**.
- **Context chooser**: 25 LightGBM boosters on **70 label-free columns** (division waves, cycle age,
  sister, blob hypotheses in a 12 µm ball, the orphan's track history).
- **Three image witnesses on the top 2 % only**: a centre-empties ratio, a TV-L1 flow-deformation witness,
  and a frozen **DINOv2 ViT-S/14** on thin XY/XZ/YZ slabs at t−2..t+2 with a logistic head (event AUC
  0.942 OOF).
- **Stacker**: LightGBM on everything; deployed rule = mean of 5 outer stacks.

Precision at their R10 operating point: **stack 84.6 %** (13 fires) · chooser alone 32.4 % (34 fires) ·
reader alone 5.2 % (213 fires). Labels permuted through the whole pipeline: 0.3–2.8 % at R10 (control).
Deployed: 283 forks, 17 on annotated mothers, **13 true (76.5 %)**.

## 7. Final repair and the identity link (747 s, 44 % of the save run)

A GPU pre-pass computes a tissue-flow field and per-node image features; one engine then fuses **nine
rules**: admit missed divisions (with a hazy-video variant), merge duplicates (twins, z-stacked twins, xy
border faces), re-stitch tracks (vanishing tracks, bridges, video-edge chains), and decide links with a
learned decider. Late passes follow: a nucleus seen twice, z-stacked twins, segments re-decided in
context, holes in haze, fragment re-stitching.

**Identity link** — a LightGBM probability per candidate link from geometry, appearance and deformation
features, an identity-CNN embedding, NCC, **33 "constellation" columns describing the neighbour pattern**,
and the decider's link probability. It **cuts** low-probability links (min 0.3, margin 0.2); forks and
ambiguous nodes frozen; never adds a node or a fork; refuses edits near the outer 2 z slices or in a
3.5 µm xy band; at most 1,500 edits per video. Cost on a T4: 5.94 + 0.772·k s (k = thousands of nodes).

| read | value |
|---|---|
| identity link | +0.0031 OOF cf (q05 +0.0020; 71 videos up, 10 down); +0.0003 in-sample; 0 divisions moved |
| identity link, B vs A | 0.970 = 0.970 public; 0.946 = 0.946 private |
| repair block, two variants, vs the specialist world | 0.967 / 0.968 vs 0.965 public; **0.943 / 0.941 vs 0.936** private |
| late passes at full coverage | +0.0031 OOF cf; 0.969 public, 0.942 private |

Visible videos: 2,746 applied edits, 0 videos reverted, divisions 81 → 94.

## 8. Learned recentring, guards, safe writer

**Learned recentring.** A 3D CNN (3 → 24 → 32 → 64 → 96 channels) reads the native **13 × 41 × 41** crop
around each node plus a map of the frame's other nodes and an in-volume mask, and predicts the node-to-GT
offset per axis with a **Laplace scale**; a per-axis LightGBM stacker adds track, crowding and face
context. Trained on **930,452 node samples** from three worlds plus 46 k synthetic jitters, in
component-closed folds; shipped weights retrained on all 199. Moves capped at **2 µm per axis**, never
land within 4 µm of another node, never move away from both time neighbours by more than 0.5 µm, never
change topology. ~0.4–0.6 s per 1,000 nodes on a T4. Moved **98,793 of 122,686** visible nodes.

Reads: +0.0024 OOF cf, +0.0003 in-sample; against the world without recentring, 0.970 vs 0.969 public and
**0.945 vs 0.942 private**. *"Before the close we expected learned movers to keep only 0.18 to 0.30 of
their OOF value; this one transported better."*

**Haze guard** (hazy videos, contrast < 1.6): cancels a recentring move whose flow-compensated residuals
to parent and child grow by more than 1 µm — 2,665 nodes restored on 2 hazy visible videos, +0.0007 OOF
cf. Shipped as a named bet; both variants scored 0.970 / 0.946.

**Frame-jump correction.** The public 5-frame smoother keeps only **0.36** of a one-frame jump and drags
the four neighbours: a 5 µm jump leaves ~3.2 µm of offset against a GT that follows the image. They detect
jumps on the chain's own raw detections and undo the drag, touching no link or node: +0.0016 OOF cf,
undecided on both boards. *14th place compensated the same shifts with FFT phase correlation.*

**Safe CSV writer.** Integer coordinates, valid links, isolated and off-image nodes dropped, file re-read
before replacing it (+0.0004 OOF cf).

## 9. Runtime safety (the part worth copying wholesale)

- Before any compute: the exact MILP solver is checked, **a valid `submission.csv` is written at t+0 s**,
  32 offline wheels installed, sha256 of every weight and source verified.
- Deadlines and fail-open: final repair stops at 10.5 h, everything at 11.2 h; **every late stage returns
  its input intact**, video by video, on error or timeout; dead or stuck workers restarted (26/26 unit
  tests and fault injection pass; soak 4 lanes × 50 videos, 0 error).
- Proof of work: a canary proves each planned stage ran on each video; the last three stages re-read
  before they write.
- One early submission scored **0.000** because they submitted a different kernel version from the one
  they verified. No final submission failed.

## 10. The build history, both boards (their Table 1 — a history, not a controlled ablation)

| date | public | private | what changed |
|---|---|---|---|
| 8 Aug | 0.887 | 0.859 | pilkwang's public models inside their own inference notebook |
| 17 Aug | 0.913 | 0.897 | faithful replica of prvsiyan's public chain |
| 19 Aug | 0.930 / 0.933 | 0.899 / 0.904 | "division then edges", then the exact local re-solve |
| 24–29 Aug | 0.943 / 0.947 | 0.908 / 0.910 | a division channel in consolidation; two division additions |
| 5–7 Sept | 0.950–0.953 | 0.911–0.919 | **integer coordinates**, smoothing, false-fork purge |
| 11 Sept | 0.954 | 0.924 | two-column learned fork rule |
| 13–18 Sept | 0.955–0.958 | 0.927–0.929 | public knobs + sub-voxel centres; re-aim; DeepCenter live |
| 19 Sept | 0.964 | 0.933 | **16 TTA views on the second model: base champion** |
| 23 Sept | 0.965 | 0.936 | mitosis specialist |
| 27 Sept | 0.967 / 0.968 | **0.943 / 0.941** | final repair rules, two variants |
| 28 Sept | 0.969 / 0.970 | 0.942 / 0.945 | late repair passes at full coverage; learned recentring |
| 29 Sept | 0.970 (×4) | 0.945–0.946 | frame-jump fix; fragments and identity link; haze guard; hardening |

**Where the public→private drop came from.** The replica of the public chain dropped 0.016. Their
consolidation and knob tuning, built on the in-sample bench and the public board, took public +0.051 to
the base champion but private only +0.036, so the drop grew to 0.031. **The stages after the base
champion read +0.006 public and +0.013 private, and the drop shrank to 0.024.**

### Key one-variable pairs (~0.003 noise, ±0.001 rounding per read)

| change | public | private |
|---|---|---|
| integer → float coordinates | **−0.008** | −0.005 |
| 16 views (z-flip) on the second model | +0.006 | +0.004 |
| second-model TTA off | −0.012 | −0.003 |
| fusion 0.475 → 0.70 / → 0.30 | −0.011 / −0.004 | −0.001 / +0.001 |
| a lateral TTA phase shift (with dedup) | −0.013 | +0.001 |
| motion relink / gap closing / smoothing off | −0.004 / −0.004 / −0.002 | −0.005 / −0.007 / −0.004 |
| **every fork cut (instrument)** | **−0.048** | **−0.033** |
| mitosis specialist | +0.001 | +0.003 |
| division-admission rule removed | +0.001 | −0.001 |
| one-plane z lift of ~23 % of nodes | −0.001 | **+0.004** |
| learned recentring | +0.001 | +0.003 |

**Across 104 parent-child pairs (76 one-variable), 17 had opposite signs: 15 negative on the public and
positive on the private.** Continuity pieces held at **1.3–3×** their public size; detector knobs tuned on
the public were flat on the private. Their three final candidates all scored 0.946 — the maximum of their
110 private scores — so final selection cost nothing. **They moved 14th public → 12th private; 9 of the 13
teams ahead of them on the public finished behind.**

## 11. What did not work (their section 11)

| idea | public or bench | private | why |
|---|---|---|---|
| other TTA views (32 views, lateral and z phase shifts) | public −0.003 to −0.014 | 0 to +0.002 | 16 views was a local optimum of the public embryo |
| their own secondary detector (199-video / OOF weights) | public 0.950 / 0.942 vs 0.954 | −0.001 / −0.007 | less data, fewer epochs than the public weights |
| loss mask for unannotated cells, 10-epoch fine-tune | recall −0.0154 and −0.0015 | n/a | a fine-tune; **the full sparse-label recipe was never tried** |
| linker re-solve with collective motion (5 variants) | in-sample +0.009; public 0 up, 1 flat, 4 down | −0.001 to +0.002 | a loss on the public embryo only |
| mass re-detections (28–33 % of positions moved) | public −0.004 / −0.003 | +0.002 / +0.003 | never positive on the 199 edge term either |
| bulk division adds (13 public reads) | 1 up, 3 flat, 9 down | mean +0.0002 over 12 pairs | below the public break-even; the private one is lower |
| cutting forks near the volume faces | public −0.004 with 0 annotated divisions lost | 0.000 | **"unannotated" does not mean "false"** |
| joint re-solve of links and divisions (exact MILP) | right fork for 22 of 96 missed divisions, vs 67 with oracle cues | n/a | **the limit was the division witness, not the solver** |
| written forecasts before public reads | 0.968 → 0.965, 0.970 → 0.967, 0.972 → 0.969 | n/a | their bench-to-public conversion had no out-of-sample skill |

*"Several of these failed only on the public embryo: on the private they were flat or slightly positive."*

## 12. Their comparison to 3rd place, and the density confound

3rd place is ahead by 0.007 public and **0.021 private**. Their public→private drop was 0.024, yu4u's
0.010; the median among teams at 0.965+ public was ~0.027. What yu4u did differently: own detectors
(ImageNet-pretrained 2.5D encoders + 3D SegResNet) with a loss built for sparse labels; a supervised
parent model reading raw and flow-aligned frames; **one LP/MILP choosing nodes, links and divisions
together on a surrogate of the metric**; one ablation ladder on fixed OOF detections scored by the
official metric.

> Validation is the best-supported cause of our bad decisions, not the whole gap.

Division Jaccard: yu4u **0.540 CV / 0.47 private**; theirs 0.409 in-sample, **0.194 OOF**, at most ~0.33
private. *"If that held for the final chain, the division term alone would explain about 0.014 of the
0.021 private gap."*

**On density.** yu4u's probe found the private embryo sparser than both training embryos (median 60–100
detections/frame against 114 and 355). Their in-sample videos are *easiest when sparse*, so reweighting
toward that density predicted a private **above** the public by ~0.02 — it came out 0.024 below. The
reweighting puts 81 % of its weight on 6bba and 75 % on copy videos, so **density, embryo and copies are
confounded: reweighting an in-sample world does not correct an in-sample bias.**

### Their idea / our attempt (their Table)

| their idea | 12th's attempt | outcome |
|---|---|---|
| OOF detector shipped | 50 epochs on ~99 videos | public 0.942 (−0.012), private −0.007 |
| "probability of being evaluated" | a learned propensity, AUC 0.645 / 0.715 across embryos | 0.49 on the decider's head; **never wired** |
| short-component removal | the same filter, same order | it also erased real cells |
| coordinate refinement | learned recentring, shipped | +0.0024 OOF cf, +0.003 private; **theirs is a non-learned affine motion model** |

## 13. Their lessons, verbatim in substance

- **Own your base models, or build the OOF world on day 1.** A bench in which one stage has seen the
  videos should decide nothing.
- **Keep one ablation ladder on one fixed OOF world**, scored by the official metric. They priced pieces
  in different worlds, one of which lacked the whole repair block.
- **The mechanism predicts transport better than the bench, and each embryo judges differently.** On the
  public embryo: continuity additions paid ~1:1, merges/re-links/upstream re-tunes paid zero or less, and
  destroying real structure cost **3–10×** its train price. On the private: continuity pieces held at
  1.3–3× their public size and linker re-solves averaged ~zero.
- **Optimise the metric in one place, with a strong division witness.** Late passes validated one by one
  cannot trade edges against divisions, and **37 of their 39 "stolen" daughters were one node for two
  bodies, which no solver on the same nodes can fix.**
- **Do not tune detector knobs on the public board.**
- **Never decide on one display unit.** They removed a division-admission rule twice on +0.001 public
  reads; the private moved −0.001 and −0.002.
- **Execution discipline is cheap.**

### Their own caveats (section 13)

| where | says | actually |
|---|---|---|
| the OOF world | "out-of-fold" | weaker networks, no final-repair rules: a **lower bound** |
| mitosis specialist, 76.5 % | nested OOF models | read on an **in-sample graph**; never measured on a new embryo |
| private pairs | one variable each | the recentring pair also changes the CSV writer version; the z-lift pair is confounded with run coverage |
| local scorer vs Kaggle | same code | floats read +0.0004 in-sample and −0.008 public; a truncation probe reproduces the sign, not the size |
| TTA ladder 1/8/16/32 views | 0.952 / 0.958 / 0.964 / 0.958 | the 32-view read sits on another parent (0.961); only 8 → 16 is a clean step |

## 14. How they worked

Direction, video review and every strategic call by a human who wrote no code; **the code, the metric
replay, the measurements and the submissions were done by one main Claude Code agent** ("ATHENA") under a
strict rule — the main agent does and coordinates everything, through skills per project part with
precise permissions and resource limits, delegating to builders, adversarial verifiers and a judge:
**> 4,200 sub-agent runs and 478 workflows since 23 Aug**. ChatGPT as external research adviser; Gemini
Deep Research and NotebookLM for literature. 112 submissions in 52 days.

(288th place, topic #744512, also ran their daily probe loop with Claude Code — and lost 21 submission
slots to a machine that was off.)
