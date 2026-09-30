# 18th place (ymg_aq) — private 0.942, from the public notebook we also forked

Source: Kaggle discussion topic **#744531**, "18th Place Solution: Lineage Graph Refinement Focused on
Cell Divisions", posted 2026-09-30 06:39Z. Faithful synthesis; numbers are theirs.

**Why this write-up matters more to us than the 3rd place's does.** They started from a *public
notebook* — `flexonafft` "Biohub Harmonic Fusion" v30, public 0.947, the same lineage we forked — kept
its detector and candidate graph untouched, and rebuilt only what comes after. Total weight of
everything they added: **about 9 MB**, loaded from a private Dataset. No new detector, no large model,
no GPU-heavy training. That reached **private 0.942** against our 0.918 from the same starting family.
It is an existence proof that ~0.024 of private score sat downstream of a detector we had concluded was
the bottleneck.

## The framing they started from

Scored on all 199 training videos, the inherited public pipeline reads **0.9116**, with division
**TP/FP/FN = 23 / 102 / 128** against 151 GT divisions. Their reading: *"Edges were largely right;
divisions were mostly missed."* So nearly all effort went to divisions.

Four stated principles:

1. **Go where the headroom is** — the inherited pipeline recovered 23 of 151 GT divisions.
2. **Use parts that survive sparse labels** — GT covers ~**2.8%** of estimated nodes and holds only 151
   divisions, so: light LightGBM/CatBoost heads on top of existing model outputs, TabPFN for few-sample
   problems, and a *small* CNN pretrained on external data. Not retraining large models.
3. **Judge the graph you submit** — later steps (daughter-track completion) change the graph, so the
   final division decision is computed on the finished submission graph.
4. **Never delete nodes, and decide on all 199 videos.**

## Stage table (their Table 1; CV gain = change in all-199-video CV when the stage was introduced)

| stage | public notebook | theirs | CV gain |
|---|---|---|---|
| edge probability | SimpleNodeTransformer + fwd/bwd harmonic fusion | replaced by a **341-feature LightGBM** | **+0.021** |
| edge selection | ILP (appear / disappear / division costs) | per-frame capacitated bipartite matching | (in the +0.021) |
| coordinates | detector output | **3-axis LightGBM correction** | **+0.020** |
| division base probability | geometric division rules | 303-feature CatBoost on parent–daughter triplets | (in the +0.020) |
| division image model | — | **3D CNN pretrained on external data** + time-direction model | **+0.021** |
| division stacking | — | **TabPFN 3.5** stacking of expert heads (α=0.5) + wide head | **+0.006** |
| division adoption | costs inside the ILP | greedy competition with ordinary edges (strength 4) | **+0.006** |
| daughter tracks | short-track handling, gap filling | reconnect orphan daughter tracks to their parent (min length 3) | |
| division re-check | — | judge on the completed graph; below q=0.3 revert to an ordinary edge | |
| lineage | — | no re-division within 12 frames on one lineage (MILP max-weight compatible set) | |
| post-processing | re-tuned on Train | **not used; no node or track deletion** | |

They note the CV protocol changed over time, so these are rough magnitudes.

## The mechanisms, in order of what they cost us

### Edge rescoring — the public probability becomes one feature among 341

Features: the public edge probability, **HOCT** (`ctc_v0`, `general_v1`) edge scores, **Trackastra**
(`ctc`) edge scores, nucleus shape and intensity from **Cellpose** and from the image, and motion
context over neighbouring frames → a 238-feature base edge LightGBM. They then fine-tuned the upper
layers of the **OrganoidTracker 2** division CNN and added **103 of its intermediate descriptors**,
giving the 341-feature "ordinary-edge head". It **fully replaced** the public edge probability (blend
weight 1.0). Training set: ~1.4 M rows; the base head saw 125,889 positive / 650,448 negative candidate
edges.

**The label choice is the load-bearing part:** they trained on *"every edge the official metric can
judge, not only GT edges"* — because GT covers ~2.8% of nodes, and treating only GT edges as positives
teaches the model that unscored edges are negatives.

Selection: minimum-cost capacitated bipartite matching between neighbouring frames instead of the ILP.
A parent may take up to two children; edges below p=0.2 unused; two children on one parent costs a
penalty of 20. Rescoring + matching took CV 0.9116 → **0.9327**.

### Coordinate correction — a learned per-axis shift

LightGBM, **Huber loss, one model per axis, 86 features**, 130,934 training rows with updated GT
matching. **The applied shift is 0.5 × the prediction, capped at 3 µm**, against a metric that matches
nodes within 7 µm. With the daughter-pair CatBoost this took CV 0.9327 → **0.9528**.

### External data for divisions — the answer to "151 divisions is too few"

- **DivisionCubeNet** — 3D CNN, 3 conv layers, GroupNorm, **60,449 parameters**. Crops 8³ cubes at
  t = −1, 0, +1, **+3** relative to the parent, **aligned to the axis between the two daughters**.
  Pretrained on *self-generated synthetic 1→2 division images*, trained further on **Linajea zebrafish
  160328 (70 real divisions, 419 negatives)**, finally fine-tuned on the competition data (243
  positives). Inference averages two views (original + flip).
- **Time-direction model** — a CatBoost (138 features) that tells whether a Linajea division is played
  forwards or backwards. A dividing cell goes 1 → 2; reversed it goes 2 → 1. **Uses no competition
  labels at all.** Trained on 489 Linajea samples only.

Blended in logit space: `logit p_teacher = (1−β)·logit p_CNN + β·logit p_time` with β=0.5, then
`logit p_base = 0.5·logit p_pair + 0.375·logit p_teacher + 0.125·logit p_time`. **+0.021 CV.**

### TabPFN stacking — few positives, many features

Separate experts: image TabPFN (112 features), temporal TabPFN (192), a **CELLECT** head (145), and the
OrganoidTracker CNN. Their predictions plus appearance and assignment context form 309 features that
**TabPFN 3.5** stacks — public pretrained weights, training data passed only as *context*: **1,139 rows,
115 positives**, outer fold excluded. Blended with the base probability at α=0.5. A separate 284-feature
"wide head" catches candidates outside the stacking head's query range, admitted only above 0.99.
**+0.006 CV.**

### Division adoption — divisions must out-bid the ordinary edges they displace

Adopting a division removes the ordinary edges attached to its parent and daughters. Gain of a division:

```
g = s · (logit p − logit 0.7)
```

`s` = adoption strength; adopt greedily only when the gain beats the ordinary edges replaced; a cell may
not divide again within two frames. **Raising s from 0.5 to 4 gave +0.002 CV and moved public 0.968 →
0.970** — *"divisions supported by the models stopped losing to competing ordinary edges."*

### Daughter-track completion, then re-check on the finished graph

Right after a division the daughters are small and close, so detection and linking break. An orphan
daughter track that persists long enough is reconnected to its parent through an unused candidate edge
(p ≥ 0.8), requiring the daughters to be ≥ 3 µm apart and their separation to grow by ≥ 1 µm. Lowering
the minimum track length 5 → 3 gained one TP.

The re-check then runs **after** completion, on the submitted graph: 93 features — 42 describing the
graph (track length, confidence, whether completion was used) and **51 describing parent–daughter motion
after subtracting the surrounding tissue motion, at horizons of 1, 3, 7 and 15 frames**. TabPFN scores
these, blended with an 86-feature geometry critic:
`logit p_final = 0.75·logit p_motion + 0.25·logit p_geom`. Below q=0.3 the weaker arm is cut and
replaced by an ordinary edge. **No cell node is deleted.**

## Node deletion is a private-set trap — with numbers

> Deleting 10% raised CV but lowered the LB sharply: **−0.015 on Public and about −0.06 on Private**.
> Fewer nodes help the node-count adjustment, but correct edges are lost with them, and CV on sparse GT
> hides that loss.

Their 9/17 submission carried 10% track deletion: public 0.942 / private **0.868**. The same models
without it, next day: public 0.957 / private **0.927**.

## CV design — the part that reads as a verdict on ours

Train holds only two embryos, so their main ruler is **embryo-wise holdout**: train the added parts on
one embryo, predict the other, both directions, and score the **199 final CSVs together with the
official metric**. Video-level 5-fold was a secondary check. They flag the honest caveat: the public
upstream models were trained on all of Train, so both CVs are *conditional* OOF for the added parts only.

> Changes that gained +0.005 to +0.012 on 8–16-video checks repeatedly turned into −0.001 to −0.003 on
> all 199 videos. We therefore always decided on the final CSVs of all 199 videos.

One division is worth roughly **0.0003–0.0007**.

### Correlations (their Table 6; LB values as displayed, 3 decimals)

| submissions | n | pair | Pearson r | Spearman ρ |
|---|---|---|---|---|
| all | 56 | public – private | 0.92 | 0.45 |
| excluding the 3 with track deletion | 53 | public – private | 0.67 | 0.35 |
| from 9/23, with embryo CV | 34 | public – private | **0.09** | **−0.23** |
| from 9/23 | 34 | CV – public | 0.03 | 0.07 |
| from 9/23 | 34 | CV – private | **0.34** | 0.23 |

The 34 late submissions are small changes to β and thresholds (public spans ≤0.005, private ≤0.007). In
that range **public and private are essentially uncorrelated, and so are CV and public**; only CV and
private correlate at all. Submissions tied at the top public 0.971 span CV 0.957–0.963 and private
0.939–0.941.

> Trusting CV on all 199 videos tracked Private better than chasing Public differences of 0.001–0.002.

## Milestones (their Table 5)

| date | main change | embryo CV | public | private |
|---|---|---|---|---|
| — | public notebook (Harmonic Fusion v30) | (0.9116) | 0.947 | — |
| 9/17 | their heads (edges, daughter pairs, point correction) **+ 10% track deletion** | — | 0.942 | 0.868 |
| 9/18 | same models **without** track deletion | — | 0.957 | 0.927 |
| 9/18 | external division CNN + time-direction model | — | 0.964 | 0.939 |
| 9/20 | TabPFN division stacking, wide head, daughter completion | 0.9605 | 0.965 | 0.942 |
| 9/23 | 2-view CNN × time-model teacher + post-completion re-check | 0.9614 | 0.966 | 0.942 |
| 9/24 | adoption strength 4, length-3 completion, rebuilt motion critic, lineage constraint | 0.9647 | 0.970 | 0.942 |

Private reached 0.942 on 9/20 and then sat between 0.936 and 0.943 for nine days while public kept
rising.

## Final selection

Three submissions share code and models, differing only in the teacher-fusion β:

| | β | embryo CV | 5-fold CV | div TP/FP/FN | public | private |
|---|---|---|---|---|---|---|
| base solution | 0.5 | **0.96468** | 0.96151 | 76/33/75 | 0.970 | **0.942** |
| final 1 | 0.75 | 0.96163 | 0.96094 | 72/37/79 | 0.971 | 0.941 |
| final 2 | 0.625 | 0.96279 | 0.96131 | 73/34/78 | 0.971 | 0.941 |

Raising β added false divisions and lowered CV yet raised public by 0.001. They picked the two
public-best arms; **on private the base solution was 0.001 higher, as CV had predicted.**

## External resources (all public, licences as they state them)

| resource | use | licence |
|---|---|---|
| Harmonic Fusion v30 (Tracking Support Pack v10, TemporalUNet3D seed314159 v2, DeepCenter v5) | detection + candidate graph, inherited | public notebook; input Datasets CC0-1.0 |
| HOCT `ctc_v0` / `general_v1` | edge and division **features** | MIT |
| Trackastra `ctc` | edge and division **features** | BSD-3-Clause |
| Cellpose | nucleus shape features | BSD-3-Clause |
| Linajea zebrafish 160328 (70 divisions, 419 negatives) | CNN pretraining, time-direction model | BSD-3-Clause |
| synthetic 1→2 division images | CNN pretraining | self-generated |
| OrganoidTracker 2 division CNN | descriptors, expert head | CC BY 4.0 |
| CELLECT | expert head | GPL-2.0 |
| TabPFN 3.5 Fast (public weights) | stacking, wide and critic heads (context only) | TABPFN-3.5 Non-Commercial, *explicitly permits Kaggle-style competitions* |

## Read against our own record

- **HOCT and Trackastra appear here as *features into a tabular model*, never as an association head.**
  We built HOCT (`celltrack/edge/hoct_edge_transformer.py`), refuted it as a head, and never fed its
  scores to a ranker.
- **CELLECT appears as an expert head.** Our CELLECT work was a long detector training run
  (`celltrack-cellect-long-washed-out`).
- **External data never entered our planning at all.** Linajea + OrganoidTracker + synthetic + TabPFN
  weights is the direct answer to the "151 divisions" wall we treated as a hard denominator.
- **The 10%-deletion number is the hardest confirmation our prune-axis closure ever got** — see
  `interpretations/celltrack/converging/2026-09-30_private_lb_post_mortem.md` §3.
