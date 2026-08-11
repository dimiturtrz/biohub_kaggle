# Mitosis detection PRECISION — what separates a real fork from a crowded false one

**Date**: 2026-08-11
**Status**: settled for the precision-cue question; partially unresolved for division-rate priors (no
source publishes a per-frame-per-cell rate directly — rates below are DERIVED from published counts)
**Question**: our score is `adjusted_edge_jaccard + 0.1 * division_jaccard` and we earn ~0 from the
division term. On the dense test movie there are 3 annotated divisions; affinity-based recovery found
the true fork but emitted 807 false ones, and geometry gates tight enough to kill the 807 also kill the
real one (daughters 13.5 µm apart, parent→2nd-daughter 6.08 µm). So: **what cue, other than geometry,
distinguishes a genuine mitotic fork?**

## Method + trust convention

Every row carries a trust level. They mean exactly this:

| Trust | Meaning |
|---|---|
| **SOURCE** | I downloaded the PDF / raw source file and read the actual bytes containing the claim. Quotes are verbatim from that text. |
| **FETCH** | Obtained via WebFetch's page summariser (a small model reading the HTML). Not verbatim-verified. Treat numbers as indicative, re-verify before betting on them. |
| **DERIVED** | Arithmetic I did on SOURCE-level published counts. The inputs are verbatim; the ratio is mine. |
| **UNVERIFIED** | Could not load / not stated in the source. Reported as a gap, never as a number. |

PDFs read locally with PyMuPDF: `arxiv.org/pdf/2403.15011`, `arxiv.org/pdf/2208.11467`,
`ecva.net/.../09819.pdf` (Trackastra), `publications.ri.cmu.edu/.../ISBI11_Huh.pdf`,
`arxiv.org/pdf/2004.12531`. Source read from GitHub raw: `weigertlab/trackastra`, `funkelab/linajea`.
Licences read from the GitHub API `license` field, not from prose.

---

## TL;DR — the three cues that actually carry precision

1. **A learned cell-state classifier on a 3D patch STACK — "is this cell a parent / daughter /
   continuation / polar body" — is the only mechanism in the literature with a measured, large
   false-division reduction, and it is validated in 3D, including 3D light-sheet.** Hirsch et al.
   (MICCAI 2022, arXiv:2208.11467) add a 3D ResNet18 over a `[5, 32, 32, 32]` input (5 timepoints ×
   32³ voxels) to Linajea. On 3D light-sheet C. elegans, **false-positive divisions drop 1.5 → 0.20 per
   1000 GT edges (7.5×)**; on 3D confocal, **0.89 → 0.053 (17×)**, at the cost of FNdiv 0.26 → 0.40.
   That is precision bought with a small, acceptable recall loss — exactly our shape of problem. MIT
   licence. [F1]
2. **Mitotic appearance is separable in 3D from shape + intensity + texture alone, without any
   tracking.** Cell-cycle-phase classification on 3D confocal Drosophila embryogenesis reaches >90%
   accuracy over 5 phases with a 9-feature SVM (sphericity, surface area, intensity mean/SD,
   intensity-centroid vs geometric-centroid offset, Haralick texture, volume). This is a cue we do not
   use at all, it is orthogonal to geometry, and it is measured on 3D developmental tissue. [F2]
3. **Deferred decision over a temporal horizon.** The mitosis-aware MHT (arXiv:2403.15011, IEEE TMI
   2025) keeps a false fork as a *hypothesis* and kills it 3 frames later using an Erlang minimum-
   lifetime cost — the paper's own Fig. 3 walks through exactly the "false division at frame 2,
   corrected a-posteriori at frame 5" case. The mechanism is right; **the validation is 2D-only, on
   nine 2D CTC datasets, with a 2D-Gaussian density motion model.** [F5]

Also load-bearing, and cheap: **Trackastra's `λ_div = 10` is a real, verbatim-confirmed loss weight on
dividing-cell rows, and the shipped code additionally oversamples training windows containing
divisions by default.** Two independent knobs we have never turned. [F4]

---

## F1 — Learned cell-state classifier (parent/daughter/continuation/polar body), 3D, measured

Hirsch, Malin-Mayor, Santella, Preibisch, Kainmueller, Funke, *Tracking by weakly-supervised learning
and graph optimization for whole-embryo C. elegans lineages*, MICCAI 2022 — arXiv:2208.11467.

| Fact | Detail | Trust |
|---|---|---|
| Classes | "parent cell (i.e., a cell that is about to undergo cell division), daughter cell (cell that just divided), continuation cell (cell track that continues without division) and polar body" | SOURCE (verbatim) |
| **The parent class is the MOTHER-BEFORE-SPLIT cue** | The classifier is asked to recognise a cell *about to divide*, from its own appearance — precisely cue #1 in our question | SOURCE |
| Architecture | "We train a 3d ResNet18 [6] with 3d convolutions for this task" | SOURCE (verbatim) |
| Input | "Input size (in voxels): [5, 8, 64, 64] (mskcc-confocal), [5, 32, 32, 32] (nih-ls)" — i.e. **5 timepoints × a 3D patch**, so appearance AND short-horizon temporal context in one tensor | SOURCE (verbatim) |
| Training | Adam, batch 64 / 16, lr 5e-4 → 5e-6 after 20k iters, "cross entropy or focal loss", dropout, no batchnorm, global average pooling, 60k iters, best ckpt by val, **no TTA** | SOURCE |
| How it enters the solver | class scores become node-label costs in the ILP, plus **novel constraints**: `y_parent,u + y_edge,e − y_daughter,v ≤ 1` and `y_daughter,v + y_edge,e − y_parent,u ≤ 1` — a selected edge out of a parent *must* land on a daughter | SOURCE (verbatim) |
| Extra weights | `w_parent, w_daughter, w_continue` scale the class scores; `w_division` is a constant division cost "contributed by each selected parent" | SOURCE |
| **3D light-sheet result (nih-ls, 270 frames)** | FPdiv 1.5 → **0.20** per 1000 GT edges; FNdiv 0.40 → 0.49; div-error sum 1.86 → **0.69**; FP edges 12 → 13, FN 6.5 → 5.3 | SOURCE (Table 1) |
| **3D confocal result (mskcc-confocal, 270 frames)** | FPdiv 0.89 → **0.053** (17×); FNdiv 0.26 → 0.40; div sum 1.2 → **0.46**; TRA 0.99418 → 0.99480 | SOURCE (Table 1) |
| Ablation isolating the classifier | Table 2 row without csc: FPdiv 0.89, div-sum 1.2 → with csc: FPdiv 0.053–0.11, div-sum 0.38–0.46. Text: "the inclusion of the classifier in linajea+csc+ssvm lowers them drastically" | SOURCE (verbatim) |
| Dimensionality | **3D throughout**: "three confocal and three lightsheet recordings of C. elegans, all fully annotated"; nih-ls is "isotropic lightsheet" | SOURCE (verbatim) |
| Scale | 270 frames, ~570 cells in last frame, 52k cell instances per sample | SOURCE |
| Division-sampling during training | "at least 25% of iterations have to contain a division" (tracking U-Net) | SOURCE (verbatim) |
| Temporal leniency in evaluation | "Divisions that are off by one frame compared to the annotations are not counted as errors" | SOURCE (verbatim) |
| Code + licence | https://github.com/funkelab/linajea — **MIT** (GitHub API `license.spdx_id = MIT`) | SOURCE |
| CTC standing | headed Fluo-N3DH-CE (3D C. elegans) DET/TRA at submission; "These results ... were generated using our cell state classifier" | SOURCE (verbatim) |
| Cost knobs in the shipped code | `linajea/config/solve.py`: `weight_division`, `division_constant`, `cell_state_key`; example config sets `weight_division = [-8, -11]`, `division_constant = [6.0, 2.5]` | SOURCE (raw file) |
| Division evaluation with frame slack | `linajea/evaluation/division_evaluation.py` matches divisions across a `frame_buffer` of ±N frames and emits TP/FP/FN/precision/recall per leniency level | SOURCE (raw file) |
| Prior Linajea (Nat Biotech, Malin-Mayor et al.) has a separate division classifier? | **No** — division is implicit via movement vectors both daughters point back to the parent; divisions enter the ILP through a split indicator with a constant cost | FETCH (PMC7614077) |

**Why this is the answer to our problem.** The 807 false forks are geometric coincidences: two cells
that happen to sit near a third. A parent-class classifier does not look at where the daughters are —
it looks at whether the mother *looks like a cell in mitosis*. The 807 mothers are ordinary interphase
nuclei; the 1 real one is not. And the ILP constraint above is the piece that converts a soft score
into a hard filter without a distance cap: a fork is only allowed out of a node the classifier calls
`parent`.

## F2 — Mitotic appearance is separable in 3D from static features alone

Du, Sudar, et al. (via PMC3278834), *Cell cycle phase classification in 3D in vivo microscopy of
Drosophila embryogenesis*.

| Fact | Detail | Trust |
|---|---|---|
| Task | 5-class: interphase + prophase, metaphase, anaphase, telophase | FETCH |
| Modality / dimensionality | **3D** time-lapse confocal (Zeiss 5 Live), 66–70 slices per stack; "3D intensity, shape and texture features" | FETCH |
| Feature set | 42 features reduced to 9: **shape** (sphericity, surface area, eccentricity), **intensity** (mean, SD, and the deviation between the intensity-weighted centroid and the geometric centroid), **texture** (Haralick: homogeneity mean, information measures of correlation, difference-variance mean, entropy mean), **volume** (voxel count) | FETCH |
| Classifier | weighted SVM (weighting for class imbalance) | FETCH |
| Accuracy | >90% overall; 90.29% gastrulation, 92.40% syncytial blastoderm; per-phase 83.83% (anaphase) – 93.46% (interphase) | FETCH |
| Licence / code | UNVERIFIED — no repo located | UNVERIFIED |

**Read**: chromatin condensation is measurable as *texture* (Haralick entropy/homogeneity), and mitotic
rounding as *sphericity*; the intensity-weighted-vs-geometric centroid offset is a cheap proxy for
"chromatin has gone asymmetric/condensed inside the nucleus". None of these need a network — they are
computable per candidate mother from the raw volume we already load. This is the cheapest possible
first experiment on cue #1.

Corroborating appearance facts from other modalities (2D, mechanism only — do not port the numbers):

| Fact | Source | Trust |
|---|---|---|
| "mitotic cells are typically much brighter than non-mitotic cells" — candidate patches are extracted by brightness thresholding alone, set "not to miss actual mitotic events" (recall-first candidate stage) | Huh et al., ISBI 2011, 2D phase-contrast | SOURCE (verbatim) |
| Mitosis causes "changes in cell shape, size, and brightness" that break segmentation | Huh et al. | SOURCE (verbatim) |
| Adding mitosis detection to a tracker cut missed mother–daughter relations by 42% and switched tracks by 21% overall (48 C2C12 populations, 4 conditions, ratio paired t-test p ≤ 0.056 … <0.0001) | Huh et al. | SOURCE (verbatim) |
| Daughters'-symmetry + mother-daughters'-dissimilarity features, EM-clustered, unsupervised | Mitodix, Gilad et al., Bioinformatics 35(15):2644 | FETCH |

## F3 — Mitodix: the one method reporting PRECISION 1.0 in 3D (at low recall)

| Fact | Detail | Trust |
|---|---|---|
| Features | daughters' symmetry `s₁,₂ʷ` (weighted Pearson correlation after symmetry-axis extraction) and mother-daughters' dissimilarity `S_m,1,2 = max(s₁,ₘ, s₂,ₘ) / s₁,₂,ₘ` | FETCH |
| Classifier | unsupervised EM over 2 clusters in that 2D feature space; posterior probability per candidate triplet | FETCH |
| Candidate stage | patch-based spatio-temporal **intensity differences** + Otsu; neighbourhood from a stochastic Delaunay triangulation | FETCH |
| **3D results** | Fluo-N3DH-SIM+01: TP 14, FP 0, FN 23 → **precision 1.0**, recall 0.38, F1 0.55. Fluo-N3DH-SIM+02: TP 23, FP 0, FN 17 → **precision 1.0**, recall 0.58, F1 0.73 | FETCH (Table 2) |
| 2D results for contrast | HeLa01 P 0.97 / R 0.91 / F1 0.94; MCF-10A P 0.95 / R 0.68 / F1 0.78 | FETCH |
| 3D caveat | the method "assumes planar cell division symmetry" in 3D | FETCH |
| Runtime | 18–65 s per frame | FETCH |
| Code / licence | github.com/topazgl/mitodix — **MATLAB, no licence file stated → NOT mountable** | FETCH + SOURCE (README read) |

**Read**: this is the only published operating point that matches our need shape — zero false
divisions in 3D, at 38–58% recall. For a metric weighted `0.1 × division_jaccard` with 3 GT divisions,
catching 1–2 with **zero** false forks beats catching 3 among 807. The MATLAB/licence status means we
port the *idea* (symmetry between the two daughters' appearance; mother must look unlike either
daughter alone), not the code.

## F4 — Trackastra `λ_div = 10`: verified, and there are TWO knobs, not one

| Fact | Detail | Trust |
|---|---|---|
| It is a **loss** weight | Eq. 6 weight matrix W: `1 + λ_div` when `deg⁺(v_kᵢ) = 2` (dividing cells), `1 + λ_cont` when `deg⁺ = 1`, `1` otherwise | SOURCE (verbatim, ECVA PDF) |
| Values | "We choose Δt = 2, λ_div = 10 and λ_cont = 1 as fixed hyperparameters. This choice effectively up-weights the loss for cell divisions and continuing tracks, and removes the loss for associations that are not used during the linking step." | SOURCE (verbatim) |
| It is applied to **matrix entries**, not samples | the weight multiplies the BCE loss of the association-matrix cells belonging to a dividing parent's row | SOURCE |
| Dataset-dependence | "Due to the comparatively lower rate of cell divisions, we choose to reduce the impact of divisions in the loss reweighting W by setting λ_div = 2 for this dataset" (DeepCell) | SOURCE (verbatim) |
| **Shipped code disagrees with the paper** | `scripts/train.py:182` `div_upweight: float = 20` (class default); `scripts/train.py:1076` `--div_upweight` CLI **default = 2**. Paper says 10. Three different values across paper/class/CLI. | SOURCE (raw file) |
| Exact implementation | `train.py:266`: `loss_weight = 1 + 1.0 * normal_tracks + self.div_upweight * division_tracks`, where `division_tracks = block_sum > 2` and `block_sum = A * (blockwise_sum(A, dim=-1) + blockwise_sum(A, dim=-2))`. Computed under `torch.no_grad()`. | SOURCE (raw file) |
| **Second, independent knob** | `--weight_by_ndivs`, **default True**, "Oversample windows that contain divisions" — a sampler-level up-weight on top of the loss-level one | SOURCE (raw file) |
| What it buys — measured? | **No λ_div ablation exists in the paper.** Division quality is reported as FP divs / FN divs (Tables 1, 3) and Div F1 (Table 2), but never against λ_div. **We cannot attribute a number to the weight itself.** | SOURCE (absence verified by grep) |
| Division F1 context (2D, DeepCell) | GT-seg: Caliban 0.97, Trackastra-ILP 0.94, Trackastra-greedy 0.90, Baxter 0.72, CellTrackerGNN 0.18. Caliban-seg: 0.92 / 0.79 / 0.71 / 0.60 / 0.13 | SOURCE (Table 2) |
| Paper dimensionality | **2D only**: "while the presented results are limited to 2D datasets, Trackastra is expected to scale well to 3D datasets since the architecture does not require to process dense images" | SOURCE (verbatim) |
| **But the released weights are not 2D-only** | `trackastra/model/pretrained.json` lists a `ctc` model with `"dimensionality": [2, 3]`, trained on "All Cell Tracking Challenge 2d+3d datasets with available GT and ERR_SEG"; "successor of the winning model of the ISBI 2024 CTC generalizable linking challenge" | SOURCE (raw file) |
| Licences | code `weigertlab/trackastra` **BSD-3-Clause**; weights `weigertlab/trackastra-models` **BSD-3-Clause** (GitHub API). Mountable. | SOURCE |
| No dedicated division head | divisions emerge from the association matrix + parental softmax (one-to-many allowed, many-to-one not) + `deg⁺(v) ≤ 2` in the ILP | SOURCE / FETCH |

**Read**: λ_div is real and cheap to copy, but nobody has measured what it buys — the honest framing is
"the frontier method thinks divisions deserve 10× loss weight and oversampled windows", not "λ_div is
worth X". Note it up-weights *recall* of divisions during training; it is not itself a precision
mechanism. Our binding constraint is precision, so expect λ_div alone to make the 807 worse, not
better, unless paired with a parent-gate.

## F5 — The Erlang cell-cycle-age mitosis cost: real, but 2D-only, and the 3D flag stands

Kaiser et al., *Cell Tracking according to Biological Needs — Strong Mitosis-aware Multi-Hypothesis
Tracker with Aleatoric Uncertainty*, arXiv:2403.15011, IEEE TMI (DOI 10.1109/TMI.2025.3583148).

| Fact | Detail | Trust |
|---|---|---|
| Erlang is real | "the interval between successive mitotic events can be approximated by an Erlang distribution in homogeneous cell cultures (Yates et al., 2017; Paul et al., 2024)" | SOURCE (verbatim) |
| The cost | `c_M,i,h = −log(∫_{−∞}^{Age(i)} Erlang_{α,β}(t) dt)` if the age is known, `0` otherwise — "with the current lifetime Age(i) of the cell (which may also be unknown)"; "penalizes hypotheses that imply implausibly short cell life cycles" | SOURCE (verbatim, Eq. 13) |
| Parameterisation | α, β "can be approximated from the data by regressing the exponential proliferation rate, as done, for instance, in (Paul et al., 2024). For short sequences of length K that rarely contain cell splits and do not allow reasonable approximations, we set **α = K and β = 1/K** to penalize cell splits with relatively high costs." | SOURCE (verbatim) |
| Mechanism vs our problem | it is a **temporal-persistence** filter in disguise: a fork whose daughter is itself implicated in another event too soon becomes expensive. Fig. 3 walks a false mitosis detected at frame k=2 being corrected using posterior knowledge from frame k=5. | SOURCE |
| Ablation | "The most expressive results can be observed without explicit mitosis costs in setting 3). On the long and complex sequences BF-C2DL-HSC and -MuSC, all metrics collapse significantly when no long-term consistency preserving mitosis costs are incorporated. On the shorter sequence PhC-C2DL-PSC, the effect is also visible but with a lower impact. The mitosis costs do not impact short sequences with short proliferation trees" | SOURCE (verbatim) |
| **Dimensionality — the flag was right** | all nine datasets are 2D: BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa, Fluo-C2DL-MSC, Fluo-N2DH-GOWT1, Fluo-N2DL-HeLa, PhC-C2DH-U373, PhC-C2DL-PSC, Fluo-N2DH-SIM+. **No 3D dataset.** | SOURCE |
| It is not merely "untested in 3D" — it is 2D-*coded* | the density representation is explicitly planar: objects "described by 2D Gaussians"; "position and motion of a detection j are described by discrete 2D [densities]". Porting requires re-deriving the uncertainty machinery in 3D. | SOURCE (verbatim) |
| Division metrics used | BC(i) — branching correctness with i frames of tolerance — and CCA (overlap of predicted vs GT life-cycle distributions). No division precision/recall. | SOURCE |
| Applicability caveat for us | the cost is **null when age is unknown**, and our sequences are short with few divisions — exactly the regime the paper says the mitosis cost "does not impact" | SOURCE |
| Code / licence | https://github.com/TimoK93/BiologicalNeeds — **MIT** (GitHub API) | SOURCE |

**Verdict on item 5**: mechanism verified, 3D **not** validated, and the implementation is 2D-specific.
Worse, the paper's own ablation says the cost is inert on short sequences with short proliferation
trees — which is our movie (3 divisions). **De-prioritise.** Keep the *idea* (a division's plausibility
depends on what happens several frames later) and take it via the cheaper temporal filter in F6.

## F6 — Temporal consistency as a division filter: horizons actually used

| Method | Temporal horizon | Dimensionality | Numbers | Trust |
|---|---|---|---|---|
| Hirsch cell-state classifier | **5 frames** stacked as classifier input (`[5, 32, 32, 32]`) | 3D light-sheet + 3D confocal | FPdiv 1.5→0.20 (light-sheet) | SOURCE |
| Kaiser MHT | hypotheses retained across frames, resolved a-posteriori (Fig. 3 example spans k=2→k=5, i.e. **3 frames**) | 2D only | ablation qualitative | SOURCE |
| MDMLM 3DCNN (arXiv:2004.12531) | candidate sequence treated as a volume with **time as the z axis**, V-Net input `128×128×16` → **16-frame** window, sliding stride 8; TP if within **6 frames** temporally and 15 px spatially | 2D+t (NOT volumetric 3D) | "improved the F1-scores over 20%" over compared methods under difficult conditions | SOURCE (verbatim) |
| RDLA++ anchor-free 4D detector (PMC7908657) | bidirectional ConvLSTM over **7 timepoints**; 2.5D spatially (3 consecutive slices) | **4D** two-photon (3D volumes + t), mammalian epidermis | **P 0.834 / R 0.875 / F1 0.855**; volumes 480×480×37; 16 datasets | FETCH |
| Huh EDCRF | candidate patch **sequences** built by linking overlapping bright patches across consecutive frames; CRF labels which frame is the birth event | 2D phase-contrast | 42% fewer missed mother-daughter relations | SOURCE |
| Linajea evaluation | `frame_buffer` — divisions matched within ±N frames when scoring | 3D | n/a (evaluation, not filter) | SOURCE |

**The direct answer to item 3**: yes, temporal context is universally used, but almost nobody
implements it as an explicit "both daughters must survive K frames" rule. The literature instead
(a) feeds a multi-frame stack to the classifier (Hirsch: 5; RDLA++: 7; MDMLM: 16), or (b) defers the
decision inside a global optimiser (MHT / ILP). I found **no** paper stating a daughter-persistence
threshold in frames as a post-hoc filter. If we implement one it is a defensible engineering choice,
not a cited one — record it as ours.

## F7 — Division-rate priors: no published per-frame-per-cell rate; here are DERIVED anchors

No source I could load publishes "fraction of cells dividing per frame" as a stated quantity. The CTC
10-years paper defines a dataset property `Mit` = "the number of division events ... computed based on
the gold standard tracking annotations", but the per-dataset values live only in a colour-coded figure
and are not transcribed numerically [FETCH, PMC10333123] — **UNVERIFIED as numbers**.

What is publishable are counts, from which rates follow:

| Dataset | Published counts (verbatim) | Derived rate | Trust |
|---|---|---|---|
| DynamicNuclearNet / DeepCell (2D nuclei, 130 videos) | "roughly 600k cells ... and showing roughly 2k divisions" | **0.33% of cell-instances** are a division | SOURCE counts / DERIVED rate |
| Trackastra HeLa test (2D) | "two videos with a mean of 16835 edges and 151 divisions" | **0.90% of edges** are a division edge | SOURCE / DERIVED |
| Trackastra Bacteria test (2D) | "six videos with a mean of 2912 edges and 236 divisions" | **8.1% of edges** — high-proliferation regime, an upper bound | SOURCE / DERIVED |
| mskcc-confocal C. elegans (3D, 270 frames) | "approximately 570 cells in the last frame and 52k in total per sample" | embryo goes from a handful to ~570 cells → ≈ 5.7×10² divisions over 52k cell-instances ≈ **1.1% of cell-instances** | SOURCE counts / DERIVED rate |
| Our training data (for comparison) | 151 divisions across 199 videos (0.76/video) | not comparable without cell-frame counts — compute cells×frames to place it on this scale | ours |

**Usable prior**: developmental/3D embryonic data sits near **~1% of cell-instances**, low-proliferation
2D nuclei near **0.3%**, bacteria near **8%**. For a principled cap on our dense movie:
`max_divisions ≈ 0.01 × (n_cells × n_frame_transitions)` as an *upper* bound, and note that our own
observed 3 divisions on that movie is the ground truth to sanity-check it against. Any cap that admits
807 forks is off by orders of magnitude against every rate in this table — that is the strongest
sourced statement available, and it is the argument for a cap, not the cap's value.

## F8 — What else was checked and found not to help

| Item | Finding | Trust |
|---|---|---|
| Trackastra dedicated division head | Does not exist. Divisions come from the association matrix + parental softmax + `deg⁺(v) ≤ 2` ILP constraint | SOURCE / FETCH |
| Linajea (Nat Biotech version) division classifier | Does not exist there either; it is the MICCAI follow-up (F1) that adds it | FETCH |
| Ultrack (Bragantini et al., Biohub — same lab as this competition's data) | ILP allows "up to 2 out-going segments (i.e. division)" with a division penalty `w_δ`; **no appearance-based division scoring**. Notes competitor "JAN-US ... they classify the cell states to detect division and assist the ILP formulation" — i.e. Ultrack's own paper points at F1 as the division-precision method | SOURCE (PDF read) |
| Mitodix as mountable code | MATLAB, no licence file → cannot ship | SOURCE (README) |
| StarDist 3D pretrained model | not revisited here; the prior note that it does not exist stands | n/a |

---

## Recommended next experiments (ordered by cost, not by hope)

No impact estimates are given — no source measured any of these on our data or metric.

1. **Cheapest, tests cue #1 with no training**: compute the F2 feature set (sphericity, volume,
   intensity mean/SD, intensity-weighted-vs-geometric centroid offset, 3D Haralick entropy/homogeneity)
   for the mother candidate at t−1 for all 808 candidate forks on the dense movie, and look at whether
   the true one separates. This is a plot, not a model, and it either shows separation or it does not.
2. **If it separates**: train the F1 classifier shape — 3D ResNet18 over a `[5, D, H, W]` stack centred
   on the candidate mother, 4 classes (parent/daughter/continuation/other), focal loss — and use
   `p(parent)` as a hard gate on which nodes may fork, i.e. the Hirsch ILP constraint, not a distance cap.
3. **Orthogonal and nearly free**: turn on both Trackastra knobs in our own training (loss weight on
   division rows, oversample windows containing divisions). Recall-side, so pair with (2).
4. **Cap the fork count** using F7's ~1% cell-instance rate rather than a guessed number, and rank the
   survivors by appearance score rather than geometry — the point of F3 is that a precision-1.0 /
   recall-0.4 operating point is worth more to `0.1 × division_jaccard` than a recall-1.0 / precision-0.001 one.

## Sources

| ID | URL |
|---|---|
| S1 | https://arxiv.org/abs/2208.11467 (PDF read locally) — Hirsch et al., MICCAI 2022 |
| S2 | https://github.com/funkelab/linajea — MIT |
| S3 | https://pmc.ncbi.nlm.nih.gov/articles/PMC3278834/ — 3D Drosophila cell-cycle phase SVM |
| S4 | https://academic.oup.com/bioinformatics/article/35/15/2644/5259190 + https://pmc.ncbi.nlm.nih.gov/articles/PMC6662301/ — Mitodix |
| S5 | https://github.com/topazgl/mitodix — MATLAB, no licence |
| S6 | https://arxiv.org/abs/2405.15700 · https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf — Trackastra (PDF read locally) |
| S7 | https://github.com/weigertlab/trackastra (BSD-3) · https://github.com/weigertlab/trackastra-models (BSD-3) · raw `scripts/train.py`, `trackastra/model/pretrained.json` |
| S8 | https://arxiv.org/abs/2403.15011 (PDF read locally) · https://github.com/TimoK93/BiologicalNeeds (MIT) · DOI 10.1109/TMI.2025.3583148 |
| S9 | https://arxiv.org/pdf/2004.12531 — MDMLM (PDF read locally) · https://github.com/naivete5656/MDMLM |
| S10 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7908657/ — RDLA++ 4D mitosis detection, CC-BY-4.0 |
| S11 | https://publications.ri.cmu.edu/storage/publications/pub_files/2011/0/ISBI11_Huh.pdf — Huh et al. ISBI 2011 (PDF read locally) |
| S12 | https://pmc.ncbi.nlm.nih.gov/articles/PMC10333123/ — CTC 10 years (Maška et al., Nat Methods 2023) |
| S13 | https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/ — Linajea, Malin-Mayor et al. |
