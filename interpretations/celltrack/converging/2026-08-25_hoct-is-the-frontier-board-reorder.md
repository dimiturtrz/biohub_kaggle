# HOCT is the frontier — research round + board reorder (2026-08-25)

**Trigger:** owner asked for another research round + reorder the pile. Result is not incremental — it
promotes one lever above all others and folds three separate beads into it.

## The find

**Higher-Order Cell Tracking Transformer (HOCT)** — arxiv 2607.11754 (Jul 2026), Bragantini & Theodoro
(Royer lab). On **Fluo-N3DH-CE** (our regime's public analog): **CLB 0.950 (1st) · LNK 0.986 (1st) ·
BIO 0.913 (1st)**, and 1st in CLB/LNK/BIO across all 16 CTC datasets. That is the **winning cluster**
(0.945–0.953). Compact, self-contained, **no pretrained image encoder** — the cleanest possible replicate
target under bead hmlo ("replicate the frontier before porting more parts").

Royer lab also owns Ultrack (the competition's namesake, Nat Methods 2025) and the baseline
(TemporalUNet3D + SimpleNodeTransformer = our joint arch). HOCT is their current linking frontier.

## Exact architecture (extracted from the paper)

- **Node encoder:** input d=19 pure-geometry features (spatiotemporal position, equivalent diameter,
  intensity stats, inertia tensor, border distance) → proj to C=288 → Lₙ=4 self-attention layers. Each
  layer applies **3D RoPE** (learnable per-head frequencies + learnable reflection) to Q/K on node
  position pᵢ; masks pairs farther than distance τ (sparsity).
- **Edge tokens:** for candidate edge e=(i,j), hₑ⁽⁰⁾ = MLPgather([zᵢ ∥ zⱼ]). Refined through Lₑ=4 layers:
  first = cross-attention (learned edge query fₑ vs hₑ), rest = self-attention **among edge tokens** (the
  higher-order part — edges attend to edges).
- **Line-to-line distance bias (the key innovation):** for edge pair (eₘ,eₙ), dₗᵢₙₑ = min Euclidean
  distance between their 3D line segments, injected as a per-head learnable bias:
  Aₕ(m,n) = q̂·k̂/√D + σₕ·softplus(αₕ)·dₗᵢₙₑ. **σₕ ∈ {+1,−1} alternates across heads** →
  attractive heads (nearby edges gain attention) + repulsive heads (distant edges gain attention).
  Edge RoPE position = midpoint (pᵢ+pⱼ)/2.
- **Parental softmax:** pₑ = exp(ℓₑ) / (1 + Σ_{e'∈C(j,Δt)} exp(ℓₑ')). The constant **1 = implicit
  no-parent** slot → identical to our `SlackRow` (association_objective.py).
- **Variable appearance prob** derived from edge logits (lowers new-track penalty, no extra loss term).
- **Two-pass tracklet ILP:** (1) solve ILP on Δt=1 edges → high-confidence tracklets; (2) collapse each
  tracklet to a meta-node, re-solve over all edges. = the **Ultrack ILP seat, built in**.
- **Loss:** focal γ=3.5, division weight 3.5×, on the parental-softmax probabilities.

## STATUS CORRECTION — HOCT is NOT new here; it is a HALF-TRIED asset

Owner caught an overclaim in the first draft. HOCT is **bead 1blg**, audited 2026-08-21, and the
"key innovation" line-to-line edge-to-edge bias was **already built and trained**, not untried:

- **1blg.1 (build: 3D-RoPE node self-attn + edge-to-edge line-to-line bias, σ±1 heads) = CLOSED.**
- **1blg.2 (O(E²) edge-to-edge inference on dense movies) = CLOSED.**
- **Trained verdict (`joint_hoct_gn_v2.pt`, GN, from-scratch):** faithful 0.7234, node-R 0.9015,
  **P_true 0.204 < P_chosen 0.266 → "does NOT close confusor, no better than pack. KILL this use."**

So the reorder is NOT "adopt a new frontier method" — it is "an already-built frontier head sits at a
**SOFT** kill and its own verdict names the re-test." The kill's named confound (per the 4-check
discriminator): trained only to **epoch 0.5/3, from-scratch, on the (1,4,4) undertrained repr, no oracle
control** — a SOFT kill, not a refutation. 1blg's verdict states it explicitly: *"Asset re-testable:
full 3-epoch train + better detected-distribution repr. Next lever = detection-side repr (finer-data
re-pool per confusor-at-voxel-limit) OR frontier dense-model replicate (hmlo)."*

**finer122 IS that finer-data re-pool, running now.** So the live move is not a build — it is to
**re-run the EXISTING HOCT head on finer122's repr, fully trained**, the exact re-test the bead
prescribed. What the paper fetch genuinely ADDED beyond the 2026-08-21 audit: dual-sign σ heads confirmed
(had it), the **two-pass tracklet ILP** (folds in the Ultrack seat — new), variable-appearance-prob from
logits (new, cheap), and the **0.950 CLB score confirmation** (winning cluster).

## Board reorder (the pile)

**#1 lever: RE-RUN the existing HOCT head on finer122's repr, fully trained (bead 1blg, re-open the
soft kill).** Not a build — `hoct_edge_transformer.py` (3D-RoPE node attn `_Rope3D`, edge-to-edge attn
:157, line-to-line bias) already exists and trained once. The re-test the bead's own verdict named. It
relates to the other beads:
- Ultrack ILP seat (we0z, g89y) → HOCT's **two-pass tracklet solver is this**, and it is the ONE piece
  the prior HOCT arm did NOT include (that arm used the standard linker). Adding it is the cheapest new
  variable. Fold in.
- Parental-softmax (SlackRow) → already in the joint trainer as `--slack-links`; was training in the
  refuted arm. Keep.
- CELLECT (54qc) → the **coupled no-code arm** (slack + nce contrastive on finer122) is a DIFFERENT head
  (contrastive-shaped embedding vs edge-to-edge attention) → a genuinely decorrelated second probe of the
  same "better repr + parental reject" hypothesis. Run BOTH as arms once finer122 lands; cheap.

**finer122 (running) is the substrate both arms attach to** — A-fixed detector + better repr. The prior
HOCT kill's named confound was exactly this repr; finer122 is the fix, so the re-test is licensed, not a
re-litigation of a hard kill.

## Plan (audit mostly DONE on 1blg; WORK gated on finer122 checkpoint + GPU)

Already built/audited (do NOT rebuild): HOCT head (1blg.1 ✓), edge-to-edge inference (1blg.2 ✓),
SlackRow parental-softmax, edge transformer training loop, EdgeGap candidate enumeration, ILP linker
(motile). **The ONLY genuinely-new deltas vs the refuted arm:** (a) **finer122 repr** (the point);
(b) **full 3-epoch train** (the refuted arm died at epoch 0.5); (c) **two-pass tracklet ILP** wrapper over
motile (the Ultrack fold-in — the one architectural piece the prior arm lacked); (d) optional
variable-appearance-prob from logits (cheap). d=19 geometry features + focal γ3.5/div3.5 were already in
the refuted arm.

**Gate (non-inverting, per 5e3u):** n=37 P_true→P_chosen on the confusor set (the SAME instrument that
soft-killed it at 0.204<0.266 — does finer122's repr + full train move it past P_chosen this time?),
merge_referee (A held by finer122), net_new_over_champion (decorrelation). Aggregate proxy BANNED >0.89.

**Price:** ~0 build (re-run existing head; two-pass ILP wrapper is the only small new code, and we0z/g89y
already scope it). Train ~1–3h/arm when GPU frees. Gain: the published 0.950 head, on the directional
axis (B) it targets, on the repr (finer122) the prior kill explicitly lacked. **Honest ceiling: this is a
re-test of a SOFT-killed asset under its own prescribed conditions — high-value but NOT a fresh frontier
adoption. If P_true stays flat under finer122 + full train, the "confusor is detection-side, unreachable
by any association head" conclusion hardens (a real result either way).**

## Secondary research notes

- **Competition organism may be zebrafish** (search: "high-resolution videos of zebrafish embryos", Royer
  lab, largest public tracking dataset) — our local proxy movies are Fluo-N3DH-CE C. elegans analogs. If
  the LB data is a different organism than our proxy, that is a **domain gap that could contribute to the
  proxy-LB inversion** already observed empirically. HYPOTHESIS ONLY — verify against the competition data
  card before acting; do not rebuild the proxy on one search snippet. Filed as a question, not a finding.
- Competition metric (repo): edge + division detection, TP/FP/FN, micro-averaged Jaccard, combined final
  = matches our adjJ+divJ proxy. Voxel scale 1.625/0.40625/0.40625 µm = our (1,4,4) anisotropy exactly.
- Sparse supervision confirmed (only GT edges backprop; unannotated cells ignored) — consistent with our
  "precision mostly uncounted" finding; identity-precision (the confusor) is the biting axis.
- Adjacent SOTA (lower priority): EmbSAM (SAM + boundary loc, Dec 2025), DELICATE (label enhancement,
  Nov 2024), ASR displacement-vector-field segmentation (F1 0.8956, 116 C.elegans stacks) — all
  SEGMENTATION-side; we are detection-fine (node R 0.98), so low leverage.
</content>
</invoke>
