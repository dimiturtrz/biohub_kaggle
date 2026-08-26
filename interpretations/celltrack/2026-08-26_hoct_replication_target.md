# HOCT replication target — the edge architecture is already built; the gap is training + substrate + solver

2026-08-26. Synthesis of the HOCT replication target (arxiv 2607.11754, Bragantini/Theodoro/Royer, #1 CTC
Linking on Fluo-N3DH-CE: CLB 0.950 / LNK 0.986 / BIO 0.913). Written after (a) fetching the paper method +
ablation and (b) a line-by-line audit of our own `celltrack/models/hoct_edge_transformer.py`. It supersedes
the framing that HOCT is a from-paper build. **It is not — the edge architecture already exists in the repo
and is faithful. The 0.7234 soft-kill was harness (training/substrate/solver), not a refutation of the
method.**

## The paper mechanism (verbatim from arxiv 2607.11754v1)

Two-stage association head feeding an ILP:

1. **3D-RoPE node self-attention.** `Ln=4` layers per frame; rotary PE over `(z,y,x)`, learnable per-head
   frequencies. QK reads relative position (⇒ velocity across frames) natively.
2. **Edge-to-edge self-attention.** Each in-gate candidate edge is a token, `h_e^(0)=MLP([z_i‖z_j])` (Eq.1);
   `Le=4` layers; edges attend to edges (higher-order). Geometric bias Eq.3
   `A_h(m,n) = q̂·k̂/√D + σ_h·softplus(α_h)·d_line(e_m,e_n)`, `σ_h∈{+1,−1}` alternating across heads
   (attractive/repulsive), `α_h` learnable init≈0, `d_line` = min Euclidean distance between the two 3D edge
   segments.
3. **Parental softmax** (Eq.4) `p_e = exp(ℓ_e)/(1 + Σ_{e'∈C(j,Δt)} exp(ℓ_e'))` — the `1` is the implicit
   no-parent / variable-appearance option.
4. **Two-pass tracklet solver.** Pass1: solve ILP on Δt=1 edges → high-confidence tracklets. Pass2: collapse
   each tracklet to a meta-node, re-solve over all edges (long links); `λ=0.5` temporal decay.

## The ablation decides what carries it (Table 2, CTC avg splits — the load-bearing finding)

| edge-stage variant | CLB | LNK | BIO |
|---|---|---|---|
| FAGCN | 0.909 | 0.981 | 0.837 |
| GAT | 0.916 | 0.982 | 0.851 |
| **Edge-Transformer (global attn, NO geometric bias)** | **0.922** | 0.982 | 0.862 |
| **HOCT (full, σ-bias)** | **0.926** | 0.983 | 0.867 |

- **The edge-to-edge global self-attention is the primary carrier** (~+0.6–1.3% CLB, GNN→Edge-Transformer).
  The paper's own reason: the candidate line-graph is **non-homophilic** (adjusted homophily ≈0.01) — an
  edge's true neighbours are NOT its graph-adjacent edges, so message-passing GNNs plateau; global edge-to-edge
  attention is what beats them. This is precisely why our node-only `SimpleNodeTransformer` caps out.
- **The σ-alternating line-to-line bias is a small topper** (~+0.4% CLB, 0.922→0.926). Real but minor. Do NOT
  build it first thinking it's the carrier.
- **The edge stage caps at 0.926, not 0.950.** The published 0.950 (Table 3 official submission) is another
  ~2.4% from the FULL pipeline: their detector + finer decode + the SOLVER. Table 1 (ILP): long links +
  **two-pass tracklet solver** + **variable-appearance** all necessary → 0.920; variable-appearance alone
  ~+0.5% CLB. The two attention blocks ALONE will not reach 0.950.

## What we already have (audited line-by-line: `celltrack/models/hoct_edge_transformer.py`)

The full faithful edge architecture is BUILT:
- `_Rope3D` :56-90 — 3D RoPE, learnable `log_freqs`; applied to Q,K at `_NodeAttentionBlock` :120.
- `_EdgeAttentionBlock` :129-166 — edge-to-edge self-attn `einsum("hid,hjd->hij", q, k)` :159; edges are
  tokens (`edge_in` :303 = `MLP([node_t[src]‖node_t1[tgt]])`).
- σ-alternating bias :140-142 (`torch.where(arange%2==0,1,-1)`, `softplus(α)`), applied :161
  `bias = sign·softplus(α)·dline`; `dline` = `_segment_distance` :168-194 (Ericson clamped line-to-line).
- Candidate-edge gate :295 (physical `gate_um=12`) — the frontier's motion-gate for O(E²) tractability.
- Drop-in for `SimpleNodeTransformer`: same 6-arg forward, same `.proj`, same `(…,N_t,N_t1)` logit shape —
  mounts at the three existing wiring sites, `head="hoct"` in `joint_config.py`.

## What we lack (the real gap — three pieces, none of them the edge attention)

1. **A properly-trained checkpoint.** The 0.7234 soft-kill (`joint_hoct_gn_v2.pt`) trained to epoch **0.5/3**
   — grossly undertrained — on the **(1,4,4)** repr with our detector. Classic faulty-TRAINING + faulty-
   SUBSTRATE, not a method refutation. (`interpretations/celltrack/converging/2026-08-25_hoct-is-the-frontier-board-reorder.md`:44-53.)
2. **Two-pass tracklet ILP + variable-appearance.** `celltrack/linkers/ilp_linking.py` is single-pass (grep
   `tracklet|two.pass|meta.node` = 0 hits). Table 1 says this is separately necessary. Solver-side, not
   head-side — a wrapper over motile.
3. **Finer decode / their detector.** The 0.926→0.950 bulk. Our substrate is (1,4,4); finer122 (1,2,2) is the
   co-adapt substrate.

Parental softmax (Eq.4) exists as `SlackRow` in `celltrack/losses/association_objective.py` but is UNTESTED
on the edge-to-edge head (0.7234 used focal γ3.5/div3.5). Its faithful NO-GO ([[celltrack-slackrow-faithful-nogo]])
was on node-only repr — harness-suspect, re-test on the edge substrate, don't carry the refutation over.

## Reconciling the "B FLAT on finer122" prior

[[celltrack-hoct-retest-on-finer122-not-new]] recorded a finer122 HOCT co-adapt with "A HOLDS, B FLAT (finer
repr alone doesn't close confusor)." This is NOT a refutation of the stack: (a) it lacked the two-pass tracklet
ILP (Table 1 necessary); (b) likely under-trained per the same recipe. The ablation says the confusor-closing
lift is DISTRIBUTED across edge-attn + two-pass-solver + finer-decode — any single piece in isolation can read
FLAT. The verdict must be on the STACK, not a component. This is the standing lesson: a well-argued idea that
reads flat implicates the harness (training/substrate/solver) before the idea.

## Build order (by ablation-carried value) and the gate

1. **Train the existing `hoct` head to convergence on finer122 (1,2,2)** — the built arch × the new substrate,
   full epochs not 0.5/3. Hours, not days.
2. **Two-pass tracklet ILP wrapper over motile** + variable-appearance cost — the one genuinely-missing
   architectural piece; separate primary per Table 1.
3. **σ-bias / RoPE already present** — verify they're active (α not collapsed to 0 after real training).
4. **Parental softmax on the edge head** — re-test, cheap, don't assume dead.
5. Finer decode / detector quality — the 0.926→0.950 bulk, largest remaining.

**GATE (SOTA-replicate rule):** reproduce the edge-stage **0.926** first — that's how we learn what carries it
— BEFORE merging our banked 0.924 global ILP + DeepCenter recovery. Do not submit on a proxy number (inverts
above 0.89); gate the submission on a GT-free fire-check + a mechanism clearing 0.924. Owner "don't be hasty
with submissions" is a hard gate.

## Status

Banked best = **0.924** (global ILP linker, [[celltrack-linker-is-the-gap]]). WIN goal (first place,
0.945–0.953) open; bar to advance = 0.924. HOCT is the top model-replication lever
([[celltrack-true-bottleneck-is-dense-regime-model-gap]]); the reframe is that ~90% of it is already built and
the remaining gap is training + finer substrate + two-pass solver, not a from-paper architecture build.
[[celltrack-hoct-replicate-target]] [[celltrack-hoct-association-head-refuted-detection-side]]
[[celltrack-frontier-gather-names-rope-and-parental-softmax]]
