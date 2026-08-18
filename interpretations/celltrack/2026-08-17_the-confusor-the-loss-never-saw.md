# The confusor the loss never saw

*2026-08-17 — from-scratch edge head, why it mislinks and the one fix that is mechanism-correct*

## The number

From-scratch edge head fits the sanity check perfectly (AUC 1.0) yet on real crowded
frames `inverted_fraction ~0.98–0.99`: the affinity itself ranks a confident wrong
near-neighbour above the true, often far-moved, successor. This is pre-linker — the
representation, not the solver.

Same-ruler proxy ladder:

- joint-from-scratch (`joint_scratch_v1`): **0.697** plateau, node R 0.882 (detection converged)
- synth-pretrain → real (`joint_ptsynth_real_v1`): **0.7213**, +0.0246 (clears 0.02 floor, under 1.5% bar — directional)
- pilkwang warm, zero training: **0.847** (LB 0.892–0.899)

Detection is not the cap (node R 0.882 near ceiling 0.906). Linking is, and linking
tracks the pretrained representation.

## The mechanism

`celltrack/data/synthetic_pairs.py` names the root cause: on the dense movie **every
mislinked partner is an UNANNOTATED detection**. With ~1–2% of cells annotated, the
detections that actually beat us never enter a batch as negatives. Sparse real
annotation *cannot supply the confusors*. The head is trained on a candidate set that
excludes exactly the cases it loses on — so a wrong link costs no gradient, and the
head grows confidently wrong (P_chosen 0.44 ≫ P_true 0.11).

**The scarcity, measured (2026-08-17).** The joint loader logs the training corpus directly:
`with a rival 2.6%` — only **2.6% of GT training pairs have a competing candidate in-gate**.
The head sees an *uncontested* choice 97.4% of the time. This is the confusor scarcity as a
single number, sharper than "~1–2% of cells annotated": the gradient that would teach
discrimination is present on 1 pair in 40. More epochs on this corpus re-present the same
2.6% — budget cannot manufacture contest. A complete-candidate synthetic corpus raises the
contest rate to ~100% by construction; that is precisely what it buys.

## Two dead ends (paid for, recorded)

- **Hard-negative mining on real** (`--hard-negative-weight`): puts gradient on cases
  already won by ~4 logits. Reproduced this session (`joint_hardneg_v1`): proxy sank
  0.7213 → 0.6919 → 0.6833, inv rose to 0.99, P_chosen rose to 0.78. The margins were
  already met in training; the loss term was ~0.005. The confusors aren't hard *examples
  in the set* — they're **absent from the set**. Mining the present ones can't help.
- **gpu-scene for the motion lever** (`gpu_scene.py`): 2-frame pairs, no t-1 →
  `PairSample` carries no velocity → structurally cannot supply `--prior-velocity`. Fine
  for detection robustness; wrong tool for the confusor lever.

## The fix is a co-dependent pair, not a ladder

1. **`--synthetic-fraction <1.0`** — the static `synthetic_pairs` corpus (6-frame). Every
   cell of a synthetic frame is labelled → the candidate set is complete by
   construction → the crowded near-neighbours ARE in the edge matrix as true negatives.
   This is the only thing that puts the real confusors into the loss. Mix, never
   synth-only (synth-only under a frozen backbone + no det loss DAMAGED it 0.9273 →
   0.9058).
2. **`--prior-velocity`** — feeds each source its t-1→t displacement so the head predicts
   WHERE the successor lands instead of picking nearest. A motion-blind head (velocity
   OFF across the ENTIRE from-scratch chain) can't learn to follow motion no matter the
   supervision.

The isolation probe (`joint_priorvel_v1`, velocity alone on real) sits flat at 0.6967,
inv 0.95 — the input helps ranking a little but can't move the proxy, because real
annotation lacks the fast-mover tail the head would learn the motion prior from. The
6-frame synthetic corpus supplies both the tail AND makes velocity computable. **The two
levers converge on the one corpus** — that convergence is the finding, not either lever
alone.

Decisive arm: `--init-weights joint_scratch_v1.pt --synthetic-fraction 0.4
--prior-velocity`, under joint_detector's unfrozen backbone + detection loss. Watch inv:
a sharp drop is the confusor+velocity pair landing.

See also `2026-08-16_from-scratch-ceiling-resolved-lever-is-pretraining.md` and
`2026-08-13_motion-in-model-and-solver.md`.
