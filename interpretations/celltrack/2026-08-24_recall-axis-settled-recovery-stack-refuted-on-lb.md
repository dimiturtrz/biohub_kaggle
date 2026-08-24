# The recall axis is settled — the recovery stack does not clear the bar

**2026-08-24 · three Kaggle submissions (55725428, 55725982, 55726521) vs the 0.900 champion**
**Verdict:** recall-via-recovery-stack REFUTED on the LB. Recall axis exhausted for the 0.910 goal.

## The question

The champion is `thr0.80` dual-seed flow at **0.900** public. Memory
[[celltrack-0027-carrier-is-deepcenter-recovery-stack]] read the public 0.915 pack as a **third CC0
recovery stack** (gap-close bridge + DeepCenter-confirmed synthetic insert + division forks) mounted on a
**high-threshold** base — i.e. the +0.027 over a naive high-thr detector was a *recall carrier*, recovered
links a conservative base drops. That predicted the real lever was a **coupled** high-thr detector +
recovery stack, and that a recovery stack on the *low*-thr champion would be inert (the links are already
there). Three submissions were designed to separate those.

## What was measured

| Sub | Mechanism | Public |
|-----|-----------|--------|
| `55725428` ArmA | champion **thr0.80** + two-frame density bridge (gap2) | **0.900** |
| `55725982` Opt2 | **thr0.99** dual-seed flow + density-gated gap2 bridge, **no** DeepCenter | 0.895 |
| `55726521` Opt3 | **thr0.99** + **FULL** recovery (gap2 + DeepCenter synth-insert + div forks) | 0.898 |

## What it means

The prediction split cleanly and **the wrong half won**:

- **Low-thr + recovery = tie (0.900).** ArmA is exactly champion — the gap2 bridge on a base that already
  has the links adds nothing. Predicted, confirmed. Recovery is inert on a high-recall base.
- **High-thr + recovery = LOSS (0.898 < 0.900).** The coupled path memory predicted as *the lever*
  **regresses**. The full DeepCenter-confirmed stack on a thr0.99 base recovers less than the thr0.80 base
  already had — it does not reconstruct the champion's recall, let alone exceed it.
- **DeepCenter confirm is not free recall.** Opt2 (0.895) → Opt3 (0.898): adding the full DeepCenter
  synth-insert + div forks buys +0.003 over the bare density bridge, still −0.002 under champion. The
  confirmer trims false inserts but cannot manufacture the true links a low threshold kept for free.

So the +0.027 carrier is **not** portable as "high-thr + recovery". Whatever the public 0.915 pack does, it
is not this recovery stack on a high-thr base — that configuration underperforms the simplest champion.

## What it kills, what survives

**Killed:** recall via gap-bridge / recovery-stack, on either base. High-thr+recovery loses, low-thr+recovery
ties. No configuration in this family clears 0.900, none approaches 0.910. The recall axis is spent.

**Also dead (same session, GT-free):** the consensus multi-config ensemble. `consensus_linker` union-Jaccard
over 8 config subsets sits at 0.9079–0.9093 vs champion ~0.9082 — a **+0.001 sub-noise** separation. The
union adds ~12–15k links but they are overwhelmingly agreement, not champion-missing recall. Not slot-worthy.

**Survives:** the gap is not recall, it is **association discrimination** — the confusor picks the nearer
wrong rival over the true successor sitting in a consistent drift direction
([[celltrack-relative-pe-radial-confusor-directional]]). That is a *precision* defect on links the base
already proposes, orthogonal to everything measured here. The live independent mechanism is the
**directional relative-PE** (`DirectionalAttentionBias`, committed `282bda5`): a per-head term over the
offset *direction* the shipped radial `|Δ|` bias is structurally blind to. Built, gated
(in-loop `mean_p_true` + post-hoc `DenseDiagnosis` inverted-fraction), queued behind the shared card.

## Cost

Three LB slots (recall axis) + one CPU consensus run. The three subs were the honest price of retiring a
primary hypothesis — recall looked like the cheapest path to 0.910 and it is now closed with data, not
argument. No further recall submissions justified.
