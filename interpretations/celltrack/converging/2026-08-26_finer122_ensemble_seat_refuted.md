# finer122 ensemble seat — REFUTED by direct sufficiency test

2026-08-26. Two convergent verdicts from the finer122+ILP GPU session (b101ac) close the finer122
ensemble-seat thesis. My decorrelation instrument (edge-agreement between finer122 and the 0.900-tying
champion configs) was killed as **moot** before finishing — agreement is necessary-not-sufficient for a
seat, and sufficiency was tested directly.

## The two numbers (8-movie CV, faithful ruler)
- **finer122_ilp standalone = 0.6555** vs best single `shipped_thr097` = 0.9113 → **−0.256**, well below
  the 0.89 proxy-inversion line = a real crash, not noise.
- **consensus subsets all regress**: vote≥4 = 0.9092 (−0.0021); vote≥2..7 = −0.0035 to −0.0044. None beats
  the best single member.

## Mechanism (corrected — audit caught a false alarm)
The `ilp` linker DOES price affinity (`ILPLinker` cost = `distance − affinity_bonus·P`,
`ilp_linking.py:44,107`; builder passes `affinity_bonus=20`, `linkers.py:538`). The
`linkers.py:334` "ilp ignores mounted affinity" log line is a **false alarm** — it fires on
`_BONUS_READERS` membership but the builder wires the bonus regardless. So 0.6555 is an **affinity-QUALITY
crash at the finer grid (bottleneck B)**, not linker blindness: finer122 un-merges detection (bottleneck A,
real) but its association head is too weak at 8× node density to link the denser nodes (P_true 0.078).
`finer122_joint` is therefore **not** a fresh lever — it hits the same association wall.

## Why the seat is dead regardless of agreement
Decorrelation licenses a seat only if the seat ADDS score. b101ac mounted the members directly
(consensus_detached) and every subset subtracts. vote≥2 net_new_vs_champion = 15908 links, but unionJ
0.9078 < 0.9113 and the union is not submittable — the **same pattern that LB-refuted every recovery-stack
arm**: big proxy-blind net-new, scores ≤ base. A member scoring 0.6555 standalone that regresses every
consensus subset is not mountable at any edge-agreement value. Running my decorr to completion would have
spent 3h more CPU on a number that cannot move the verdict.

## Forward
The pool is one flow-tracker recipe space → saturated. The next member must come from **outside** that
space = the bottleneck-B build (directional line-to-line association head at finer grid) — the hard open
problem, gated on card + a champion LB, not another thr/flow variant and not a plumbing fix. The finer122
recall axis is real but harvestable only once association is solved at finer density. Replication path
stays dead (0.902, no code home). Banked LB = 0.900; no TRIED mechanism demonstrated-by-us >0.910.
