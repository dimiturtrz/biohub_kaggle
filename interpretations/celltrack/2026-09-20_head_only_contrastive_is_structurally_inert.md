# The nce flag was inert in BOTH arms — what separated them was the detector, before training started

2026-09-20. Arm `hoct_finer122_nce` (1.86h, early stop, best proxy 0.6239, gate 0.6877 — FAILED).
This overturns the 2026-08-28 finding it was built to act on.

## What was believed

2026-08-28 found that the entire finer122 head family trained with `--contrastive-weight` /
`--hard-negative-weight` omitted, which `LossCfg` defaults to 0.0 — a dark term. avl8 (same joint
architecture, nce ON, pilkwang detector) reached proxy 0.79 while every finer122 head plateaued ~0.61. The
conclusion drawn was that **the contrastive term was the difference**, that every finer122 head verdict "rode a
crippled objective", and that the never-run cell was finer122 + nce ON. This arm ran that cell.

It failed. But the diagnosis is not "the cell is dead" — it is that **the comparison that motivated it was never
a comparison.**

## The nce term cannot train anything in EITHER arm

The weights landed (`train loss 2.5167 (edge 0.1093 det 0.0972 nce 1.5399 hn 0.0252)`), yet across **72,096
steps** `nce` moved 1.5399 -> 1.5496 while `edge` swung 0.065-0.18. The code says why:

| site | fact |
|---|---|
| `joint_assembly.py:77` | the `--detector-from` path calls `model.detector.requires_grad_(False)` |
| `joint_config.py:291` | `contrastive_site` defaults to `ContrastiveSite.FEATURES` |
| `contrastive_term.py:59-63` | at `FEATURES` the term builds no projection, so it holds **zero parameters** |
| `info_nce.py:57-62` | it contrasts backbone features directly — "no head in between" |

With the trunk frozen and the term parameter-free, InfoNCE's gradient reaches nothing trainable.
`weight x constant` is a constant.

**And avl8 was frozen too** — `logs/avl8_20260827T013435Z.log`:
`warm detector from ...pilkwang_jm.pt, fresh hoct head (detector frozen)`. Its `nce` likewise barely moved
(2.4670 -> 2.3603). So the contrastive term was inert in the 0.79 arm exactly as in the 0.62 arm. **It cannot be
what separated them.** The 2026-08-28 attribution was wrong: nce-ON versus nce-OFF was never the live variable,
because nce-ON does nothing under a frozen detector.

Nor does moving the site rescue it. `PROJECTION` / `DETACHED_PROJECTION` give the term its own weights, but
those feed an auxiliary embedding the edge head never reads — it would train something and still could not
change a tracked output. **No `--contrastive-site` makes a contrastive objective act on a frozen trunk.**

## What actually separated them: the detector, measured before training

Both runs initialise a **fresh, untrained** HOCT head, so the init proxy prices the detector alone:

| run | detector | **init proxy** | best proxy | gain from all training |
|---|---|---|---|---|
| avl8 | pilkwang_jm | **0.7875** | 0.8054 | **+0.018** |
| today | finer122_coadapt_long | **0.6164** | 0.6239 | **+0.008** |

The entire 0.79-vs-0.62 gap is present **before a single gradient step**. Head training then moves the number
by under 0.02 in the better arm and under 0.01 in the worse one — both at or below the 0.01-0.02 noise floor.

This is the same verdict as [[celltrack-hmlo-replicate-0888-gap-is-DETECTOR]], reached from a new direction:
**the edge head is not a lever on this substrate; the detector sets the score.** finer122's features do not
support association, and no head trained on top of them recovers it.

## A second confound, for the record

The two runs did not even share a training corpus:

| run | corpus | pairs | candidates/frame | in-gate mean | with a rival |
|---|---|---|---|---|---|
| avl8 | detected pairs | 1,895 | 11.7 | 2.19 | **53.1%** |
| today | GT pairs | 18,024 | 7.1 | **1.00** | **2.6%** |

Today's arm trained on GT-sparse data where the median source has **exactly one** in-gate candidate. An
association objective needs something to choose between; at rival-rate 2.6% there is nothing to associate. This
is the documented failure of [[celltrack-pmkf-trains-GT-sparse-not-detected-crowd]], and the missing
`--detected-videos` that [[celltrack-e9ci-untried-but-needs-density-negs]] named as the prerequisite.

So today's run is invalid as a test of the finer122 *head* axis. But the detector conclusion above is drawn
from the **init** proxies, which are taken before any training and are untouched by the corpus choice.

## The head anti-learns, which is consistent

| window | proxy | inv | P_true | P_chosen | top1 |
|---|---|---|---|---|---|
| init | 0.6164 | 0.61 | 0.209 | 0.264 | 0.259 |
| epoch 1 | **0.6239** | 0.66 | 0.201 | 0.265 | 0.137 |
| epoch 2 | 0.6049 | 0.76 | 0.190 | 0.308 | 0.171 |
| epoch 3 | 0.5859 | 0.86 | 0.145 | 0.388 | 0.116 |
| epoch 4 | 0.5721 | **0.92** | 0.139 | **0.475** | 0.065 |

Inverted fraction climbs to 0.92 and `P_chosen` nearly doubles while `P_true` falls — the head grows
monotonically *more confident in the wrong edge*. Training on a corpus with no rivals teaches it to be
confident, not to be right; best score is epoch 1 and everything after is decline.

Also note the pre-registered gate's second half was never discriminating: `top1 >> 0.012` was satisfied **at
init** (0.259, untrained). A gate half a random head passes measures nothing.

## The inertness is scoped to the head-only family — and freezing is the real cost

A sweep of every run log for `detector frozen` returns **six** runs, ever: `avl8`, `avl8hnfix`,
`finer122_hoct_headonly`, `hoct_real_converge`, `jointarm_cheapfirstcut`, and today's. Everything else trains
the trunk, where the contrastive term is live. So [[celltrack-nce-lifts-joint-hn-knife-edge]] ("nce lifts
joint 0.61 -> 0.87") **holds** — its arms were joint. The dead flag is a property of head-only runs, which is
precisely the family every finer122 head verdict belongs to.

That comparison prices freezing itself:

| construction | best proxy |
|---|---|
| frozen head-only, good detector (avl8) | 0.8024 |
| joint (cellect_long / detcorpus_dw0 / confB_ctrl2) | 0.8475 / 0.8567 / **0.8630** |

**Freezing costs ~0.05 — more than any head choice has ever bought (+0.018).** The head-only construction was
adopted to make arms cheap, and it caps the number below where the joint runs already sit.

## Verdict and what NOT to re-queue

CLOSED — "finer122 + nce ON" as a distinct cell. It is not distinct: nce ON and nce OFF are the same run under
a frozen trunk. Every finer122 head verdict the 2026-08-28 note invalidated is hereby **re-validated** —
`rzvw` (0.6187), coadapt faithful (0.779), the HOCT head-bound referee — none of them rode a crippled
objective, because avl8 rode the identical one to 0.79.

NOT worth re-queuing — the same arm with `--detected-videos`. It would fix the corpus confound, but it is
priced out: avl8, with a *good* detector and the right corpus, bought **+0.018** over its init. finer122 starts
at 0.6164; the same gain lands ~0.635 against a 0.6877 gate. ~2h of GPU for a mechanism whose own best case
falls 0.05 short.

STILL OPEN — the JOINT ARM's coupling was never tested, and now has no cheap gate. Any real test must unfreeze
the trunk, because that is the only construction in which the contrastive half exists at all.

## How to apply

Before attributing an A/B gap to the variable you changed, check the **init** numbers: if the arms already
differ before training, the variable you named is not the one that moved it. Here the whole gap was in the
initialisation, and a month of head-axis reasoning was spent on a flag that could not fire.

And before pricing an ablation, ask **which parameters the term's gradient can reach**. A loss can be weighted,
logged, non-zero and wholly inert. "The flag is on" is not "the mechanism fired" — the check is free, static,
and would have saved both this run and the verdicts it wrongly overturned.
