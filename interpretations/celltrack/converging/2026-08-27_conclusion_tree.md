# Conclusion tree — the one living synthesis (supersedes scattered per-run memories)

2026-08-27. **Purpose:** the memory base exploded to ~80 one-off leaves, each an experiment, no synthesis
layer. This is the tree: observations → conclusions → the remaining levers, per config axis. When a new
result lands, fold it into the matching node here and update the leaf verdict — do NOT spawn another
isolated memory. Per-run detail stays in `interpretations/celltrack/<date>_*.md` + git history; this doc is
the index that decides *what to run next*.

Verdict tags: **BANK** (shipped/true) · **REFUTED** (killed on real board or sound mechanism) ·
**HARNESS** (killed but the kill was our setup — undertrained / wrong grid / wrong ruler; re-testable) ·
**AUDIT-FLAT** (mechanism-bounded before building) · **OPEN** (live, un-run or un-scored).

---

## NEWS 2026-09-22 14:04Z — every finer-grid refutation was read through a threshold that is wrong by construction

Three warm arms at `--downsample 1 2 2` all collapsed the same way inside one 500-step window: node count
5–10× while node recall stayed flat. Arm 2 (`--det-weight 0.1`) refuted the gradient-magnitude explanation —
raw det loss still 9.1223, ratio still +7.95. Arm 3 (`--lora`, pack base frozen **including BN affine**)
refuted the BN-affine explanation — counts 16946/34685/28426/33387 → 177984/156555/173923/191878. Three
freeze configurations, one signature. That rules out the training and points at the read-out.

Neither the target nor the decode is buggy, and that is the finding:

- `targets.py:50` — `sigma = spacing.anisotropic_radius(scale_um)`. A **physical** radius, so at (1,2,2) the
  Gaussian is 2× wider in y/x and covers **4× the in-plane positive voxels**. Correct, and self-consistent.
- `peaks.py` — `radius = rint(2.0 * spacing.anisotropic_radius(scale_um)) // 2`. The NMS window uses the
  *same* physical conversion. Also correct.
- `tunet.py:186` — `_logit_floor(threshold)` turns a **fixed probability constant** into the floor, **with no
  grid term at all.**

So a head trained at (1,2,2) learns a target whose positive fraction is 4× higher, its output prior
recalibrates upward to match, and the same physical confidence then clears a constant fitted to a (1,4,4)
head in 4× more voxels. Flat recall, exploding count, immune to loss weighting, immune to freezing the base —
every symptom follows, and none of them is about the grid being hard.

Swept on arm 1's trained state (`finer122_warm_coadapt.resume.pt`) via `celltrack.eval.joint_eval
--threshold`, **no training at all**:

| threshold | faithful | node recall | ratio |
|---|---|---|---|
| 0.96875 (shipped) | 0.3598 | 0.9745 | +2.700 |
| 0.99 | 0.3597 | 0.9731 | +2.519 |
| **0.999** | **0.5241** | 0.8605 | **+0.382** |
| 0.9999 | 0.0915 | 0.0921 | −0.960 |

**+0.164 from one constant.** At 0.999 the density ratio lands at +0.382 — init's +0.42 — which is the
prediction the mechanism makes, not a fitted point. Note also that the trained head's node recall at the
shipped threshold is **0.9745 against init's 0.948**: the finer grid *improves* cell-finding, which is the
axis the tree says we need. What it costs is peaks, and peaks are a threshold question.

**SETTLED 14:11Z — the optimum is 0.9985 and the seat still loses to not training.** The refined sweep on the
same trained state closes the bracket:

| threshold | faithful | node recall | ratio |
|---|---|---|---|
| 0.995 | 0.3675 | 0.9722 | +1.984 |
| 0.997 | 0.4364 | 0.9736 | +1.630 |
| **0.9985** | **0.5303** | 0.9412 | **+0.868** |
| 0.999 | 0.5241 | 0.8605 | +0.382 |
| 0.9995 | 0.3866 | 0.5499 | −0.410 |

A clean single-peaked curve, **0.5303 at 0.9985**, cliff on both sides.

**RETRACTED 14:44Z — that was read off three 500-step arms with a converged checkpoint of the same recipe
sitting on disk.** Sweeping `finer122_coadapt_long.resume.pt` (the converged co-adapt referee arm, also never
read at a correct threshold) gives 0.6442 @0.96875 → 0.6530 @0.98 → 0.6917 @0.99 → 0.7613 @0.995 → **0.7935
@0.997** → 0.7563 @0.9985 → 0.6889 @0.999 → 0.5186 @0.9995. That **beats the untrained graft**, so warm
co-adapt at (1,2,2) *does* beat not
training once converged and read correctly. The 500-step arms' 0.53 is a **step count, not a recipe verdict** —
the sentence this block originally carried ("the training makes the seat worse") was one sweep from being
written into the tree as a finding.

The mechanism survives that retraction and is in fact confirmed twice over, on the **asymmetry**: the converged
arm's optimum moves (0.96875 → ≥0.9985) because a head trained at (1,2,2) sees the 4×-denser target and
recalibrates, while the untrained graft's optimum **stays at the shipped 0.96875** (0.7267 → 0.7179 @0.975 →
0.7160 @0.985 → 0.7122 @0.99, monotone falling) because it is still a (1,4,4)-calibrated head. Predicted, not
fitted. *(Loose end, not to be conflated: the graft reads 0.7267 where 0.6408 was recorded at the same
threshold. Different measurement, not a threshold effect; unexplained.)*

The peak carries its own corroboration: at 0.997 the density ratio is **+0.026**, i.e. the decoded node count
essentially *matches GT*. The principled operating point and the empirical optimum coincide, which is what
distinguishes a corrected constant from a tuned one. On the same weights the shipped constant sits at ratio
+0.528. Sibling read: `hoct_finer122_nce` peaks at **0.6960 @0.99**, *below* plain converged co-adapt — HOCT
does not help at (1,2,2).

**Net for the day on this seat, zero training, weights that already existed: recorded 0.6408 → 0.7935.**

**What this does NOT claim.** The best finer point is **0.7935** against a champion at **~0.9375 on this same
proxy** — still ~0.14 short, and no local arm closes 0.14 in seven days. The direction changed; the gap did not. Fixing the threshold makes the seat
*measurable* for the first time; it does not make it competitive, and nothing here is a submission path. The
claim is narrower and stronger than a win:
**the finer seat has never once been measured on its own operating point**, so none of its recorded kills
mean what they say. Standing rule earned the hard way — [[celltrack-constant-audit-faithful-maxgap2-lever]],
[[celltrack-donor-constants-dont-transplant-without-their-component]] — now has a third instance: **a
constant fitted at one grid is not a constant at another, and reading an arm through it is a harness fault,
not a result.**

---

## NEWS 2026-09-22 13:52Z — the public ceiling is BELOW us, and the LB top is not reachable from any kernel

The LB moved while the loop optimised knobs. Top 12 on 2026-09-22: **0.974 / 0.973 / 0.970 / 0.968 / 0.968 /
0.967 / 0.967 / 0.967 / 0.966 / 0.966 / 0.964 / 0.964**, eight of them submitted **today**. That is not one
outlier to dismiss as a metric hack — it is a crowded tier **0.017–0.027 above our banked 0.947**.

Two kernels were missing from the 41-entry harvest inventory, both by `amanatar`, both fresh:

| kernel | votes | last run | vs our 0947 | its own comment |
|---|---|---|---|---|
| `biohub-geometric-fusion` | 100 | 2026-09-21 | 111 vs 113 keys; **3** new elements, 1 constant delta | `LB 0.913` |
| `biohub-metric-hack-last-call` | 25 | 2026-09-22 | + 11 keys on the above | `LB 0.913` |

`geometric-fusion` — the **most-voted kernel in the competition**, posted yesterday — differences to our
champion at `BIOHUB_LEAF_PRUNE_MIN_EDGE_PROB=0.0`, `BIOHUB_PPSWEEP_EXTENDED=1`,
`BIOHUB_PPSWEEP_PREFIX_GUARD=1`, and `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.20 vs 0.25. Everything else is
artifact-path plumbing. This is the **fourth** independent confirmation of
[[celltrack-e60-frontier-kernels-are-our-champion-downgraded]].

And the hack is not readable either: `metric-hack-last-call` adds 11 keys of prune/repair plumbing with
**every actuating value shipped at `0.0`** — `SEG_PRUNE_MIN_PROB`, `OUTPUT_MIN_EDGE_PROB`,
`LINEFIT_MAX_SHIFT_UM`, `COUNT_EXCESS_FRAC`, `MOTION_RELINK_VEL_EMA`, `REPAIR_PARENT_MAX_UM`,
`GAP_CLOSE_DIV_UM` all zero. The author published the wiring and withheld the constants.

**What this settles.** The public kernel axis is **closed end to end**: the best public work sits at 0.913
declared / 0.947 realised, we are at the top of it, and no published element remains unharvested. The
0.964+ tier comes from something nobody has released. So the remaining week cannot be spent reading kernels,
and it cannot be spent on post-processing either — prune (E47/48/52/53), gap repair (E55), division (E61),
selection (E43/44), gate (E57) and edge deletion (E58) are each priced shut with a ceiling. **The only axis
this tree still lists as open and sized for 0.02+ is the dense-regime detector/model gap** — finer decode
plus centre-offset, the pair the 0.945 argument calls jointly necessary. The entry below is the first
un-confounded measurement on that seat in the project's history, and it exists because two recorded
"refutations" turned out to be from-scratch runs.

Floor is handled and needs no further work: final-2 stays `celltrack-public-0947` + `celltrack-public-v1329f`
(E67 measured the decorrelation, 0.9299 node Jaccard at 7 µm). Submissions are not the scarce resource — 5/day
against 7 days is 35, and the final-2 pick is already argued. **Card time is the scarce resource, and it
belongs on the model axis.**

---

## NEWS 2026-09-22 13:29Z — the finer-grid ceiling was the HEAD, and an untrained warm init proves it in 0.27h

The finer-decode seat had two converged numbers against it — `finer122_coadapt_long` **0.6150** (3.8h, early
stopped at 4.33 epochs) and `finer122_coadapt_v3` **0.5881** (1.70h) — and both were read as "finer decode
costs association". Both logs open with `init proxy 0.0000 | node R 0.000 | 0 nodes`. **They were from
scratch, both.** The warm form (bd `cvsb`) had never run, and `joint_assembly.py:185` explains why it is
legal: the grid guard is on `--detector-from` only; the `--warm-start` branch hands the pack detector *and*
transformer to `JointModel(.., downsample)` with no check at all.

Run it, and the first eval is the whole finding:

```
init proxy 0.6408 (sel 0.6408) | node R 0.948 ratio +0.42 | mislinks 176 | sanity AUC 1.0000
```

**The pilkwang head, untrained, decoded on (1,2,2), scores above 3.8h of from-scratch co-adaptation.** The
finer grid never broke the pack head. What those two runs measured was a from-scratch head too weak to learn
association at 8× candidate density — a head-strength ceiling wearing a grid-cost costume.

### the training then destroyed it, and the mechanism is physical

| step | det loss | node R | ratio | proxy |
|---|---|---|---|---|
| init | — | 0.948 | **+0.42** | **0.6408** |
| 500 | **9.2000** | 0.947 | +7.79 | 0.0820 |
| 1000 | 0.3948 | 0.949 | +8.06 | 0.0659 |
| 2500 | 0.1387 | 0.950 | +8.13 | 0.0808 |

One 500-step window takes it to **8× over-detection**, and it never returns: the loss recovers (9.2 → 0.14)
while the density stays pinned at +8. Edge loss is tiny and flat (0.0048) the whole way — association is not
what is failing.

The from-scratch runs converge to ratio **+1.18 / +0.99 against the same target**, so the target is sound.
Detection was *already good* at init (ratio +0.42), so what the training does is destroy a working head.

**CORRECTION 13:52Z — the gradient-magnitude story above is refuted by its own follow-up.** The first
explanation was geometric: a head trained at (1,4,4) paints a blob of fixed *physical* size across 4× the
voxels at (1,2,2), the per-voxel balanced BCE is enormous on contact (det loss 9.2), and `lr × 9.2` wrecks
the head in one window. Arm 2 damped exactly that term — `--det-weight 0.1` — and the collapse is
**identical**: raw det loss still **9.1223** at window 1, ratio still **+7.95**, proxy declining monotonically
0.0770 → 0.0683 → 0.0594 → 0.0540 with `best` stuck at the init value. Ten times less gradient, same
explosion. Magnitude is not the binding mechanism.

What the numbers actually say: **node recall sits flat at 0.91–0.95 while the node count goes 8×.** That is
not a learned change, it is a **threshold shift** — the head still finds the same cells, it just fires far
more widely. A channel-wise scale is the shortest path to that signature, and the code names the channel-wise
scale that is still free to move: `freeze_backbone_norm` suppresses the running-stat update and the
batch-stat normalisation, but its own docstring (`joint_assembly.py:133-136`) says *"the affine weight/bias
keep `requires_grad` and still train"*. γ/β scale whole channels. So the flag exonerated above is exonerated
only for the **init eval** — it does not freeze the detector during training, and `--det-weight` cannot reach
γ because 10× less gradient still moves it over 500 steps.

Two harness notes worth keeping. `--patience 5` at `--eval-every 500` killed this at step 2500 of 36048 —
**0.07 of a 2-epoch schedule**; the long from-scratch run needed 4.33 epochs to reach its best. And
`--freeze-backbone-norm` is exonerated here: the init eval runs BN in eval mode on the same frozen stats and
returns 0.6408, so frozen BN is not what the training broke.

**Arm 2 ran and was killed at 13:52Z**, refuted on its own pre-registration at window 1 (see the correction
above) and declining monotonically with `best` never leaving the init value. **Arm 3 is up:** `--lora`, which
freezes the pack base *outright* — BN affine included — and trains low-rank adapters only. It is the test
that discriminates the BN-affine hypothesis from everything else, and it is exactly the re-test the frozen
finer graft was already flagged for (`celltrack-finer-graft-frozen-refuted`: "LoRA-retestable"). Same init,
`0.6408`. Pass = ratio holds near +0.4 and proxy climbs off 0.6408, and the finer seat has a live warm arm
for the first time. Fail = the seat closes on a mechanism rather than a confound, which is itself worth the
0.3h with a week to the deadline.

---

## NEWS 2026-09-22 10:48Z — the two-pass ILP lever is two-thirds already-refuted, and the live third was never the plan

The tree names **two-pass tracklet ILP** as one of the three jointly-necessary 0.945 levers, and `wfh5` is the
bead that would build it. Auditing that bead's own build plan before writing a line: pass-2 is specified as
*"solve gap/division edges over the contracted graph"*, and both of those legs are closed already.

- **gap leg — dead by construction.** E55: every GT edge is **dt=1**. A contracted `t-1 → t+1` bridge cannot
  MATCH a GT edge at any weight, so no amount of solver quality makes it score. Not a tuning question.
- **division leg — dead.** E61 + `hq93`: division recall is unreachable from post-processing. `division_cost`
  produced the first non-`fp` movement ever (fn 24 → 21) and still cost −0.0027 / −0.0018 on sets where
  `div_jac = 0.0000` in every arm.
- **dt=1 re-selection leg — LIVE, and the bead never named it.** Contracting a high-confidence tracklet to a
  meta-node changes what a dt=1 association *costs*: a swap that reads as one edge locally gets priced against
  a 60-frame tracklet once contracted. That is a selection change on dt=1 edges, so it is metric-visible — and
  it is exactly where the local signal points (native inverted-fraction 1.00 → 0.61 with selection **flat at
  0.6081**: the discrimination exists and selection does not spend it).

Two things fall out. First, the donor's 0.926 → 0.950 carry leans on multi-frame gaps **its** GT has and ours
does not — the same shape as *donor constants don't transplant without their component*, one level up: a donor
*mechanism* doesn't transplant without the donor's GT topology. Second, the solve is not in anything we
control — `champion_submission.py:1467` only does `IndexedRXGraph.from_geff`, there is no solver call in the
extract — so pass-1 stays library-internal and pass-2 can only act on the output graph.

### and then the third leg closes too, on numbers we already owned

Re-scoping the bead to the dt=1 leg was still one step short. Pulling E58 to write the instrument's counters
is what killed it, on the same corpus E54's bar was measured on:

- **E54 measured wrong-association = 0.** A re-selection pass exists to fix mis-selected edges. There are
  none to fix.
- **E58: of the 79 in-scope broken GT edges, the champion already links 51** between *other* nodes inside the
  same 7 µm rulers. Those are not selection failures — the scorer's Hungarian represents the GT cell with a
  **different real track's** node (its decoys carry 1.84–1.88 edges, 0 of 102 isolated). Contraction changes
  which *edges* are picked; it cannot change which *node* the Hungarian matches. Wrong layer.
- **Ceiling arithmetic: 79 − 51 = 28 edges, against E54's 35-edge bar.** Even a perfect dt=1 re-selection over
  everything that remains misses the 1.5% bar. Same denominator for bar and ceiling, so this comparison is
  legitimate.

**`wfh5` → AUDIT-FLAT, P3.** All three legs priced out before a line was written. And the tree's own lever
list needs the correction: *two-pass tracklet ILP* was carried in as one of three jointly-necessary 0.945
levers **on the donor's authority**, and it does not transplant — their carry needs multi-frame GT gaps and
mis-selected associations, and our corpus has neither (dt=1-only GT, wrong-association = 0). The remaining
0.945 levers are variable-appearance and finer decode; the third seat is empty, not merely untested.

What this does *not* rule out: contraction as a **training-time** signal (tracklet context inside the model,
where it can move localization rather than selection) is untouched by this — the kill is of post-hoc
contraction over a solved graph. Consistent with E58's own verdict that the one live axis is the
center-offset head, which a CPU cannot reach.

Cost of this audit: **zero runs**, and it retired a P1 build plus the blind 3-4h kernel run it would have
opened with.

### removing one seat voids the joint-necessity argument, in both directions

The 0.945 stack was never three toppers: the donor's ablation says two-pass tracklet ILP + variable-appearance
+ finer decode are **jointly necessary**, and we have only ever tested them solo. That framing is what kept
three solo refutations from closing the campaign. Pull one seat out and the argument stops working *as an
argument* — it cannot be silently renumbered to "two remaining levers".

Worth stating what the other two actually are, because the list flattered them:

- **variable-appearance IS SlackRow** (parental softmax `1 + Σexp`), already **REFUTED SOLO** at −0.0107,
  i.e. inside the noise floor. It is not a separate topper at all: its stated mechanism is the *recall-cost
  escape* for a sharp edge head, so it only means anything **coupled** to HOCT.
- That coupling has been tested once — `rzvw`, head-only on a frozen finer122 detector — and it read
  **NO-GO** (train flat at proxy 0.6187, faithful 0.767 against champion 0.921). Head-only/frozen is a
  harness-limited test, so the coupling is *not* refuted outright; it is untested at co-adapt.

Two readings, and honesty requires carrying both. (a) The donor's carry leans on a leg our corpus does not
have, so their 0.945 is not reachable by their route here — the campaign is closed as a *replication*.
(b) Alternatively our corpus is simply *easier* on the solver axis (dt=1-only GT, wrong-association = 0),
which would mean the remaining weight sits on finer decode and the HOCT×slack coupling, not that the ceiling
moved. Nothing measured so far distinguishes these, and the only experiment that would is the co-adapt
coupling arm — which needs the card and a real training budget, neither of which is available or justified
at 7 days out with 0.947 banked.

**Practical consequence: no GPU arm is justified right now, and that is a finding, not an idleness.** The
shipped 0.947 stands, the remaining spend goes to the graded LB pair already in flight and the final-2
selection. The lever list should be read as *empty of cheap moves*, with the co-adapt coupling arm as the
one named, expensive, un-run thing — not as two levers waiting to be picked up.

---

## NEWS 2026-09-22 10:12Z — E67: the hedge's decorrelation is 0.861 node-Jaccard, and it was measured, not argued

Full write-up: [E67](../2026-09-22_hedge_decorrelation_measured.md). Three findings, in the order they came:

- **The hedge was invisible to the settings diff** — `celltrack-public-v1329f.ipynb` is a 68k wrapper that
  base64+gunzips a **175k payload** and shells out. `theirs=0`, guard refused (correctly). Decoded, the diff
  is small: they add only plumbing keys, `SAFE_DIV_THRESHOLD` 0.20 vs **0.25**, and we add
  `DEEPCENTER_TTA` / `SECONDARY_EDGE_FEATURE_TTA` / `VALIDATOR_*` / `PPSWEEP_*`. **Decode a blob kernel
  before concluding anything about it.**
- **The foreign graft is real but guarded off on a quarter of the corpus.** v1329f mounts
  `josephadamski91/biohub-v1327-w3-real-model` with frame-local mean/std alignment and clamp [0.5, 2.0]; a
  retention guard (`minimum_retention 0.9`, count-based) falls back to `untouched_v1290_primary_d4`. Movie
  `44b6_0b24845f`: **98/100 frames fell back**. Other three: 0 fallbacks, median retention 0.987–1.002. Where
  the graft moved the count, it was vetoed; where allowed, it barely moved the count — and a *count* ratio
  near 1.0 does not prove the sets match, so the graft's real share stays unresolved.
- **The overlap number I first computed was my own harness artefact.** Exact-coordinate matching gave node
  J **0.547** / edge J **0.207**, stable under rounding — and wrong, because detections are *displaced*
  (recall 0.824 @2.87µm vs 0.9863 @7µm). With µm tolerance: node J **0.7274 @1µm · 0.8610 @2.87µm ·
  0.9299 @7µm**. The edge 0.207 was the same artefact squared (0.547² ≈ 0.30).

**Net: final-2 stays `0947` + `v1329f`** — a genuinely different detection field (~7% of nodes unshared at
7µm, ~14% at 2.87µm), not a near-duplicate that dies with the champion. But the decorrelation is
**detector + inference stack jointly**, with the graft confounded against the TTA/validator/ppsweep deltas.
**Never measure node overlap on exact coordinates in this project — always a µm tolerance.**

---

## NEWS 2026-09-22 09:33Z — the harvest diff was reading half a kernel, and six E66 constants were fallbacks

Every kernel carries each constant **twice** — an `os.environ[KEY] = value` pin and the default of the
`os.environ.get(KEY, default)` that consumes it — and the pin wins. `tools/kernel_env_diff.py` parsed only
pins. Consequences, both now fixed in `7c6326e`:

- **A donor that ships its method as consumer defaults looked empty.** `biohub-v1-infer` reads ~70 keys
  while assigning 8, and reported `theirs=8`, one delta, "no harvest". Parsing both layers: our own visible
  surface **64 → 113** keys, v1-infer's constant deltas **1 → 31**.
- **E66 quoted six fallbacks as though they were the champion's pins.** `SAFE_DIV_MAX_UM` 4.7 vs the pinned
  **9.0**, sister 7.2 vs **14.0**, symmetry tau 0.0 vs **0.6**, `PP_SELECT_MARGIN` 0.002 vs **0.001**. The
  counters and the in-flight arms are untouched (counters are measured; arms write pins). The mechanism
  gets *stronger*: the geometry gates are already wide, and the champion pinned the deepcenter veto
  **above** its own consumer fallback (0.12 → 0.20), so the 0.15 arm walks back toward the library's value
  instead of inventing one.
- The tool now prints `PIN SHADOWS A DIFFERENT FALLBACK` — **41 keys** in the champion shadow this way.
  Never quote a `get` default as what ran.

**What did NOT change: evg0942 re-run through the fixed parser still shows zero candidate elements.** The
repeated "public kernels are our champion downgraded" verdict (E60, E66) was real, not a tool artifact.
v1-infer's 31 deltas are not donor information either — it speaks our own vocabulary (`SECONDARY_LINK_MODE`,
`GAP2_*`), so it is an *ancestor of ours*, and its constants are our own older settings.

---

## NEWS 2026-09-22 09:22Z — E66: the safe-division axis is GATE-bound not budget-bound, and two graded LB arms are flying

`kaggle kernels output` on the banked champion hands over `run_stats.csv` — every safe-division counter,
per movie. The axis has been argued about from the outside for weeks; this prices it from the inside.
Four test-proxy movies: 730 geometric candidates → deepcenter veto **rejects 414 (57%)** → 316 accepted →
167 symmetry-rejected → 149 candidates → **124 divisions added**, and `safe_division_skipped_cap = 0` on
every movie. The chain reconciles to the digit, so the reading is not a guess.

Two consequences. **The frac caps never bind** (0.008 / 0.004), so a gate loosened upstream reaches the
output — same shape E61 found for `candidacy=steal`, now read off counters instead of inferred from a
score. And **the champion already injects 124 divisions the local proxy prices at nothing** (`div_jac ≈ 0`
there), so this is the one gate where a submission buys information rather than confirming a proxy.

`SAFE_DIV_DIVERGE_UM = 2.25` turns out to be **c3** — the donor's derived physical quantity, already
shipped at its derived value — and its pool is 4172, ten times bigger. Left alone anyway: buying candidates
by lowering a derived constant is the exact knob-twiddle the donor-constant rule forbids. The threshold is
not derived, so it is the gate that may move.

**Sign verified before spending anything, and it was backwards from how I first stated it.**
`deepcenter_accept_repair_point` rejects when `score < threshold` → higher is STRICTER. My staged arm had
been 0.20 → 0.25, described as "loosens the veto"; it would have tightened it. Restaged at 0.15. The donor
`evg0942` ships 0.25 and scores 0.942 to our 0.947 — weak and confounded, but the same direction.

In flight, a graded pair on one pool so the day reads sign AND magnitude: `…-safediv015` (0.20 → 0.15,
admits the band) and `…-dcvetooff` (`VETO=0`, admits all 414). Each differs from the champion in exactly one
setting per `tools.kernel_env_diff`; `PP_CANDIDATES` touches neither key, so the kernel's own sweep cannot
overwrite the pin. Both flat against 0.947 ⇒ the division post-processing block is metric-invisible at any
looseness and the axis closes for good. Detail: `interpretations/celltrack/2026-09-22_safe_division_gate_budget.md`.

Also this tick: the 3-leg pmkf chain's leg1 scored its first **full official** eval — **0.5699** against the
champion's 0.8463 on the same ruler (edge 0.5696, div 0.0035). The detected-crowd recipe is 0.28 below, not
a candidate; the earlier per-leg micro deltas (+0.0988 → +0.0131, collapsing 7.5x) were never going to
close that. The waiter armed on `logs/pmkf_chain_legs.log` was watching a 0-byte file and could never have
fired — the legs log to `logs/pmkf_leg_<ts>.log`; stopped it and read the ledger directly.

---

## NEWS 2026-09-22 09:05Z — E65: the 0.942 public kernel is our 0947 with two constants changed, and the division axis re-closed

Two closures this tick, both by reading before running.

**The `evg0942` harvest is empty, and the emptiness is the finding.** `research/frontier_kernels/evg0942/`
(`biohub-0-942-lb-proxy-score-0-9417`, same author as the already-harvested `0-927-lb`) had sat un-diffed
since 2026-09-20. Parsing every `os.environ["BIOHUB_*"]` out of both and differencing the SETS:

| | count | |
|---|---|---|
| they set, we do not | **0** | no element to harvest |
| both set, different value | **2** | `DEEPCENTER_CHECKPOINT` (a path), `DEEPCENTER_SAFE_DIV_THRESHOLD` ours 0.20 vs theirs **0.25** |
| we set, they do not | 8 | `MOTION_RELINK_TIGHT_UM=6.0`, `GAP2_MAX_STEP_UM=4.4`, `GAP_CLOSE_REUSE_UM=3.2`, + artifact plumbing |

So a 0.942 public kernel is our banked 0.947 **minus** three post-processing constants, with a looser
deepcenter division veto. It is behind us by 0.005 and there is nothing in it we do not already run —
including `BIOHUB_EDGE_FEATURE_TTA=1`, the one element that looked new in the raw diff (dihedral-averaging
the UNet *feature* map that feeds edge scoring, not just the detection logits). That flag is already on in
`celltrack-public-0947`. This is the third time E60's pattern holds: **a public score at or near ours is our
own kernel re-published, and the diff to run is env-constant SETS, never the notebook.** The only open
question the diff raises is a single constant in the direction the loser moved it, which is not a submission.

**`hq93` (division recall, +0.0971 headroom) dropped P0 → P3.** The lever E61 named — re-parenting inside
the linker objective — exists, is built, and already ran: `LinkerConfig.division_cost` opens a parallel
`SOURCE → ("out", row)` arc priced `appearance[row] + surcharge`, and on `--divisions 4` it moved the **fn**
side for the first time on record (24 → 21; every earlier division experiment could only move fp, ceiling
+0.0011), 0.9243 → 0.9296. It then lost on its denominator: on sets not selected for divisions,
`div_jac = 0.0000` in *every* arm including baseline, so the arc's cost appears naked — default proxy
**−0.0027**, CV-8 **−0.0018**, monotone in how much fork it buys. Both ends hang on one quantity nothing
local measures: the division-bearing fraction of the hidden corpus, weighted by `tp+fp+fn` per video. The arc
stays as code (it is the only construct that can express a fork) and is re-tested only if the detection side
moves. **A headroom can be real, correctly located, and unclaimable** — the ruler that can see the term is a
set selected to contain it.

Also closed: `4u5v` — the confusor-eval substrate guard is verified landed (`synthetic_confusor_eval.py:147`,
mirror at `synthetic_scene.py:431`, pinned by `test_score_rejects_mismatched_substrate`). That guard rejects
exactly the checkpoint `wfh5` named as its gate, so `wfh5`'s gate was repointed to the `ModelEvaluator` native
proxy — where the verdict it already returned (native inv 1.00 → 0.61, **selection flat at 0.6081**) is the
affirmative case for a two-pass solver: the head learned the discrimination and selection did not spend it.

## NEWS 2026-09-22 08:23Z — the detected-crowd arm scored its FIRST epoch for two runs; fixing that bought +0.0988

Once the memory work let the arm run ten epochs, it scored **0.3765**. That number is not a ten-epoch
number — `edge_predictor_best.pth` had an mtime five minutes after launch. The donor trainer
(`$PACK/scripts/train_unet_transformer.py:1165`) selects on `score = test_acc * test_recall`, a **detection**
metric, and this arm freezes the UNet: across all ten epochs `best` printed **0.9957 to four digits**.
`is_best` fires once, there is no `_last` file, and epochs 2-9 were trained and discarded.

`_edge_selected` in `scratchpad/pmkf_detected_swap.py` now wraps `evaluate` and writes its own checkpoint
whenever the **edge** loss improves — the only quantity a frozen-detector arm can move. The donor's file is
left intact so the two selections stay comparable; `pmkf_leg.sh` resumes from the edge-selected one, because
resuming from the detection-selected one silently discards the previous leg's training.

| | 0.3765 leg | edge-selected leg |
|---|---|---|
| MICRO score | 0.3765 | **0.4753** |
| `44b6_ddf577ad` | 0.1439 | 0.3238 |
| `6bba_57b7cc1e` | 0.1383 | 0.2388 |
| scored epoch | 1 | 4 (`test_loss` 0.00343) |

Both matched rows move on the **same denominator**, so this is not a row-count artefact, and +0.0988 is ~5x
the noise floor. The confound is stated rather than hidden: the leg fixed selection **and** added ~2400 steps
on top of the 0.3765 weights, so +0.0988 is not attributable to selection alone. The reading both stories
share is the actionable one — **more steps still buy score; the arm is compute-bound, not data-bound.**

Two things to carry. **A "best checkpoint" ranks the metric the DONOR chose, for the donor's setup** — change
what trains (freeze, head swap, partial finetune) and that metric may no longer be movable, at which point the
selector is ranking eval noise. It fails silently: the epoch table looks healthy the whole way. **Check the
checkpoint's mtime against the launch time before believing a score is the end-of-run number.** Division is
dead from this arm regardless: `division_jaccard=0.0088`, divFP 19-67 per movie against divTP 0-1.

Leg 3 launched 08:23:43Z on the same config, resuming from the edge-selected weights. Gap to the champion is
**0.371**; at +0.099/leg that is ~4 legs if the curve held linear, which it will not — the per-leg delta
halving is the signal to stop buying epochs.

## NEWS 2026-09-22 07:20Z — the detected-crowd arm was never too big for the card; the head was `e**2` four times

No score in this entry — an ENABLING fix. `vutl` (the HOCT edge head trained on DETECTED candidate nodes
rather than the GT-sparse ~17/frame it had been getting) OOMed three times in a row, and each traceback was a
different allocation. The block turns out to hold **four separate `e**2` terms**, each invisible until the
one in front of it was fixed:

| term | fix | commit |
|---|---|---|
| the `(h, e, e)` score matrix | attend a tile of query rows at a time | `fa7b61d` |
| the tiles' **retained softmax** — tiling alone does NOT help, every tile's output is kept | recompute each tile in backward (`torch.utils.checkpoint`) | `74cc438` |
| the parametric segment-distance solve's ~10 dense planes | row-tiled under `no_grad` — both biases are functions of the COORDINATES alone, so no gradient flows through them | `74cc438` |
| the two `(e, e)` **results**, 3.64 GiB each in fp32 | store each at the width its consumer reads: `dline` fp16 (a µm distance scaling a learned bias), `neighbour` **bool** (a hard in/out test, not a plane of `0.0`/`-1e9`); widen inside `_attend` at tile size | `95e31dd` |

One block forward+backward at the true edge count: **7.3 GiB of biases alone → 3.21**, block peak **3.93 GiB**.
The arm now sits at **12507 MiB** of the 15.92 GiB allowed and trains at 1.05 s/it.

**The tracebacks also re-priced the regime.** The 3.64 GiB allocation is `count**2 * 4`, so the detected-crowd
pass gates **~31,200 candidate edges per frame pair**, not the ~21k the tiling budget had been sized against —
**1.5x low**. That number was free, printed in an error message, and it is the honest measure of what
"detected-crowd" means next to the ~17 GT nodes/frame the head used to train on
(`celltrack-pmkf-trains-gt-sparse-not-detected-crowd`).

Two lessons worth carrying past this arm. **An OOM traceback names ONE allocation site — it is not a size
budget for the layer.** An `e**2` layer has an `e**2` term per tensor it builds, and each is a separate fix;
stopping after the first one only moves the traceback. And **three of the four needed no layout change at
all** — only the observation that memory was being kept for a backward pass that did not want it, or at a
width the consumer never reads. Checking what the BACKWARD retains, and whether the tensor carries a gradient
at all, was worth more here than any tiling arithmetic.

**Still open, and it is a compute lever as well as a memory one:** `neighbour` is a HARD local mask, so the
dense `(e, e)` attention is mostly wasted work in exactly the regime this head exists for. Tiling + recompute
+ narrow storage is the stopgap; local/blocked attention is the fix.

Arm is running (10 epochs x 600 iters, resumed from step 1200 — `pmkf_leg.sh` does export `PMKF_RESUME`; the
bead's "no resume support" note was scoped to a different script). Gate is the script's own: if the dense-row
`edge_jac` lifts off at all, commit full.

---

## NEWS 2026-09-22 00:05Z — the three pending arms scored, and two of them are EXACT ties

| submission | public LB | vs base 0.947 |
|---|---|---|
| `det096` (detection threshold 0.96) | **0.947** | 0.000 |
| `gaploose` (gapfill at its repair ceiling: min_score 0.35, peak_radius 5.0um, allow_synthetic, filler 240 -> 1165 edges) | **0.947** | 0.000 |
| `synthsec` (synthetic secondary head) | 0.926 | **-0.021** |

Two exact ties are more informative than a small loss would be.

**`gaploose` adds 925 edges and the metric does not notice.** Not "costs a little" — costs NOTHING, to three
digits, while quintupling the filler. That is E55 confirmed end-to-end on the real ruler: a bridge edge
t-1 -> t+1 can never BE an annotated edge (every GT edge has dt=1), so it is neither charged nor credited. The
gap-repair axis is not merely priced out, it is INVISIBLE. Do not spend another arm on gapfill parameters.

**`det096` is a tie, not a loss.** Memory carried "threshold move LB-refuted" from an earlier arm; at 0.96 the
move is free. The detector's operating point is on a FLAT stretch of the LB, which is the same message E40/E41
gave from the other side (the "missed" cells are displaced, not absent — so a threshold that admits more of
them admits displaced duplicates that the matcher then discards). Threshold is not a lever in either direction.

**Final-2 pick unchanged.** There are now FOUR arms sitting at exactly 0.947 (base, `gapfill`, `det096`,
`gaploose`) and all four share the champion's detector, so none of them decorrelates from any other — a tie
on the public LB between arms with a shared component is one submission, not four. `celltrack-public-v1329f`
(0.939, different detector) stays the hedge for exactly the reason recorded before: rank a hedge by WHICH
component decorrelates, never by score proximity or disagreement percentage.

## NEWS 2026-09-21 20:50Z — E64: the fork axis is worth +0.1263 and the annotation holds every bit of it

The arm below was built and run. It is **edge-free as argued** (eTP +8, eFP −4, eFN −8) and worth
**+0.0031**, not the +0.0286 the projection implied — the ceiling was read on a *different* operating point,
against a division ruler holding 35 units, while at the shipped point the annotation holds 11 divisions in
total. **Re-price a ceiling at the operating point before planning on it** — the same lesson E61 already
taught about the c3 veto, paid twice now.

Beside it, the sharper measurement. An oracle that culls every predicted fork the annotation does not divide
at, keeping the branch the annotation agrees with, moves the local official ruler **0.8463 → 0.9726
(+0.1263)** — the largest number this project has measured offline. It decomposes into a **+0.0977 division
term** and a **+0.0286 edge term**, and the whole of it is DISCRIMINATION: the identical cull run **blind**
is **+0.0034**. The blind arm deletes all 254 charged false forks and still scores divJ **0.0000**, because
the division term is a ratio `k/(11+m)` — deletion alone can never buy it.

Graded on the champion's own predictions (`celltrack/eval/fork_rank_report.py`):

- **which fork to cut** — every existing `fork_ranking` strategy and their compositions; best reaches div
  Jaccard **0.10** vs the oracle's 1.00, worth **+0.008** on my T=4 labels, ~**+0.0036** on the metric's real
  T=11. Ranking is not the bottleneck.
- **which branch to keep** — of 7185 forks the annotation resolves exactly **245** to one branch (a fork with
  no matched branch grades nothing; one with two is a real division where cutting is wrong either way).
  Shipped rule (likeliest) **0.5878**, best of five GT-free cues **0.6000**, chance **0.50**. Three edges of
  245 separate them.
- **`survival` is the WORST of the five (0.5551).** E39's blip argument predicts it should win. So the surplus
  branch at a resolved fork is not a blip — it *looks like a continuation*, which is
  [[celltrack-confusor-pairwise-unresolvable-needs-multiframe]] arriving from a new direction.

**Post-processing on this axis is CLOSED.** What remains is what E61 already named: re-parenting inside the
LINKER, where ending a continuation in favour of a division is priced against the whole assignment. Any arm
there must argue it changes the candidate distribution or the cost structure — not the choice rule, which is
chance-bound. `BranchAccuracy.of` prices a future model-side branch scorer against 0.5878 in one CPU run.

Detail: `interpretations/celltrack/2026-09-21_fork_axis_is_annotation_bound.md`.

---

## NEWS 2026-09-21 01:20Z — the division re-parent is edge-FREE, which makes it the largest priced lever we hold

**SUPERSEDED by the E64 entry above: the ceiling below was read at the wrong operating point. Measured
+0.0031, not +0.0286.**

No GPU, no run — a reading of the metric source that re-prices `hq93`'s one surviving remedy. E61 left the
**10 of 25** missed divisions whose daughter is already claimed by another predicted track
(`DivisionScoring._daughters_all_claimed`), and left the edge-side cost unpriced; the bead's own caveat
guessed a fork rewrites edges and that this is what sank the global division-ILP at −0.055.

`ChargedLinks._countable` (`core/metrics/edges.py:124-140`) settles it: a link counts when **the annotation
continues through one of its endpoints**. A GT daughter is such a node, so the thief's link into her is
charged today and, not being on an annotated edge, scores **FP**. Re-parenting deletes an FP and adds a TP.
Unlike [[E58]]'s 1.85-GT-edges-lost-per-1-gained, this deletion is not *score*-selected — it is named by the
division topology and wrong by construction.

Ceiling on the division ruler (tp=1 fp=10 fn=24, denominator 35): the 10 re-parented → 11/35 = **+0.0286 on
the combined score**, ~2× the 1.5% bar, with a positive-signed edge term. Every past division sweep was
bounded at +0.0011 because all of them moved the fp side.

**Two corrections this forces.** (1) "DIVISION axis CLOSED" stays retired — and now has a *sized* successor,
not just a reopened door. (2) The −0.055 of the global division-ILP is re-read as a **precision** failure,
not a structural cost of forking (that arm emitted 3114 forks for 7 recoveries). This does not license
re-running it: the build is a linker candidacy that can choose to end a continuation in favour of a
division, and it must be measured on the combined score — what makes a correct re-parent free is exactly
what makes a wrong one expensive.

Detail: `interpretations/celltrack/2026-09-20_e61_division_recall_is_the_unclaimed_term.md`.

---

## NEWS 2026-09-21 00:40Z — E63: the offset head learns what the annotation can teach, which is not the thing it is for

Campaign A — the last lever memory listed as open — is built, trained, measured and **CLOSED at the data**.
The head is real (analytic null 5.85um → **2.90um**, monotone, backbone grad norm 1.26), but over 25 train
videos / 2307 frames / 6609 centres the nearest-neighbour distance is **24.54um median**, only **1.3%** of
pairs sit inside one supervision box, and only **2.4%** of multi-centre frames contain a Voronoi seam at
all. ~98% of the supervision is isolated single cells, so the merged-blob case the head exists to resolve is
not in the training signal. Ring stratification says the same thing from the other side: **0.364um** at the
centre voxel, **7.961um** in the outer shell — worst exactly where a seam lives.

Two method lessons outrank the verdict.

**The matched control is what turns an attribution into a measurement.** The proxy decline under the head
was first written up as the known warm-finetune collapse. The `offset_weight=0.0` control refutes that: it
sits at **0.843–0.870** across seven reads while the offset arm sits at **0.795–0.821**, four matched reads
entirely below the control's band. The term itself is the cost.

**And the control then corrected me a second time.** Its 0.8704 read as a climb, so it was resumed to 12000
steps — 0.8435 / 0.8628 / 0.8544 / 0.8509, no trend. So "the recipe gains +0.026" was reading the top of a
band as a result; the honest statement is a **plateau near 0.855**, and a detector that is **not
under-trained**. The reusable part is a ruler: a single detector-arm proxy eval carries **±0.027** of
scatter, so the 0.01–0.02 floor — measured on the *tracker* proxy — does not transfer to this one. Any past
detector A/B decided on one eval inside that band decided nothing.

Live remedy, filed not started: supervise the offset field against **detected** cells (~980/frame) rather
than annotated ones, the same shape as the `pmkf` finding (bd `vutl`). Offline-unpriceable per E59.

Donor housekeeping the same night: **`evg0942` inventoried and closed as already-harvested.** Its header
says `'public 0.939 base'`, its constants ARE ours (`SAFE_DIV_DIVERGE_UM=2.25` = our c3,
`OUTPUT_LINEFIT_WEIGHT=0.8` = our smooth, `OUTPUT_MIN_TRACK_LEN=6`), and the one element that looked new —
gap closing that INSERTS a synthetic node, so both new edges are dt=1 and GT-matchable, the hole E55 leaves
open — is `SyntheticGap` in `celltrack/postproc/gap_closer.py`, already wired into the submission path. bd
`nfpt` closed unstarted for the same reason: E61 already measured the c3 cue as a score (3 of 25) against
the gate we ship (4 of 25).

---

## NEWS 2026-09-20 18:10Z — E62: the donors' fuse weight reduces to switching the reverse pass off, and the alignment suspicion is closed

Both frontier kernels that run a bidirectional edge fuse weight the reverse direction far below half —
evgendvorkin 0.30, `evg0942` 0.15 — while we take the plain harmonic mean, which is the same fuse at
`w=0.5`. A month-old issue flagged the gap and was never run. Seven cells on test-4, with the shipped
arithmetic as its own control: **w=0.5 0.9375 (best)** · w=0.30 0.9361 · w=0.15 **0.9343** · bidirectional
off **0.9343**. The 0.15 cell returns the OFF cell to every digit — clamped 0.9317, bonus +0.0026 — and that
is arithmetic, not a bug: `f*r/((1-w)r + wf) → f` as `w→0`. **On our candidate distribution the donor
constant does not mean "weight the reverse lightly"; it means "do not run the reverse pass".** One line of
algebra predicted that cell before the GPU did. Reduce a donor constant at our own numbers before spending
time on it.

`align_reverse_moments` — the reverse logits put on the forward centre and spread, as `evg0942` does — is
**inert** (identical to 4dp at 0.15 and 0.30, −0.0005 at 0.5). It was built to test a real standing
suspicion: that every bidirectional arm read flat because the fuse shifted probability mass under a fixed
candidate threshold. **That explanation is now closed**; what remains is that the reverse pass carries
little the forward pass does not. Full spread 0.0032, sub-floor — no lever either way, and the direction
that would have been a win is the one we were already in. Closes bd `xoap` + `3gyw`. Detail:
`interpretations/celltrack/2026-09-20_e62_bidirectional_fuse_weight.md`.

A harness note worth carrying: the first run of this sweep produced nothing. `logger.info` writes to
**stderr** while mlflow's chatter goes to stdout, so a stdout-only redirect left a log with every header
present, no traceback, and no scores — 22 minutes of correct GPU work discarded. **A silent log reads
exactly like a silent crash; check the first expected log line exists before trusting an arm.**

---

## NEWS 2026-09-20 16:55Z — division recall is not reachable from post-processing, and a ceiling number lied about where the cost was

**(Revised 18:10Z with the closing arms — the first version of this entry headlined "the veto deletes six
true divisions". That number was real but REGIME-SPECIFIC, and it does not survive at the operating point.)**

E61 priced the division-recovery stage to its ceiling and then re-priced the winner where it actually runs.
At the ceiling — every knob permissive, budget uncapped — `require_c3_divergence` ON reaches **7 of 25**
missed divisions and OFF reaches **13**, with `fork_rejected` 5 → 0, which reads as the veto censoring six
true forks. **At the SHIPPED operating point that cost is zero.** The decisive 2×2, recovered of 25:

| `require_c3_divergence` | `prefer_divergence` | recovered |
|---|---|---|
| true | false | **4** (shipped) |
| true | true | 3 |
| false | true | 3 |
| false | false | **1** |

The gate beats the same cue used as a score, and both beat nothing. Under an uncapped geometric candidacy
the veto was the only thing rationing a flooded proposal set; at shipped candidacy the hard threshold is
simply the better estimator. **Price a stage's ceiling, then re-price the winner at the operating point —
the two regimes disagreed here and the ceiling one was the misleading answer.**

**The instrument lesson generalises past divisions.** I had read `fork_rejected=5` as "the right mother forked
and the wrong daughter won the sort" — a ranking failure. `core/metrics/divisions.py:129-136` assigns that
stage **post hoc on the OUTPUT graph**: it means only that *some* fork landed near the parent, and under gates
opened two orders of magnitude the stage emits 726 forks on one movie, so one lands near almost any mother.
**A stage label read off the output is a hypothesis, not a record of what the pipeline decided** — only an A/B
against the suspected stage separates censorship from coincidence.

The built response was divergence as a **SCORE** (`DivergingDaughterRanking`, dimensionless
`(s2-s1)/max(s1,s2)`, commit `1c10595`), soft where the champion is hard. It is a **real cue** — alone it
recovers 3 where nothing recovers 1, carrying two of the gate's three — and is **strictly dominated by the
same cue as a gate**. It ships OFF.

The other half of the miss — the 10 unproposable — is not a post-processing question at all: the stage draws
its second daughter from the orphan pool only, so a division whose daughters the linker already claimed is
outside it at every setting. `candidacy=steal` attempted exactly that re-parenting and **dies at the GATE,
not the budget**. I predicted the capped arm's recovered 4→2 came from steal candidates crowding out true
forks under a binding budget, and registered the uncapped arm as the test: it recovers **1** — worse — with
**4 spurious forks across eight movies**, where displacement would have produced hundreds. So the proposals
never reach the ranking; the physical gates and C3 destroy them first. The mechanism is coherent with E54c:
a daughter another track already holds was claimed *because she looks like a continuation*, which is
precisely what a division-shaped gate exists to reject. What remains is re-parenting inside the **linker's
objective** — a different component, a different cost.

Still **no submittable gain**; 0.947 stands. Detail:
`interpretations/celltrack/2026-09-20_e61_division_recall_is_the_unclaimed_term.md`.

---

## NEWS 2026-09-20 13:41Z — the finer122 joint+nce arm collapsed its detector, so the gate it was built to read never got a reading

**HARNESS, not REFUTED.** `launch_finer122_joint_nce.sh` — the correctly-specified never-run cell (finer122
substrate, trunk trainable, contrastive ON, detected corpus) — read **proxy 0.0000 at BOTH eval windows**,
from an init of 0.6150. Node recall went to exactly 0.000 while the emitted node count roughly DOUBLED
(53k → 97k on `44b6_e57ff5c6`): the probability map saturates, NMS returns a lattice of peaks, and almost
none of it lands within 7µm of a cell. `det` loss RISES across the windows (0.1460 → 0.1574) while the total
falls — the optimizer buys contrastive progress with the detector. `sanity AUC` holds ~0.995, which is what
says this is the detector and not the harness: the affinity head still discriminates on the same forward pass.

The registered gate said "below 0.70 = the substrate is the bound and the finer122 axis closes for good".
**It does not close.** The arm never tested the objective on a working detector; it tested whether
`--contrastive-weight 1.5` at `--lr 5e-5` with an unfrozen trunk destroys a finer122 detector. It does, in
947 steps. 1.5 was not a guess — it is the weight recovered exactly from the joint family where nce lifted
0.61 → 0.87. What is new underneath it is the (1,2,2) decode, which quadruples the candidate count the term
sums over. That is a *hypothesis* for the imbalance (nce 1.30 vs det 0.15), not a measured cause.

Re-test, if ever: scale the weight by the grid ratio rather than picking a new one by feel, and assert node
recall at the FIRST window so an arm like this dies in 4 minutes. **Not scheduled** — it is ~1.5h, not a
submission arm by its own registration, and the ruler work outranks it. Detail:
`interpretations/celltrack/2026-09-20_joint_nce_finer122_detector_collapse.md`.

Unaffected: the 120× linker swap below, measured on this same run. It is a post-processing timing result,
independent of detector quality — and it is the only reason two eval windows fit in 18 minutes instead of
two hours, which is how the collapse was caught the same afternoon.

---

## NEWS 2026-09-20 13:20Z — the thing eating the schedule was never the training: the flow solve was 120x off

Timed a tracker pass by stage and found `LinkerStage` is **~75% of it**, and on the eval movies this arm
actually uses, **406.7s and 290.3s** per movie against `AffinityDivisionRecovery 1.4s | ShortTrackFilter 0.5s
| DensityGapBridge 0.0s | LinefitSmoother 0.0s`. Four movies per eval window × 16 windows = the EVAL fold, not
the gradient steps, is what a training arm spends its wall-clock on. Cause: `nx.network_simplex` is pure
Python and scales ~n² on this shape (1.53s at 6k detections, 5.92s at 18k, 156,907 nodes on a real movie).

The network is an assignment LP (unit capacities + bipartite incidence ⇒ totally unimodular), so a C++
min-cost flow solves it **exactly**, not approximately. Two rewrites make OR-tools' non-negative-cost API
accept a priced transition (`distance − bonus·P` is routinely negative): **forced units + a BYPASS arc**
(without the bypass, isolation is taxed and the solver buys +1123 spurious links at 18k) and a **uniform
shift** (exactly one arc consumes each unit at an in-node, so one constant on all three moves the objective
by `shift·count` and cannot move the argmin). Verified: identical objective AND identical selected edge set,
**5.92s → 0.05s (120×)** — `tests/unit/celltrack/linkers/flow_solving.py` asserts the edge-set equality over
random layered shapes with costs spanning zero. networkx stays as the fallback because the submission kernel
installs a fixed wheel set (see `bd` — shipping the ortools wheel in the kit is filed, not done).

**What this changes: the PRICE of every future arm**, not any score. An eval window drops from ~20 min to
~3 min, so a training arm is now gradient-bound instead of eval-bound. It does **not** unblock `kiw1`
(two-pass ILP): that is gated on a discriminator over the 2405 CONTESTED edges at a 1.4% base rate, which is
an accuracy problem, not a budget one. Landed `6ebed4c`; the running arm was restarted at 16:19 local to pick
it up, since the old process had loaded the pre-swap code and was paying the 406.7s per movie in full.

---

## NEWS 2026-09-20 12:35Z — the nce flag was INERT in both arms; a month of head-axis reasoning chased a dead variable

Ran the "never-run cell" (`hoct_finer122_nce`: frozen finer122 + fresh HOCT head + contrastive ON, 1.86 h).
Best proxy **0.6239** against a 0.6877 gate — FAILED. But the failure refutes the 2026-08-28 *framing*, not
the cell, and it re-validates a pile of verdicts that note had thrown out.

**The contrastive term cannot train anything under a frozen trunk.** `joint_assembly.py:77` freezes the
detector; `contrastive_site` defaults to `FEATURES` (`joint_config.py:291`); at `FEATURES` the term builds no
projection and holds **zero parameters** (`contrastive_term.py:59-63`). `nce` moved 1.5399 → 1.5496 over
72,096 steps. **avl8 was frozen too** — and its `nce` likewise barely moved (2.4670 → 2.3603). So nce was
equally dark in the 0.79 arm and the 0.62 arm; it was **never the live variable**. No `--contrastive-site`
rescues it (PROJECTION trains an auxiliary embedding the edge head never reads).

**What separated them is the DETECTOR, and it is measurable before training.** Both arms init a *fresh
untrained* head, so init proxy prices the detector alone:

| run | detector | **init proxy** | best | gain from all training |
|---|---|---|---|---|
| avl8 | pilkwang_jm | **0.7875** | 0.8054 | **+0.018** |
| today | finer122_coadapt_long | **0.6164** | 0.6239 | **+0.008** |

The whole 0.79-vs-0.62 gap predates the first gradient step, and head training moves the number by under 0.02
in the better arm — at the noise floor. Independent confirmation of the hmlo verdict: **the edge head is not a
lever on this substrate; the detector sets the score.**

**RE-VALIDATED** (the 2026-08-28 note wrongly voided these): `rzvw` 0.6187 · finer-coadapt faithful 0.779 ·
finer122 HOCT head-only control · the HOCT HEAD-BOUND referee. None rode a crippled objective — avl8 rode the
identical one to 0.79.

**Scope of the inertness, audited:** a sweep of every log for `detector frozen` returns **six** runs ever
(avl8, avl8hnfix, finer122_hoct_headonly, hoct_real_converge, jointarm_cheapfirstcut, hoct_finer122_nce).
Everything else is JOINT, where the term is live — so [[celltrack-nce-lifts-joint-hn-knife-edge]] **holds**.
The dead flag is a property of *head-only* runs, i.e. exactly the finer122 verdict family. And that sweep
prices freezing itself: frozen head-only with a good detector tops out at **0.8024** (avl8) while joint runs
reach **0.8475 / 0.8567 / 0.8630** (cellect_long, detcorpus_dw0, confB_ctrl2). **Freezing costs ~0.05 — more
than any head choice has ever bought (+0.018).** The construction adopted to make arms cheap caps them below
where the joint family already sits.

Second confound, recorded so it is not mistaken for the cause: today's arm trained on GT pairs (in-gate mean
**1.00**, rival **2.6%**) where avl8 used the detected-pair corpus (2.19 / **53.1%**) — the missing
`--detected-videos`. That voids today's run as a *head*-axis test but not the detector read, which comes from
the inits. Re-running it with the right corpus is **priced out**: avl8's best-case head gain of +0.018 from
0.6164 lands ~0.635, still 0.05 short of the gate, for ~2 h of GPU.

Full write-up: `interpretations/celltrack/2026-09-20_head_only_contrastive_is_structurally_inert.md`.

**Lesson, general:** before attributing an A/B gap to the variable you changed, read the **init** numbers. If
the arms already differ untrained, the variable you named is not what moved them.

---

## NEWS 2026-09-20 09:38Z — CAMPAIGN CLOSED: ship 0.947. The last lever dies on an oracle we already measured

Decision taken with the owner: **stop experimenting, ship what is banked.** GPU stood down.

**bd `zpq7` (centre-offset head) is CLOSED — on a ceiling, not on a failed run.** The framing that it "can only
be priced by training it" was too pessimistic about our own data: E59 had already re-solved the linker with
**corrected GT coordinates**, which is perfect localization and strictly better than any trained head. It bought
**+2 GT edges = +0.0015**. Against a ~35-edge bar (E54) and a 0.01–0.02 noise floor, the axis is an order of
magnitude under the floor *at oracle*. A 0.67 h arm cannot buy a measurable number.

The mechanism is not refuted — E57's diagnosis stands, broken-edge endpoints sit **4.08 µm** off versus 1.46 µm
for recovered ones. What is refuted is that fixing localization **changes linker decisions**: E58 found the
champion already holds **51 of 79** in-scope broken edges, so better coordinates mostly re-confirm edges it
already has. Reopen only with a mechanism that turns localization into **new** edges, not better-placed
existing ones.

**FINAL SUBMISSION — one manual UI action, no API exists. Before 2026-09-29:**
- `dimiturnt/celltrack-public-0947` — **0.947**, the champion
- `dimiturnt/celltrack-public-v1329f` — **0.939**, the hedge

v1329f is chosen over `readmit` (0.946) deliberately: readmit scores higher but **shares 0947's detector**, so it
fails on the same inputs and decorrelates on nothing. A hedge earns its slot by failing *differently*, not by
scoring second-best.

**Closed axes, for the record:** detection, prune, gate, division, ensemble, gap-repair, linker knobs, CPU
post-processing, kernel harvest (E60), and the discussion channel (07:20Z). Nothing offline remains.

---

## NEWS 2026-09-20 07:20Z — the Kaggle DISCUSSION channel is unreachable at BOTH levels; the offline-harvest axis is now closed end to end

The record had the forum *listing* closed as JS-rendered (NOTICE_BOARD:14, 09-19). Today's probe extends that:
an **individual thread fetched by id** (`discussion/741749`, the hikaggler 0.939 write-up already cited in our
own notes) also returns **EMPTY**. Kaggle discussion pages are not server-rendered at any level, so no
WebFetch/WebSearch path in this harness reaches a post body — only titles and third-party GitHub mirrors.

Consequence, stated plainly so it is not re-probed: **the 0.964–0.974 band has no reachable public write-up.**
Search re-surfaces only what NOTICE_BOARD already triaged (`JunhaoLiXD` 0.944, `kito2718` roadmap-only,
`sota1111` eval-only, `royerlab/kaggle-cell-tracking-competition` = the host baseline + `metrics.md`, both
already mined in `deep_dives/2026-07-31` and `2026-08-03`). I re-ran the listing fetch before finding that
note — two wasted calls, and the reason the closure is now written at thread level too.

With [[E60]] closing the kernel harvest and this closing the forum, **every offline source of a new element is
exhausted.** What remains is not findable, only trainable: the centre-offset head (bd `zpq7`), which E59 proved
cannot be priced offline. It needs the GPU, which is parked.

---

## NEWS 2026-09-20 07:14Z — E60: the six top-voted unharvested kernels ARE our champion, with worse constants

Six public kernels we had never pulled (263/97/82/78/58/30 votes), triaged by **diffing their `BIOHUB_*`
env constants against the banked 0947** instead of reading four near-identical 222 KB notebooks — a
megabyte of donor in one `comm`. Write-up: `interpretations/celltrack/2026-09-20_e60_the_frontier_moved_back_to_us.md`.

| kernel (votes) | vs our 0947 |
|---|---|
| `biohub-harmonic-fusion` (263), `-v3` (78), `lineage-forge` (97), `lf-dctta` (82) | identical except `DET_THRESHOLD` 0.960→**0.965**, `PPSWEEP_SELECT_MARGIN` 0.0005→**0.001**, and **no `MOTION_RELINK_TIGHT_UM=5.5`**. Its own header: `'public 0.939 base'` |
| `biohub-0-95` (58) | **SCORER EXPLOIT** — the kirneo hack again (hub@`t=-1000`, `MAX_COMPONENTS=1400`, `FORKS=5`), already at tree lines 701 / 1391-1396. Not portable, not submitted |
| `biohub-lf-hoctveto-div-b` (30) | **`BIOHUB_HOCT_VETO=2`** — the only new element |

**The bidirectional harmonic fusion is OURS, shipped, and already in the 0947 we hold.**
`TrackerConfig.shipped()` carries `bidirectional_edges=True`; the fuse lives at
`celltrack/edges/blended_edge_scoring.py:191-194`; priced −0.0027 under the assignment linker and **+0.004
on the LB under flow** a month ago. Zero to harvest.

**`HOCT_VETO` is pre-refuted, three legs, before any run.** It runs HOCT general_v0 (royerlab, arXiv
2607.11754) over the champion's final nodes and drops every edge HOCT does not also propose. (1) Author's
own number **+0.0040 [+0.0006, +0.0058]** — under our 0.01–0.02 floor. (2) It is an edge DELETION and **E54
measured wrong-association = 0**: a dropped edge is a GT edge (loss) or an unannotated one (E53,
metric-invisible); E47 caps the prune axis at **+0.021 even perfect**, E58 measured **1.85 lost per 1
gained**. (3) Its stated gain is division-side, and the division axis is **LB-flat**. Filed and closed as
bd `0jaw`, a do-not-re-harvest record.

**Net: zero shippable elements, and one reframing** — the public list holds nothing we have not priced, and
a public score above 0.947 is not by itself evidence of a better tracker. The remaining wall-clock belongs
to the centre-offset head (Campaign A), which E59 proved cannot be priced offline at all.

---

## NEWS 2026-09-20 06:54Z — E59: the re-solve prices perfect localization at TWO edges, and 76/76 was a tautology

E57 asked for a re-solve, because snapping a node onto its GT coordinate is a no-op for the metric by
construction. `kaggle/localization_resolve.py` runs it. Arms differ in ONE thing — the coordinates the
**linker** may see; every arm is scored on the **original** positions, so node matching, node-count term
and estimated count are identical across arms and the whole delta lands on edges. The harness reproduces
the champion at **2022 recovered / 105 broken**, E54's figures exactly.

| arm (distance-Hungarian at the champion's 14 µm) | recovered | broken | score |
|---|---|---|---|
| champion edges (reference level) | 2022 | 105 | 0.8932 |
| shrink 0.0 — control | 1951 | 176 | 0.8043 |
| shrink 0.5 — half the error corrected | 1946 | 181 | 0.7990 |
| shrink 1.0 — **perfect coordinates** | 1953 | **174** | 0.8059 |

**+2 edges, +0.0015**, with the half-arm at −0.0053: non-monotone and under the floor, the shape of a null.
The ranking oracle implied 76 edges ≈ **+0.036**; the re-solve returns 1/38th.

**And row four was never evidence.** It ranks candidates by distance to `gt_xyz[target]` — the node the
scorer matched to that target is within 7 µm of that point by construction, under a Hungarian that would
have preferred any nearer rival. It asks "is the node matched to this cell the closest node to this cell?"
The 4.08 vs 1.46 µm gap survives as a **correlation** (crowded regions produce both), not as a lever.

**This does NOT close the centre-offset head** — and the reason is the instrument's own limit, one level
above the one E57 named. Only **2172 of 122808** nodes (1.8 %) have a GT coordinate to snap onto, and
E54c's thief is an **unannotated** track that keeps its wrong position in every arm. A trained head moves
the whole field; this oracle corrects one side of a two-body problem, so +2 is a lower bound from a
one-sided correction. What it *does* close is **pricing the head offline at all**: there is no GT for the
nodes whose displacement does the damage. Train it and measure, or leave it — no third option.

Net: *"localization is the only channel left"* loses its causal support and reverts to a plausible
mechanism backed by a correlation, indistinguishable offline from the crowding story.
Detail: `interpretations/celltrack/2026-09-20_e59_the_snap_oracle_cannot_price_localization.md`.

---

## NEWS 2026-09-20 06:43Z — the public dense GT is NOT our embryos: axis CLOSED in one CPU hour

The 06:40Z entry below is **resolved and partly wrong**, and the correction is worth more than the find.

`kaggle/public_gt_registration.py`. A crop is anonymised — integer voxel coordinates, clock restarted at
zero, no translation transform — but its GT chains run 40–100 frames, and a chain's sequence of step
LENGTHS survives any unknown translation or rotation. Slide that sequence against every public track.

**Positive control first.** Take real public tracks, round them onto the crop's voxel grid exactly as an
anonymiser would, and feed them back in: **20 of 20 recovered**. The probe works under the quantisation
the real question has to survive.

| public volume | steps | tracks | crops matched (of 8 tested) |
|---|---|---|---|
| ZSNS001 | 20,092,258 | 1,454,396 | 1 — and only the weakest needle (spread 3.28 µm), 29 hits = chance |
| ZSNS003 | 3,888,503 | 167,995 | **0** |
| ZSNS004 | 5,225,573 | 285,300 | **0** |
| ZSNS005 | — | — | **0** |

Strides 2, 3 and 4 swept on ZSNS003 in case a competition frame advances several published timepoints:
**0 at every stride.** A true source would give exactly one hit on a strong needle; instead the seven
*strong* needles (spread 4.45–11.06 µm) score zero everywhere, and the single sub-threshold needle scores
29 — the signature of chance, not provenance.

**The two train embryos are not the publicly annotated ZSNS001/003/004/005.** Strictly: no public track
reproduces a crop's motion signature to within one voxel. The imaging could still be shared with the
coordinates re-curated, but nothing in the public files is usable as our labels.

**Two things close at once.** The dense-supervision axis is dead — E54c's thieves and E58's decoys have no
published track to inherit, so the centre-offset head must be trained on what we have. And the *leakage*
route closes with it: not even the train crops resolve against the public set, so there was never an answer
key to refuse. Good. Bead `03sn` closed.

Two traps avoided, both worth keeping: the first run reported 2 of 8 matches, and both were chains whose
every step sat inside one tolerance band — a needle with **no power**, matching any slow track in the
embryo. And the first matcher slid its window across concatenated tracks, so a window could span two
unrelated cells. Fixing those turned a 25 % "hit rate" into a clean zero.

Cost: ~1.4 GB of public CSV, one CPU hour, zero GPU, zero submissions.

---

## NEWS 2026-09-20 06:40Z — the competition GT is a SPARSE SUBSAMPLE of a DENSE public annotation (SUPERSEDED — see 06:43Z)

Triaging the last three untriaged public kernels turned up `giorgosi/zsns00{4,5}` — not a tracker at all,
but a **data harvester**. It pulls the source volumes from
`public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/` and, with them, a per-volume
**`ZSNS00N_tracks.csv`**. Probed live (range request, no full download):

```
ZSNS001_tracks.csv   Content-Length: 890,580,527
track_id,t,z,y,x,parent_track_id
270656,99,357.12,172.527,578.163,270655
228090,9,308.76,138.724,482.022,-1
```

**That is our exact GT schema — divisions and all — for the WHOLE embryo.** Track ids run past 312000.
The competition's own GT is ~17 annotated cells per frame
([[celltrack-pmkf-trains-gt-sparse-not-detected-crowd]]). Published (image.sc / FEBS): train spans **two
embryos**, the hidden test is a **third**; `*_tracks.csv` exists for ZSNS001/003/004/005 (002 has none).

**This reframes the entire chain the last six experiments built.** E54c found every thief to be a real,
median-61-frame, **0/126-annotated** track; E58 found the decoy nodes to be mid-track members of a second
real track. Those tracks are almost certainly *annotated in this CSV*. The labels we have been calling
missing are published.

**Two uses, and only one of them is ours:**

- **LEGITIMATE — dense training supervision.** Register each `44b6_*` train crop against its source volume
  and the sparse crop GT becomes dense. That feeds exactly the one live lever
  ([[celltrack-campaign-a-offset-head-build-map]], the center-offset head): the reason a decoy sits 4.08 µm
  off (E57) is that nothing ever supervised the cell it actually belongs to. Registration needs **no image
  download** — the crop's `.geff` gives (t,z,y,x) of its annotated cells, and recovering an integer t-offset
  plus a 3D translation that embeds that sparse point cloud in the dense one is pure CPU point-set
  matching on the CSVs.
- **REFUSED — test-side lookup.** The same registration run against the *test* crops would read the answer
  out of a public file. Same class as the coordinate perturbation refused in E58, and not on the table.

**Status: found, not spent.** The consumer is training, and the GPU is parked by owner directive with the
entry deadline two days out. Recorded as the largest un-taken axis on the board, ahead of every knob —
per the standing rule that data outsizes knobs. Bead filed. Crop-geometry note: the train zarr carries
`scale [1.0, 1.625, 0.40625, 0.40625]` µm and **no translation transform**, so provenance is not in the
metadata; it has to be recovered from the point clouds.

Cost so far: one HTTP range request. ZSNS001's CSV is downloading to
`D:/data/volumetric/microscopy/reference/zebrahub/` for the registration probe.

---

## NEWS 2026-09-20 06:32Z — the published synth 5-fold does NOT price bd `rf80`, and our own parked weights beat it

`bhpepper/biohub-synthetic-5fold-ensemble-v1` publishes an `ensemble_summary.json` (2 KB, free read) for a
Stage-2 synthetic fine-tune of the 8.35 MB `edge_predictor`: 5 folds + an SWA member.

| | value |
|---|---|
| mean proxy score | 0.9672 |
| mean recall | 0.9679 |
| fold spread (best 0.9720 / worst 0.9585) | **0.0135** |
| best_epoch per fold | 15, 28, 14, 8, 23 |
| param drift vs stage-1 base | 0.164 |

Three free reads, no download of the 5×8.35 MB:

1. **It cannot price `rf80`.** There is **no non-synthetic control arm** in the file — every number is a
   synthetic fine-tune. A forum claim of +0.012 from synth pretrain stays unpriced; this publishes the
   *level*, never the *delta*. (The same trap as our own 20+10-epoch A/B, which scored `acc*recall` with no
   matched base.)
2. **The recipe's own fold spread is 0.0135** — at or above our 0.01–0.02 noise floor. So even *with* a
   control, a synth delta of the claimed +0.012 size would be unreadable against this recipe's own seed
   scatter. `best_epoch` scattered 8→28 says the early-stop is picking noise, not a convergence point.
3. **Our parked weights already beat it.** `rf80`'s note records `synth_pre80` finished 09-19 13:00Z at
   **0.9783** (80 ep) and is sitting on disk at
   `external/frontier_ds/biohub-tracking-support-pack-50ep-v1/repo/weights/synth_pre80/split_0/edge_predictor_best.pth`.
   Different proxy, not strictly comparable — but there is no argument that their 0.9672 mean is a *better*
   donor than a checkpoint we already own. **Donor triage: do not pull the weights.**

Same verdict covers `easonyanyan/biohub-exposure-bias-weights-v1` (same 8.35 MB `edge_predictor` shape,
four epoch checkpoints, no summary at all). Both are **edge-head** donors, and the edge head is not our
failure mode: E54b row 4 gives **76/76** from plain distance once coordinates are right, E57 puts the broken
endpoints **4.08 µm** off, E46 puts appearance below chance. A better edge head reading wrong coordinates is
a pre-registered ~null on the axis we actually measured. `rf80` stays OPEN but demoted — and its remaining
step (real-data fine-tune warm from `synth_pre80`) is GPU work, which is parked by owner directive.

Cost: one 2 KB read. Zero GPU, zero submissions.

---

## NEWS 2026-09-20 06:22Z — E6 (packet grouping) CLOSED by argument: the grouping is inert, the knobs are pre-registered nulls

`newwang12/biohub-v1-grouped` re-ran 06:01Z today and sat at **DONOR-UNMEASURED — "would need a slot"**
(E6, table row below). It does not need one. Reading it:

1. **The packet grouping does nothing.** `GROUP_OVERRIDES = {'low': {}, 'middle': {}, 'high': {}}`, and
   preflight *asserts* it empty. `graph_packet_features` / `classify_packets` (the xy radial-shell profile
   and severity model) compute a per-packet label that is then applied to **no parameter**. The whole
   grouping apparatus is instrumentation. Whatever the kernel scores, the grouping is not why.
2. **The real delta is two knobs**, named in `EXPERIMENT_NAME = 'R3_zero_only_minlen4'`:
   - `EXPERIMENT_ZERO_MAX_UM = 0.0` → in the patched `motion_relink_edges`, `if zero_max is not None and
     prob == 0.0 and (zero_max == 0.0 or raw > zero_max): continue` — a **precision veto: never motion-relink
     through a pair the learned edge model scored exactly 0.0**.
   - `OUTPUT_MIN_TRACK_LEN = 4`. (`EXPERIMENT_KEEP_MIN_PROB = None` — the second knob is off.)

**Both are nulls by results we already own.** The zero-prob veto only *removes* motion-relink edges, and
E54 measured `wrong-association = 0` on all four public-GT movies — the pipeline never links two
*annotated* cells wrongly — so every edge this veto can delete joins at least one unannotated node, which
E53 showed is neither TP nor FP. Metric-invisible. And minlen4 is *looser* than the champion's own minimum
component of **6** (E52), so it can only re-admit short tracks, priced by E53/E58 at the negligible
per-node term. Pre-registered expected gain ≈ 0 on both; not run, not submitted.

Incidental corroboration: this kernel's `BASE_PARAMETERS` carry `DET_THRESHOLD 0.965` and
`OUTPUT_EDGE_MAX_UM 14.0` — the champion's constants, independently confirming the 14 µm candidate radius
E58 read out of the staged kernel.

Rule: **before pricing a donor element, check whether its headline mechanism is actually wired to
anything.** The grouping had a classifier, a severity score and a receipt, and an empty override dict.

---

## NEWS 2026-09-20 06:18Z — final-2 SETTLED: readmit is the better buy and the wrong hedge (bd `9c61`)

E38's table (below, 06:00Z block) makes `readmit` look like the obvious second seat: LB **0.946**
(−0.001, against v1329f's −0.008) at **0.8690** union agreement — **80 % of v1329f's decorrelation for an
eighth of the LB cost**. Rejected anyway, on E38's own point 4.

A variance hedge must not share the component that can fail. `readmit` is a 0947 **post-processing**
variant: it re-adds peaks 0947's *own* detector scored ≥0.965 and discarded, so its extra nodes come from
that detector's discard pile and inherit its calibration whole. `v1329f` is a different foundation with an
independent node set (123485 vs 122808). E38 point 4 says disagreement scales with the **node-set**
difference, not with any association setting, and E36/E37/E57/E58 all put the failure axis in
detection/localization. So readmit's 13.1 % is post-processing edges hanging off the same, possibly
mis-tuned, nodes — it hedges the part that is not at risk. The same argument rejects `gapfill` (0.8792,
and E55 proves its dt>1 bridges can never be a TP) and `fp16` (0.9956 — no hedge at all).

**Final 2 = `celltrack-public-0947` + `celltrack-public-v1329f`.** Zero runs, zero submissions. Caveat:
readmit's node count is inferred from its mechanism, not measured — its submission.csv was never pulled.

---

## NEWS 2026-09-20 06:13Z — E58: the champion already holds 51 of 79 broken edges, one node to the side

`representative_alternates.py` collects every predicted node inside the 7 µm ruler of a GT source and of a
GT target and asks whether any predicted edge joins the sets. Recovered edges (control) read **1.000**;
**broken edges read 0.646 — 51 of 79**. The tracker HAS the edge; the scorer's Hungarian represents those
two cells by a different pair. Broken edges are duplicate-enriched too: 1.71 candidate nodes near the
source vs 1.23. 51 clears E54's 35-edge bar.

The obvious harvest is then killed by the same probe: the decoys that win the assignment carry
**1.84 / 1.88 edges and 0 of 102 are isolated** — mid-track nodes of a second REAL track (the champion
already prunes isolated ones). Deleting one loses ~1.85 edges to gain 1. Blanket merge is worse: **23 %**
of all nodes (28197/122808) have a same-frame in-ruler neighbour. Node count itself is *not* the
constraint — `jac*(1-0.1*(t_pred-t_true)/t_true)` is per-node, ~0.1/t_true each.

This is **E54c at node resolution**: two genuine tracks within one ruler-width, the annotated cell
represented by the unannotated track's node, because that node is mis-placed by ~4 µm (E57). **All CPU
post-processing on this axis is closed.** Perturbing decoy coordinates to dodge the assignment would work
and is a scorer exploit — not submitted. The honest version is the **center-offset head** (Campaign A),
which needs training. Detail: `interpretations/celltrack/2026-09-20_e58_the_champion_already_holds_the_edge.md`.

---

## NEWS 2026-09-20 06:06Z — the gate was never the blocker; widening it is strictly negative

One 40 s CPU run, no solve. `cue_oracle.py` hardcodes `ASSOCIATION_GATE_UM = 10.0` — **our** linker's
constant, not the champion's, which caps output edges at `OUTPUT_EDGE_MAX_UM = 14.0`. Re-ranked at
10 / 14 / 20 µm: "true successor outside the gate" **15 → 1 → 0**, so E54b's fifteen were **already inside
the gate the champion uses** and were mis-ranked, not un-gated — "unreachable by any solver" was an
artifact of the instrument's own constant. Selection bucket grows 61 → **76**.

And widening buys nothing: distance ranks the true successor first on **exactly 10 cases at every gate**
(10/61, 10/75, 10/76 — absolute count FLAT, rate 0.164 → 0.132), oracle velocity *drops* 10 → 9. Every
newly-admitted candidate is a rival. The ceiling row stays **76/76**: every selection failure is
gate-reachable and plain distance solves all of them given correct coordinates. **bd `ittt` CLOSED**; the
last route by which the gap could have been a linker constant is gone, and **localization is the only
channel left**. Detail: `interpretations/celltrack/2026-09-20_e57_localization_not_selection.md`.

---

## NEWS 2026-09-20 05:58Z — E57: it was LOCALIZATION all along, and E54b had already proved it

Read at the official 7 µm ruler, over the same 105 broken and 2022 recovered GT edges: a **broken** edge's
endpoints sit **median 4.08 µm** from the cell (46 % beyond 5 µm), a **recovered** edge's **1.46 µm** (2 %).
Three times worse, one denominator, no ruler swap. The official matcher forgives 7 µm — about one cell
width — so a node half a cell off still counts as *detected*, and its dead edge is filed under
**selection**. The detection/selection split E54 reported is a property of the **tolerance**, not of the
tracker: 26/76 at 7 µm, 641/20 at 2.87 µm.

The closer was banked five hours earlier and under-read. `cue_oracle.py`'s fourth row —
**distance to the true GT position of the target ranks the true successor first 61/61**, against 10/61 for
plain distance and 10/61 for oracle velocity. **With correct coordinates the simplest discriminator there
is, is perfect.** E54c's "structure right, discriminator missing" therefore restates as **the
discriminator is present and is being fed nodes a cell diameter off**. E54b even named the mechanism and
left it: 15 more true successors fall *outside* the 10 µm gate because "matched predicted positions carry
up to 7 µm localization slack each". `gate_um = 10` is the maximum **GT** step — right for clean
coordinates, systematically undersized for the predicted ones it is actually applied to.

A detector with **no weights at all** reaches part of the gap. `kaggle/classical_blindspot.py` harvests
the multi-scale DoG from `fabriciodasilva/biohub-dodecatiad-cell-tracking` (CPU-only) — the one detector
family that cannot share the learned pool's bias, which is what E36's *flat-across-nine-caches* 0.824
demanded. Over all 292 frames of the four public-GT movies: champion recall **0.7747** at cell separation
but **0.9863** at the official ruler; of the 494 cells missed at 2.87 µm, a model-free detection lands
within ruler on **0.1336** against **0.0486** for a frame-shifted matched null → **excess +0.0850 = 42
cells**, at **0.7× the champion's detection count**. Fewer detections, not more, so the rescue is not
bought with false positives; and stride-5 gave +0.0909, so it is not a sampling artifact.

**Node RECALL at the scorer's ruler is CLOSED** — six misses in 2193, model-free excess of one cell. The
open channel is **LOCALIZATION**, which the 7 µm matcher is built not to see and which edges die on.
Cheapest first: **widen `gate_um` and re-solve** (a constant, no training) · then a center-offset head
priced against the 42 · **never** swap the detector wholesale (finer122 graft 0.7503, ILP swap 0.6555).
Do not price this as 42 recovered edges: the snap-to-GT oracle is a **no-op by construction** — the
broken/recovered verdict depends only on the node matching and the ILP has already run — so causality
needs a re-solve. Three CPU runs, ~25 min, zero GPU, zero submissions.
`interpretations/celltrack/2026-09-20_e57_localization_not_selection.md` · bd `n2xg`.

---

## NEWS 2026-09-20 05:36Z — E56: the broken-edge count is ANTI-correlated with the board

Full write-up: `interpretations/celltrack/2026-09-20_e56_cross_tracker_edge_agreement.md`.

- Built `error_budget.py --rival`: compares two cached submissions **GT edge by GT edge** over the same
  2127 public-GT edges. **fp16 is the control and it behaves** — its broken set is **100 % nested** inside
  the champion's, confirming E38's 0.9956 union at edge resolution. The instrument discriminates a
  same-family variant from a decorrelated one.
- The v1329 family rescues **39 champion-only edges**: ruler-stable (44/38/39/37 at 2.87/5/7/10 µm),
  **33/39 selection** (17 swap, 9 source-rival, 7 target-rival), **33/39 inside the dense movie**
  `6bba_05db0fb1`.
- **CORRECTION to commit `0d7dc7d`**, which called v1329f and v50 "two unrelated recipes". They are not:
  v50's `v1329_submission.csv` is **byte-identical** to v1329f's submission and both carry node count
  123485. v50 = v1329 + one stage. **n = 1 family witness, not 2.**
- **The deflation.** Broken GT edges vs LB across all four banked submissions:
  105 → **0.947** · 104 → 0.945 · 78 → 0.939 · 77 → 0.907. **The champion breaks the MOST and scores the
  BEST**; the ordering is exactly reversed. Not the E51 saturation story — these are the *same four
  movies the board scores*. Our pooled harness sides with the count, not the board (champion 0.8932 vs
  v1329f 0.9263, 0.033 the wrong way against the board's 0.008).
- The confound is visible and uncontrolled: the rivals also carry **677 more nodes**, which E53's signed
  node-count term prices. v50 sizes the sensitivity — same node count as v1329f, 90 fewer edges,
  **−0.032**. So the reading is not "recovering GT edges hurts" but **"the score is not dominated by
  GT-edge recovery, and this harness cannot separate the two effects."**
- **E54's bar survives with an explicit clause: recover 35 GT edges *without adding nodes*.** That was
  implicit before and is now measured. **A cross-tracker merge is NOT licensed by an edge-count win** —
  adopting a rival's edge means adopting endpoints the champion lacks, i.e. the exact term the board
  punishes. Never price a cross-tracker change with E54's ~2.3 units/edge; it was derived *within* one
  tracker with node count held fixed.
- **ADDENDUM, same tick — the 33 selection fixes are node-free BY CONSTRUCTION, so the axis is alive.**
  `classify_broken` tests rules in order with the detection rules first, so a swap/rival label is only
  reachable when **both** GT endpoints already matched a champion node. All 33 connect nodes the champion
  **already has**; importing them costs **zero** nodes. **The 677-node penalty belongs to the rival's
  DETECTOR, not to the edges** — the two were confounded and separate cleanly. Swap fixes are edge-neutral
  too, and E54c put the collateral at zero (0/126 thieves annotated). That returns the 33 to the
  within-tracker regime where ~2.3 units/edge holds: **≈1.4%, just under the bar**, oracle-assumed →
  **directional, composes.** **A RE-RANKING axis, not a merge axis** — the champion lacks the
  discriminator (E54c) and v1329 is an existence proof one exists; transfer needs no GT at test time.
  Affordable only because **twrp** freed 4010s (rival predict ~930s). bd `x6bd` blocks `kiw1`.
- **SECOND ADDENDUM, same tick — AXIS CLOSED. Node-free is not the same as free.** `kaggle/edge_transfer.py`
  asks the transfer question the way a *submission* must: align rival nodes onto champion nodes by
  **distance alone, no GT**, then sort every rival edge. At the official 7 µm ruler: unaligned 3831,
  redundant 112643, **CLEAN 13**, **CONTESTED 2405** (at 2.87 µm: 7070 / 110273 / 11 / 1538). Clean
  additions are **11–13 across four movies** = sub-noise, worth nothing. **And the 33 are not among them**
  — swap / source-rival / target-rival each mean a champion slot is ALREADY TAKEN, so all 33 are contested
  **by definition**. The harvest is not "add 33 edges", it is **"pick the right 33 out of 2405"** = a
  **1.4 % base rate**, each wrong pick breaking a GT edge *and* planting an FP (E54, one axis). Exactly
  E48/E49's spec, now binding on transfer proposals: their GBM hit FP-recall 0.0846 vs 0.75 required on an
  *easier* base rate. **Borrowing a rival's edges does not supply the discriminator — it supplies 2405
  suggestions, 98.6 % wrong.** Corrects the first addendum: node-free right, "alive" too generous.
- **Also banked from that census: 94.7 % of the rival's edges are REDUNDANT** (112643 / 118892). Two
  separately-trained families agree on nineteen edges in twenty — the ensemble-saturation closure measured
  at **edge resolution** for the first time, rather than through a score. Cost to close the whole axis:
  **one CPU run, zero submissions, zero GPU.**

---

## NEWS 2026-09-20 05:17Z — E55: gap repair cannot score (GT is all dt=1), and the LB already said so

- E54 measured **0 GT edges with `dt != 1`**. So a gap-repair bridge connects `t-1` to `t+1` and
  **matches no GT edge, ever** — best case metric-invisible (E53), worst case a clean FP when both
  endpoints are annotated. It does not restore the two real GT edges either. **E37's repair axis and the
  `max_gap = 2` lever are both retired by argument, for zero submission slots.**
- **The board already told us:** `gapfill` = **0.947, an exact tie with the base champion**. That tie was
  sitting there unread. **Pre-registered:** `gaploose` (filler 240 → 1165) returns **≤ 0.947**; above it
  would refute this and force a re-measure of the dt=1 count.
- **`fp16` = 0.945**, below the champion — E38's "no slot" call from the 0.9956 edge union, confirmed.
  **Final-2 stays 0947 + v1329f.** det096 / synthsec / gaploose still pending.
- **Standing question for every edge-adding proposal from now on: *can the edge it adds ever BE a GT
  edge?*** Anything spanning more than one frame answers no. (Recovering a missed detection's NODE is a
  different proposal — the 26/105 detection slice.)

---

## NEWS 2026-09-20 05:13Z — E54c: the error is a SPLICE with an unlabelled REAL track

- **0 of 126 slot-stealing rivals is an annotated cell** (`error_budget.py --thieves`). On E54's 76
  selection failures: source slot 63 unannotated / 13 free / **0 annotated**; target slot identical.
- Harness suspect cleared first: the thieves sit **median 6.92 µm from the nearest GT node of any kind,
  0 % within 2 µm** — one cell-separation. Not mislocalized labelled cells.
- **And they are REAL CELLS, measured, not junk.** Predicted track length: thieves **median 61 frames,
  0 % ≤3 frames, 67 % ≥50** vs GT-matched 74 and all-predicted 44. Not one blip; they out-persist the
  average predicted node.
- **SELF-CORRECTION, recorded deliberately.** I first read "0 annotated thieves" as "no conflict exists,
  so re-costing cannot help". The track-length measurement refutes that: the slot **is** claimed, by the
  thief's own real trajectory, which GT merely does not label. The ILP constrains it correctly already.
- **So: structure right, DISCRIMINATOR missing.** The failure is a splice between a labelled track and an
  unlabelled real one, and position, appearance (E46, below chance) and motion (E54b, oracle) all fail to
  separate two real cells.
- **Narrows the standing conclusion from "dense detector" to "identity representation".** The thief
  *should* be detected — better detection does not remove it. And a same-feature re-solve cannot either.

---

## NEWS 2026-09-20 04:52Z — the error budget in metric units: ONE axis, 105 broken GT edges

- **E54: `wrong-association = 0` on all four movies, verified by a direct recount.** Predicted edges with
  BOTH endpoints matched to annotated GT nodes but not a GT pair: 0 / 0 / 0 / 0. Among cells the
  annotators labelled, the global ILP links them to each other correctly **every time**. Every one of the
  141 FP edges is single-endpoint-matched — an annotated cell continued into a detection GT lacks.
  **Retires a framing: the confusor is not a competition between two tracked cells.**
- **99 % of FPs (139/141) are incident to an endpoint of a GT edge the tracker failed to recover.** FP and
  FN are not independent axes — they are two views of the same **105 broken GT edges**, each costing
  ~**2.3 metric units** (105 FN + 139 FP). E47's ~2× asymmetry, derived, and general to every missed link.
- **Budget:** pooled adjusted **0.8932** (raw 0.8915), tp 2022. **83 of the 105 are on the one dense
  movie.** Perfect-fix ceilings OVERLAP and must not be added: all-105 → 0.9991; fp→0 alone +0.059;
  frag alone +0.035; detection alone +0.012.
- **The 105, partitioned (`--shapes`): 76 (72 %) are PURE SELECTION** — every endpoint detected, the
  candidate present, the solver ranked another partner above it. **50 are outright swaps** (both endpoints
  linked elsewhere) + 13 source-rival + 13 target-rival. Only **26 detection**, **3 division**, and
  **0 GT gaps** (`dt ≠ 1` never occurs). 76 vs a 35-edge bar → **`kiw1` licensed with 2× headroom.**
- **Broken edges move 4.08 µm median vs 1.46 µm recovered; 46 % > 5 µm vs 2 %** — E45's "outranked band is
  fast cells" now holds on the champion's metric-visible errors.
- **MOTION CONTINUITY IS ORACLE-REFUTED** (`kaggle/cue_oracle.py`, same tick). Ranking every in-gate
  candidate in the next frame: distance **10/61** · predicted-tracklet velocity **8/61** ·
  **ORACLE velocity from the cell's TRUE GT trajectory 10/61** · true-GT-position ceiling 61/61.
  **Handing the ranker the actual past trajectory buys nothing over plain distance** → the estimate is not
  the problem, and no Kalman / tracklet second pass recovers these. I wrote the motion premise into E54
  and refuted it 20 min later; the doc carries the correction.
- **Cue inventory now CLOSED: position, appearance (E46, below chance) and motion all fail on these 61.**
  The information is absent from the detected representation → pairwise-unresolvable keystone
  re-confirmed with motion added AND an oracle. **Do not build `kiw1` on the motion argument.** What
  remains is a joint-assignment argument (50/76 are swaps) — but the champion already solves a global
  ILP, so the COSTS are wrong, not the structure. Re-points at the dense-regime DETECTION model.
- **15 of the 76 have their true successor outside the 10 µm gate** — unreachable by any solver; matched
  predicted positions carry up to 7 µm localization slack each. **GATE WIDENING PRICED AND CLOSED same
  tick**: 12 µm → 3 unreachable, 15 µm → 0, but **distance still ranks exactly 10 first at every gate**
  (10/61 → 10/73 → 10/76). Widening admits the edges without making them RANKABLE, while adding
  confusors everywhere else. No-op at best.
- **Consequences.** (1) Do NOT chase FP suppression as its own programme — 99 % are symptoms; deleting an
  FP without supplying the right link converts a 2.3-unit error into a 1.3-unit one at best. (2) Every
  future proposal must state **how many of the 105 it moves**; the 1.5 % bar ≈ **35 recovered GT edges**.
  (3) Detection is genuinely smallest (26/105), consistent with E40/E41. (4) `kiw1` (two-pass tracklet
  ILP) is the axis, now sized.
- 85.8-99.8 % of predicted edges are metric-**invisible** — E53 restated as a measurement.
- `kaggle/error_budget.py` · `interpretations/celltrack/2026-09-20_e54_error_budget_in_metric_units.md`.
  Caveat: E51 in-sample, magnitudes are ceilings; the shape is structural.

## NEWS 2026-09-20 04:45Z — the prune axis was never a tracking axis (read from the scorer's source)

- **E53: an edge between two UNANNOTATED detections is neither TP nor FP.** `metrics.py:55-114` and the
  champion's faithful copy (`champion_submission.py:3584-3620`): an edge counts only if an endpoint
  matched a GT node. A spurious detection away from annotated cells is **metric-invisible** except
  through `adjusted = jac * (1 - 0.1*(t_pred - t_true)/t_true)`, where `t_true` is the geff's **estimated
  true node count** and the term is SIGNED — `t_pred < t_true` multiplies the jaccard UP.
- **Champion already sits at/below `t_true` on 3 of 4 movies** (25637/25755, 20729/32795, 6152/6362,
  70290/69800) → no over-detection penalty left to recover.
- **Re-reads the prune axis:** E47's +0.1139 oracle is mostly that multiplier, not recovered edges;
  E48/E49's target label was not "is this an FP" but "did an annotator pick this cell" (98.4 % of their
  rows were unlabelled REAL cells); E52's cut buys ≤ +0.009, under the noise floor, against ~2× that in
  collateral. **Axis closed for a better reason than E48/E49 gave.**
- **The label is partly predictable** — on the two sparsest movies annotation is a thin slab in y (GT std
  5.7 / 8.4 µm vs 30.7 / 24.9 for all detections). **Not building it**: deleting real cells for the
  node-count bonus games sparse GT and improves no tracking. Any future prune must name its channel —
  recovered edges (real) or the multiplier (not). Real FP headroom = edges TOUCHING annotated cells, i.e.
  the mislink/confusor problem (E45/E46).
  `interpretations/celltrack/2026-09-20_e53_prune_axis_is_an_annotation_artifact.md`

## NEWS 2026-09-20 04:40Z — a harvested element was already in the champion; and a submission can't price a prune

- **E52: SHORT5 is a NO-OP on our output.** `howonkang/biohub-0947-short5-prepp-r1` deletes every linked
  component of ≤5 nodes from the raw geff. Minimum component size in the champion's submission is
  **exactly 6** on all four movies — `OUTPUT_MIN_TRACK_LEN = 6` already binds, and no short division
  component survives for `OUTPUT_KEEP_DIVISION_COMPONENTS = 1` to spare. Cuts at ≤2/≤3/≤5 remove **0
  nodes**. Its only remaining claim is as a PRE-postprocessing pass (i.e. "repair is net-harmful on short
  components"), which sits inside E37's ~1.1 % repair ceiling → sub-bar, no slot owed.
  `interpretations/celltrack/2026-09-20_e52_short5_is_a_noop.md`
- **The denominator rule this exposed — general.** Only **1.39 %** of predicted nodes match any GT cell at
  2.87 µm (2.21 % at 7 µm). That is annotation SPARSITY (E51's fracs 0.0016–0.135), not a 98 % FP rate.
  So **a cached submission cannot measure a pruner's FP-recall** — its negative class is "unannotated",
  not "false". Collateral stays valid (denominator = matched GT). Price prunes on a candidate set with GT
  correspondence, as E47–E49 did.

## NEWS 2026-09-20 04:30Z — the proxy is in-sample, and its own split cannot say how much

- **E50 (`g89y` precondition): both gates FAIL under an intensity proxy, but the proxy is confounded.**
  Containment @2.87 µm **0.7498 < 0.824** detector; best FP-recall @2 % collateral **0.0203 vs 0.75**
  (E47), with `height` a flat 0.0000 at every cut. Per the pre-registration this is NULL-leaning, NOT a
  refutation of `g89y` — the real build uses a trained contour head. A fixed `FOREGROUND_QUANTILE = 0.98`
  first produced a *fake* containment of 0.6047 by cutting above every GT cell on one bright-background
  movie; `proxy_validity()` now refuses such a movie on its own arithmetic. Otsu admits the cells and
  costs >3 min/frame — **no threshold is both valid and affordable.**
  → `interpretations/celltrack/2026-09-20_e50_hierarchy_precondition.md`
- **E51: the public affinity model trained on ALL 199 videos → every proxy movie is affinity-in-sample.**
  Manifest is literally `unet_transformer_alltrain_seed314159_v1`, its 40 "test" videos a strict subset of
  its 199 train. DeepCenter held out 128 — but the four proxy movies it DID train on are exactly the four
  `44b6` ones, which are also the four sparsest annotations (frac ≤0.029). In-train adj_jac 0.9189 vs
  held-out 0.8996 **attributes to nothing**: in-train ≡ `44b6` ≡ sparsest is perfectly collinear on this
  split. **Mechanism for a rule we held only empirically** — local association headroom is optimistic, so
  an offline association win need not transfer (dw0 +0.0267 local / −0.028 LB is the shape). It
  *strengthens* E49: AUC 0.6380/0.6622 were in-sample, i.e. an optimistic ceiling that still failed 9×.
  → `interpretations/celltrack/2026-09-20_e51_proxy_contamination.md`

---

## NEWS 2026-09-20 03:30Z — the board compressed; the last standing lever is runtime-clear

- **The field moved and we did not.** Public LB re-listed: 0.974 / 0.973 / 0.969 / 0.968 / 0.967 /
  0.966 — and **0.964 is only about rank 12**. Our banked 0.947 no longer sits near the top; the gap to
  the front is ~0.027, which is far above the 0.01–0.02 noise floor and is the same dense-regime model
  gap E42–E49 keep arriving at from new directions. Entry deadline **09-22**, final submission **09-29**.
- **Donor `fabriciodasilva/biohub-dodecatiad` REJECTED on inspection, no run spent.** Read in full: DoG
  blob detector + Hungarian/velocity linker + gap-close + a second-daughter division heuristic, on
  numpy/scipy/zarr only. **No ultrack, motile, tracksdata, trackastra; no hierarchy, contour, watershed,
  ILP, appearance, embedding or affinity anywhere in the file.** Every element is a strict subset of what
  we already run, and its one detector idea is the axis E40 closed (DoG union 0.139 vs 0.450, n=3206).
  Recorded in bd `r5ce` as do-not-re-harvest. Einstein criterion paid for itself: a read, not a run.
- **`g89y`'s runtime gate is FOREGROUND-DEPENDENT, and my earlier "closed in its favour" was wrong.**
  The 4.5 s/frame figure holds only at a 2 %-of-voxels foreground. That threshold is *invalid on some
  movies*: where the background is bright and hazy (raw median 1115 vs 114), the GT cells sit BELOW the
  cut, so 0 % of them can be contained — the number measures the threshold, not the hierarchy. Admitting
  those cells (Otsu) inflates foreground to 21–48 % and costs **>3 min/frame** — ~20–40× — which blows
  the ~27000 s headroom by 4×. So the honest statement is: hierarchy cost scales with foreground volume,
  and the cheap regime is the one that misses cells. Budget is an argument against `g89y` again until
  the trained contour head shows it can be both tight and cell-containing.
- **Ultrack's own published mechanism is exactly g89y's claim** (Nature Methods 2025): candidate
  segmentations from *multiple* algorithms, with temporal consistency used to **select** among them.
  That is the one axis E48/E49 explicitly could not reach, because selection scores nested contours our
  candidate set never contained. E50 (`kaggle/hierarchy_precondition.py`) is measuring it now.

## NEWS 2026-09-20 00:49Z — where the campaign stands

- **Board:** ours 0.947 (`celltrack-public-0947`, tied by `-gapfill`), then readmit 0.946, v1329f 0.939,
  v50 0.907, our own banked 0.924. LB top 0.974. Deadline 09-29. **5 slots, reset 00:00 UTC.**
- **Pending LB:** det096 (56370794), synthsec (56370924), fp16 (56372434). **fp16 is now decided
  offline** — 0.9956 of the edge union with 0947, 257 disputed edges (E38) — its score is a formality.
- **Post-processing on the 0947 base is priced out (E37).** The repair the E36 residue asks for is
  already shipping in `-gapfill`; its ceiling is ~1.1 % more edges because only 935 bridgeable
  end→start pairs exist, and 39 % of dangling ends are cells leaving the imaged volume. Two free
  instrumented kernels (`gapdiag`, `gaploose`) are in flight to settle which gate discards the rest;
  **neither is a submission candidate** at that ceiling.
- **The final-2 hedge is real and keeps its slot (E38):** 0947 and v1329f dispute ~16 % of the edge
  union (10356 / 10700 unique edges), so they are two genuinely different graphs, not one in two hats.
- **The live lever is unchanged in SIZE and corrected in TARGET (E40/E41/E41(b)).** E36's 2.5 % of edges
  is independently reproduced — 3.30 % on a cache whose recall is 0.697–0.750 vs E36's 0.824 — so the
  lever is real and detection is still the axis. What was wrong is *which cells it points at*. E36's
  mechanism (a GT node borrows a **neighbouring** cell's detection) fails: missed cells sit 24.8 µm from
  the nearest GT neighbour, the same as found cells, and a matched null (frame t+50) shows the detection
  3.83 µm away is the cell's **own**, displaced — own-frame closer than chance in 0.807 of cases, and
  only 3 % of these cells lack any local intensity excess. So the 2.87 µm population (n=3864, 40.3 % of
  edges — impossible against a 0.92 LB) is **localization scatter**, while the population that actually
  costs score is the **288 nodes blind at the official 7 µm ruler (2.25 %)**, 13× smaller. Aim a recovery
  pass at those, not at the 2.87 µm set. A learning-free DoG union is separately dead on the conditional
  (0.156 vs its own 0.414 marginal). The card stays parked by explicit instruction; nothing starts
  without asking.
- **The second addressable block is SOLVER-side and mostly spent (E43/E44).** Edge fragmentation is a
  *selection* failure, not a candidate one: 99.51 % of fragmented GT edges already carry a proposed
  candidate with an affinity (only 59 = 0.48 % have none, and those are median 11.72 µm — outside the
  10 µm gate). So it needs no detector pass. But the obvious knob is spent: sweeping termination cost
  shows a non-monotone curve peaking at 1.0–2.0, and raising it to d5er's proposed 3/4/6 **loses**
  recall (0.8589 → 0.8293) because expensive termination makes nodes grab FP rivals. What stays live is
  the **11.95 % of true edges the affinity head ranks below a rival, by a median margin of 0.0985** —
  an affinity-*ranking* problem, reachable by a global solver or a better head, never by a cost knob.
- **That surviving band is PAIRWISE-bound, and the next lever is tracklet-level (E45/E46).** The band is
  a fast-cell population (5.7× enriched); on it, distance picks true 0.1767, oracle velocity buys +0.065,
  and raw appearance reads **0.4853 vs a 0.8230 mutual-best control** — 0.4345 at matched displacement,
  *below chance*. The confusor is a near-static lookalike and is an **unmatched FP in 0.9698** of
  contests: every local pairwise cue selects it, and it has no continuation for the pair to see. So the
  argued levers are **segment-SELECTION ILP (bd `g89y`, prune the duplicate)** and the **two-pass
  tracklet ILP (bd `kiw1`, out-compete it with a sequence)** — never a better edge feature. (E46 is a
  replication of a 2026-08-23 memory that the index had dropped; see the E46 block.)
- **The OFFLINE PRUNE AXIS IS CLOSED — the joint ceiling is 9× short (E49).** E48 killed two scores one
  at a time; E49 gave the inference its best shot: eight features (intensity, the affinity field's
  best-out/best-in/margin/degree, crowding, temporal support t±1), a GBM free to combine them, **split by
  MOVIE**. Joint **AUC 0.6622 — only +0.026 over its best single feature — FP-recall 0.0846 at the
  required 2 % collateral.** Not a tuning gap. The **nulls carry the news**: degree 0.4278, crowding
  0.4815, **temporal support 0.4837/0.4838**. "A spurious detection has no continuation" — the strongest
  remaining intuition, and the one bd `kiw1` runs on — is worth nothing, because E46's confusor is a
  near-static lookalike sitting where the cell WAS and has the same t±1 neighbour a real cell has. The
  pairwise-unresolvable finding and the temporal-support null are one fact seen twice. bd `g89y` is NOT
  refuted by this: hierarchy selection scores nested contours the candidate set never contained, so it is
  outside the measurement's reach — but it must now supply information that is not a function of any of
  those eight, which is the sharpest form its precondition has taken.
  Detail: `interpretations/celltrack/2026-09-20_e49_offline_prune_axis_is_closed.md`.
- **The diagonal is now MEASURED, and no score we hold reaches the operating point (E48).** Center-voxel
  intensity: AUC 0.5221, and its **FP-recall equals its collateral to three decimals at every cut**
  (0.0222 at 0.02) — E47's q = r diagonal, literally, which upgrades the recovery-stack retro-explanation
  from argument to measurement. Best incident affinity is better and still 13× short: AUC 0.6380,
  FP-recall **0.0578 against the 0.75 required**. That failure was pre-registered with its reason — the
  E46 lookalike's signature IS high affinity to the source, so a want-to-link score cannot demote it.
  Read together: real-vs-FP status is not recoverable from anything DOWNSTREAM of the detector, which is
  the dense-model-gap conclusion reached from a new direction (convergence, not a new finding). `g89y` is
  sharpened, not closed — it must carry information neither score has, and the bar is a number, not a
  direction. Caveat: the cache has no per-detection score column, so the detector score itself is
  refuted only by analogy — measure it opportunistically on the next GPU pass.
- **The prune axis is priced, and COLLATERAL is what binds it (E47).** Oracle FP-deletion rescues 95.3 %
  of the band (0.1195 → 0.0056, net **+0.1139** offline) — the ceiling is not the problem. But deleting a
  real cell costs ~2× its own size (both its edges), so at 5 % collateral even a PERFECT pruner nets
  +0.021, while at 2 % collateral FP-recall 0.75 earns **+0.035**. The requirement is ASYMMETRIC, not
  aggressive. This retro-explains the LB-refuted recovery stack by mechanism: a threshold is
  intensity-ordered, so it buys recall and collateral on the SAME axis ~1:1, and E41 puts the at-risk
  real cells exactly there — it lands on the losing diagonal by construction, not by mistuning.
  **Precondition before building `g89y`:** check whether hierarchy selection's survival decision
  correlates with intensity; if it does, it inherits the same coupling.
- **Runtime is solved and is no longer a reason to avoid anything (E31/E32).** `-fast` = 1590 s vs 6286 s,
  byte-identical submission; hidden test is 4 videos at 9.93 predict-minutes, ~20x headroom in a 9 h kernel.
- **The ensemble idea is dead for the donors we hold (E33).** v1329f loses 98.9 % of the 961 genuinely
  contested parents; v50's inner `v1329_submission.csv` is byte-identical to v1329f so it is not an
  independent third voter. Final-2 = 0947 + v1329f as private-LB VARIANCE hedging only, never a merge.
- **The one live, un-run lever:** each fork carries ~3–4 k nodes the other has no node for within 5 µm
  (0947-only 3118, v1329f-only 3806), and **both sets are fully track-embedded — mean degree 1.73/1.76,
  ~0 % isolated — so neither is FP noise.** Which set is real is undecidable from submissions alone and
  needs GT: run v1329f's detector on holdout20 and score the unique nodes against labels. That is the next
  GPU run, and it is the only thing left that could plausibly be worth ≥1.5 %.

## NEWS 2026-09-18 — frontier went public; the tree's "no public >0.927" premise is OBSOLETE

- Public kernels now plateau at **0.947**; alfonso1799 V50 claims **0.9605** (pilkwang UNet+node-transformer
  primary, temporal-unet3d secondary, DeepCenter veto, v1327 w3 real-model, tracksdata ILP + heavy post-proc).
  Its guard report admits `leaderboard_feedback_used_for_configuration: True` → overfit-risk on private.
  Forked unchanged as `celltrack-public-v50` / `celltrack-public-0947`; LB pending (watcher auto-submits).
- **Consequence:** every LEVEL 3–4 verdict below was measured on OUR model family. The frontier model is a
  different (stronger) substrate — our replicate-first rule now points at *its* weights + trainer
  (`external/frontier_ds/…support-pack…/repo`), not our pmkf/HOCT stack. Build-past levers go on top of it.
- **Open build-past (bd, P1):** synthetic CC0 pretrain → real fine-tune of the frontier model (forum +0.012–0.018).
  Not a repeat: prior synth nulls were our generator / frozen UNet / from-scratch joint. Synth frames carry
  ~300–560 labelled nodes vs ~17 sparse on real → supplies the complete-candidate supervision
  ([[celltrack-pmkf-trains-gt-sparse-not-detected-crowd]]) our real labels lack. Held-out: 20 movies
  (7×44b6, 13×6bba) — the public weights trained on all 199, so both arms must retrain.
- **Ruler (09-19):** the public weights, LEAKED on holdout20, with default predict (`--use-ilp`, thr 0.99) score
  **0.8842** (edgeJ 0.889, divJ 0 of 17, nodeR 0.992). They clear the 20 movies' labels but score ~0.06 below
  the LB plateau, so the local number is a relative ruler only: compare the A/B arms with each other, never
  with the LB. Divisions score zero at default settings — the post-proc stack carries the LB gap.
- **V50's last layer is public-LB surgery (09-19).** After the V1329 foundation it (a) linearizes EVERY fork
  (keeps the nearer child — a general "predict no divisions" policy) and (b) re-adds ≤2 divisions per ≥30k-node
  movie through ~10 thresholds fit to one public event (`6bba_05db0fb1`, P=20025→D=20865). (b) is overfit but
  tiny; (a) is the real private-LB bet. Hedge fork `celltrack-public-v1329f` = foundation only (after
  ozermehmet's candidate pair); the V50−v1329f LB delta prices the surgery. Final-2 pick = one of each.
- **V50 fork scores 0.907 on OUR LB (09-19), below the banked 0.924.** The claimed 0.947/0.9605 did not reproduce.
  The visible-test run completed cleanly (V1329 took 25 min, all hashes matched, no traceback). But the V1327
  adapted-detector guard fell back to the untouched V1290 primary on 98/100 frames of `44b6_0b24845f` (median
  retention 0.76): per its own guard, the headline detector is unfit on a dense 44b6 movie. The hidden-run log is
  not downloadable, so the cause of the drop is not established.
- **v1329f (foundation, no surgery) = 0.939 on LB, the NEW CHAMPION (+0.015 over 0.924) (09-19).** Same notebook
  as V50 minus the surgery cell, so the surgery's measured price is **−0.032**. Fork linearization (drop every
  division) plus the 2-division rescue loses 0.032: the LB rewards the foundation's divisions, and "predict no
  divisions" is refuted on the LB, not just overfit. Final-2 = v1329f + one decorrelated candidate (not V50).
- **Why the surgery looked good upstream: it is fitted to the 4 visible test movies (09-19).** Those movies ship
  with GT under `train/`, so `tools/score_submission.py` scores kernel CSVs locally. V50 = **0.9605**, v1329f =
  0.9263: adjJ is equal (0.927 vs 0.926), and the whole +0.034 is division_jaccard 0.33 vs 0.0. That is one
  division TP out of **3 GT divisions** on the visible set. So V50's "0.9605" claim is a 3-division overfit that
  reverses on the hidden movies (−0.032). The visible-4 score cannot rank division policies. It also cannot tune
  the family's other knobs, because the pack's weights trained on all199, which includes these movies. The LB
  is the only clean ruler for this family.
- **thtennant forked 0947 three ways (09-19, unscored).** All three are post-process edits on the same "0.939 base +
  holdout-selected post-process" notebook (`research/frontier_kernels/thtennant_*`):
  - `gapfill` bridges gaps ≤3 frames with the detector's sub-threshold peaks (score ≥0.5).
  - `readmit` re-adds discarded peaks (score ≥0.965) within 4 µm of an open track end.
  - `divprec` tightens the sister symmetry τ from 0.6 to 0.4.
  The first two add recall, but node recall is already about 0.98 on the visible movies. So each is priced at
  ≤0.01, below the floor. Probe on the LB only once 0947's own score lands; without the base score, a variant's
  score can't be read.
- **0947 fork = 0.947 on LB, the NEW CHAMPION (+0.008 over v1329f, +0.023 over our own 0.924) (09-19).** It is
  reyhanksatria's notebook with three input paths changed; the model is Pilkwang's pack. So the gain is upstream's,
  not ours. Now that the base has a score, thtennant's `gapfill` and `readmit` variants are pushed as LB probes
  (`kaggle/kernels/celltrack-public-0947-*`). Final-2 default = 0947 + v1329f.
- **Forum 741749 (hikaggler, own 0.939): node-count calibration predicts LB, model ranking doesn't; synth pretrain is
  the one lever (09-19).**
  - The N_pred ratio term moved their LB every time. A one-point recall change did not.
  - Post-proc settings chosen locally carried over to the LB. The local ranking of two trained models did not.
  - Their synth recipe: CC0 José Freitas sequences, only 497 of 2174, pretrain detector + linker 80 epochs, then
    60 epochs on real. 24-video hold-out: 0.9146 → 0.9269 (+0.012).
    - Using all 2174 sequences at equal gradient steps scored −0.005.
    - Pretraining pushes N_pred up and doesn't help divisions.
  - Our synth A/B (from_synth 0.9063 < scratch 0.9127, trainer acc·recall) is **INCONCLUSIVE, not refuted**. It used a
    different recipe: 20 pretrain + 10 real epochs vs their 80 + 60, and 0.006 on the trainer metric is noise.
  - Re-run at their recipe: `synth_pre80` launched 09:03Z (580 seqs, ≈3.5 h). The 60-epoch real fine-tune (≈20 h) is
    gated on the pretrain converging.
- **New public forks (09-19), unscored:**
  - thtennant's det096 line, vs 0947:
    - DET_THRESHOLD 0.965 → 0.96, which raises N_pred, the term hikaggler says moves the LB.
    - frame cache 48, ILP timeout 1200 s, repair deadline 27000 s. "fast-tight60" is runtime engineering, not a model change.
    - gapfill-det096 adds a flow-seeded motion relink (`MOTION_RELINK_FLOW_*`).
  - noisyislands trains a TabPFN division classifier (divisions ≤0.002 LB per hikaggler).
  - newwang12 `biohub-v1-grouped` (10 votes), arnav170 `biohub-reid3`, ghazarosbarseghyan91 `biohub-dae-alpha-0-17`:
    V9-lineage hosts self-citing 0.939 — host LB is irrelevant, they are ELEMENT DONORS (inventory below).
  - beraterolelk `0-947-lb-biohub-deepcenter-ilp-tracker`: same 3 pinned weights; DET 0.96 + relink TIGHT 5.5 +
    PPSWEEP margin 0.0005; titled 0.947 = TIE with 0947 → det096 knob alone likely sub-floor.
- **ELEMENT INVENTORY vs 0947 (harvest elements, not notebooks; 09-19).** Diff = `BIOHUB_*` keys + new `def`s per
  public kernel (scratch `elements.py`). Each row = transplant candidate into the 0947 base:
  | # | element | donor | port | status |
  |---|---|---|---|---|
  | E1 | gap-fill from low-threshold detection pool (`fill_gaps_from_low_detections`, LOWDET 0.3, ≤3% added) | thtennant gapfill | code | **LB 0.947 = TIE with 0947** (probe 56354350) → sub-floor, not a lever alone |
  | E2 | flow-seeded motion relink (`MOTION_RELINK_FLOW_*`, K 12, radius 40 µm) | thtennant gapfill/readmit/divprec | code | inside E1/E3 probes |
  | E3 | readmit discarded detections (score ≥0.965, r 4 µm) | thtennant readmit | code | **LB 0.946 = TIE/−0.001** (probe 56354352) → sub-floor |
  | E4 | division precision: sister symmetry τ 0.6→0.4, sister-min 0 | thtennant divprec | knob | untested |
  | E5 | learned re-ID appearance descriptors as pair feature (sweep picks REID_WEIGHT 4.0) | arnav170 reid3 | code | **DONOR-NULL** (its own `reid_report.json`, 5656 held-out sources): GBM top-1 0.98568 = transformer top-1 0.98568 exactly; perm-importance tf_prob 0.202 AUC, every descriptor block ≤0.0014 ⇒ appearance adds nothing past tf_prob. Sweep adjJ +0.004 (8 vids) sub-floor; LOCAL re-run reproduces it (base 0.9260 → best reid8_tight55_relaxed9 0.9312, +0.005; proxy spread driven by 8-video divJ noise). Don't port. |
  | E6 | packet grouping post-process (`grouped_postprocess`, xy radial shell profile) | newwang12 grouped | code | **CLOSED 09-20 06:22Z, no slot spent**: the grouping is INERT (`GROUP_OVERRIDES` asserted empty — the classifier drives no parameter). Real delta = zero-prob motion-relink veto + `OUTPUT_MIN_TRACK_LEN` 4, both pre-registered nulls by E54 (`wrong-association = 0`) / E53 / E52. See NEWS 06:22Z. |
  | E7 | test-time denoising-AE prefilter (30 steps, α 0.17) | ghazarosbarseghyan91 dae | code | **DONOR-INCONCLUSIVE** (its validator vs reid3 `base` on the 4 shared stems): adjJ 0.8811 vs 0.8858 (−0.005), per-stem ±0.06, spurious −15 %, missed +1.25 ⇒ no carry evidence; low priority |
  | E8 | sub-voxel centroid refinement (`refine_all_centroids`) | evgendvorkin 0.927 | code | **MECHANISM-NULL, not submitted**: metric `DistanceMatching(max_distance=7.0 µm)` vs refine shift ≤ sub-voxel (<1 µm), and both donor + 0947 round output to int voxels. Ported anyway (`kaggle/element_transplant.py`, variant `celltrack-public-0947-refine`, smoke-tested) — the CLI is the reusable transplant harness. |
  | E9 | DET 0.965→0.96 | thtennant det096 / beraterolelk | knob | beraterolelk tie ⇒ sub-floor |
  | E10 | runtime budget: frame cache 48, ILP timeout 1200, deadline degrade | thtennant fast | code | ENABLER — headroom for stacking E5–E8 in 9 h |
  | E11 | TabPFN division classifier | noisyislands | code | low ceiling (divisions ≤0.002); donor log shows only `LightGBM outer accuracy: 1.000` on 1082 division rows (saturated ⇒ likely leaky/trivial labels), no track score ⇒ UNMEASURED, don't port |
  | E13 | XGBoost division classifier (13 GT-only features) | noisyislands xgboost-division-events | code | same family as E11; divisions ≤0.002 LB → skip |
  | E14 | linker 'association MLP' on candidate distance only | noisyislands linker-association-mlp | code | pilot, weaker than 0947's tf_prob (E5 showed even rich descriptors add 0 past tf_prob) → skip |
  | X1 | SCORER EXPLOIT, not an element: fake hub node t=−1000 → roots of top-1400 components + 5 chained off-image (−10000) fake divisions appended to submission | codezzzsleep 095-owned-validation (09-19) | — | **WON'T PORT**: games the metric, no tracking change; same family as the kirneo off-image hack (0 divJ for us); a host fix would void it on private |
  | E15 | none — verbatim fork | pawanmali zhincez947-fork-v1 | — | code + env + data sources byte-identical to our 0947 base → nothing to harvest |
  | E16 | none — strict subset of 0947 (single seed, no TTA, no DeepCenter veto; thr 0.985, tighter repair caps) | binasalama learned-unet-transformer-ilp-gap-recovery (09-19 re-pull, 0-line diff vs stored) | none stated | inference-only "closing note" of its line → nothing to harvest |
  | E17 | none — param deltas only: GAP_CLOSE 5.8 (0947 5.0), TIGHT 5.5 (6.0), DeepCenter div veto OFF (0947 ON) | gautiermarti deepcenter-unet3d (v29, same 3 mounts as 0947) | self-reported val n=4 0.943; table stops at v8 LB 0.934 | same lineage as 0947; its v30 "TTA link-logit fusion" is a note, not code; tight55 already in detthr probe |
  | E18 | 402-epoch support-pack edge predictor (vs pilkwang 50ep), thr 0.99, ILP app 1.0 / disapp 2.0 | yongjilyu ct-sp402 | none | the one WEIGHTS donor seen (longer-trained model = our model gap) but `yongjilyu/biohub-ct-inference-pack` is PRIVATE (API 403) → unharvestable; watch for it going public |
  | E19 | none — older 0947 ancestor (det 0.960, ILP div 1.2, safe-div thr 0.25) | chukkkk lb-942 | 0.942 | superseded by 0947 |
  | E20 | none — no model mounts, kinematic heuristic tracker | avikdas567 3d-kinematic | none | below learned base → skip |
  | E21 | = E2 (flow-seeded relink) + tight55 + safe-div 0.25 + validator off | thtennant flow2-v1 (09-19 re-list) | env | already inside gapfill 0.947 tie → skip |
  | E23 | leonixis v1/v2/v3 pipelines: organizers' unet_transformer 40-ep from scratch (local 0.780, LB strict 0.871–0.913); p400/p314159 = pilkwang's exact SHAs | leonixis v1-infer (09-19 re-list) | weights | same recipe, far weaker than the pilkwang pair → no seat → skip |
  | E22 | none — pseudocode scaffold (ResUNet3D + MCMF linker + motion GPT), driver commented out, no weights/score | umarshad notebook82c6959503 | none | from-scratch, unrunnable → skip |
  E12 probe (14:33Z): reid3 own divdiag — `retro6_nnk2` leaves OWNED at 6/12 (retro does not reach owned cases); steal never
  run by donor. Sized: owned 6/12 FN, divJ 0.23 → ≤~0.6 ⇒ ≤ +0.037 proxy IF steal is clean (adj cost unknown). Probe =
  `kaggle/donor_probe.py` (re-runs donor's OWN validator sweep with our configs, no slot) → kernel
  `celltrack-reid3-steal-probe` v1: steal ratio 1.5/2/3, owner-um 3/4.5, no-reattach, composed w/ reid3 selected.
  Gate: div_tp up, div_fp flat, adj loss ≤0.0005.
  LOCAL VALIDATOR (15:00Z): `kaggle/local_kernel.py` runs any kernel on the 5090 (remaps /kaggle paths; `--spec` =
  donor_probe candidates; `--env KEY=VAL` pins env). CORRECTION: kernel re-materializes its repo + predict wipes output
  ⇒ resume states did NOT cache GPU predictions; `kaggle/local_predict_cache.py` now caches per-video geffs keyed by
  argv+BIOHUB_* env+weights fingerprint (`--candidate-shard I/K` splits the CPU PP sweep across processes on one cache).
  Test predict 4 videos = 2.8 min locally. **FIDELITY PASS (15:20Z):** local base over the donor's 8 stems = donor
  `validator_results.csv` (division counts identical, |Δadj| ≤ 5e-5) ⇒ local PP ranking is trustworthy.
  GPU profile: predict = U-Net bs=1 → transformer bs=1 → CPU ILP serial per video ⇒ GPU idles during ILP (bursty
  util). Fix = in-process pipelining (ILP of video k on CPU thread while GPU runs k+1) + batched windows — also Kaggle
  runtime headroom for stacking. **PIPELINED PREDICT (16:00Z, `local_predict_patch.py`):** worker-thread ILP +
  frame prefetch + numpy edge candidates → geffs BIT-IDENTICAL to cache (4/4); wall 3.01 vs 2.94 min = no win
  because GPU side (~120 s/4 videos, bs=1 encodes) now dominates. Batched 7-view dihedral TTA is NOT exact (node
  counts differ 0.01–0.1%/video — cuDNN algo per batch size flips threshold-edge detections). **Timing 16:32Z (4 test videos, CPU sweeps running
  alongside — same load for both arms):** fp32 bs=1 120.2 GPU-s · batched-TTA fp32 151.5 (+26%, SLOWER — bs=7 at 64³
  loses to bs=1; keep off) · bf16 autocast + batched 97.2 (−36% vs batched, −19% vs baseline). bf16 geff node counts
  differ −0.03…+0.15% vs fp32 — same order as batched noise; score effect unmeasured. Speed only pays if it buys an
  ensemble seat inside the Kaggle runtime. **bf16 unbatched, N=20 (19:45Z): adj 0.9025 vs fp32 0.9022, 23.8 vs 29.2
  GPU-s/video (−18.5%) → accuracy-neutral.** But the Kaggle kernel runs on T4 (sm75, no native bf16), so the shippable
  form is fp16 autocast. **fp16, N=20 (20:44Z): adj 0.9018 (−0.0004), missed GT 446, 16.8 GPU-s/video (−42%), no
  NaN → accuracy-neutral.** Shipped as `celltrack-public-0947-fp16` (`kaggle/amp_variant.py`) — the Kaggle runtime vs
  0947's 6286 s says how much room it frees for a third member. **Kaggle fp16 run (22:16Z, T4, submitted):
  test predict 9.68→5.15 min (−47 %), validator predict 16.4→7.6 min (−54 %), wall 6286→5590 s (−11 % — the visible
  run is install + CPU post-proc bound). The GPU half is what a third member would buy time from, and fp16 nearly
  halves it; the hidden run has many more videos, so its GPU share (and the saving) is larger.**

  **E24 — the kernel is 77 % SELF-VALIDATOR, not model. Stage profile of the fp16 log (5590 s total):
  install+weights 313 · test predict 301 · test PP + write submission 639 · validator predict 499 ·
  validator base PP 440 · 7 candidate PP sweeps 3034 · validator rewrite 352. The shipped submission
  costs ~1250 s; the held-out self-validation block costs 4325 s and its only product is choosing one
  PP candidate — `tight55` (`MOTION_RELINK_TIGHT_UM 5.5`) in BOTH the fp32 and fp16 runs, with every
  candidate inside ±0.002 adj (noise). Pinning that override and setting `BIOHUB_VALIDATOR_ENABLE=0`
  reproduces the same config by code (the rewrite applies exactly that one key over the env base, and
  the drift guard does not cover it), so `celltrack-public-0947-fast` (`kaggle/env_variant.py`) should
  emit a byte-identical submission in ~1600 s. A kernel RUN costs no submission slot, so this is a free
  probe: pushed 22:36Z, verify by diffing its `submission.csv` against 0947's.

  **E48 — THE DIAGONAL, MEASURED: NEITHER SCORE WE HOLD REACHES E47'S OPERATING POINT.**
  E47 argued that a threshold fails because its decision variable is intensity-ordered.
  `kaggle/prune_score_quality.py` measures it on n=723133 detections (real 12482, FP 710651), scoring the
  only two rankings the cache and volumes provide, at the collateral E47 requires (the cut is placed on
  the real detections' own quantile, so collateral is exact by construction). **Center-voxel intensity:
  AUC(real>FP) 0.5221**, and FP-recall **equals collateral at every cut** — 0.0222 at 0.02, 0.0476 at
  0.05, 0.1011 at 0.10. That is the q = r diagonal exactly, which is what a coin-flip AUC means in
  operating terms, and every point on it is a loss by E47's grid: the recovery stack sat there by
  construction. **Best incident affinity: AUC 0.6380**, FP-recall **0.0578 against the 0.75 required —
  13× short.** Pre-registered as a failure, with the reason stated first: the E46 lookalike's signature
  is HIGH affinity to the source (it sits where the cell was), so a score that ranks by how much the
  tracker wants to link cannot demote the one detection it most wants to link. Strong reading:
  real-vs-FP is not recoverable from anything downstream of the detector — the dense-model-gap
  conclusion from a new direction, a convergence to cite rather than a discovery to claim. For `g89y`
  this raises the bar from "off the intensity axis" to a number: FP-recall 0.75 at 2 % collateral, which
  every future prune proposal should be asked for FIRST — it is cheap and it killed two scores in an
  afternoon. **Caveats:** the cache carries no per-detection score column (`coords` = `[t,z,y,x]`), so the
  DETECTOR score the recovery stack actually thresholded is refuted only by analogy — get it on the next
  GPU pass, do not wake the GPU for it; center-voxel is not integrated intensity (a better feature moves
  0.5221 somewhat, not to the 0.9-plus the operating point implies); our tunet substrate, FP field 0.972.
  Detail: `interpretations/celltrack/2026-09-20_e48_no_score_we_have_can_prune.md`.

  **E47 — THE PRUNE AXIS IS PRICED: THE CEILING IS BIG (+0.1139) AND COLLATERAL IS WHAT SPENDS IT.**
  E46 pointed at bd `g89y` because the confusor is an FP. "Prunable in principle" is not a licence to
  build a GPU detector pass, so `kaggle/prune_operating_point.py` prices it on CPU from the cache first.
  ORACLE (delete every unmatched detection): band 0.1195 → 0.0056, **95.3 % rescued, net +0.1139** offline
  mutual-best — but it deletes 97.2 % of detections and keeps 0.37 % of candidate edges, so it licenses
  nothing alone. The grid is the decision: **collateral enters at ~2× its own size** (a deleted cell
  takes the true edge on BOTH sides), so `q=0.05` nets only **+0.021 even at perfect FP-recall**, while
  `q=0.02` needs FP-recall ~0.5 to break even and earns **+0.035 at 0.75**. The spec is ASYMMETRIC — an
  order of magnitude more careful with real cells than thorough with FPs. **This retro-explains the
  LB-refuted recovery stack (≤0.900) by mechanism:** a threshold is intensity-ordered, so it buys recall
  and collateral on the SAME axis at ~1:1, and E41 puts the at-risk real cells precisely there (2.4×
  fainter, 46.9 % bottom decile) — it lands on the losing diagonal by construction, not by mistuning.
  `g89y` survives ONLY because hierarchy selection picks among nested contours by consistency rather than
  a global intensity cut; that is now the **cheap precondition to test before the build** — does its
  survival decision correlate with intensity? +0.035 clears the 1.5 % bar with room and the oracle leaves
  +0.11 of headroom, but this is the offline association proxy on OUR tunet caches (FP field 0.972),
  not the LB. Detail: `interpretations/celltrack/2026-09-20_e47_what_a_pruner_must_achieve.md`.

  **E46 — THE BAND IS NOT APPEARANCE-BOUND, IT IS PAIRWISE-BOUND; THE CONFUSOR IS A NEAR-STATIC FP
  LOOKALIKE.** *Replication, not discovery:* `celltrack-confusor-pairwise-unresolvable-needs-multiframe`
  (2026-08-23) already had this — appearance top-1 0.115 vs 0.260 chance, source correlating better with
  the rival (+0.808) than the true successor (+0.540), "only global multi-frame breaks it". It was
  MISSING FROM THE MEMORY INDEX, so the prior-art check did not surface it; index repaired, and the cost
  was re-deriving a known result. E46 adds power (n=37 one movie → n=1226 across 20), a
  displacement-matched control, and one correction: the rival is an FP, not a cell.
  Re-tested at n=1226 with the E43 denominator, CPU-only, raw NCC on cell-sized cubes
  (`kaggle/outranked_band_appearance.py`): band picks true **0.4853**, mutual-best control **0.8230** —
  so appearance works, and fails exactly where it is needed. Not fast-cell decorrelation either: at
  MATCHED displacement ≥3.46, mutual-best holds **0.7182** (n=330) while the band falls to **0.4345**
  (n=695), *below chance*. Below chance is the mechanism — the rival looks MORE like the source than the
  true target does, which composes with E45's "true edge farther in 0.7984" into one picture: the
  confusor is a near-static lookalike near where the cell was, while the true cell moved far and changed.
  **Every local pairwise cue selects the same wrong target; three of four now measure below chance.** So
  a learned embedding on the same pairwise view inherits the geometry — what separates true from
  lookalike is that the lookalike has NO CONTINUATION, a tracklet-level fact. **And the lookalike is an
  unmatched FP in 0.9698 of contests (n=992, only 30 are real GT-matched cells)** — so it is PRUNABLE,
  correcting the 2026-08-23 read that it was a distinct slower cell. This closes the pairwise axis on
  mechanism and points at bd `g89y` (segment-SELECTION ILP — structural pruning, since the threshold
  route is LB-refuted at ≤0.900) with the two-pass tracklet ILP (bd `kiw1`) as the other half of the same
  fix; both are jointly-necessary 0.945 levers. Detail: `interpretations/celltrack/2026-09-20_e46_appearance_on_the_outranked_band.md`.

  **E45 — THE OUTRANKED BAND IS A FAST-CELL POPULATION (5.7× enriched), AND MOTION IS DEAD ON IT AT THE
  ORACLE.** E44 left the 11.95 % outranked band as the one live reading of the fragmentation block. This
  characterises it and then prices the cue it appears to ask for. All CPU, on the same cache.

  *It is a displacement population.* Against a marginal outranked rate of 0.1195:

  | | n | mean disp | q50 | q90 |
  |---|---|---|---|---|
  | ALL true candidate edges | 11879 | 1.726 | 1.414 | 3.464 |
  | MUTUAL-BEST | 10459 | 1.441 | 1.414 | 2.828 |
  | **OUTRANKED** | **1420** | **3.831** | **3.742** | **5.831** |

  **P(outranked | displacement > the all-true q90 of 3.46) = 0.6769 against a 0.1195 marginal — 5.7×.**
  The affinity head's ranking failure IS a fast-cell failure: where the cell moves far, a nearer rival
  takes the link.

  *So a distance prior is worse than nothing.* In the outranked band the true edge is **farther** than
  its rival in 0.7984 of cases (true median 4.12 vs rival 2.24 voxels); among non-tied pairs distance
  picks the true edge **0.1767 of the time against 0.5 chance**. Adding a distance cost to the linker
  would actively select the confusor. This independently reproduces the below-chance signature already
  recorded for motion-in-ILP, now localised to the band that actually matters.

  *And velocity does not rescue it — tested at the ORACLE.* If the cell merely moved, the cue should be
  distance from `x_t + (x_t - x_{t-1})`, built here from **GT** positions, so it upper-bounds any learned
  motion feature. On the 950 band edges whose source has a GT predecessor: oracle velocity picks the true
  target **0.6400**, static distance **0.5747** in the same frame — velocity buys **+0.065** — with a
  median margin improvement of **+0.041 µm** and `frac>0 = 0.5053`, i.e. a coin flip on WHICH edges it
  helps (the mean +0.645 is a tail artifact). **Caveat, and it runs in my favour:** this frame measures
  from the GT source position to detections, and the true detection was itself selected as the one
  nearest the GT tail, so it is biased TOWARD the true edge — which is why static reads 0.5747 here and
  0.1767 on the unbiased cache column. The bias being optimistic is what makes the result usable: the
  true ceiling is **at most** 0.64. An oracle that barely separates needs no learned version.

  **Conclusion: the outranked band is not reachable by any geometric cue — not distance, not velocity,
  not at the oracle.** It is an appearance/affinity-quality problem, consistent with
  `celltrack-four-cues-fail-confusor-detection-side` and with the confusor sitting at the voxel
  resolution limit. Do not re-file a motion or distance term for it. Note this also explains the older
  "motion_distance lever aggregate-small" reading without contradicting it: aggregate-small is exactly
  what a real effect concentrated on 11.95 % of edges looks like — here the concentration is real and
  the cue is still dead, so the axis closes on mechanism rather than on dilution.

  **E44 — THE TERMINATION KNOB POINTS THE WRONG WAY. Raising disappearance cost past ~1.5 LOSES GT
  edges, so bd `d5er`'s proposed 3/4/6 direction is refuted on its own mechanism.** E43 put the
  fragmentation block on a solver knob; this prices the knob. The cached candidate graph is pure dt=1,
  so the faithful model is a per-frame-pair global assignment with a constant termination cost per
  unmatched node (costs `-log(affinity)`, termination as a square augmentation, decomposed over the
  connected components *of each frame pair* — components over the whole movie chain all 100 frames into
  one blob and buy nothing). Sweeping `term` over 8 values, CPU-only, no detector pass:

  | term | 0.25 | 0.5 | **1.0** | **1.5** | **2.0** | 3.0 | 4.0 | 6.0 |
  |---|---|---|---|---|---|---|---|---|
  | GT-edge recall | 0.7643 | 0.8455 | **0.8582** | **0.8589** | **0.8581** | 0.8532 | 0.8454 | 0.8293 |

  The curve is **non-monotone with a broad shallow plateau at 1.0–2.0**, and every step above it costs
  recall — 6.0 gives up 0.0296 against the peak, an order of magnitude over the 0.01–0.02 noise floor.
  d5er's hypothesis was that cheap termination (`BIOHUB_ILP_DISAPPEARANCE_WEIGHT=2`) lets the solver
  abandon tracks it should continue, so 3/4/6 would buy edges back. The mechanism **backfires**: with a
  global assignment, expensive termination forces every node to link to *something*, and in a field
  carrying ~980 FP per frame against ~17 GT cells the something it grabs is an FP rival that steals the
  true target. The shipped weight of 2 already sits on the plateau — **this knob is spent, not mis-set.**

  *Both axes degrade together, now measured rather than inferred.* Counting mislinks over the honest
  denominator — links whose two endpoints are BOTH GT-matched detections — gives 14 (0.0013) at term
  1.0, 18 (0.0017) at 1.5, 20 (0.0019) at 2.0, and 43 (0.0042) at 6.0. So climbing from the plateau to
  6.0 costs 0.0296 recall AND multiplies mislinks by 2.4. There is no trade to exploit in either
  direction. (Precision among real cells is near-perfect throughout — ~10.6 k GT-matched pairs linked
  against tens of mislinks — which independently reproduces d5er's original "mislinks only 74".)

  *Two denominators, because the naive ones lie.* The sweep's first `mislinks` column (taken links that
  are not GT edges) sat at 0.98 for every value of `term` and measured nothing — almost every link is
  FP-to-FP and was never a mistake about a real cell; the honest version is the one above
  (`kaggle/termination_cost_sweep.py`). And the recall LEVEL
  (0.859) is not comparable to the real tracker's 0.9475: this model has no distance gate and no
  division term. **The shape across `term` is the result; the level is not.** Caveat as E43: this is our
  tunet cache (node recall 0.697–0.750), not 0947's 0.824.

  *What survives.* E43's 11.95 % outranked band is untouched by this — those true edges lose to a
  **rival**, not to termination, and a termination cost cannot flip a swap. That band, with its median
  margin of 0.0985, remains the one live reading of the fragmentation block, and it is an
  affinity-*ranking* problem (global solver, or a better head) rather than a cost knob.

  **E43 — FRAGMENTATION IS A *SELECTION* FAILURE, NOT A CANDIDATE FAILURE. The 4.46 % block is on the
  table and the solver declines it — so it is solver-reachable, and reachable WITHOUT A DETECTOR PASS.**
  E42 closed the detection axis on the 0947 base, which left bd `d5er`'s fragmentation block as the
  largest addressable population still standing: an edge whose **both endpoints are detected AND matched**
  yet carries no predicted link. That is an omission, not a confusion — but "omission" hides two bugs
  with opposite fixes, and nobody had separated them:

  - **CANDIDATE** — the pair was never proposed. Then no solver knob reaches it; the fix is a wider gate.
  - **SELECTION** — the pair was proposed, with an affinity, and the solver preferred to terminate. Then
    the disappearance/termination cost reaches it, and the affinity head's *ranking* bounds how much.

  The `real_scratch` cache settles this for free: it stores **487238 proposed edges with their
  affinities**, so the candidate graph the linker saw is on disk. No GPU, no detector re-run.
  Instrument: `kaggle/fragment_selection_gate.py` (CPU, ~1 min, 20 movies, n=12346 GT edges).

  1. **The gate passes decisively — it is SELECTION.** Of GT edges whose endpoints both match at the
     official 7 µm ruler (11938 = 96.70 %), **99.51 % already have a candidate edge with an affinity**.
     Only **59 (0.48 % of all GT edges)** have none, and those are median **11.72 µm** displacement —
     outside the 10 µm gate, i.e. gate-bound and far too small to matter. Candidate generation is NOT
     the bug. Every fragmented edge was on the table.
  2. **The affinity head is mostly right, and wrong by a thin margin.** The true edge is **mutual-best**
     (top-ranked out of its source AND into its target) in **88.05 %** of cases. The **11.95 %** that are
     outranked lose by a **median margin of only 0.0985** (true 0.285 vs rival 0.510). A thin margin is
     precisely what a *global* solver flips and a greedy one cannot — which is the mechanism behind
     `celltrack-linker-is-the-gap` (global ILP broke the 0.902 wall). It also means these are not
     hopeless: they are near-misses, not confident errors.
  3. **The weak tail is real and it is ranked-down, not just low.** **12.48 %** of true candidate edges
     carry affinity **below coin-flip** (q10 = 0.433), and of those only **23.01 %** are mutual-best.
     So a true edge scored under 0.5 usually *also* loses its ranking — low affinity and bad ranking are
     the same population, not two independent taxes.

  **What this licenses and what it does not.** The 11.95 % outranked band is larger than the 4.99 %
  fragmentation rate, so fragmentation sits *inside* it — consistent, and it means the termination cost
  is not the whole story. Raising it recovers a true edge that lost **to nothing**; it cannot fix one that
  lost **to a rival** (that is a swap, and swaps show up as mislinks, of which d5er counts only 74).
  **Caveats that must travel:** measured on OUR `real_scratch` cache (node recall 0.697–0.750), not the
  0947 base (0.824); and the 7 µm matching here is nearest-detection, not the official metric's global
  assignment.

  **E40 — THE DoG UNION IS DEAD ON THE CONDITIONAL, AND E27'S KILL WAS SCALE-BOUND. Five CPU-only
  instruments, no GPU.** E39 left detection as the only axis with a ≥1.5 % mechanism, priced as
  GPU-blocked. It is not: a union with an *independent* detector needs no local card (a Kaggle fork run
  is free), and the gate — does the second detector find what the first missed — runs CPU-only against
  the cached `real_scratch` coords and the train `.geff`s.

  1. **E27's DoG kill was a SCALE artifact, and correcting it does not save the axis.** E27 refuted DoG
     as *preprocessing* at the default 4 µm band; the band is the whole result. GT recall at cell
     separation (2.87 µm): default `(2.83, 4.0, 5.66)` = **0.16–0.30** — E27's number — against a 2 µm
     band `(1.4, 2.0, 2.83)` = **0.62–0.74**, versus the learned detector's 0.75 on the same frames. So
     the learning-free detector is *competitive*, which is exactly what makes the next line decisive.
  2. **But the errors are DEPENDENT, so the union pays nothing.** n=256 GT over 8 movies: cache 0.7500,
     DoG 0.4141, union 0.7891 (+0.0391 marginal). The conditional kills it: **P(DoG finds it | cache
     missed) = 0.1562** against DoG's own 0.4141 marginal — DoG is **2.7× LESS** likely to find a cell
     the learned detector missed than to find an average cell. Shared blind spot, not complementary
     coverage. A union member must beat its own marginal on the conditional; this one is a third of it.
     **Widened 12.5× and it gets stronger, not weaker:** n=3206 over 20 movies, cache 0.6974, DoG 0.4498,
     union 0.7396 (+0.0421), and the conditional **0.1392 against a 0.4498 marginal = 3.2× less likely**.
     Not a small-sample verdict.
  3. **My own explanation for that was then falsified. The missed cells are VISIBLE.** I expected
     signal-absent (both detectors blind to dim cells), which would have closed the single-frame axis
     entirely. Measured local background-subtracted contrast at every GT centre: only **3.0 %** of
     missed cells have no local intensity excess, median contrast 1.29 vs 2.00 for found, and ~64 % are
     as bright as routinely-found cells. Bright, not blind.
  4. **Nor is it a merge with a neighbour.** n=3864 missed: **72.5 % have a detection within 5 µm**,
     median distance **3.83 µm** — but nearest *GT* neighbour is 24.8 µm for missed cells vs 23.2 µm for
     found. Identical. There is no second cell nearby to merge with.
  5. **The matched null says the nearby detection is the cell's OWN.** With ~854 detections in a 104 µm
     cube, 3.83 µm could be chance, so: score GT against detections from frame `(t+50) % 100` — same
     count, same spatial distribution, same density, identity destroyed. REAL nearest median **3.83 µm**
     vs NULL **7.53 µm**; own-frame closer in **0.807** of cases (0.5 = chance); REAL median dz −3.25
     vs NULL +0.00. The association is real.

  **E41 — WHAT THAT MEANS FOR E36, INCLUDING THE PART OF E40 THAT DOES NOT SURVIVE ITS OWN AUDIT.**

  - **Self-correction first.** I read the missed-subset signature (median dz = −3.25 µm = exactly −2
    z-voxels) as a discrete offset. It is **partly tautological**: one z-voxel is 1.625 µm, so
    conditioning on `distance > 2.87 µm` mechanically selects |dz| ≥ 2 voxels whenever the error is
    z-dominated. The per-axis voxel histogram over **all** GT (not the missed subset) is **unimodal at
    0** — dz `−2:10.9 % −1:28.4 % 0:35.0 % +1:15.0 % +2:3.9 %` — so there is **no −2-voxel mode and no
    plumbing bug**. I looked for one and it is not there.
  - **What survives unconditioned is a systematic NEGATIVE-Z BIAS, ~2.5:1** (−1 vs +1: 28.4/15.0; −2 vs
    +2: 10.9/3.9), mean dz −0.57 µm over all GT. dy/dx are near-symmetric by comparison. This is real
    and not a selection artifact — but it is **0.35 of a z-voxel**, and the official matcher is at
    7 µm, so a constant z correction is free to apply and there is **no mechanism by which it delivers
    1.5 %**. Directional at best; recorded, not planned around.
  - **The load-bearing consequence is for E36's MECHANISM, not its arithmetic.** E36 explained the
    0.824-vs-0.998 recall gap as a GT node *borrowing a neighbouring cell's detection* and still
    scoring found at 7 µm. On this cache that explanation **does not hold**: the nearest GT neighbour is
    24.8 µm away, so there is no neighbour to borrow from, and the null shows the matched detection is
    the cell's own, displaced. The 2.87 µm "recall" number is therefore measuring **localization
    scatter, not missing cells** — it is a ruler-choice, and recall at the official 7 µm really is
    ~1.0. **E36's 574-edge (2.5 %) detector-caused edge loss was derived from that mechanism and now
    needs re-derivation before any GPU is spent on it.** bd `lwrr` is re-priced on that basis; bd
    `1ocj` closes negative.
  - **Caveats that must travel with this.** (a) The cache is `real_scratch` = OUR tunet detector, not
    0947; this cache reads 0.697–0.750 where E36 reads 0.824, so the config differs and extending to
    0947 is an inference, not a measurement. (b) REAL and NULL **mean** dz are near-identical (−1.19 vs
    −1.27) while only the medians separate — the mean does not discriminate here and I have not
    explained why. (c) Every displacement discussed is 3–4 µm, i.e. **inside** the official 7 µm
    matcher, so nothing here claims a score effect on its own.

  **E41(b) — SELF-CORRECTION WITHIN THE HOUR: E36's SIZE SURVIVES. I OVERCLAIMED. The mechanism was
  wrong; the number was not.** E41 said the 574-edge (2.5 %) lever "needs re-deriving", implying it would
  shrink. Re-derived (bd `og4x`, CPU-only, same cache): a GT edge counted dead when either endpoint has
  nothing within the ruler —

  | ruler | GT nodes with nothing in reach | GT edges killed |
  |---|---|---|
  | 2.87 µm | 3864 / 12777 = 30.2 % | 4976 / 12346 = **40.3 %** |
  | 5.00 µm | 1062 = 8.3 % | 1502 = 12.2 % |
  | 7.00 µm (official) | 288 = **2.25 %** | 408 = **3.30 %** |

  - **40.3 % edge loss is impossible against a 0.92 LB**, which is the cleanest possible proof that 2.87 µm
    is the wrong ruler — exactly what E41 argued.
  - **But at the ruler that scores, the loss is 3.30 %**, on a cache reading recall 0.697–0.750 against
    E36's 0.824. A better detector lands lower, i.e. **right on E36's 2.5 %.** So E36's *size* is
    independently reproduced; only its *story* (borrowing a neighbour's detection) was wrong.
  - **And this SHARPENS the lever rather than closing it.** The 2.25 % of GT nodes with nothing within
    7 µm are genuinely, officially undetected — no displacement explanation available, since the ruler is
    the official one. That is a real recall gap with real material behind it, and it is the same
    population E40 measured as only 2.1 % blind at 10 µm. Detection stays the live axis, still
    GPU-priced, still parked. bd `lwrr` is therefore **re-instated, not retired** — with the correct
    population (nodes blind at 7 µm, n=288 here) rather than the 2.87 µm population (n=3864) that is
    13× larger and mostly localization scatter. Targeting the wrong one of those two is the actual
    mistake E36 would have caused.

  **E42 — THE 3.30 % LEVER IS REAL, AND IT LIVES IN THE FAINTEST DECILE — WHICH IS THE ONE LEVER
  ALREADY LB-REFUTED. The detection axis closes on an ARGUMENT, not on "the GPU is parked".** E41(b)
  localised the officially-costly population to the 288 GT nodes blind at 7 µm. Contrast measured at
  those centres against an equal-sized found sample **from the same frames** (so illumination, depth and
  movie are controlled):

  | | blind at 7 µm (n=288) | found, same frames (n=288) |
  |---|---|---|
  | contrast q10/25/50/75/90 | −0.02 / 0.12 / **0.29** / 0.66 / 1.18 | 0.27 / 0.42 / **0.70** / 1.27 / 1.91 |
  | no local intensity excess | **10.8 %** | 0.7 % |
  | at/below found's 10th pct | **46.9 %** | — |
  | touching the volume border | 9.7 % | — |

  1. **These are a different population from E40's.** The 2.87 µm "missed" cells were only 3.0 %
     excess-free at median contrast 1.29 — bright, and displaced. The 7 µm blind cells are 10.8 %
     excess-free at median **0.29**, i.e. **2.4× fainter than what the detector finds in the very same
     frame**. So the two rulers are not two readings of one gap; they name two unrelated failures, and
     only this one costs score.
  2. **Material exists — 89 % do have a local excess — so this is a detector SENSITIVITY gap, not
     blindness.** But 46.9 % sit in the bottom decile of what the detector already accepts, which means
     reaching them is a **threshold** move, not an architecture move.
  3. **And the threshold move is the one thing already refuted on the real board.** The recovery-stack
     arm is LB-refuted at ≤0.900 precisely because a threshold low enough to recover faint cells starves
     precision, and the candidate distribution already carries ~980 FP/frame against ~17 GT/frame with a
     +26 % surplus sizing a 5.5 % swap. Recovering 3.30 % of edges by admitting the faintest decile buys
     them at an FP rate that the same measurements say costs more than it pays.
  4. **What this does NOT close.** A detector that separates these cells *without* lowering the
     threshold — better sensitivity at equal precision, i.e. a genuinely better dense-regime model — is
     untouched by this and remains the standing frontier-replication bet. What closes is the cheap
     version: no threshold sweep, no recovery stack, no post-hoc recall pass on the 0947 base is owed a
     card. bd `lwrr` closes on that argument rather than on availability.
  5. **Honest gap in the evidence:** the cached `.npz` carries coords and edges but **no confidence**, so
     I could not draw the actual precision/recall trade at the threshold that would admit these 288 —
     the FP cost above is carried over from the prior FP-competition and recovery-stack measurements,
     not measured here. That is the one thing that would make this quantitative rather than argued.

  **E37(d) — THE REJECT HISTOGRAM, MEASURED. Pair existence is the binder, the peak-side gates are
  second, and loosening them takes the filler to its ceiling — where it is still ~1 %.** Both free
  instrumented kernels returned (00:54Z). `gapdiag` (stock gates) emits a submission with node and
  edge counts **identical** to `-gapfill` (123232 / 119036), so the instrumentation is score-neutral
  and the fork is faithful. Summed over the four test movies, at g=3 (the loosest gate):

  | | gapdiag (stock) | gaploose (loosened) |
  |---|---|---|
  | dangling ends considered | 5030 | 4873 |
  | **ends with NO start inside the gate** | **2740 (54 %)** | **2703 (55 %)** |
  | in-gate pairs (raw, many-to-many) | 1322 | 1130 |
  | rejected by the context filter | 717 | 619 |
  | rejected: no peak chain exists | 577 | 333 |
  | **pairs accepted at g=3** | **27** | **148** |
  | filler total: nodes / edges | 167 / 240 | **814 / 1165** |

  1. **E37(b)'s offline ceiling is confirmed by the kernel's own counters: 54 % of dangling ends
     have no candidate restart inside the euclidean gate at all**, three frames out. No knob reaches
     those; the partner is not there to be found.
  2. **Among the pairs that DO exist, the peak-side gates were genuinely binding** — loosening score
     (0.5→0.35), radius (3.5→5.0 µm) and synthetic tolerance (0→1) cut `chain_none` 577→333 and took
     accepted g=3 pairs 27→148, a **4.9× filler yield** (240→1165 edges). So the named suspect was
     real; it was just never the ceiling.
  3. **And it lands exactly on E37(b)'s predicted ceiling, which is the point.** gaploose's total
     change over base is +1693 nodes / +1894 edges = **1.60 % of predicted edges**, against the
     predicted perfect-filler cap of ~1870 edges / 1.6 %. The mechanism is now exhausted by
     construction: there is no third arm here, because there is nothing left to collect.
  4. **Corrections to E37 as first written:** the shipped `-gapfill` delta is +424 nodes / **+488**
     edges, not +534; and only **167 / 240** of that is the low-detection filler — the rest comes
     from the single-frame and gap-2 closers, which run in the same stage and which I had folded in.
     The instrumented number is the attributable one.
  5. **Submitted anyway, and deliberately** (slot spent 00:57Z, 4 remaining): the arm sits at its own
     measured ceiling, no competing use existed for the slot, and it settles the one thing the
     artefacts cannot — whether repair yield converts to score at all. Expect a tie or a small loss:
     193 of its nodes are SYNTHETIC straight-line interpolations with no detector evidence behind
     them. A tie would say the filler's edges are right and the axis is simply too small; a loss
     would say the fabricated nodes cost more than the true ones earn.

  **E39 — THE "ONLY LEVER LEFT WORTH ≥1.5 %" PRICES OUT AT ~0.9 %, AND IT CAN BE PRICED WITHOUT
  THE GPU. The fork-unique nodes are 73–87 % LOCALIZATION BLIPS, not missed cells.** The 09-19 NEWS
  block called the fork-unique node sets "the only thing left that could plausibly be worth ≥1.5 %"
  and sent it to GT on holdout20 (bd `d8rs`, a GPU run). It can be decomposed GT-free first: assign
  every node to its own fork's track (connected component of that fork's edges), match nodes to the
  other fork per frame, and report each track's *unmatched fraction*
  (`scratchpad/unique_tracks.py`, CPU, seconds). At MATCH_UM = 2.0 µm:

  | | v1329f-only | 0947-only |
  |---|---|---|
  | unmatched nodes | 7987 | 7311 |
  | **whole track invisible to the other fork** | **162 tracks / 1194 nodes (14.9 %)** | **54 / 415 (5.7 %)** |
  | majority-unmatched | 115 / 980 (12.3 %) | 80 / 560 (7.7 %) |
  | minority (blips inside a track both forks have) | 1992 / 5813 (**72.8 %**) | 1934 / 6336 (**86.7 %**) |

  1. **Most fork-unique nodes are not missed cells — they are the same cell localized differently.**
     A "blip" sits inside a track the other fork also has, and since BOTH forks' edges are all dt=1
     with zero gaps, the other fork did not miss that frame: it placed the node >2 µm away. That is
     a position disagreement, and it costs nothing at the official 7 µm matcher.
  2. **The tolerance sweep proves the split rather than assuming it.** Re-run at the GT NN floor
     (2.87 µm): the blip population collapses 7987 → 5947 and 7311 → 5275, while the whole-invisible
     track count barely moves, 162 → 156 and 54 → 53. A real recall difference is tolerance-
     insensitive; a localization difference is not. The two populations behave as claimed.
  3. **So the union's ceiling is the whole-invisible tracks, and they are small.** 1194 nodes in 162
     tracks is ~1 % of the node table and, at ~(len − 1) edges per track, **~1032 edges = 0.87 % of
     the 118548 predicted** — *if every one of them is a true cell and the union costs nothing
     elsewhere*. The reverse direction is 415 nodes / ~361 edges = 0.30 %. **Both are under the
     1.5 % bar before any GT is consulted**, so bd `d8rs` cannot license a ≥1.5 % move and drops off
     the critical path. (It stays worth running as an *instrument* if the card is ever free: which
     fork's unique tracks are real is the cleanest read we have on which detector to build on.)
  4. **The asymmetry is the interesting residue.** v1329f finds 3× as many whole cells 0947 misses
     as the other way round (162 vs 54), yet scores 0.008 LOWER on the LB. Either those tracks are
     false positives, or v1329f pays for them elsewhere — the same tension the E33 ensemble read
     left open, now with a size on it.

  Rule: **price a GPU run's ceiling from the artefacts it would be scoring before spending the
  card on it.** Recorded 09-20.

  **E38 — fp16 IS PROVEN PRECISION-NEUTRAL WITHOUT ITS LB SCORE, AND THE FINAL-2 HEDGE IS REAL
  AFTER ALL — BUT ONLY ON THE HONEST DENOMINATOR.** Pairwise disagreement of every banked
  submission against 0947, via the instrument that already exists for exactly this
  (`kaggle/tracker_agreement.py`, CPU, seconds — I wrote a duplicate in scratch before finding it):

  | vs 0947 | LB | shared edges / union | a_only | b_only |
  |---|---|---|---|---|
  | `fp16` | pending | **0.9956** | 257 | 268 |
  | `0947-gapfill` | 0.947 | 0.8792 | 7389 | 7877 |
  | `0947-readmit` | 0.946 | 0.8690 | 7618 | 9108 |
  | `v1329f` | 0.939 | **0.8371** | 10356 | 10700 |
  | `v50` | 0.907 | 0.8368 | 10417 | 10671 |

  1. **fp16 shares 99.56 % of the union with 0947 — 257 disputed edges out of 118548.**
     Half-precision is neutral *as a mechanism*, established offline for zero slots. Its pending
     LB score is a formality: anything but 0.947 would indict the ruler, not the model.
  2. **v1329f disputes ~10.4 k of 0947's edges — 16 % of the union.** The final-2 pair really is
     two different graphs, and the hedge is worth its slot (E33's pick stands, unchanged).
  3. **My first pass at this said 1.6 %, and it was wrong in the familiar way.** I measured edge
     agreement *conditioned on node pairs that matched*, which throws away precisely the region
     where the two forks disagree — the nodes one has and the other does not. The conditioned
     number (0.984) and the union number (0.837) differ by **10×**, and the conditioned one is
     the flattering one. Fifth occurrence of the denominator error in this campaign; the tell was
     that I reported a disagreement rate whose denominator was itself selected for agreement.
  4. **Where the forks separate is detection.** v50 and v1329f have identical node counts (123485)
     and agree with 0947 to within 0.0003 of each other, consistent with v50 being v1329f plus the
     surgery cell. Across the table the disagreement scales with the node-set difference, not with
     any association setting — the same conclusion E36 and E37 reach from the other side.

  Rule: **a disagreement rate measured only where two outputs already agree is not a disagreement
  rate.** Compare on the union, and check whether an instrument for it is already in the repo
  before writing a second one. Recorded 09-20.

  **E37 — THE E36-MATCHED REPAIR IS ALREADY BUILT AND ALREADY SHIPPING; IT JUST YIELDS 10 %. The
  0.947 tie of `celltrack-public-0947-gapfill` was a YIELD result, not a neutrality result, and we
  had read it as the latter.** Its `fill_gaps_from_low_detections` is exactly what E36 asks for: it
  bridges a dangling track end at `t` to a dangling start at `t+g+1` through the detector's
  SUB-THRESHOLD peaks, inserting real nodes where a cell was missed. Offline on the two cached
  submissions, CPU-only: the base 0947 graph has **122808 nodes / 118548 edges, every edge dt=1, zero
  gap edges, zero isolated nodes, and 4260 no-parent + 4384 no-child dangling ends**. The filler's
  pool is ample — **39523 free sub-threshold peaks ≥ 0.5** across the four test videos, after
  excluding those already on a node. Yet the fork's submission differs from the base by **+424 nodes
  (+0.35 %) and +534 edges (+0.45 %)** — an order of magnitude below the 1.5 % bar, which is precisely
  why the board returned the same 0.947 to four decimal places. The 3 % add cap (`~3700` nodes) is
  NOT what binds; something upstream of it rejects ~90 % of the dangling pairs. The gate chain, in
  order: the euclidean gate `GAPFILL_STEP_UM × (g+1)` (10 µm at g=1); `context_ok`'s direction test;
  and `chain_for`, which needs a free peak within **`GAPFILL_PEAK_RADIUS_UM = 3.5 µm`** of the
  straight-line sample at **EVERY** missing frame, since `GAPFILL_ALLOW_SYNTHETIC = 0`. E36 measured
  the loose endpoints' nearest prediction at **4.892 µm median — past that 3.5 µm radius**, which
  makes the radius the named suspect rather than a knob picked off a list. Two FREE kernel runs
  (no submission slot, no local GPU) settle it: `celltrack-public-0947-gapdiag` adds a reject counter
  at each gate at stock values, and `celltrack-public-0947-gaploose` runs the same counters with
  `MIN_SCORE 0.5→0.35`, `PEAK_RADIUS 3.5→5.0 µm`, `ALLOW_SYNTHETIC 0→1`. **Rule: a tie against a
  base is only evidence about a mechanism once you have measured how much of the mechanism reached
  the output — diff the artefacts, don't read the score.** Pushed 00:36Z 09-20; bd E37.

  **E37(b) — SELF-CORRECTION, SAME HOUR, AND IT CAPS THE ARM I JUST BUILT. The peak radius is not
  what binds; the PAIRS do not exist.** Asking the base graph directly how many dangling ends have
  ANY dangling start within the euclidean gate `5 µm × (g+1)`, by optimal assignment per frame:
  **g=1 → 50 pairs, g=2 → 316, g=3 → 569, total 935**, against **3469 interior dangling ends and
  3176 interior starts**. So a PERFECT filler at `MAX_GAP=3` could add at most ~935 bridges ≈ 1870
  edges = **1.6 % of predicted edges**, and the shipped fork already banked 534 of them. The whole
  remaining headroom on this mechanism is **~+1.1 %** — below the 1.5 % bar — and loosening the peak
  radius can only ever collect part of it. My "yield is 10 %" reading used the dangling-end count as
  the denominator when the reachable denominator is the in-gate pair count; against 935 the fork is
  already running at roughly half. **The real shape of the loss is worse for post-processing and
  better-aligned with E36: 2500 of the 3469 dangling ends have no plausible restart within three
  frames at all.** Those tracks do not resume — the cell stops being detected for a long stretch, not
  for a frame — so no bridging rule reaches them and no gate widening is licensed (5 µm/frame already
  exceeds the GT median step of 1.82 µm and the 10 µm at g=1 is the max observed GT step). **This is
  the same detector-weights conclusion E36 reached, now with a post-processing ceiling attached to
  it: ~1.1 % is ALL that repair-side work can buy on the 0947 base.** The two free runs still finish
  (they cost nothing and give the exact reject histogram), but `gaploose` is now priced as a
  sub-bar directional arm, not a submission candidate on its own.

  **E37(c) — AND A THIRD OF THOSE ENDS ARE NOT LOSSES AT ALL: CELLS LEAVE THE FIELD.** Dangling ends
  are strongly enriched at the volume boundary against the node population they come from (within 6
  voxels in y/x or 2 in z, on a ~63×254×254 volume): **0.791 vs 0.161** on `44b6_0113de3b`, **0.402
  vs 0.119** on `6bba_05db0fb1`, **0.364 vs 0.201** on `6bba_05b6850b` — 1363 of the 3469 ends, ~39 %,
  are cells exiting the imaged volume, which no tracker should link and no repair should invent.
  The one exception is **`44b6_0b24845f`, flat at 0.21 vs 0.176**: its 1217 ends are genuinely
  INTERIOR terminations, it holds the largest sub-threshold pool (26152 free peaks of the 39523) and
  it took the largest share of the filler's gain (+287 of the +424 nodes). **So the repair headroom
  is not spread over the test set; it is concentrated in one of the four videos**, which also caps
  how much any post-processing change can move a four-video LB score. Correcting E37(b) downward
  with this: the addressable interior ends are ~2100, of which only the in-gate subset is reachable
  at all. Cheap rule earned here: **before sizing a repair, subtract the losses that are physically
  correct — a track that walks out of the volume is not a broken track.**

  **E36 — THE OMISSION RESIDUE IS DETECTION, NOT ASSOCIATION, AND NODE RECALL AT 7 µm HIDES IT.
  Reverses the tree's standing "the lever is association, not detection" reading — for the 0947
  champion, measured, CPU-only.** E35 counted 713 in-gate fragmented GT edges and read 638 of them as
  slot competition. That reading was still too generous to the association axis: it never checked
  whether the GT endpoints were matched to the RIGHT prediction. They mostly are not. Match residual
  (GT node → its assigned prediction) is **3.362 µm median / 6.148 p90 on fragmented edges vs 1.724 /
  3.35 on linked ones** — the fragmented endpoints sit past the **2.87 µm GT nearest-neighbour floor**,
  so the 7 µm matcher is pairing them with a NEIGHBOURING CELL's detection. Gating on a confident match
  (both endpoints inside the floor) leaves **139 of 713**: `disappearance` 33 · `source_stole` 57 ·
  `target_taken` 24 · `both_reassigned` 25. The other **574 (81 %)** are matcher stretch.
  `_nearest_prediction` then settles what the stretch means, ignoring the assignment entirely: of the
  **827 loose endpoints, only 7 have ANY prediction within 2.87 µm**, median nearest **4.892 µm**. So
  they are **MISSING DETECTIONS (99.2 %), not contested ones** — no assignment or cost change can reach
  them. **Sizing: detection-caused edge loss 574/23080 = 2.5 % of edges, which CLEARS the 1.5 % bar;
  genuine association residue 106/23080 = 0.46 % and disappearance 33/23080 = 0.14 %, both far under
  it.** The mechanism that hid this: the official node metric also matches at 7 µm, so a GT node whose
  own cell was never detected still scores as a detected node by borrowing a neighbour's detection —
  **node recall stays ~1.0 while every edge through that node dies.** That is exactly how
  [[celltrack-local-official-metric-harness-reproduces-LB]]'s "recall 0.998 DEAD" survived: recall was
  never measured at a tolerance that could see a cell-swap. **Measured directly: recall at cell
  separation is 0.824** — 4292 of 24399 GT nodes (17.6 %) have NO prediction within 2.87 µm — against
  the ~0.998 the 7 µm ruler reports. **It is FLAT across all nine full 40-video caches this project has
  ever produced: 0.8173 – 0.8304, a 1.3-point spread**, so every post-processing and ILP knob we have
  swept leaves it untouched; it is a detector-weights property, GPU-priced. Note the honest bound: the
  metric LAUNDERS most of those misses, because the borrowed neighbour detection is itself correctly
  linked in its own track, so the edge maps onto a linked pair and scores as present. **The realised
  cost is the 574 edges (2.5 %), not the 4292 nodes (17.6 %)** — the node figure is the size of the
  detector's real blindness, the edge figure is what the board can see. Also finishes the distance-knob question
  (bd P1, filed this session): on the confident 139 the winner is shorter in 72 % of cases, true step
  5.139 µm vs winner 3.25 µm — but the population it governs is 0.46 % of edges, so an ILP distance
  reweight joins the disappearance weight as sub-bar. **Every remaining ILP cost knob is retired by
  size.** The live lever on the 0947 base is recall of *undetected cells in crowds* — which is the same
  bottleneck A ([[celltrack-two-bottlenecks-detector-separation-head]], finer decode) the tree already
  names, now sized on the CHAMPION rather than on our own detector, and now with an edge-loss price
  attached. **Rule earned: when the metric's matcher is looser than the object's own separation, a
  high recall is not evidence the object was found.**

  **E35 — THE DISAPPEARANCE SWEEP IS A 0.32 % EXPERIMENT; DON'T PAY THE GPU ARM. And the omission story
  IS the confusor story: 86 % of recoverable omissions are slot COMPETITION. CPU instrument, run before
  the run.** `fragment_audit.py` decomposes each in-gate fragmented GT edge by what the PREDICTED graph
  did with its two endpoints (`_verdict`): if the source ended its track and the target took no parent,
  the solver paid a disappearance and the disappearance weight is the binding knob; if either slot went
  to some other node, it is a competition failure no cost on disappearance can repair. Over the 40
  cached validator videos, **23080 matched GT edges, 739 fragmented (3.2 %), 713 in gate:
  `disappearance` 75 · `source_stole` 195 · `target_taken` 172 · `both_reassigned` 271.** Only **11 %**
  of the recoverable omissions are disappearances — **75/23080 = 0.32 % of edges is the ENTIRE ceiling of
  `logs/run_ilpdisapp.sh`**, five times under the 1.5 % bar, so the sweep is retired before it cost the
  ~23 min GPU arm E34 priced. The other **638 = 2.8 % of edges** are slot competition — a node other than
  the true one won the successor or the parent — which lands back on the confusor/association axis the
  whole tree keeps converging to; `both_reassigned` (271, the largest single class) means BOTH endpoints
  were re-used elsewhere. `_rival_gap` then sizes the winner: median **3.25 µm**, only **55/638 (8.6 %)
  within 1.6 µm** (a decode duplicate of the true cell) and **330/638 (52 %) beyond 3 µm** — past the
  2.87 µm GT nearest-neighbour floor, so **the slot-winner is usually a GENUINELY DIFFERENT CELL, not a
  duplicate detection.** Finer decode / NMS is therefore NOT the repair for this residue; edge-affinity
  discrimination is. Supersedes the E26/E28 reading of omission as absence — it is mostly DISPLACEMENT.
  **Methodology correction (self-inflicted, third denominator slip this session):** the first pass used
  greedy `cKDTree` NN at 5 µm, which lets several GT nodes claim one prediction and INVENTS fragments;
  fragmentation swung 0.62 % → 0.92 % → 1.91 % across 2/3/5 µm. `match_nodes` now mirrors the official
  ruler — `core/metrics/matching.py` `DistanceMatcher`, an OPTIMAL per-frame `linear_sum_assignment`
  under `_MAX_DISTANCE_UM = 7.0`. All numbers above are at that faithful setting. The RETIREMENT was
  stable across every tolerance tested (disappearance 21/85, 35/166, 64/397, 75/713 — always a small
  minority); the magnitudes were not. **Rule earned: match with the metric's own matcher, or the rate
  you report is a property of your tolerance, not of the tracker.**

  **E34 — PLUMBING CORRECTION: E29's candidate cache has NEVER ONCE FIRED, so the "ILP arms are free"
  saving is projected, not banked. `find runs -name '*.candidates.npz'` returns **0 files** across all
  21 prediction-cache dirs (which hold 40 `.geff` + 40 `.retention.jsonl` each and no candidates). Cause
  is ordering, not a bug: the `_save_candidates` code landed in `kaggle/local_predict_patch.py` at commit
  `6c1424a` 23:06Z, while the only run since (`secedge_0.30`) started 22:41Z and finished predicting
  ~23:04Z — it ran the older patch. Confirmed by grepping its restored `tracking_repo` predict script:
  0 hits for `_save_candidates`, but the CANDIDATES→vectorized rewrite IS present, so it was patched, by
  the previous revision. The patch itself is sound: applying it to a pristine
  `external/frontier_ds/biohub-tracking-support-pack-50ep-v1` script parses (`ast.parse` OK) and puts
  `_save_candidates(_candidates, coords, edges)` in the `else` branch directly after `predict_video`,
  exactly at the GPU/CPU seam. **Consequence for `logs/run_ilpdisapp.sh`: arm 1 pays the FULL GPU
  prediction (~23 min at N=20, per the secedge timings), and only arms 2-4 are CPU.** The sweep is
  ~1 GPU-arm + 3 CPU-arms, not 4 CPU-arms. Also corrects the kill accounting: `secedge_0.30` DID reach
  the end — `submission.csv` 241937 rows, `validator_results.csv` written, `run.log` ends at
  `FINAL: 22002 nodes, 21298 edges` — the only missing artifact is `ppsweep_results.csv`.
  **Rule earned: a cache is not a saving until a file exists on disk; assert the artifact, not the code.**

  **E33 — REFUTED, and it cost no slot and no GPU: v1329f has NO ensemble seat, because where the two
  trackers actually disagree the champion is right. `tracker_agreement.py --disputes` isolates the only
  edges a combination rule could ever swap — nodes BOTH trackers gave a parent to, but a different one —
  and there are **961 of them, not the 6.1 k E32 estimated** (the rest of the "a_only/b_only" counts were
  unmatched-node artifacts, the same denominator error E32 already had to correct once). On those 961:
  **0947 median step 2.071 µm vs v1329f 8.286 µm, and 0947 picks the shorter step in 98.86 %**; only 24
  disputes are within 1 µm of each other. GT median step is 1.82 µm with p99 7.2 µm and max 9.96
  ([[celltrack-gate-um-is-derived-from-displacement]]), so v1329f's contested picks sit at the gate and are
  physically implausible — this is not a tie to break, it is v1329f being wrong, and it is presumably WHY it
  scores 0.939 against 0947's 0.947. **Net for the whole donor: 428 orphan-fill edges (0.2 %) plus 961
  disputes it loses 99 % of. There is no headroom to combine.** Consequence for the plan: the final-2
  default of 0947 + v1329f stands only as SUBMISSION diversification against private-LB variance, never as
  a merge; and the 0.947→0.974 gap is NOT reachable by combining the public forks we hold — the remaining
  levers stay detection-side, where E32(i) put ~3–7 k nodes each fork finds and the other does not.**

  **E32 — THE ENSEMBLE SEAT IS OPEN, AND RUNTIME WAS NEVER THE THING BLOCKING IT. Two corrections in
  one measurement. (a) SIZE: the hidden test is **4 videos**, `predict_minutes_total` 9.93, and the
  validator-free kernel finishes in 1590 s of a 9 h budget — **~20x headroom**. "The models are too slow
  to ensemble" is false; we can afford several full passes in one kernel. (b) DIVERSITY: `kaggle/tracker_agreement.py`
  matches nodes between two submitted trackers within 2 µm per frame and compares edge sets. 0947 (LB 0.947)
  vs v1329f (LB 0.939) share **108192 edges, with 10356 0947-only and 10700 v1329f-only — shared_fraction_of_union
  0.8371**. Two trackers 0.008 apart on the board disagree about **one edge in six**. That is not the
  saturated pool of [[ensemble-diversity-has-a-quality-floor]] — that verdict was measured inside OUR
  single recipe; these are two independent public forks. The seat is open; what is unproven is the
  COMBINATION RULE. Note the merge needs no re-run: both submissions already exist as kernel outputs, so a
  merge kernel can attach them as inputs at ~zero GPU.
  **CORRECTION, same sitting — I first called the ~21 k disputed edges the headroom, and that is the wrong
  denominator.** `kaggle/merge_submissions.py` (constraint-safe union in the champion's frame) admits only
  **263 of 10700** donor edges at 2 µm and **428** at 5 µm, and the rejection histogram says why: **9492
  fail `unmatched_endpoint`** and 956 fail `target_has_parent`. Two facts follow. (i) Most of the
  0947/v1329f disagreement is **DETECTION-side** — each finds ~3–7 k nodes the other has no node for —
  which is the same detection-side verdict the rest of this tree keeps landing on, now measured between two
  frontier forks instead of inside our own. (ii) **Edge UNION is structurally dead as a rule**: the champion
  already assigns a parent to nearly every node, so a union can only fill orphans, and orphans are ~0.2 % of
  edges. The live rule is a **SWAP**, not a union — at 5 µm the two trackers assign *different* parents on
  **6141 / 6485** matched-node edges (~5.3 % of the graph), and a 2-member vote cannot break that tie. So
  the next step is a THIRD independent voter (v50, LB 0.907, the only non-0947-family fork we hold;
  gapfill/readmit are 0947 variants and therefore correlated voters), with the rule "flip 0947's parent only
  where the third member agrees with v1329f against it". 5.3 % contested is above the bar; the flip count is
  the number to measure before spending a slot.**

  **E31 — BANK: the validator is 74 % of the kernel's wall-clock, and removing it is OUTPUT-NEUTRAL.
  `celltrack-public-0947-fast` (= 0947 with `BIOHUB_VALIDATOR_ENABLE=0`, staged by `kaggle/env_variant.py`)
  finished in **1590 s vs the base kernel's 6286 s** and its `submission.csv` is **byte-identical**
  (`cmp` clean, 241357 lines both). Not an approximation that needs an LB confirm — the same bytes score
  the same. E30 independently found the field running `VALIDATOR_ENABLE=0` too. **Consequence: ~4700 s
  of the kernel budget is now free at zero cost to the shipped score**, which is what makes anything
  second-pass affordable on Kaggle, where E29's local candidate cache does NOT reach. Spend it on a
  SECOND ILP pass at a different disappearance weight with edge union (the two-pass tracklet ILP that
  [[celltrack-refuted-axis-is-the-long-run-lever]] flags as a jointly-necessary 0.945 lever) — not on a
  third detection vote, which E25 measured inert, and not on a base-family ensemble member, which
  [[celltrack-base-decorr-has-no-finer-member-redundant]] already called NO-GO.**

  **E30 — donor triage round 2 (23:12Z), three kernels, zero new arms — but one numeric.
  `newwang12/biohub-v1-grouped` (11 votes) and `leonixis/biohub-v1-infer` are both 0947-family
  (177 / 77 hits on `harmonic_association|DeepCenter|ILPSolver|BIOHUB_`); an env-block diff against
  `logs/0947_src.py` returns exactly ONE substantive numeric difference —
  `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` **0.25 vs our 0.20** — plus `BIOHUB_VALIDATOR_ENABLE=0`, which
  independently confirms the validator-free `-fast` variant is what the field runs. Neither posts a score,
  so 0.25 is a hint, not evidence; it is a DETECTION-side (DeepCenter division veto) knob, hence upstream
  of the E29 seam and still GPU-priced — queue it BEHIND the disappearance sweep, not ahead of it.
  `noisyislands/biohub-linker-association-mlp` is a 6-feature `EdgeMLP(6→16→16→1)` over GT-node geometry
  with injected synthetic dups/noise: that is simultaneously
  [[celltrack-four-cues-fail-confusor-detection-side]] (geometry-only cues rank the true successor at or
  below chance) and [[celltrack-pmkf-trains-gt-sparse-not-detected-crowd]] (trains on the GT-sparse pool,
  not the ~980-FP/frame detected pool). Refuted family, no score posted, NO arm. Einstein criterion held
  three times in one pass.**

  **E29 — the ILP-cost axis is no longer GPU-priced. `predict_video` returns `(coords, edges)` — the
  candidate graph with `edge_prob` — and everything under it (`build_graph` + `td.solvers.ILPSolver`) is
  pure CPU, so the GPU half knows nothing about the solver weights. `local_predict_patch.py` now caches
  the candidate graph on a key built WITHOUT the `--ilp-*` args (and without `BIOHUB_ILP_*`), so the
  first arm pays ~55 min of prediction and every later cost arm re-solves in seconds plus the PP sweep.
  A 4-arm disappearance sweep drops from ~3.7 h to ~1 h, and — the part that matters for rigor — the
  matched control at the shipped weight 2.0 is now FREE, off the identical candidate set, instead of
  being compared against a differently-seeded baseline. Staged as `logs/run_ilpdisapp.sh` (3.5 GPU,
  then 2.0 / 1.4 / 6.0 on CPU). Note this does NOT reach the Kaggle side, which always predicts fresh;
  it makes the local search cheap enough to spend one submission on a weight we actually chose.**
  **E28 — THE FRAGMENTED EDGES ARE REACHABLE: the ILP had the candidate and declined it. E26 said 1460
  GT edges have both endpoints detected and matched with no predicted link, but "matched" does not imply
  "on the solver's table" — an edge longer than the candidate gate was never offered. `kaggle/fragment_audit.py`
  walks the cached validator geffs against GT (per-frame nearest match ≤5 µm, µm from voxel × (1.625,
  0.40625, 0.40625)): of 22042 matched-endpoint GT edges, 417 are fragmented and **393 of them (94 %)
  sit inside the 10 µm gate**; only 24 are out of reach. So the loss is a SOLVER-COST decision, not a
  candidate-generation miss — which is what licenses the disappearance-weight arm rather than a gate
  widening. And the shape names the direction: fragmented edges have median step **2.73 µm** against
  **1.82 µm** for the edges that did link. The ILP is dropping the LONGER true steps, i.e. termination
  is out-competing continuation exactly where continuation costs most. Raising `ILP_DISAPPEARANCE_WEIGHT`
  above 2 is the mechanism-matched move; 1.4 (the hack kernel's value, E27) predicts MORE fragmentation
  and is the control, not the candidate.**
  **E27 — donor triage 2026-09-19 22:50Z. `codezzzsleep/biohub-095-owned-validation` (claims 0.95) is a
  METRIC HACK, not a method: it appends a hub node at `t=-1000, z=y=x=-10000`, links the 1400 largest
  components' roots to it, then chains 5 synthetic fork triples (`MAX_COMPONENTS=1400`, `FORKS=5`) —
  same family as `kirneo_metric-hack-last-call-update`, already recorded as farming zero divJ. NOT
  submittable, and its "0.95" is not evidence about any mechanism. One legit datum survives it: its ILP
  weights are edge −1.0 / appearance 0.0 / **disappearance 1.4**, against our 2. Someone else tuned this
  axis and landed BELOW us — the opposite direction from E26's prediction — so the arm is two-sided
  (3.5 and 1.4), not one. `binasalama/…-ilp-gap-recovery` is our own env family at stock defaults
  (disapp 0.1) = no new information. `fabriciodasilva/biohub-dodecatiad` is a from-scratch DoG-blob
  detector + greedy motion linker, no learned model, no posted score: structurally out-of-recipe (an
  ensemble-seat shape) but built on the DoG filter our own normalization axis already refuted as LOWERING
  centre detectability — not worth GPU ahead of the disappearance arms.**
  **E26 — THE MISSING EDGES ARE OMISSIONS, NOT CONFUSIONS. `validator_results.csv` already carried the
  decomposition and we had never summed it. Over the 40 held-out videos (base config, fp16 run):
  GT edges 47170 · edge recall 0.9475 · edge_fn 2476 = 1016 lost to detection (41 %) + 1460 FRAGMENTED
  (59 %) · `wrong_association_edges` = **74**. Mislinks are ~nil. The standing story — "association
  under crowding / confusor disambiguation" — is not what the remaining loss is made of: 1460 GT edges
  have BOTH endpoints detected and matched, and simply no predicted edge. That is 3.1 % of all edges,
  the largest single addressable block we have measured, and it is a RECALL problem at the linker.
  Mechanism candidate, never swept: `ILP_DISAPPEARANCE_WEIGHT = 2` (with appearance 0) prices ending a
  track cheaply, so the ILP is free to drop a marginal link; with only 74 wrong edges there is enormous
  headroom to pay more for continuation. It lives inside `predict_unet_transformer.py`, so it needs a
  GPU re-run (~55 min/arm), queued behind the edge-weight arms. Division in the same table: recall
  0.217 (26/120), precision 0.361 — measured at n=120 GT divisions, not the n=7 that the old
  "division postproc dead" verdict rested on. Gate knobs are already swept flat (bead yesg, 9 knobs
  ±0.003), so any division re-entry must be CANDIDACY, not gating.
  CAUTION: the `spurious_pred_nodes` column sums to 1.72 M against 1.77 M predicted nodes and 836
  missed GT nodes — it is not a per-node FP count. Do not build on that column without deriving it.**

  **E25 — `SECONDARY_DETECTION_WEIGHT` is INERT (local, fp16, N=20, 44 videos). 0.6 → adj 0.9021 /
  missed GT 418 · 0.8 (shipped) → 0.9018 / 446 · 0.95 → 0.9021 / 454. Spread 0.0003 = noise across a
  36-cell missed-GT swing: the blend moves detection RECALL and the score does not follow. Two readings,
  same direction — the two detectors agree wherever it matters, and the score is association-bound, not
  detection-bound (see "score lever is association not detection"). So the freed runtime above must NOT
  buy a third DETECTION vote. Follow-up arm running 22:41Z: `SECONDARY_EDGE_WEIGHT` 0.30 / 0.05 vs the
  shipped 0.15 — the same fusion knob on the association side, never swept.**

  This retires the "6-config ensemble infeasible, ~18-24 h" bound — it was measured with the validator
  block in every member. Marginal cost of an extra member is ~940 s (predict 301 + PP 639), so 3-4
  members fit inside one 9 h kernel even before fp16. "The models are slow" was never the models.** Cache-key bug found: env paths under the work dir made every shard miss → keys now content-fingerprinted
  (f4d499b). New donors 15:25Z: noisyislands linker-MLP (2 epochs, 4 videos, ~0 negatives =
  toy) · xgboost-division (13 GT-only fork feats, negatives = random non-forks, in-sample acc 0.997 ⇒ trivial
  labels, no track score) · thtennant divprec-v1 (ppsweep selected=base, no held-out) ⇒ none carries evidence. The
  IDEA they gesture at is real but untried by us: a fork classifier trained on PREDICTED-graph candidate forks
  (labels = GT match), replacing reid3's hand SAFE_DIV_* thresholds. DENOMINATOR is the point: donor validator = 8 stems, 12 GT
  divisions ⇒ a division knob moves 1 event = ±0.08 divJ = noise. Local lets N_PER_TYPE grow (pool 71 44b6 + 128
  6bba). Caveat: primary weights' train split undocumented (deepcenter: 71/128 split) ⇒ validator may be in-sample;
  use it to RANK PP configs, not as an LB estimate. First run = reid3 unchanged, fidelity check vs donor csv.
  Plan (rev 14:05Z): donor OUTPUTS are free evidence — `kaggle kernels output <owner>/<slug>` gives each donor's own
  validator/ppsweep/reid_report; read them BEFORE porting. E5/E8 retired by that read (above); E7 inconclusive.
  Remaining: E1–E3 (pending probes) → composite best-of(E1/E3) + E10 + E4. Donor divdiag (reid3, 8 held-out stems):
  divJ 0.23, 9/12 FN, **6/12 = "2nd daughter OWNED by another track"** — the steal/reattach axis (SAFE_DIV_STEAL_*, off by
  default in reid3) is the division-side element to look at next.
  **Division axis n=40 (18:12Z, reid3, 40 held-out / 60 GT div, `probes/reid3_divaxis.json`): FLAT.** base proxy 0.9161
  (adj 0.9022, divJ 0.140, 12 TP / 26 FP / 48 FN). All 9 knobs within ±0.003: steal_r20 0.9160, steal_own45 0.9161,
  rawsucc 0.9163, tight55 0.9163, dcdiv032 0.9153 (10 TP), diverge175 0.9146 (+10 FP), nnk2 0.9182, dcdiv018 0.9178,
  reid8+relaxed9 0.9177; combo nnk2+dcdiv018+reid8 0.9193 (+0.0032 directional, +3 TP / +9 FP). 48/60 GT divisions
  missed by EVERY knob → the n=8 steal hypothesis does not hold at n=40; division recall is upstream (detection /
  association), not a postproc knob. Division term ceiling = 0.1·divJ ≈ 0.014 → axis CLOSED for PP.
  **Detection threshold (same 40):** 0.955 adj 0.9008 vs 0.965 0.9022 (−0.0014): 17 fewer missed GT nodes, +20
  fragmented edges, +12.9k pred nodes — extra nodes do not become links. 0.975: adj 0.9020 (−0.0002), missed GT 461 vs 440 @0.955 — threshold axis FLAT over 0.955–0.975 (span 0.0014); only the LB probe det096 can still move it.
  **Kaggle steal probe (reid3 own 8-video validator, T4):** base 0.9490, all 7 steal/tight knobs 0.9477–0.9511
  (adj ±0.003, 3 TP / 9 FN divisions everywhere) — same FLAT verdict as n=40 on an independent run.
- **Cross-family decorrelation (18:20Z, GT-free, 4 visible test movies):** 0947 vs v1329f submission edges agree
  81–93 % (44b6_0b24845f 0.81, 6bba_05db0fb1 0.85; node match ≤2 vox). But the v1327-w3 "real model" is the
  0947 PRIMARY + 0.09 % weight delta (unet 0.0009, transformer 0.0003 rel-L2; vs seed314159 secondary ≈1.0) —
  the disagreement is PIPELINE (post-proc/decode), not model diversity. w3-as-secondary killed on mechanism
  (would replace the one decorrelated seed with a primary clone); no run.
- **LB instruments pushed 18:24Z (expiring slots, Kaggle GPU):** `0947-det096` (DET 0.965→0.96 only — tests
  hikaggler's "N_pred moves LB" on our base) and `0947-synthsec` (secondary seed314159 → our all199_from_synth80
  ep2 ckpt, private dataset `dimiturnt/celltrack-synth80-ft2` — first structurally-different member; an ep2
  under-trained ckpt, so a loss = "under-trained seat", not "synth refuted"). Watcher `logs/kaggle/watch_0947_probes2.log`.
  Both submitted 20:16/20:26Z. NB the kernel blends DETECTION as `0.2·primary + 0.8·secondary`
  (`BIOHUB_SECONDARY_DETECTION_WEIGHT` 0.80), so synthsec swaps 80 % of the detector, not just an edge vote.
  Local instrument `secdet_{0.95,0.6}` (fp16, N=20, vs fp16 base 0.9018) running 20:48Z — never swept before.
- **synth_pre80 done 13:01Z**: synth val best 0.9783 (80 ep). Chained all199_from_synth80 started; ep1 val 0.9201 on
  the 4-movie split (not comparable to the 20-holdout A/B arms' 0.89–0.91). ~25 min/epoch.
- **Forum 740145 (hengck23):** Kaggle GT sometimes sits on cell "corners" (Ultrack-derived); many FPs lie next to a GT
  node. Their trick: at a 99% edge-recall cutoff, re-rank only the surviving candidates with a heavier module.

## ROOT

- **Goal:** WIN (1st). Winning cluster 0.945–0.962. **Banked best = 0.924 LB** (sub 55779061 = faithful
  evgendvorkin champion replica + global tracksdata ILPSolver linker).
- **The gap = 0.924 → 0.945 = ASSOCIATION-under-crowding.** NOT detection: `node_recall ≈ 0.998` is DEAD.
- **Ruler discipline:** CTC/local proxy INVERTS above 0.90 (dw0 +0.0267 held-out → −0.028 LB). The only
  non-inverting local ruler = the 6-movie official-metric harness (`scratchpad/off_eval6.py`,
  `metrics.evaluate()`, tracksdata 0.1.0rc8, max_distance 7.0µm); gate ≈ 0.94 local-official. A slot is
  spent ONLY on a named mechanism clearing 0.910, gated on this ruler or on a card-free decorrelation gate.

## LEVEL 1 — WHERE is the gap? (three branches)

- **Detection recall** → **REFUTED as a lever.** node_recall 0.998; detector over-detects ~40× (~1005
  det/fr vs GT ~6.6 annotated/fr). Recall is not missing.
- **Division** → **REFUTED as a lever.** Champion's own real-LB sweep: div_weight 0.3–3.0 all ~0.915;
  7.0 → offline divJ first-nonzero but LB DROPPED 0.914 (proxy-inversion caught in the act). TEST has ~3 GT
  divisions → 0.1·divJ ≈ 0. Division postproc / FP-fork precision does not convert.
- **Association under crowding** → **THE gap.** edge_jaccard collapses monotone with density: sparse movie
  (4.3k nodes) 1.000 → dense movie (98k nodes) 0.598. Even the champion ILP only lifts the densest
  0.598→0.6495. Everything below hangs here.

## LEVEL 2 — WHY association fails (mechanism, banked)

1. **Sparse-train / dense-infer.** Champion edge head (`SimpleNodeTransformer`) trains on GT-touching pairs
   (~17/fr) but infers on the ~1005-FP detected crowd. NUANCE (density-collapse audit): the champion's
   *external* trainer ALREADY does complete-candidate source competition (`detect_and_match` ~258/fr,
   focal-BCE `softmax(dim=0)` masked to GT-touching rows∪cols) → the naive "it never saw the crowd" is only
   half true. The dense collapse saturates on the source-competition axis that IS trained → **representation/
   capacity limit, not purely a data limit.**
2. **The confusor is DETECTION-side.** Four cues — distance, incoming-direction, raw full-res appearance,
   learned features — ALL rank the true successor BELOW CHANCE on the mislinks, unanimously. The
   discriminating info is not in the detected representation; a fraction of labels may be wrong. → any
   link-time cost that encodes motion/direction is ≤ null, plausibly negative.

## LEVEL 3 — the config surface (every axis we support → folded verdict)

### A. Link-loss FORM (per-source target competition) — `--symmetric-links` / `--slack-links` / Sinkhorn/OT / parental-softmax
**AUDIT-FLAT.** Symmetric's headline 0.9278 was a dead-constant bug (constant-removed = 0.9060 = baseline).
Where it genuinely improves association (mislinks 46→43), the **shared-trunk detection-coupling tax** charges
it back (both heads read one backbone; harder assoc gradients over-fire the detector). SlackRow faithful
re-test −0.0107 ≤ noise. Sinkhorn couples the SAME two constraints harder → same tax. Only un-flat sub-angle
= doubly-stochastic mutual-exclusion vs FP-forks, but divJ tiny. Bounded as a family.

### B. Link-time COST (motion / appearance / context added to the objective)
**REFUTED.** motion-in-ILP: motion cue ranks true successor below chance → actively misleads solver
(solver-agnostic). PoE cost-shape refuted (confusor is per-frame-greedy STRUCTURE). appearance cosine =
coin-flip on confusor (n=37). pair_context −0.011. edge_options/view_tta −0.0049 (dense needs sharper, not
averaged). The confusor info isn't in the representation → no link-time recombination recovers it.

### C. Global topology — linker name: `greedy` → `flow` → `ilp`
**BANK = ilp.** Global min-cost ILP defeats per-frame-greedy confusor STRUCTURE: 0.902 → 0.924 (the bank).
Card-free alternates below champ (assign 0.900, flow 0.695). Topology axis closed AT ilp — but see Level 4:
the winner's ilp is **two-pass tracklet**, ours is single-pass.

### D. Edge-head ARCHITECTURE — `--head {pack, hoct}`, `edge_hidden_dim`
**HARNESS on prior kills; but CEILING-BOUNDED as a solo lever.** HOCT (edge-to-edge global attn + 3D RoPE +
σ line-bias) is fully BUILT (`hoct_edge_transformer.py`), drop-in for SimpleNodeTransformer. Prior kills
(597c0f2 "REFUTED", proxy 0.7234) were epoch 0.5/3 + (1,4,4) + from-scratch = **harness**, explicitly
reclassified. **The decisive result:** frozen-detector HOCT on GT-sparse gave a *conclusive confusor
instrument-lift* (top1 0.012→0.214 = +0.202, inverted 0.988→0.583) — head class carries bottleneck B for the
first time on a non-inverting ruler. BUT faithful DROPPED 0.6877→0.6432: HOCT's sharper affinity makes the
flow linker drop marginal-but-true edges calibrated for the standard head's softer scores. **B solved on the
instrument; recall-cost erases it at the ship point.** AND the winner's own Table 2: edge-stage-alone caps at
**CLB 0.926** ≈ our 0.924. **So a perfect edge head ≈ +0.002 = noise. The edge head is not the 0.945 lever.**

- **Width trap:** every HOCT ckpt is `hidden_dim=128`. "C=256" the human approved = HEAD WIDTH; `K=256` in
  the pmkf scripts = candidates/frame (a different knob). A width-256/288 head has NEVER been trained — but
  width is unlikely to be the lever when the ceiling itself is 0.926.

### E. Detection RESOLUTION — `--downsample 1,2,2` (finer122) vs 1,4,4
**HARNESS-then-BOUNDED.** finer(1,2,2) separates to 1.0µm vs (1,4,4) merges <2.0µm (oracle). Frozen-graft
REFUTED (off-grid, +60.9% peak inflation = fabricated fragments = harness). Co-adapt un-merges +2625
sub-1.6µm cells, physics-validated GT-free (bottleneck A HOLDS). BUT **B FLAT** (from-scratch/co-adapt head
too weak at 8× density) AND **GT nearest-neighbor floor = 2.87µm, ZERO cells <2µm across 131k frames** → sub-
1.6µm un-merges are FP splits, not recall. A-axis real but bounded; needs B solved AT finer to matter.
> **FOLD 2026-08-28 — the "B FLAT" finer-head verdicts are HARNESS, not capacity.** Every finer122
> edge-head checkpoint (coadapt_long standard, hoct_converge, AND the 3.D-CHEAP-CUT rzvw head-only run)
> trained with contrastive `nce=0 hn=0` — LossCfg defaults both to 0.0 (joint_config.py:288,298) and the
> launch scripts omit the flags. All three plateau proxy ~0.61, inv 0.97 = the degenerate untrained-affinity
> signature. avl8 (same joint arch, nce ON): proxy 0.79 / node R 0.92 / inv 0.54. So B-FLAT-at-finer was
> never tested with the association objective's key term ON. **Un-refutes** finer-head B AND the HOCT×slack
> coupling (3.D cheap-cut). The never-run cell = finer122 detector + nce ON → staged `launch_hoct_finer122_nce.sh`
> (contrastive 1.5, hard-neg 0.015 — recovered EXACT from avl8 loss decomposition, == nwfi record). CAVEAT:
> this is an INSTRUMENT to close the axis honestly (does nce flip finer inv 0.97→~0.5?), NOT a >0.945 bet —
> the 3.D ceiling (0.926 ≈ 0.924) still bounds any solo edge head. If the instrument clears the axis, the win
> is still the Terminal-#2 JOINT stack, not this arm alone.

### F. Data DISTRIBUTION — train on detected crowd (`--detected-videos` / dw0)
**REFUTED on the board.** Feeding the SimpleNodeTransformer head the complete detected crowd: +0.0267
held-out → **LB 0.872** (proxy anti-correlated at top). The data-half individually failed. pmkf's open bet
= does swapping to HOCT arch + freezing the detector flip this — untested, but capped by D's 0.926 ceiling.

### G. ENSEMBLE / decorrelation
**Common-mode → bounded.** All candidates are evgendvorkin variants making the SAME confusor errors.
Decorrelation instrument: unionJ 0.9093 < 0.910 (leans NO-GO). One-recipe pool saturates within noise. An
ensemble seat requires a member from OUTSIDE the recipe (a structurally-different detector), not another
copy. A genuine 2-model seat (champion SimpleNodeTransformer + a working HOCT) only pays off IF HOCT first
becomes a pipeline win — which D says it can't, alone.

## LEVEL 4 — the winner's THREE 0.945-levers, and why each looks flat SOLO on our data

The winner (Table 1/3) reaches 0.945+ via three **jointly-necessary** pieces, none of them the edge head.
We have tested each **in isolation** and banked a partial-refutation for each — which is exactly the trap,
because the winner says they are jointly necessary (→ 0.920 only together):

| winner lever | our banked solo result | tag |
|---|---|---|
| **Two-pass tracklet ILP** (Δt=1 tracklets → meta-node re-solve) | our GT: 128883 edges 100% dt=1, ZERO gap-edges → two-pass fires only on rare detector-miss + confusor-FP | HARNESS/UNTRIED (single-pass now) |
| **Variable-appearance** (parental softmax `1+Σexp` = SlackRow) | SlackRow faithful −0.0107 ≤ noise | REFUTED SOLO |
| **Finer decode** (0.926→0.950 bulk, detector-side) | un-merges physics-valid but B FLAT, GT NN floor 2.87µm | HARNESS/BOUNDED SOLO |

**THE conclusion of the tree:** we have been mining the EDGE-HEAD axis (HOCT, target-competition, detected-
crowd, confusor), which the winner's own ablation caps at 0.926 ≈ our bank. Every lever that actually reaches
0.945 is in the SOLVER + variable-appearance + finer-decode STACK, and we have only ever tested those **one
at a time and killed each in isolation** — against a source (Table 1) that says they only work together.
This matches the standing meta-conclusion: no TRIED mechanism has been DEMONSTRATED-BY-US >0.910, and every
kill on the association axis was HARNESS-bound. The path is **replicate the winner's FULL stack jointly**
(two-pass tracklet ILP + variable-appearance + finer decode, co-trained), not another solo edge-head arm.

## TERMINAL — remaining levers, ranked by plausible size × un-refutedness

1. **Two-pass tracklet ILP** (solver-side). Untried; winner marks it necessary. **NOT card-free** (bd wfh5
   audit 2026-08-26): shippable path WRAPS tracksdata.predict() INSIDE the kernel — our local motile
   `ILPLinker` is 7apb-unshippable and tracksdata doesn't import locally → build+validate are kernel-bound.
   wfh5 also imposes a GPU referee-gate BEFORE the build (produce hoct affinity → confusor referee under
   flow+ilp → does a global solver convert the hoct inv-gain to fewer mislinks vs pilkwang 143?). Risk: jm68
   sizing = our GT has zero gap-edges → two-pass may barely fire on the metric. GPU + wheel-gated, not a
   card-free advance.
2. **Full-stack joint replication** (finer decode + HOCT edge head + two-pass ILP + variable-appearance,
   co-trained). The only path the winner's ablation actually supports to 0.945. Expensive, GPU + wheel-gated,
   multi-arm. This is the real bet, not any single piece.
3. **pmkf / HOCT-on-detected-crowd frozen** (bd `biohub_kaggle-pmkf`, OPEN, trained today, UN-SCORED on
   off_eval6). Worth SCORING because the confusor instrument-lift is real and the 6-movie number is cheap —
   but D's 0.926 ceiling means best-case ≈ noise over 0.924. Score it to CLOSE the axis, do not build on it.
4. Everything in Level 3.A/B/F/G — refuted or bounded; do not re-open without a new mechanism.

## JOINT ARM — the staged plan for Terminal #2 (card-free design; launch is wheel+card-gated)

The winner's three levers are jointly necessary (Table 1 → 0.920 together); we killed each SOLO. The arm
that could actually reach 0.945 co-trains them AND co-designs the linker recalibration that erased the HOCT
gain last time. Spec (so the launch is designed, not improvised):

- **Substrate:** finer122 (1,2,2). Detector = warm from `finer122_coadapt_long.pt` (already un-merges +2625
  physics-valid), `--freeze-backbone-norm` (BN-pollution guard; joint_cli:152). Bottleneck A already held here.
- **Head:** `--head hoct` (edge-to-edge attn + RoPE; built, `hoct_edge_transformer.py`). Read width off ckpt
  (9f33205); train to CONVERGENCE (prior kills were epoch 0.5/3 = harness). This attacks B (confusor).
- **Variable-appearance = the recall-cost FIX, not a separate topper (KEY mechanism, 2026-08-27):**
  `--slack-links` (SlackRow = parental softmax `p=exp(ℓ)/(1+Σexp)`). The `1` is the NO-PARENT escape valve.
  HOCT-solo dropped recall (0.6877→0.6432) because a SHARP head with NO escape FORCES every cell to link →
  over-commits → drops marginal-true edges when uncertain. The `1+` lets low-confidence cells stay unlinked
  gracefully. So HOCT (sharp) and variable-appearance (escape) are **COUPLED, not additive** — the sharp
  head NEEDS the valve to not over-commit; the valve with no sharp head to regulate is inert (explains
  slack-solo −0.0107). This is the precise mechanism of "jointly necessary." Predicts: the joint arm's
  success hinges on the HOCT×slack coupling, testable by ablating the `1+` term within the joint arm.
- **Residual linker recalibration (secondary):** after the escape valve restores graceful non-linking, any
  remaining distribution mismatch (flow thr/disappearance-cost tuned for the softer standard head) is a
  smaller sweep, not the primary fix. A joint arm that lifts faithful past 0.6877 WHILE holding top1 >> 0.012
  is the first real B-driven pipeline win.
- **Solver:** two-pass tracklet ILP (bd wfh5), kernel-side (wraps tracksdata.predict()). Referee-gated FIRST
  (produce hoct affinity → confusor referee under flow+ilp → mislinks < pilkwang 143?). Build only if it
  converts.
- **Oracle (non-inverting gate, in order):** (1) confusor referee mislinks < 143 = affinity real; (2)
  off_eval6 6-movie micro ≥ 0.94 local-official; (3) only then a slot. Proxy above 0.90 is BLIND — never gate
  on it.
- **CHEAP FIRST-CUT (flags audited 2026-08-27, all compose — no launch crash):** the coupling hypothesis is
  testable WITHOUT the expensive co-adapt. `--detector-from finer122_coadapt_long.pt --head hoct
  --edge-hidden-dim 256 --slack-links --downsample 1 2 2` = warm+FREEZE the finer122 detector, fresh-init the
  HOCT head, train HEAD-ONLY (no detector backprop → fast, ~head-only budget not full-train). This is the
  exact 0825 frozen-detector design PLUS the slack valve — one variable added. Gate: does faithful hold/beat
  0.6877 (vs solo-HOCT's 0.6432) while top1 stays >> 0.012? If YES → the valve fixes the recall-cost, coupling
  confirmed, escalate to co-adapt + two-pass. If NO → coupling refuted cheaply, saved the multi-hour arm.
  (256/4=64 = RoPE-valid; config default-comments winner=288, also valid.) Guard: `--detector-from` loads BN
  finer122 → keep `--norm batch` (warm guard rejects `group`).
- **Launch order:** (0) CHEAP head-only frozen arm above — coupling gate. → if converts, (1) referee-gate
  hoct affinity (wfh5, GPU inference) → (2) GPU co-adapt joint train (finer+hoct+slack, converged,
  recalibrated linker) → (3) off_eval6 → (4) slot. Each step gates the next; do not skip.
- **Cost:** cheap arm = head-only GPU (short); full bet = multi-hour GPU. Both wheel+card-gated. The cheap
  arm is the highest-ratio next GPU spend — price it FIRST on wheel-grant.
- **CHEAP-CUT INVALIDATED 2026-08-28:** the rzvw head-only run above executed with `nce=hn=0` (flags
  omitted) → trained FLAT (proxy 0.6187, top1 0.20) = the nce-off plateau, NOT a coupling refutation. The
  coupling gate must be RE-RUN with `--contrastive-weight 1.5 --hard-negative-weight 0.015`. Staged as
  `launch_hoct_finer122_nce.sh` (frozen coadapt_long detector + hoct head + nce ON). Card-gated behind knee
  raddino (~0900z). Gate unchanged: faithful holds/beats 0.6877 AND top1 >> 0.012 = coupling confirmed.

## What is DEAD — do not re-suggest
motion-in-ILP · division postproc/FP-fork · directional-PE (detection-side) · consensus copy-ensemble ·
link-loss form family solo ·
appearance/pair_context/view_tta · finer sub-1.6µm as recall.
