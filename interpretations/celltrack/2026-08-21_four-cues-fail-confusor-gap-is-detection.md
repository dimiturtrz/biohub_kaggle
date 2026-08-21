# Four independent cues all fail the confusor — the gap is detection, not association

> **CORRECTION 2026-08-21 (same day): the "detection" verdict below is REFUTED on the right denominator.**
> The fate decomposition (`celltrack.eval.dense_diagnosis`, 1183 annotated edges) shows detection-limited
> (`ENDPOINT_MISSING`) = **9 edges = 0.76%** — below the 1.5% bar even at oracle. The detection-side read
> came from `dir_oracle` *site density* (4x-denser detected vs annotated), which is the WRONG denominator:
> site count ≠ edge fate. The dense loss (51 edges) is `MISLINK_CONFLICT` 27 (53%) + `MISLINK_FREE` 10
> (20%) + `ENDPOINT_MISSING` 9 (18%) + `SKIP` 5 (10%). The global-flow linker aimed at the CONFLICT share
> was already built and refuted (bead tf5, +0.0007) — it follows the same affinity cost, and 97.3% of
> mislinks have P_chosen 0.581 >> P_true 0.079. **True bottleneck = affinity discrimination; the
> disambiguating info is not in the pooled (1,4,4) representation.** The four-cue result below stands (no
> handcrafted cue over pooled features separates it); only its "it's detection" framing is wrong. See the
> "Fate decomposition" section at the bottom.

*2026-08-21 · CPU, detected-centre ranking on dense movie `6bba_05db0fb1` (74347 nodes), no Kaggle spend*

## Question

The dense confusor (association head prefers a near rival to the true far successor) is the whole
remaining gap. Distance, learned features, and full-resolution appearance were each refuted on it.
One geometric cue was untested: **heading continuation** — does the true successor continue the
source's incoming direction better than the rival (HOCT's line-to-line edge bias)? If yes, an
edge-to-edge stage is worth building; if no, that build dies before it starts.

## Measurement

Mirror of `appearance_identity.InGateRanking` scoring candidates by `cos(incoming, candidate−source)`
instead of appearance, where `incoming = source − predecessor` in the **prediction** graph (the input
a real head would have). Run on the tracker's own mislinks (`CandidatePairs.mislinked`) with a correct-edge
control. `scratchpad/dir_rank.py`.

```
mislinked (confusor) n=  31 | direction TOP-1 true 0.161 (chance 0.255) | true>rival 0.194 | median rank 3.0 / 4.4 cands
correct              n=1084 | direction TOP-1 true 0.536 (chance 0.287) | true>rival 0.691 | median rank 1.0 / 3.9 cands
```

## Verdict

**Direction REFUTED as a confusor lever.** It is a real cue on correct edges (top-1 0.536 vs 0.287
chance, true>rival 69%, median rank 1) — and **below chance on the confusor** (0.161 < 0.255, true>rival
only 19%, true successor median rank 3 of 4.4). The true successor is the *worst*-continuing candidate;
the rival continues the heading better.

This makes the pattern unanimous across **four independent cues** — distance, appearance (full-res),
velocity prior, and now direction — each of which **separates the correct edges and fails the same
confusor set**. Four cues failing identically is not four wrong ideas; it is evidence the disambiguating
information is **not present in the detected/pooled representation**.

Which link broke: the **IDEA** (direction-as-confusor-lever), with a named confound — the incoming
heading is taken from the prediction graph, which is itself corrupted on the confusor (a prior mislink
or a mislocalised detection gives a garbage heading). It does NOT rule out direction computed on a
*clean* tracklet — but a clean tracklet requires good detections first, so detection is upstream.

## The reframe — it's detection, not association

The confusor is **~4× denser on detected centres than annotated** (31 vs ~8 structural sites, the
annotated-graph oracle in `scratchpad/dir_oracle.py`), and mislink endpoints sit **2.1× farther** from
their annotated centres than the population (`2026-08-12_appearance-identity`). The near-rivals that
break association are **manufactured by spurious and mislocalised detections** — they barely exist on
the annotation. Every association-side cue fails because the candidate field it scores is corrupted.

**Consequence.** Stop building association heads over the current detected inputs — four cues prove
that ceiling. The dense-regime lever is **detection precision/localisation** (fewer spurious peaks,
tighter centres in crowding), which removes the confusor rivals at the source. Converges with
`celltrack-appearance-refuted-right-denominator` (gap is detection-side) and
`celltrack-TRUE-bottleneck-is-dense-regime-model-gap`. HOCT's dense-regime win may come from its
detector, not its fancy edge stage — worth re-reading the paper for detector quality, not just the
line-to-line bias.

## What it does NOT rule out

- Direction on a CLEAN tracklet (annotated-quality incoming heading) is untested — but circular
  (needs good detection first).
- A LEARNED head combining weak cues jointly could still beat any single cue — but the input-information
  argument (four cues, all below/at chance on the confusor) bounds how much is recoverable from the
  pooled detected representation. Un-pooled full-resolution input to the associator remains the one
  untested representation axis.

## Follow-up: rival real-cell vs artifact — inconclusive (harness-bound)

`scratchpad/rival_support.py` asked directly: is a confusor rival a real annotated cell (physical
ambiguity ceiling) or a detection artifact (detector lever)? Matched-rate vs annotation, controlled:

```
BASELINE detections matched to annotation = 1.8% (1224/69899)  -> GT so sparse the instrument is near-blind
mislink (confusor) n=  37 | rival matched  5.4% (2/37)   | true-target matched 100% (37/37)
correct (control)  n=1102 | rival matched  1.0% (11/1102)| true-target matched 100%
```

Nominal read (rival 5.4% > baseline 1.8% AND > correct-rival 1.0%) points at REAL-cell ambiguity, but
**n=2 matched events — underpowered, no bar cleared.** Which link broke: **TEST** — baseline 1.8% means
matched-rate is dominated by GT sparsity, not discrimination. Does NOT overturn the detection-side read
(dir_oracle n=31, 4x-denser, is the stronger denominator); both co-exist (sites detection-manufactured,
a rival landing near an annotation weakly enriched). Verdict: instrument too weak to referee artifact-vs-
real on the both-detected set; the 4x-denser oracle stands as the primary read.

## Campaign floor reached

Every cheap local lever is now refuted-with-mechanism (4 cues, PoE, linker swaps, head-retrain-frozen,
full-backbone LoRA, dw0 data, GN+slack+synth). The dw0 data lever — the only held-out proxy WIN
(+0.0267) — **lost on the real board** (commit 28f3978, −0.028). `joint_slack_gn_v1.pt` finished
(early-stop, best proxy 0.5413, confusor inv 0.97 unmoved) and does not beat shipped; keeps only the
BN→GN recall-salvage finding. The detector-precision lever is **licensed by measurement** (annotated
centres carry ~4x fewer confusor sites → a perfect detector removes ~75% at source) but the remaining
moves are all large BUILDS whose confirmation needs a board submission (currently disallowed) or the
403-blocked 2-UNet asset. No further cheap measurement redirects this — the next step is a strategic
call (commit to the detector build / unblock the asset / permit one submission), not another probe.

## Fate decomposition — the right denominator, and it kills the detection detour

`python -m celltrack.eval.dense_diagnosis --movie 6bba_05db0fb1` on the shipped tracker, 1183 annotated
edges (the module built exactly to name the next lever off measured fractions):

```
CORRECT          1132  95.7%
SKIP                5   0.4%   -> gap-bridge target
MISLINK_CONFLICT   27   2.3%   -> true target claimed by another source (global-assignment target)
MISLINK_FREE       10   0.8%   -> affinity ranks wrong-real-neighbour above truth
ENDPOINT_MISSING    9   0.8%   -> DETECTION-limited (no linker recovers)
mislink signal: 37 edges, affinity-inverted = 97.3%  [mean P_true=0.079 vs P_chosen=0.581]
```

Dense loss = 51 edges. **Detection-limited = 9 = 0.76%, below the 1.5% bar at oracle** → the detector is
not the lever; my `dir_oracle` site-density read measured the wrong denominator. The loss is
assignment/affinity: CONFLICT 53% + FREE 20% = 73% of it is the 37-edge confusor set, on which the
affinity is inverted 97.3% (P_chosen 0.581 >> P_true 0.079).

**Global-flow already refuted the CONFLICT share.** Bead tf5 built a track-level min-cost-flow linker
over the same affinity cost: peak +0.0007 (noise). Mechanism: the solver reassigns by cost, but the cost
IS the inverted affinity — when the competing source also prefers the rival, no reassignment frees the
true target. So CONFLICT is not solver-recoverable; it is affinity-bound like FREE.

**Verdict — the bottleneck is affinity discrimination, and the disambiguating information is not in the
pooled (1,4,4) detected representation.** Confirmed from three independent directions: (1) four handcrafted
cues over the pooled repr all fail the confusor, (2) the fate decomposition puts 73% of the dense loss on
an inverted affinity, not detection, (3) a global linker over that affinity recovers nothing. This
re-centers on the pre-detour conclusion (`celltrack-TRUE-bottleneck-is-dense-regime-model-gap`,
`celltrack-proxy-saturated-affinity-quality-bound`): the only mechanism-sound lever left is a **richer
INPUT representation to the affinity** — un-pooled/full-resolution local context or relative geometric PE
(HOCT-style 3D RoPE node attention, bead-referenced in `celltrack-frontier-gather-names-rope`) — so the
head can see the sub-resolution directional detail that pooling to (1,4,4) destroys. Raw-patch cosine
(coin-flip, refuted) is NOT that test — it is a similarity metric, not a learned head over un-pooled
features. That build, validated on the local dense-movie proxy (no submission), is the next real arm.
