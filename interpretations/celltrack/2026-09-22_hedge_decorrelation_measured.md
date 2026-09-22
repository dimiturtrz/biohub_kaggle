# E67 — what the hedge actually decorrelates, measured instead of argued

**2026-09-22.** The final-2 pair (`celltrack-public-0947` at 0.947 + `celltrack-public-v1329f` at 0.939) was
chosen on the argument that the hedge decorrelates at the *detector*, the component every other analysis
names as the true bottleneck. This prices that argument from the hedge's own run output. Two things were
wrong on the way in, and both were mine.

## The hedge was invisible to the harvest diff because its source is a gzip+base64 blob

`kernel_env_diff` reported `theirs=0` on `celltrack-public-v1329f.ipynb` and the empty-side guard refused —
correctly. The notebook is a 68k-char wrapper with 3 defs that base64-decodes and gunzips a **175k-char
payload** (185 `os.environ`, 71 defs) and shells out to it. Decoding it first, the real diff against the
champion is small:

- **they set, we do not:** only `BIOHUB_V1329_SUBMISSION_PATH`, `BIOHUB_V1329_WORKING_DIR` (plumbing)
- **constant deltas:** `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.20 vs **0.25**, plus the artifact slug
- **we set, they do not:** `DEEPCENTER_TTA`, `SECONDARY_EDGE_FEATURE_TTA(_WEIGHT)`, the whole `VALIDATOR_*`
  block, `PPSWEEP_*`

So on settings the hedge is *the champion minus TTA/validator/ppsweep, with the safe-div veto stricter* —
and, separately, plus a foreign model.

## The foreign graft is real but guarded, and on one movie it is switched off

The hedge mounts `josephadamski91/biohub-v1327-w3-real-model/model.pth` and adapts it with frame-local
mean/std alignment and a scale clamp `[0.5, 2.0]` (its own receipt: *"256 released-data centered-W3 updates
from exact V1274; frozen transformer and BN"*). A retention guard then compares the blended candidate count
to the untouched primary's and falls back when it drops:

```
BIOHUB_RETENTION_GUARD {"minimum_retention": 0.9, "retention": 0.7548,
  "guard_fallback": "untouched_v1290_primary_d4", "use_primary": true, ...}
```

Per movie (`v1329_terminal_candidate_receipt.json`): `44b6_0b24845f` → **fallback on 98 of 100 frames**
(median retention 0.759); the other three → **0 fallback frames**, median retention 0.987 / 1.000 / 1.002.
Where the graft moved the candidate count, the guard vetoed it; where it was allowed, it barely moved the
count. Note what this counter is and is not: it is a **count** ratio, so ≈1.0 does not prove the candidate
*sets* match — the graft's effect on those three movies is unmeasured by it, not shown to be zero.

## Measuring the decorrelation on the outputs — and the ruler that got it wrong first

Both submissions are on disk, so the overlap is directly measurable. Matching nodes on **exact coordinates**
gives node Jaccard **0.547** and edge Jaccard **0.207** — stable under rounding to 2, 1 and 0 decimals, so
not float noise, and tempting to read as "enormously decorrelated".

That reading is wrong, and it is the standing trap in this project: detections are *displaced*, not absent
(node recall 0.824 at 2.87 µm vs 0.9863 at 7 µm). Exact-coordinate equality is a ruler that cannot see a
sub-micron shift. Re-matched with a distance tolerance, greedy-nearest per `(dataset, t)`:

| tolerance | matched | node Jaccard | frac of champion nodes matched |
|---|---|---|---|
| 1.0 µm | 103713 | 0.7274 | 0.8445 |
| 2.87 µm | 113952 | **0.8610** | 0.9279 |
| 7.0 µm | 118671 | **0.9299** | 0.9663 |

(champion 122808 nodes, hedge 123485.) The edge Jaccard of 0.207 needs no separate explanation: an edge
needs *both* endpoints to match, and 0.547² ≈ 0.30, so it was tracking the same artefact.

## What this changes

**The hedge survives, on a number instead of an argument.** v1329f is a genuinely different detection field
— ~7% of nodes unshared at 7 µm, ~14% at 2.87 µm — not a near-duplicate that would die alongside the
champion in a shakeup. Final-2 stays `0947` + `v1329f`.

**But the decorrelation is not cleanly attributable to the foreign graft.** The graft is guarded off on a
quarter of the corpus, and the TTA / validator / ppsweep deltas are confounded into every difference
measured above. "Ranks hedges by WHICH component decorrelates" is still the right rule, and on this evidence
the honest answer for v1329f is *detector plus inference-time stack, jointly, with the graft's share
unresolved*.

**Two harness lessons.** A kernel that ships its source as a base64 blob is invisible to a settings diff, so
the diff's empty-side guard is what stands between us and a silent "nothing to harvest" — decode first. And
an overlap measured on exact coordinates is a lower bound, not an overlap; any node comparison in this
project needs a µm tolerance, chosen against the GT nearest-neighbour floor.
