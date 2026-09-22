# The safe-division gate budget, read off the champion's own run stats

**E66, 2026-09-22.** `run_stats.csv` from the banked `celltrack-public-0947` run (fetched with
`kaggle kernels output`) carries every safe-division counter, per movie. This is the first time the
axis has been priced from the inside rather than argued about from the outside.

Four test-proxy movies (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`), totals:

| stage | count | note |
|---|---|---|
| geometric candidates | 730 | after distance/sister/mutual-NN/divergence gates |
| `REQUIRE_DIVERGENCE` rejected | **4172** | the biggest pool on the axis by 10x |
| mutual-NN rejected | 377 | |
| deepcenter safe-div checked | 730 | = geometric candidates exactly |
| deepcenter accepted | 316 | |
| deepcenter **rejected** | **414** | 57% of everything that reaches it |
| sister-symmetry rejected | 167 | |
| candidates | 149 | = 316 − 167, reconciles exactly |
| **divisions added** | **124** | |
| `skipped_cap` | **0** | on all four movies |

## Two things this settles

**The frac caps do not bind.** `SAFE_DIV_FRAME_FRAC_CAP = 0.008` and `SAFE_DIV_GLOBAL_FRAC_CAP = 0.004`
never truncate: `safe_division_skipped_cap = 0` everywhere. So a gate loosened upstream *does* reach the
output — the axis is gate-bound, not budget-bound. This is the same shape E61 found for
`candidacy=steal` ("dies at the GATE not the budget"), now confirmed from the counters instead of inferred
from a score.

**The champion already injects 124 divisions that the local proxy prices at nothing.** `div_jac ≈ 0` on
these movies, so all 124 are invisible to every local number we have. Whatever they are worth — and they
are worth *something*, because 0.947 is the banked score with them in — only the LB can say. That makes the
deepcenter threshold the one gate on the axis where a submission buys real information rather than
confirming a proxy.

## Correction: the gate constants above were read off the fallbacks, not the pins

A kernel carries each constant twice — an `os.environ[KEY] = value` pin near the top and the default of
the `os.environ.get(KEY, default)` that consumes it — and the pin wins. This document originally quoted the
fallbacks. The champion's *running* safe-division geometry is considerably wider than stated:

| constant | fallback (quoted here at first) | pin (what ran) |
|---|---|---|
| `SAFE_DIV_MAX_UM` | 4.7 | **9.0** |
| `SAFE_DIV_SISTER_MAX_UM` | 7.2 | **14.0** |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.008 | **0.0076** |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.004 | **0.00375** |
| `SAFE_DIV_SISTER_SYMMETRY_TAU` | 0.0 | **0.6** |
| `PP_SELECT_MARGIN` | 0.002 | **0.001** |

Nothing in the counter table or the in-flight arms changes — counters are measured, and the arms write
pins, which win. The mechanism gets *stronger*: with the geometry already wide, the champion pinned the
deepcenter veto **above** its own consumer fallback (0.12 → 0.20), so the 0.15 arm walks back toward the
library's value rather than inventing a number. `tools/kernel_env_diff.py` now parses `get` defaults as
effective settings and prints a `PIN SHADOWS A DIFFERENT FALLBACK` section, so a fallback cannot be quoted
as a pin again. Forty-one keys in the champion shadow this way.

## The gate that can move, and the one that cannot

`SAFE_DIV_DIVERGE_UM = 2.25` is **c3**, the donor's physical divergence quantity, already transplanted at
its derived value. Its rejection pool is 4172 — 10x the deepcenter pool — but lowering a derived physical
constant to buy candidates is exactly the knob-twiddle
`celltrack-donor-constants-dont-transplant-without-their-component` warns about. Left alone.

`DEEPCENTER_SAFE_DIV_THRESHOLD` is not derived; it is a heatmap-score cut at 0.20 rejecting 414 of 730. A
true daughter is faint and newly separated — a weak centre score is its *signature*, which is the mechanism
for suspecting this cut is too tight. The sign was verified in the consuming code before anything was
spent: `deepcenter_accept_repair_point` rejects when `score < threshold`, so **higher is stricter**. The
donor kernel `evg0942` ships 0.25 (stricter) and scores 0.942 against our 0.947 — weak, confounded, but
pointing the same way.

## The graded pair in flight

Two arms, same axis, same pool, different amplitude — so the pair reads sign *and* magnitude scaling from
one day of free slots:

- `celltrack-public-0947-safediv015` — threshold 0.20 → 0.15. Admits the band; modest.
- `celltrack-public-0947-dcvetooff` — `DEEPCENTER_SAFE_DIV_VETO=0`. Admits the whole 414; maximal.

Both verified by `tools.kernel_env_diff` to differ from the champion in exactly one setting, and the
numeric drift guard was confirmed re-pinned for the threshold arm (the boolean is not in
`_EXPECTED_NUMERIC`, so it needs no pin). `PP_CANDIDATES` touches neither key, so the kernel's own
post-processing sweep cannot overwrite either pin before `write_test_submission`.

**What each outcome means.** If the maximal arm moves the LB and the modest one moves it less, the axis is
live and monotone and there are ~35 free slots left before the deadline to walk it. If both land flat
against 0.947, the safe-division block is confirmed metric-invisible at *any* looseness and the axis closes
for the last time — which is worth knowing, because it is the last post-processing gate on the division
side that had not been priced on the real ruler.

Neither arm is a ship candidate. The banked champion and its hedge are untouched.
