# Motion structure, and why the mislinks are not one failure

*2026-08-12 — analysis only, no training, no submission. Tools:
`celltrack.analysis.motion_statistics`, `celltrack.analysis.localisation`.*

The campaign has known the *displacement distribution* since the linker gate was derived (median 1.67 µm,
p99 7.20, max 9.96) and has never measured the motion's **structure**. Four motion-shaped attempts had been
refuted — a damped-velocity linker cost, motion-predicted geometry inside the flow solver, a prior-velocity
feature on the edge head, and the e9b velocity cost knobs — and all four estimate velocity from **one**
previous step, where the signal is the size of the noise. Whether that was the reason, or whether the motion
carries no direction at all, was never separated.

## What cells actually do

Measured on annotated tracks across the eight CV movies (ground truth, so no linker error contaminates it).

| movie | MSD exp | stationary | turn cos | frame drift | residual |
|---|---|---|---|---|---|
| 44b6_0113de3b | 1.79 | 44% | +0.61 | — | — |
| 44b6_0b24845f | 1.93 | 65% | +0.49 | — | — |
| 6bba_05b6850b | 1.25 | 88% | −0.02 | 0.98 µm (61%) | 1.01 µm |
| **6bba_05db0fb1** | 1.59 | 78% | +0.22 | 1.48 µm (65%) | 1.33 µm |
| 44b6_341df25f | 1.63 | 84% | +0.38 | 1.96 µm (83%) | 1.57 µm |
| 44b6_e57ff5c6 | 1.44 | 38% | +0.45 | 5.09 µm (92%) | 1.83 µm |
| 6bba_969618f6 | 1.83 | 62% | +0.62 | 2.16 µm (88%) | 1.08 µm |
| 6bba_fc83837d | 1.65 | 75% | +0.40 | 1.63 µm (71%) | 1.39 µm |

**Motion is superdiffusive.** MSD exponent median ~1.64 against 1.0 for a random walk and 2.0 for a straight
line. Turn cosine is positive on seven of eight, and rises with step length on five of seven — the
signal-to-noise control, since a long step is better determined than a short one. Localisation noise adds a
*constant* to the MSD, which flattens short lags and biases the exponent **down**, so 1.64 is a floor.

**Most steps are not steps.** 38–88 % of displacements fall below the 2.37 µm noise floor (√2 × the measured
1.675 µm per-node localisation offset). The corpus *median* step of 1.675 µm is itself below that floor — which
is, on its own, the whole explanation for why single-frame velocity has never worked, and is pinned as a test.

**The tissue moves as a body.** Frame drift is 61–92 % of the mean step. Neighbour displacement agreement is
**equal to its same-frame shuffled control** on every movie with samples (dense +0.356 vs +0.320; e57ff5c6
+0.690 vs +0.696). Permuting which cell owns which displacement changes nothing, so at this sampling the
coherence is global drift, not local structure.

### What that does to the wider-temporal-window plan

Displacement accumulates, so multi-frame averaging genuinely beats the per-step noise floor — the mechanism
behind adding a temporal positional encoding and T>2 is sound. But **most of what accumulates is drift, and a
uniform drift shifts every candidate equally, so it cannot disambiguate anything.** The residual individual
motion after removing it is 1.0–1.8 µm, at or below the localisation noise. A wider window would largely learn
a quantity that is (a) estimable far more cheaply from the current frame and (b) useless for choosing between
candidates. The bead is downgraded on measurement rather than on opinion.

**The null is underpowered in one specific direction, and it is the interesting one.** It was measured on ~12
*annotated* cells per frame in coarse 10–40 µm bands, which cannot resolve a spatially varying field. A drift
**field** does not cancel between candidates. And it can be estimated from ~700 *detections* per frame — 60×
the samples, no labels, available at inference. That is filed as the next test.

## The mislinks are a mixture

An earlier finding left a confound open: mislinked edges' endpoints carry ~2.1× the population localisation
error (3.48 µm vs 1.675 µm), and it was unresolved whether those were true mislinks or missing detections
wearing a mislink label. Splitting the error by side settles part of it.

```
                    source      true target    rival
mislinks (n=37)    3.634 µm     3.350 µm      n=2 of 37
correct  (n=1102)  1.675 µm     1.675 µm      n=11
grid-explainable:  mislinks z/y/x 0.279/0.412/0.471  vs population 0.502/0.833/0.833
```

Both ends are elevated near-equally, so it is not a one-sided "the successor's detection is bad" defect, and it
is not quantisation (the grid-explainable share roughly halves). The rival column confirms from a second
direction that the wrong partner is almost never an annotated cell — 2 of 37.

That raised a sharper hypothesis: if both endpoints are off by ~3.5 µm in uncorrelated directions, the step
between them inherits ~√2 × 3.5 ≈ 4.9 µm of error, the same size as the ">5 µm" displacements that define these
cases — so the fast mover could be **manufactured by our own read-out**. The arbiter is the step between
*annotated* centres, which no detection touches:

```
annotated step   mislinked  n=  37   median 3.275 µm   IQR [1.285, 6.894]
                 correct    n=1132   median 1.675 µm   IQR [0.575, 2.188]
```

**Refuted.** The annotation independently says mislinked cells travel 2× the population median. The motion is
observed, and the causality runs the other way: fast cell → smeared during acquisition → peak displaced → both
endpoints off. One cause explains both the misleading distance term and the affinity being read off-centre.

**But the IQR is the finding.** The bottom quartile of mislinks has an annotated step of ~1.3 µm — cells that
barely moved. The 37 are **not one failure mode**: roughly half are genuine fast movers, and the rest fail for
some other reason. Any mechanism aimed at fast movers can address at most half, which re-sizes every proposal
on this axis and explains why single-mechanism attacks keep landing inside noise.

## Standing

- Detection is finished on every measurable axis (4 misses in 1229; resolution closed 2026-08-11).
- Association remains the whole gap, worth +0.086 on the dense movie.
- The live, cheap, inference-time candidate is the **local drift field**; the expensive one (T>2 + positional
  encoding) is downgraded but not closed.
- The 37 mislinks should stop being treated as one population in any future arm.

## Architecture note

`celltrack/analysis/` was split out of `celltrack/eval/` in the same change: eval ranks configurations, analysis
explains runs, and the two had been conflated. The seven leaf diagnosis modules moved cleanly; `dense_diagnosis`
could not, because its `Fate`/`Charge` taxonomy is shared vocabulary that `model_evaluator` and two trainers
import, so moving it wholesale would create a package cycle. Its taxonomy belongs in `core/metrics` beside the
edge counting it already depends on — filed, not bodged.
