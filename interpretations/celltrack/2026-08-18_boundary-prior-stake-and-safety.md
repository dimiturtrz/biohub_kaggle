# Activating the flow boundary prior — the stake, the selectivity, and the safety

*2026-08-18 · CPU-only, `python -m celltrack.analysis.boundary_census`, 8 annotated movies*

## Question

`BoundaryPrior` is built and unit-tested but **dark in the shipped operating point** (op93 wires no
`volume_shape`/`boundary`, so the flow linker pays the flat `disappearance_cost` everywhere). Activating
it (bd **duw4(a)**) discounts to free every track boundary the observation window already explains — a
birth in the first observed frame, or within one gate step (10 um) of a volume face. Three things decide
whether to spend a GPU affinity pass on it:

1. **Stake** — what share of real boundaries does the prior reclassify from full charge to free?
2. **Selectivity** — how much of the volume does the discount shell cover (a toothless prior frees
   everything)?
3. **Safety** — are the boundaries it leaves charged genuinely interior, or real births it would wrongly
   penalise?

## Numbers

Volume is `(z=64, y=256, x=256)` at spacing `(1.625, 0.40625, 0.40625)` um → a physically **near-isotropic
~102 um cube**. Gate = 10 um.

| movie | births | explained | interior | interior depth (gate-units past band) |
|---|---|---|---|---|
| 6bba_05b6850b | 16 | 81.2% | 18.8% | +1.76 |
| 6bba_05db0fb1 (dense) | 46 | 34.8% | 65.2% | **+1.23** |
| 6bba_969618f6 | 21 | 81.0% | 19.0% | +0.54 |
| 6bba_fc83837d | 34 | 82.4% | 17.6% | +1.95 |
| 44b6_341df25f | 7 | 100% | 0% | — |
| 44b6_e57ff5c6 | 7 | 100% | 0% | — |
| 44b6_0113de3b / 0b24845f | 2 | 100% | 0% | — |

**Boundary band = 47.6% of the volume** at the shipped 10 um margin (36% at p99 = 7.2 um, 26% at 5 um).

## Read

1. **Stake is large on 7/8 movies (70–100% of boundaries reclassified as free).** A track break costs an
   appearance plus a disappearance; freeing most true boundaries changes many linker trade-offs. Not
   sub-noise — clears the bar for one GPU pass. The dense confusor movie 05db0fb1 is the honest outlier
   (35% explained), the movie where the prior helps least.

2. **Safety holds on ground truth.** The dense movie's 65% interior births sit a **median +1.23 gate-units
   past the band edge** (~22 um from any face) and are **spread across the movie** (timepoint median 0.46,
   not clustered at t=0). They are genuine mid-volume, mid-movie track starts — divisions the annotator
   recorded as separate tracks, or re-acquisitions — NOT near-face entries the exact margin barely missed,
   and NOT annotation-onset artefacts. So the prior charging them full is CORRECT: on GT it frees the true
   edge/first-frame births and charges the true interior ones. 969618f6 is the one to watch (+0.54 deep, 4
   near-band births) but its stake is small.

3. **Selectivity is the design cost.** The discount shell is ~48% of the volume — a consequence of a 10 um
   step against a 102 um cube, not a loose margin. On GT that is fine (real entries cluster at faces). The
   residual risk is at **inference**: a linker's spurious break (a fragmented track's phantom birth) that
   lands in that 48% shell is subsidised rather than charged. That is the one thing this CPU census cannot
   see — it needs detected tracks (GPU).

## Conclusion

- **duw4(a) is launch-ready and downside-protected on GT** — activation never charges an interior birth
  MORE than the shipped flat cost (factor ∈ {0,1}, shipped flat = the interior ceiling), and the census
  confirms the boundaries it frees are real FoV/first-frame entries while the ones it charges are real
  deep-interior starts. Blocked only on one GPU affinity pass (edge affinity is not disk-cached).
- **Open, GPU-only:** measure how many *spurious* inference births land in the 48% band — that bounds the
  fragmentation-subsidy downside the GT census is blind to.
- **Queued refinement (mechanism-sound, unproven):** decouple the boundary margin from the linking gate.
  The gate is the linking *admissibility* radius (must be permissive → max step, 10 um); the boundary
  margin answers a *probabilistic* question ("could this cell have been off-field last frame") where p99
  (7.2 um) is the honest cut. They answer different questions, so gate = margin is a coincidence, not a
  principle — p99 shrinks the shell 48%→36% at ~1% miss cost on true entries. Whether it moves LB is
  GPU-measurable; file, don't ship.

See [[2026-08-18_motion-persistence-drift-vs-window]] for the sibling architecture-gate result; both
land on bd **duw4**'s thesis — every LB gain altered how the linker decides.
