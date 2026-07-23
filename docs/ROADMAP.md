# celltrack — ROADMAP

Ordered execution status. Substance for each item lives in `PLAN.md`; this file is the order + state.
The build log itself is git history.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Now

- [x] Scaffold generated (sdlc-scaffold v1.26.0).
- [x] Data downloaded + extracted to the `paths.yaml` root.
- [ ] `core/` data layer: OME-Zarr + GEFF readers, voxel↔µm scaling, submission CSV round-trip.
- [ ] Local evaluator matching the official metric, verified against the organizers' `evaluate.py`.

## Next

- [ ] Dataset statistics that settle the PLAN open questions: annotated fraction, GT nearest-neighbour
      distance in µm, per-prefix differences, division counts.
- [ ] Reproduce the organizers' baseline (U-Net detection + cross-attention linking) for an honest
      starting number.
- [ ] Error decomposition — detection recall vs. linking vs. gap closing — to pick the real lever.

## Later

- [ ] Division head, only once edge jaccard saturates (worth ≤ 0.1 of score).
- [ ] Grouped CV by acquisition prefix; multi-seed once a number moves.
