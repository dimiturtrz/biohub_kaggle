# The division arc works, and the division axis still has no result in it

`hq93` opened on a real gap: the shipped min-cost-flow linker gives each detection a unit-capacity
`("in", row) -> ("out", row)` arc, so out-degree is at most one and a division is not expressible at all.
Every division gate in the pipeline is therefore shown a continuation. The fix is one parallel arc,
`SOURCE -> ("out", row)`, priced at `appearance[row] + division` — a surcharge over that detection's own
track-start charge, so a parentless cell can never buy its own appearance through it.

That part worked. The arc is built, both flow engines agree with it open, and on the division-rich ruler it
did the two things the prior record said could not happen: the **fn** side moved (24 -> 21, where every
earlier division experiment moved only fp, ceiling +0.0011) and the edge term did not pay for it.

Then it did not survive its denominator.

## The measurements

`--divisions 4` (8 movies selected as the most division-rich per acquisition):

| arm | edge | div | score |
|---|---|---|---|
| baseline (postproc recovery only) | 0.9214 | 0.0286 | 0.9243 |
| linker arc 6.0 + postproc | 0.9219 | 0.0755 | 0.9294 |
| linker arc 6.0, postproc off | 0.9228 | 0.0682 | 0.9296 |
| postproc off, no arc | 0.9219 | 0.0000 | 0.9219 |

Surcharge sweep with postproc off: 4.5 -> 0.9249 · **6.0 -> 0.9296** · 7.5 -> 0.9253 · 9.0 -> 0.9219 (the arc
is never bought). A spike that loses 0.0045 to either neighbour.

The same arm on sets that were not selected for divisions:

| set | baseline | arc 6.0 | delta |
|---|---|---|---|
| default proxy (4 test movies) | 0.9375 | 0.9348 | **-0.0027** |
| CV-8 (4 test + 4 denser train) | 0.9099 | 0.9081 | **-0.0018** |

On both, `div_jac = 0.0000` in **every** arm, baseline included. The div term is not merely small on these
movies — it is unmeasurable, so the arc's cost appears naked in the edge term, and it is monotone in how
much fork the arc buys (default proxy: 7.0 -> 0.9366, 6.0 -> 0.9348, 5.0 -> 0.9342).

## What that means

The arc adds forks everywhere. Where GT has divisions they land on real ones and the edge term rises; where
GT has none every fork is wrong and there is nothing to pay it back. The +0.0053 was measured on a
denominator chosen to contain the thing being sold. That is the CLAUDE.md denominator failure in its exact
shape: a rate computed inside the subset that was selected for it is not a rate over the corpus.

The 6.0 peak is fitted, not derived. It coincides with `2 * disappearance_cost`, which is suggestive and was
never argued; a sharp single-point optimum swept on 8 movies does not need a derivation, it needs to be
distrusted.

## The postproc stage, measured from the same runs

`AffinityDivisionRecovery` turns out to be the mirror image, and the attribution arms priced it for free:

| set | with recovery | recovery off | delta (edge) |
|---|---|---|---|
| default proxy | 0.9375 | 0.9380 | +0.0005 |
| CV-8 | 0.9099 | 0.9109 | +0.0010 |
| divisions-4 | 0.9214 (edge) | 0.9219 (edge) | +0.0005 |

It costs edge on every set and buys `div = 0.0286` only where divisions are scoreable — net **+0.0024** on the
division-rich set, net **-0.0005 to -0.0010** on both neutral ones. Its sign flips with the same property the
arc's does.

So both ends of the division axis are controlled by one unmeasured quantity: what fraction of the hidden test
corpus is division-bearing, weighted as the metric weights it (by `tp+fp+fn` per video, so dense movies
dominate). Nothing local answers that. E61 concluded division recall was unreachable from post-processing;
this says it is reachable from inside the linker and still not worth reaching, because every delta on this
axis — 0.0005 to 0.0053 — sits an order of magnitude below the 0.01 noise floor and two below the 1.5% bar.

## Disposition

- `LinkerConfig.division_cost` ships **off** (`None` = the shipped 1-to-1 network, arc for arc). Off is the
  default already; nothing to revert.
- `AffinityDivisionRecovery` stays **on**. Removing it is a sub-floor gain of unknown sign on the real
  corpus, and it is the incumbent — a tie decides for the shipped thing.
- The arc is kept as code, not deleted. It is the only construct that can express a fork, and it is the piece
  a future detector would need: today's forks are wrong because the detector cannot tell a daughter from a
  continuation, not because the linker cannot represent one. Re-test it if the detection side ever moves.
- Do not re-sweep `division_cost` on `--divisions N`. That ruler answers "can the arc find divisions", which
  is now yes, and cannot answer "should it", which is the only open question.
