"""Admitting a candidate edge only where both directions of the affinity agree — the mutual-agreement gate.

The bidirectional fusion (`BlendedEdgeTransformerScorer.fuse`: the harmonic mean of a gap scored forwards and
backwards, so a pair scores high only where both directions agree) has a second use beyond the one it already
has. Used as a SCORE it REPLACES the forward probability in the cost, and its history says the pairing matters:
under the per-frame assignment linker that measured -0.0027 — it re-states a constraint that linker already
imposes structurally, while blunting the directional sharpness the cost feeds on — while paired with the global
flow linker it scored LB 0.895, our best. This module is the third pairing: the same quantity as an admission
FILTER, with the sharp directional `P(s->t)` left driving the cost untouched.

It targets a measured failure. On the dense movie 100% of mislinked edges are affinity-inverted: a wrong NEAR
neighbour scores 0.686 against the true, farther successor at 0.134. A wrong near neighbour typically fails
MUTUAL agreement — that near cell has its own better parent, so it loses backwards what it won forwards —
while a true successor usually does not. Gating on agreement therefore removes the candidate the cost cannot
resist, and leaves the cost's ranking of whatever survives alone.

The agreement quantity is not recomputed here: the gate consumes a second `EdgeAffinity` carrying the already
fused probabilities, so `fuse` remains the one definition of mutual agreement in the codebase.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool

from celltrack.affinity import EdgeAffinity


@dataclass(frozen=True)
class AgreementGate:
    """An admission floor on mutual agreement: a candidate edge is linkable only where both directions agree.

    `mutual` supplies the bidirectionally fused probabilities for each gap (the harmonic mean the edge scorer
    produces under `bidirectional`), aligned to the same node order as the directional affinity the cost reads.
    The gate narrows the distance gate rather than replacing it — an edge must clear both — and a gap the fused
    affinity does not score is left to geometry, exactly as an unscored gap already is for the cost.

    `floor` is a real parameter and deliberately has no derived default: a principled value is a low quantile
    (say the 5th percentile) of the mutual agreement OBSERVED ON TRUE EDGES of an annotated fold, chosen so the
    floor admits nearly every true successor while sitting above the collapsed agreement of the inverted near
    neighbour — the two distributions the dense diagnosis already separates. Until that quantile is measured on
    our own data the gate stays off (`LinkerConfig.agreement_floor = None`), so no number is silently chosen.
    """

    mutual: EdgeAffinity
    floor: float

    def narrow(self, timepoint: int, within_gate: Bool[np.ndarray, "s t"]) -> Bool[np.ndarray, "s t"]:
        """The in-gate pairs whose mutual agreement also clears the floor — the edges left admissible."""
        agreement = self.mutual.probabilities(timepoint)
        if agreement is None:
            return within_gate
        return within_gate & (agreement >= self.floor)
