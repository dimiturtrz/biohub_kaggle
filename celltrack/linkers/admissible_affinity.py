"""The parent probability CONDITIONED on the parents the solver may actually pick.

`EdgeTransformerScorer.affinities` soft-maxes the association logits across SOURCES, and the source set is
every detection of frame `t` — 700 on the dense movie. The linker then forbids all but the handful inside its
gate. So the probability priced into `cost = distance - affinity_bonus * P` is normalised over a choice set
the assignment does not have: it answers "which of the 700 cells in the previous frame is this one's parent"
when the question the solver asks is "which of the four admissible ones".

MEASURED (`celltrack.analysis.softmax_population`, dense movie, 69164 target columns, shipped operating point):
the in-gate share of each target's mass is mean 0.7796, median 0.9174, p25 0.7289, p5 0.0861. So on average
22% of the evidence sits on parents the gate excludes, and the leak is wildly uneven — a fifth of targets lose
more than a quarter of their mass and the worst twentieth lose over 90% of it.

That unevenness is the whole reason this can matter. A UNIFORM deflation would be absorbed by
`affinity_bonus`, which is a free scale on the same term; what a bonus cannot absorb is that the deflation is
worst exactly where the mass leaks — crowded targets with many out-of-gate neighbours, which is where the
dense mislinks live. Renormalising restores those columns to comparable confidence without touching the ones
that were already concentrated in-gate.

NOT THE RENORMALISATION `e9b` REFUTED. That swept the AXIS — over sources, over targets, raw logits,
bidirectional — and found every variant flat. This changes the POPULATION on the same axis, restricting the
denominator to the admissible parents. It is the only variant that corresponds to a constraint the solver
imposes rather than a choice of convention, and it is the conditional probability the cost was always meant
to price.

A column with no admissible parent at all is left exactly as it was: there is nothing to condition on, and its
pairs are about to be priced at infinity by the gate regardless.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float

_SOURCE_AXIS = 0
_NO_ADMISSIBLE_MASS = 0.0


@dataclass(frozen=True)
class AdmissibleAffinity:
    """Renormalises each target's parent probability over the sources its linker is allowed to choose."""

    @staticmethod
    def conditioned(
        probability: Float[np.ndarray, "s t"], within_gate: Bool[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"]:
        """`P(parent = i | parent is admissible)` per target column, leaving unreachable columns untouched."""
        admissible = probability * within_gate
        mass = admissible.sum(axis=_SOURCE_AXIS, keepdims=True)
        return np.divide(admissible, mass, out=probability.copy(), where=mass > _NO_ADMISSIBLE_MASS)

    @classmethod
    def applied(
        cls,
        conditioned: bool,  # noqa: FBT001
        probability: Float[np.ndarray, "s t"],
        within_gate: Bool[np.ndarray, "s t"],
    ) -> Float[np.ndarray, "s t"]:
        """The probability a linker should price, or the untouched one — the linkers' one seam, as `EvidenceRamp`.

        Both linkers price the same term, so the off state lives here once rather than as a branch in each.
        """
        return cls.conditioned(probability, within_gate) if conditioned else probability
