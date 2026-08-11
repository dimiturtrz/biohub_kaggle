"""A distance-dependent WEIGHT on the learned-evidence term: the same bonus, spent where the evidence is needed.

`cost = distance - affinity_bonus * P` weighs the head's probability by ONE scalar at every separation. Measured
(`celltrack.eval.shortcut_diagnosis`, dense movie, real detections) the two halves of that cost do not decay
together: over in-gate judged pairs the learned probability separates the true successor at every band it can be
judged in (AUC 0.972 at 4-7um, 0.950 at 7-10um, 0.970 pooled over 5-10um) while proximity collapses with range
(0.706, 0.642, 0.823). The head's value RELATIVE to distance therefore RISES with distance — and a constant bonus
weights the learned evidence LEAST exactly where it carries most, the crowded fast movers where the dense
mislinks live. One scalar cannot be right at 1um and at 8um at once, which is what a flat sweep plateau over the
bonus (15/20/25/30 -> 0.9365/0.9412/0.9390/0.9397) looks like from the inside.

This term makes the weight a function of the candidate's own separation:

    cost = distance - affinity_bonus * ramp(d) * P
    ramp(d) = max(0, 1 + strength * (d / mean_gated_distance - 1))

`mean_gated_distance` is the mean raw separation over the candidate pairs THE GATE ADMITTED in that frame gap —
self-normalising and dimensionless, the same autobalance device the affinity bonus itself is derived by, so
there is no absolute micrometre threshold and no second constant anywhere. `strength` is the single dimensionless
dose: at 0 the ramp is identically 1 and the cost is byte-identical to the shipped one, at 1 the evidence weight
is proportional to distance.

The ramp REWEIGHTS the evidence budget, it does not raise it: `mean(ramp) == 1` over the admitted pairs by
construction, so a swept `strength` moves WHERE the bonus is spent while the total spend is held — that is what
makes an arm comparable to the control instead of confounded with "more bonus".

The clamp at zero is the one guard: a candidate closer than `(1 - 1/strength)` of the mean would otherwise earn a
NEGATIVE weight, turning the head's evidence into a penalty on the very short-range pairs it is 96% right about.
There is deliberately no clamp from above — far candidates earning more weight is the mechanism, not an overshoot.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float


@dataclass(frozen=True)
class EvidenceRamp:
    """The distance-dependent multiplier on the affinity term — `1` on average, larger far, clamped at zero near.

    `strength` is dimensionless and ships OFF (`0` is the shipped cost exactly). It has no derived default: how
    fast the head's value overtakes proximity is a claim about our data, so a sweep decides it, not this file.
    """

    strength: float

    def weights(
        self, distance: Float[np.ndarray, "s t"], within_gate: Bool[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"]:
        """Each pair's evidence weight, normalised by the mean separation of the pairs this gap ADMITTED.

        A gap that admits nothing (or whose admitted pairs are all coincident) has no scale to normalise by, so
        the weight falls back to the flat `1` — the shipped cost — rather than dividing by zero.
        """
        admitted = distance[within_gate]
        mean_gated = float(admitted.mean()) if admitted.size else 0.0
        if mean_gated <= 0.0:
            return np.ones_like(distance)
        return np.maximum(0.0, 1.0 + self.strength * (distance / mean_gated - 1.0))

    @classmethod
    def applied(
        cls, ramp: "EvidenceRamp | None", distance: Float[np.ndarray, "s t"], within_gate: Bool[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"] | float:
        """The affinity term's per-pair weight, or the scalar `1.0` when no ramp is wired — the linkers' one seam.

        Every linker that prices `- affinity_bonus * P` multiplies by this instead of branching on the option itself,
        so the off state is a scalar `1.0` (an exact float multiply, hence byte-identical) in one place, not two.
        """
        return 1.0 if ramp is None else ramp.weights(distance, within_gate)
