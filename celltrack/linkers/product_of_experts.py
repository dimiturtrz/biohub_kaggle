"""Fusing a learned appearance probability with a motion+diffusion prior as ONE assignment cost.

Every linker in the family prices a candidate edge by `distance - affinity_bonus * P(source->target)`
(`motion_linking.py`): a raw Euclidean distance and a learned probability, added through a `bonus` whose only
justification is dimensional (`2 * gate`). The two terms are incommensurate — a micrometre against a unit
probability — and the weight that couples them is fitted, not derived.

This is the principled form. Read BOTH terms as log-likelihoods of the SAME event, "target `t` is the child
of source `s`", contributed by two experts that see different evidence:

- the APPEARANCE expert is the edge head's `P_edge(t|s)` — the soft learned association;
- the MOTION expert is `P_motion(t|s) ~ exp(-||t - predicted(s)||^2 / 2*sigma^2)` — a Gaussian on the residual
  between the target and where the source was predicted to land (`MotionPrediction.distances`), with `sigma`
  the residual scale a persistent-random-walk fit already measures, not a tuned constant.

A product of experts fuses them by MULTIPLYING the distributions (adding the log-probabilities) and letting the
solver read `-log P_fused` as the cost. Equal weight is the only unfitted choice: it is exactly the Bayesian
assumption that appearance and motion are conditionally independent given the true assignment, so the scale of
each expert lives INSIDE its own calibration (`sigma` for motion, a temperature for the head) rather than in a
free lambda between them. Where both experts are flat — the irreducibly ambiguous dense crowd — the product
stays flat, so the fusion refines the confident pairs and leaves the ambiguous ones soft, instead of forcing a
confidence the evidence does not support.

The primitive here is pure and optimiser-agnostic: it turns probabilities and predicted distances into a fused
cost. Wiring it into a linker's `_blended_cost` in place of the linear blend is a separate, opt-in step.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

# Below this the appearance probability is treated as "no evidence" rather than "impossible": a hard zero would
# send the log to -inf and make the pair unlinkable on appearance alone, overriding a motion expert that may be
# certain. The floor keeps a zero-probability pair priced by motion, which is the softness the fusion is for.
_PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class MotionDiffusionPrior:
    """A Gaussian motion expert: the log-likelihood that a target is where the source was predicted to go.

    `sigma_um` is the standard deviation of the residual between a target and the source's predicted position,
    in micrometres — the spread the diffusion prior allows around the prediction. It is a MEASURED quantity: a
    persistent-random-walk fit (`celltrack.analysis.motion_statistics`) reports the localisation noise and the
    unpredicted step variance whose sum is this residual scale, so the prior's width is derived from how these
    cells actually move rather than chosen. `from_walk` (to follow) will construct it from that fit directly.
    """

    sigma_um: float

    def log_likelihood(self, distances_um: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """`-||residual||^2 / (2*sigma^2)` for each pair — the Gaussian log-density up to a constant.

        `distances_um` is the residual magnitude the motion model already computes: the distance from the
        source's PREDICTED position to the target (`MotionPrediction.distances`), not the raw separation. The
        `-log(2*pi*sigma^2)` normaliser is dropped because it is one constant across every pair and cannot move
        an assignment that picks one target per source.
        """
        return -(distances_um**2) / (2.0 * self.sigma_um**2)


@dataclass(frozen=True)
class ProductOfExperts:
    """Equal-weight log-linear pool of an appearance expert and a motion expert into one assignment cost."""

    def as_log_probability(self, probabilities: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The appearance expert's log-probability, floored so a zero pair is priced by motion, not ruled out."""
        return np.log(np.maximum(probabilities, _PROBABILITY_FLOOR))

    def log_fused(
        self, log_appearance: Float[np.ndarray, "s t"], log_motion: Float[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"]:
        """The unnormalised fused log-probability `log P_edge + log P_motion` — the product of experts, equal weight.

        Left unnormalised on purpose: the partition function is one additive term the one-to-one solver does not
        read (it shifts every pair equally), so carrying it would cost a `logsumexp` per gap and change no
        assignment. A caller that needs a proper distribution can subtract the per-source `logsumexp` itself.
        """
        return log_appearance + log_motion

    def cost(
        self, log_appearance: Float[np.ndarray, "s t"], log_motion: Float[np.ndarray, "s t"]
    ) -> Float[np.ndarray, "s t"]:
        """`-log P_fused` — the per-pair cost a min-cost assignment reads, low where both experts agree."""
        return -self.log_fused(log_appearance, log_motion)
