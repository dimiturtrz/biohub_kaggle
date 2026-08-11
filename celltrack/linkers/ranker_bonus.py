"""A third cost term: the learned association re-ranker's probability, priced exactly like the other two.

`MutualBonus` established the shape — a `[0, 1]` probability multiplied by a micrometre-scale weight and
SUBTRACTED from the transition cost, so the objective reads it as evidence rather than as a filter. This term
is the same mechanism carrying a different quantity: the public CC0 re-ranker's `P(true successor)` for the
pair (`celltrack.edges.association_ranker`), which reads 22 facts about the linking state — crowding, the
candidate list, the primary link graph's degrees, and a constant-velocity motion residual — that the pairwise
affinity transformer never sees.

    cost = distance - affinity_bonus * P_forward - mutual_bonus * P_mutual - ranker_bonus * P_ranker

Why a separate weight rather than folding the ranker into the affinity: in the 0.915 public notebook the two
learned signals enter as ONE convex evidence term, `0.85 * ranker + 0.15 * edge_prob`, with the transformer
demoted to a minority share of the association signal. That split is a claim about the RATIO of the two, and
holding them as separate weights on one cost is what lets a sweep test that claim on our data instead of
assuming their number transfers (`LinkerConfig.ranker_bonus` documents the translation into our scale).

The ranker's probabilities are not computed here: this term consumes an `EdgeAffinity` carrying the already
scored gaps, so the ranker keeps one home and the linker layer never imports `celltrack.edges`.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity


@dataclass(frozen=True)
class RankerBonus:
    """The re-ranker term of the cost: `- bonus * P_ranker`, subtracted alongside the forward and mutual terms.

    `ranker` supplies the re-ranker probability for each gap, aligned to the same node order as the directional
    affinity. A gap the ranker does not score is left to the other terms and geometry, exactly as an unscored
    gap already is for them — and so is a pair outside the linker's gate, which the ranker scores as 0 because
    it was never in any source's candidate list.

    `bonus` has no derived default and the knob ships OFF, because the number that would be copied is not
    dimensionless: see `LinkerConfig.ranker_bonus` for what the public 0.85/0.15 evidence split translates to
    on our micrometre cost scale, and why the split is the thing to sweep rather than the absolute weight.
    """

    ranker: EdgeAffinity
    bonus: float

    def discount(self, timepoint: int, cost: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The cost with the re-ranker term subtracted — unchanged where the ranker does not score the gap."""
        probability = self.ranker.probabilities(timepoint)
        if probability is None:
            return cost
        return cost - self.bonus * probability
