"""A second cost term rewarding mutual agreement — the fused probability as EVIDENCE, not as a filter.

The edge scorer emits a DIRECTIONAL probability: a softmax over sources answering "for this target, which
parent?". The bidirectional fusion (`BlendedEdgeTransformerScorer.fuse`, the harmonic mean of a gap scored
forwards and backwards) emits a SYMMETRIC one: "do these two prefer each other?". Preference and agreement
are different facts about a pair, and today the linker uses one INSTEAD of the other — `bidirectional_edges`
replaces the directional probability in the cost with the fused one, so whichever is mounted, the other is
discarded.

The measured history says which role each earns. Fusion as a SCORE (replacing the forward term) measured
-0.0027 under the per-frame assignment linker but +0.004 on the leaderboard under the global flow solver, our
best result. Fusion as a GATE (`AgreementGate`) is refuted in all four pairings it was tried in. What survives
is agreement as evidence the OBJECTIVE reads rather than a filter applied before it — which is exactly a
second additive term:

    cost = distance - affinity_bonus * P_forward - bonus * P_mutual

Both terms are subtracted discounts on the same micrometre-scale cost, so an edge links where preference and
agreement TOGETHER outweigh the distance spanned, and a pair the forward pass ranks second can still win on
agreement instead of being admitted-or-excluded by it.

The agreement quantity is not recomputed here: this term consumes a second `EdgeAffinity` carrying the already
fused probabilities, so `fuse` remains the one definition of mutual agreement in the codebase.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity


@dataclass(frozen=True)
class MutualBonus:
    """The agreement term of the two-term cost: `- bonus * P_mutual`, subtracted alongside the forward term.

    `mutual` supplies the bidirectionally fused probabilities for each gap, aligned to the same node order as
    the directional affinity the forward term reads. A gap the fused affinity does not score is left to the
    forward term and geometry, exactly as an unscored gap already is for the forward term itself.

    `bonus` is a real parameter and deliberately has no derived default. Both terms multiply a `[0, 1]`
    probability against a micrometre distance, so they live on ONE scale and trade directly against each other:
    what is spendable is the total P-weight, and the question this knob asks is how much of it buys agreement
    instead of preference. A principled first probe therefore holds the budget fixed at the derived
    `affinity_bonus` (`2 * gate_um`, the dimensional P-vs-distance coupling) and splits it — `affinity_bonus +
    bonus` constant, sweeping the split — so the probe measures the SPLIT rather than confounding it with a
    change in how strongly probability outweighs distance overall. Until such a split is measured on our own
    data the term stays off (`LinkerConfig.mutual_bonus = None`), so no number is silently chosen.
    """

    mutual: EdgeAffinity
    bonus: float

    def discount(self, timepoint: int, cost: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """The cost with the agreement term subtracted — unchanged where the fused affinity does not score the gap."""
        agreement = self.mutual.probabilities(timepoint)
        if agreement is None:
            return cost
        return cost - self.bonus * agreement
