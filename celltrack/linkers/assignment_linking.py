"""A global 1-to-1 linker solved as independent per-frame assignments — the ILP's division-free case, no SCIP.

The integer-program linker (`ILPLinker`) poses the whole video as one min-cost edge selection under
`MaxParents(1)` and `MaxChildren(k)`. When `k == 1` — one parent, one child, the shipped configuration — those
constraints never couple two different frame gaps: a node's incoming edge (from `t-1`) and outgoing edge (to
`t+1`) are disjoint edge sets, so the global program separates into one independent min-cost bipartite matching
per consecutive frame pair. Each such matching is exactly what `scipy.linear_sum_assignment` solves, so this
linker reproduces the ILP's division-free optimum with numpy and scipy alone — no motile, ilpy, or SCIP, whose
offline wheels fail to solve inside the Kaggle kernel.

The objective is the frontier's: an edge costs `distance - affinity_bonus * P(s->t)`, and every cell carries a
zero-cost option to stay unlinked, so an edge is selected only where its cost is negative — where the learned
association outweighs the distance it spans. The zero-cost option is a dummy partner per node in an augmented
square cost matrix, which turns "assign or skip" into the plain perfect-matching `linear_sum_assignment` wants.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.admissible_affinity import AdmissibleAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.evidence_ramp import EvidenceRamp
from celltrack.linkers.mutual_bonus import MutualBonus
from celltrack.linkers.ranker_bonus import RankerBonus
from core.data.tracks import TrackGraph
from core.geometry import Spacing

# A forbidden pairing (a real pair beyond the gate, or a node matched to another node's skip dummy) is priced
# far above any admissible cost so the solver never selects it while a zero-cost skip is on offer.
_FORBIDDEN = 1.0e12
# With no learned association, an in-gate edge needs a reward far above the longest in-gate distance so that
# linking always beats the zero-cost skip and the distance only breaks ties (the ILP's `_LINK_REWARD`). When an
# affinity IS wired the reward is dropped to zero, so an edge is selected only where `distance - bonus*P` is
# itself negative — the frontier's prob-driven selection. Kept below `_FORBIDDEN` so an out-of-gate pair, even
# rewarded, never beats a skip.
LINK_REWARD = 1.0e9
_LINK_REWARD = LINK_REWARD  # local alias, kept so the cost expression reads as one line


@dataclass(frozen=True)
class AssignmentLinker:
    """Per-frame min-cost 1-to-1 assignment with a skip option — the ILP's `MaxChildren(1)` case without SCIP.

    Matches `ILPLinker(division=False)` edge for edge: same `distance - affinity_bonus * probability` cost,
    same gate, same "an edge is worth taking only where its cost is negative" selection. An optional `affinity`
    supplies the learned association; unset it is the pure-distance assignment.
    """

    spacing: Spacing
    max_distance_um: float
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0
    # The cost of leaving a source unlinked (the ILP's disappearance weight). Zero lets a track end freely, so
    # an edge links only where `distance - bonus*P` is negative; a positive value makes the linker prefer to
    # continue a track — it takes an edge whose cost is up to `disappearance_cost`, recovering a true successor
    # the pure-negative rule drops as a track break. The frontier prices this (appearance stays free).
    disappearance_cost: float = 0.0
    # Off by default: admit a candidate edge only where the forward and reverse normalisations agree
    # (`AgreementGate`). The cost still ranks what survives by the sharp forward probability.
    agreement: AgreementGate | None = None
    # Off by default: a second cost term rewarding mutual agreement (`MutualBonus`), so the cost reads both the
    # directional preference and the symmetric agreement instead of one replacing the other.
    mutual: MutualBonus | None = None
    # Off by default: a third cost term carrying the CC0 local-association re-ranker's probability
    # (`RankerBonus`), so the cost reads the linking-state evidence — crowding, the candidate list, the primary
    # graph's degrees, a motion residual — that a pairwise affinity has no access to.
    ranker: RankerBonus | None = None
    # Off by default (`None` = one flat evidence weight at every separation, the shipped cost byte for byte).
    # Set, the forward affinity's weight becomes distance-dependent (`EvidenceRamp`): the same budget, spent
    # more on the far candidates where the head out-separates proximity and less on the near ones where
    # proximity already decides. It reweights only the FORWARD term — the gate and the other terms are untouched.
    ramp: EvidenceRamp | None = None
    # Whether the priced probability is conditioned on the ADMISSIBLE parents (`AdmissibleAffinity`).
    admissible: bool = False

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Assign each cell to at most one successor, frame pair by frame pair, minimising total edge cost."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        edges: list[tuple[int, int]] = []
        for timepoint in np.unique(timepoints)[:-1]:
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            for source, target in self._match(int(timepoint), sources, targets, positions_um):
                edges.append((int(detections.node_ids[source]), int(detections.node_ids[target])))
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _match(
        self,
        timepoint: int,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[tuple[int, int]]:
        """Source→target row pairs for one gap: the min-cost matching that beats every node's zero-cost skip."""
        if len(sources) == 0 or len(targets) == 0:
            return []
        distance = cdist(positions_um[sources], positions_um[targets])
        cost = self._gated_cost(timepoint, distance)
        augmented = self._augment_with_skips(cost)
        rows, columns = linear_sum_assignment(augmented)
        real = (rows < len(sources)) & (columns < len(targets))
        return [(int(sources[r]), int(targets[c])) for r, c in zip(rows[real], columns[real], strict=True)]

    def _priced_affinity(self) -> EdgeAffinity | None:
        """The affinity that actually prices the cost, or None when no learned term is in play.

        The two ways of having no learned term must behave identically, and they used to not: the link reward
        switched on `affinity is None`, so a mounted affinity at `affinity_bonus = 0` fell between the modes —
        prob-driven selection with no probability to drive it. Every arc then cost a positive distance, nothing
        linked, and the short-track filter deleted the whole graph, so a sweep to zero read 0.0000 and looked
        like "the affinity is everything" when it was an empty graph. A knob's zero must be its limiting case.
        """
        return self.affinity if self.affinity_bonus != 0.0 else None

    def _gated_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """`distance - bonus*P - link_reward` within the gate, forbidden beyond it — the cost a pair competes on.

        Without an affinity the link reward makes every in-gate edge beat a skip (distance-only linking); with
        one the reward is zero so only a negative `distance - bonus*P` links — matching `ILPLinker` in both.
        A `MutualBonus` adds the agreement term to that cost; an agreement gate instead narrows WHICH pairs
        compete and never touches the cost they compete on.
        """
        within_gate = distance <= self.max_distance_um
        if self.agreement is not None:
            within_gate = self.agreement.narrow(timepoint, within_gate)
        priced_by = self._priced_affinity()
        cost = distance - (0.0 if priced_by is not None else _LINK_REWARD)
        if priced_by is not None:
            probability = priced_by.probabilities(timepoint)
            if probability is not None:
                priced_probability = AdmissibleAffinity.applied(self.admissible, probability, within_gate)
                weight = EvidenceRamp.applied(self.ramp, distance, within_gate)
                cost = cost - self.affinity_bonus * weight * priced_probability
        if self.mutual is not None:
            cost = self.mutual.discount(timepoint, cost)
        if self.ranker is not None:
            cost = self.ranker.discount(timepoint, cost)
        return np.where(within_gate, cost, _FORBIDDEN)

    def _augment_with_skips(self, cost: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s+t s+t"]:
        """Embed the `s×t` cost in a square matrix giving every node a zero-cost dummy partner (stay unlinked).

        A source's skip dummies occupy the right block, a target's the bottom block, each forbidden off its
        diagonal; the dummy-dummy corner is free. A target's skip is free (appearance costs nothing) while a
        source's skip costs `disappearance_cost`, so a source pairs a real target only when that costs less
        than leaving the track to end — the ILP's "select an edge only where its cost beats disappearance".
        """
        s, t = cost.shape
        augmented = np.zeros((s + t, s + t), dtype=np.float64)
        augmented[:s, :t] = cost
        source_skip = np.full((s, s), _FORBIDDEN)
        np.fill_diagonal(source_skip, self.disappearance_cost)
        augmented[:s, t:] = source_skip
        target_skip = np.full((t, t), _FORBIDDEN)
        np.fill_diagonal(target_skip, 0.0)
        augmented[s:, :t] = target_skip
        return augmented
