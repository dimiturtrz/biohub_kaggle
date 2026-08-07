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
_LINK_REWARD = 1.0e9


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

    def _gated_cost(self, timepoint: int, distance: Float[np.ndarray, "s t"]) -> Float[np.ndarray, "s t"]:
        """`distance - bonus*P - link_reward` within the gate, forbidden beyond it — the cost a pair competes on.

        Without an affinity the link reward makes every in-gate edge beat a skip (distance-only linking); with
        one the reward is zero so only a negative `distance - bonus*P` links — matching `ILPLinker` in both.
        """
        within_gate = distance <= self.max_distance_um
        link_reward = _LINK_REWARD if self.affinity is None else 0.0
        cost = distance - link_reward
        if self.affinity is not None and self.affinity_bonus != 0.0:
            probability = self.affinity.probabilities(timepoint)
            if probability is not None:
                cost = cost - self.affinity_bonus * probability
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
