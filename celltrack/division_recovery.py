"""Recovering the cell divisions a one-to-one assignment cannot represent — a post-link pass.

`MotionHungarianLinker` solves a bipartite matching, so every cell receives at most one successor. At a
division that is not a tuning miss but a structural one: one daughter keeps the edge, the other is left
unparented, and the fork the metric scores is never expressed at all. The competition score adds
`0.1 * division_jaccard` on top of the edge Jaccard, so a pipeline that cannot fork forfeits that entire
term however good its detector is — and the dropped daughter edge costs the edge Jaccard as well.

This pass runs after linking and proposes the missing second daughter: an unparented cell is joined to a
nearby cell in the previous frame that already has exactly one child. Two gates keep it honest. The parent
gate bounds how far a daughter can sit from its mother; the sister gate bounds how far the two daughters
can sit from each other, which is the discriminating one — a real mitosis leaves its products adjacent,
while an ordinary cell that merely lost its own link has no sister anywhere near it. A global cap bounds
how many forks the pass may invent, because a false fork is scored twice over: once against the division
term it was reaching for, and once against the much larger edge term.

Run this before `ShortTrackFilter`, whose division-preserving carve-out can only protect forks that exist
by the time it sees the graph.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial import KDTree

from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_SINGLE_CHILD = 1
_UNPARENTED = 0

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DivisionRecovery:
    """Join unparented cells to a nearby single-child parent, gated on parent and sister distance."""

    spacing: Spacing
    parent_gate_um: float
    sister_gate_um: float
    max_added_fraction: float

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with recovered division edges added."""
        recovered = self._division_edges(graph)
        if not len(recovered):
            return graph
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([graph.edges, recovered]),
        )

    def _division_edges(self, graph: TrackGraph) -> Int[np.ndarray, "d 2"]:
        """Every accepted fork, as node-id pairs, after the global cap trims the least confident."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        proposals: list[tuple[float, int, int]] = []
        for timepoint in np.unique(timepoints)[:-1]:
            proposals.extend(
                self._proposals_between(
                    np.flatnonzero(timepoints == timepoint),
                    np.flatnonzero(timepoints == timepoint + 1),
                    positions_um,
                    adjacency,
                )
            )
        limit = int(self.max_added_fraction * len(graph.edges))
        accepted = sorted(proposals)[:limit]
        if len(accepted) < len(proposals):
            logger.info("division cap bound: %d of %d proposals kept", len(accepted), len(proposals))
        if not accepted:
            return np.empty((0, 2), dtype=np.int64)
        return np.array([[graph.node_ids[parent], graph.node_ids[child]] for _, parent, child in accepted])

    def _proposals_between(
        self,
        source_rows: Int[np.ndarray, "s"],
        target_rows: Int[np.ndarray, "t"],
        positions_um: Float[np.ndarray, "n 3"],
        adjacency: Adjacency,
    ) -> list[tuple[float, int, int]]:
        """Candidate forks across one frame gap, as `(parent distance, parent row, daughter row)`."""
        parents = [row for row in source_rows.tolist() if adjacency.out_degrees[row] == _SINGLE_CHILD]
        orphans = [row for row in target_rows.tolist() if adjacency.in_degrees[row] == _UNPARENTED]
        if not parents or not orphans:
            return []
        sisters = [adjacency.successors[parent][0] for parent in parents]
        within_reach = KDTree(positions_um[parents]).query_ball_point(positions_um[orphans], r=self.parent_gate_um)
        return self._nearest_eligible(parents, sisters, orphans, within_reach, positions_um)

    def _nearest_eligible(
        self,
        parents: list[int],
        sisters: list[int],
        orphans: list[int],
        within_reach: list[list[int]],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[tuple[float, int, int]]:
        """Pick each orphan's closest sister-gated parent, letting no parent gain more than one daughter."""
        claimed: set[int] = set()
        found: list[tuple[float, int, int]] = []
        for orphan, reachable in zip(orphans, within_reach, strict=True):
            eligible = [
                (float(np.linalg.norm(positions_um[orphan] - positions_um[parents[index]])), index)
                for index in reachable
                if parents[index] not in claimed
                and np.linalg.norm(positions_um[orphan] - positions_um[sisters[index]]) <= self.sister_gate_um
            ]
            if not eligible:
                continue
            distance, index = min(eligible)
            claimed.add(parents[index])
            found.append((distance, parents[index], orphan))
        return found
