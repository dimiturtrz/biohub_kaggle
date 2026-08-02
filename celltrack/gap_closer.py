"""Bridging a one-frame detection dropout by reusing an existing unlinked node — a post-link pass.

A cell the detector misses for a single frame breaks its track in two: the track before the gap ends, and
the track after it starts, with the metric charging both the lost incoming and outgoing edge. This pass
reconnects them across a one-frame gap — an END (a node with no successor) at time t is matched to a START
(no predecessor) at t+2 when they are close enough to be the same cell, and the missing middle is supplied
by an *existing* isolated detection at t+1 sitting near their midpoint. Two edges are added, end→middle and
middle→start, recovering both dropped edges at once.

Only the reuse path is geometry: the public frontier also *inserts a synthetic* midpoint when no real
detection is near the gap, confirmed by a DeepCenter centre-prior veto. That model is not mounted here, so a
gap with no reusable node is left open rather than filled with an unconfirmed invention — the honest,
precision-safe half of the frontier's `gap_close`. Matching is one-to-one per frame (a Hungarian assignment
within the distance gate); a global cap bounds how many bridges the pass may add.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_UNLINKED = 0
_GAP_SPAN = 2  # end at t, start at t+2: exactly one missing frame between them
_BEYOND_GATE = 1.0e9


@dataclass(frozen=True)
class GapCloser:
    """Reconnect a one-frame gap through an existing isolated node near the bridged midpoint."""

    spacing: Spacing
    gate_um: float
    reuse_um: float
    max_added_fraction: float

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with reused-midpoint bridge edges added across one-frame gaps."""
        bridges = self._bridges(graph)
        if not bridges:
            return graph
        added = np.array([edge for bridge in bridges for edge in bridge], dtype=np.int64)
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([graph.edges, added]),
        )

    def _bridges(self, graph: TrackGraph) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Accepted (end→middle, middle→start) edge pairs, tightest first, within the global cap."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        ends = np.flatnonzero(adjacency.out_degrees == _UNLINKED)
        starts = np.flatnonzero(adjacency.in_degrees == _UNLINKED)
        isolated = np.flatnonzero((adjacency.in_degrees == _UNLINKED) & (adjacency.out_degrees == _UNLINKED))
        proposals = self._proposals(ends, starts, isolated, positions_um, timepoints)
        cap = int(self.max_added_fraction * len(graph.edges))
        return self._accept(sorted(proposals), graph.node_ids, cap)

    def _proposals(
        self,
        ends: Int[np.ndarray, "e"],
        starts: Int[np.ndarray, "s"],
        isolated: Int[np.ndarray, "i"],
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
    ) -> list[tuple[float, int, int, int]]:
        """Every gated end→start match with a reusable midpoint, as `(cost, end, middle, start)` rows."""
        proposals: list[tuple[float, int, int, int]] = []
        for timepoint in np.unique(timepoints):
            frame_ends = ends[timepoints[ends] == timepoint]
            frame_starts = starts[timepoints[starts] == timepoint + _GAP_SPAN]
            middles = isolated[timepoints[isolated] == timepoint + 1]
            if not len(frame_ends) or not len(frame_starts) or not len(middles):
                continue
            proposals.extend(self._match_frame(frame_ends, frame_starts, middles, positions_um))
        return proposals

    def _match_frame(
        self,
        frame_ends: Int[np.ndarray, "e"],
        frame_starts: Int[np.ndarray, "s"],
        middles: Int[np.ndarray, "m"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[tuple[float, int, int, int]]:
        """One frame's one-to-one end→start assignment, each kept only if a reusable midpoint is in reach."""
        cost = cdist(positions_um[frame_ends], positions_um[frame_starts])
        rows, columns = linear_sum_assignment(np.where(cost <= self.gate_um, cost, _BEYOND_GATE))
        matched: list[tuple[float, int, int, int]] = []
        for row, column in zip(rows, columns, strict=True):
            if cost[row, column] > self.gate_um:
                continue
            end, start = int(frame_ends[row]), int(frame_starts[column])
            middle = self._reusable_midpoint(end, start, middles, positions_um)
            if middle is not None:
                matched.append((float(cost[row, column]), end, middle, start))
        return matched

    def _reusable_midpoint(
        self, end: int, start: int, middles: Int[np.ndarray, "m"], positions_um: Float[np.ndarray, "n 3"]
    ) -> int | None:
        """The isolated node nearest the end–start midpoint, if one sits within the reuse radius."""
        midpoint = (positions_um[end] + positions_um[start]) / 2.0
        distances = np.linalg.norm(positions_um[middles] - midpoint, axis=1)
        nearest = int(np.argmin(distances))
        return int(middles[nearest]) if distances[nearest] <= self.reuse_um else None

    @staticmethod
    def _accept(
        proposals: list[tuple[float, int, int, int]], node_ids: Int[np.ndarray, "n"], cap: int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Greedily take the tightest bridges, letting no node serve as two middles/ends/starts, up to the cap."""
        used: set[int] = set()
        bridges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for _, end, middle, start in proposals:
            if len(bridges) >= cap:
                break
            if used & {end, middle, start}:
                continue
            used |= {end, middle, start}
            bridges.append(((int(node_ids[end]), int(node_ids[middle])), (int(node_ids[middle]), int(node_ids[start]))))
        return bridges
