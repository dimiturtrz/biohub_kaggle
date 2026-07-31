"""Straightening jittery track coordinates without changing the topology — a post-link cleanup.

A detector localises each cell a little differently frame to frame, so even a correctly linked track zig-zags
around the smooth path a real cell follows. The metric matches predicted to annotated centres at a 7 um
radius, so that jitter costs matches the track should have won. Pulling each interior node of a linear track
toward the straight line between its neighbours removes the wobble and lifts the match rate — and because
only a node with exactly one parent and one child is moved, and only its coordinates, the lineage topology
(and every division) is left exactly as the linker decided.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

from core.data.tracks import Adjacency, TrackGraph

# How far each interior node is pulled toward the line between its neighbours: mostly the smoothed position,
# a little of the original, so a genuinely moving cell is still tracked but the frame-to-frame wobble is gone.
_SMOOTHING = 0.8


@dataclass(frozen=True)
class LinefitSmoother:
    """Pull each linear-track interior node toward the line between its neighbours, topology untouched."""

    strength: float = _SMOOTHING

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with interior-node coordinates de-jittered; nodes, edges and divisions unchanged."""
        adjacency = Adjacency.of(graph)
        interior = np.flatnonzero((adjacency.in_degrees == 1) & (adjacency.out_degrees == 1))
        smoothed = graph.coordinates.astype(np.float64)
        positions = smoothed[:, 1:]
        for row in interior.tolist():
            neighbour_mid = (positions[adjacency.predecessors[row][0]] + positions[adjacency.successors[row][0]]) / 2
            positions[row] = (1 - self.strength) * graph.coordinates[row, 1:] + self.strength * neighbour_mid
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=self._rounded(smoothed),
            edges=graph.edges,
        )

    @staticmethod
    def _rounded(coordinates: Float[np.ndarray, "n 4"]) -> Int[np.ndarray, "n 4"]:
        """Back to non-negative integer voxel indices — coordinates the submission and metric expect."""
        return np.clip(np.rint(coordinates), 0, None).astype(np.int64)
