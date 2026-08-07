"""A linker that recovers cell divisions the one-to-one nearest-neighbour linker cannot.

Nearest-neighbour linking assigns each cell at most one successor, so at a division — one parent, two
daughters — it keeps a single daughter edge and drops the other. That lost edge is the entire gap between
what NN linking scores on perfect detections (~0.998) and 1.0, and it is structural, not a tuning miss.

This linker runs the same one-to-one assignment first, then a second pass: a daughter left unmatched by
the assignment is joined to the nearest already-matched parent within a division radius, provided that
parent does not already have two children. The result is a linker that can represent the two-child
topology a division needs, so on ground-truth nodes it exceeds the nearest-neighbour ceiling instead of
being capped by it.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Int
from scipy.spatial.distance import cdist

from celltrack.linkers.linking import NearestNeighbourLinker
from core.data.tracks import TrackGraph
from core.geometry import Spacing

_MAX_CHILDREN = 2


@dataclass(frozen=True)
class DivisionAwareLinker:
    """One-to-one nearest-neighbour linking plus a division-recovery pass."""

    spacing: Spacing
    max_distance_um: float
    division_distance_um: float

    def link(self, detections: TrackGraph) -> TrackGraph:
        """Link one-to-one, then attach each unmatched daughter to its nearest eligible parent."""
        base = NearestNeighbourLinker(spacing=self.spacing, max_distance_um=self.max_distance_um).link(detections)
        divisions = self._division_edges(detections, base)
        edges = np.concatenate([base.edges, divisions]) if len(divisions) else base.edges
        return TrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=edges)

    def _division_edges(self, detections: TrackGraph, base: TrackGraph) -> Int[np.ndarray, "d 2"]:
        """Second daughter edges: an unmatched cell joined to a nearby already-matched parent."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        children_of = self._child_counts(base)
        matched_targets = set(base.rows_of(base.edges[:, 1]).tolist())
        edges = [
            detections.node_ids[[parent, child]]
            for timepoint in np.unique(timepoints)[:-1]
            for parent, child in self._divisions_between(
                np.flatnonzero(timepoints == timepoint),
                np.flatnonzero(timepoints == timepoint + 1),
                positions_um,
                matched_targets,
                children_of,
            )
        ]
        return np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)

    def _divisions_between(
        self,
        source_rows: Int[np.ndarray, "s"],
        target_rows: Int[np.ndarray, "t"],
        positions_um: Int[np.ndarray, "n 3"],
        matched_targets: set[int],
        children_of: dict[int, int],
    ) -> list[tuple[int, int]]:
        """For one frame gap, pair each unmatched daughter with the nearest parent that has room."""
        unmatched = [row for row in target_rows.tolist() if row not in matched_targets]
        parents = [row for row in source_rows.tolist() if children_of.get(row, 0) == 1]
        if not unmatched or not parents:
            return []
        distances = cdist(positions_um[unmatched], positions_um[parents])
        pairs = []
        for daughter_index, daughter in enumerate(unmatched):
            parent_index = int(np.argmin(distances[daughter_index]))
            if distances[daughter_index, parent_index] > self.division_distance_um:
                continue
            parent = parents[parent_index]
            if children_of.get(parent, 0) < _MAX_CHILDREN:
                pairs.append((parent, daughter))
                children_of[parent] = children_of.get(parent, 0) + 1
        return pairs

    @staticmethod
    def _child_counts(base: TrackGraph) -> dict[int, int]:
        """How many children each detection already has after the one-to-one pass, in row space."""
        sources = base.rows_of(base.edges[:, 0]) if len(base.edges) else np.empty(0, dtype=np.int64)
        rows, counts = np.unique(sources, return_counts=True)
        return dict(zip(rows.tolist(), counts.tolist(), strict=True))
