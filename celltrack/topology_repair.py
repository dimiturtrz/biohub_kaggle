"""Deterministic graph cleanups that enforce the biology a linker can violate — a post-link pass.

A global or greedy linker can leave a track graph in states no real lineage occupies: an edge that skips a
frame, a cell wired to two parents, a link longer than any cell moves in one step, or a node left with no
link at all. None of these need a model to fix — they are invariants, and this pass restores them in one
sweep. Every edge must join consecutive frames and span no more than `edge_max_um` micrometres; every node
keeps at most its single closest incoming edge (a cell has one parent, though it may still divide into two
children); and any node no surviving edge touches is dropped as an unbacked detection.

Order is the point: the edge invariants and the single-parent choice run first, so pruning sees the final
edge set and removes exactly the nodes those repairs isolated. This is the frontier's `filter_output_graph`
(`enforce_next_frame`, `single_parent_repair`, `prune_isolated`, `edge_max_um`), minus nothing — all four are
pure geometry with no DeepCenter dependency.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from core.data.tracks import TrackGraph
from core.geometry import Spacing

_NEXT_FRAME = 1


@dataclass(frozen=True)
class TopologyRepair:
    """Restore per-edge and per-node lineage invariants a linker can break, then prune the nodes left isolated."""

    spacing: Spacing
    edge_max_um: float

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with only consecutive-frame, in-gate, single-parent edges and no isolated nodes."""
        return self._prune_isolated(graph, self._repaired_edges(graph))

    def _repaired_edges(self, graph: TrackGraph) -> Int[np.ndarray, "k 2"]:
        """The edges surviving the frame, gate and single-parent invariants, in their original order."""
        edge_rows = graph.edge_rows()
        positions_um = self.spacing.to_micrometres(graph.positions())
        displacement = np.linalg.norm(positions_um[edge_rows[:, 1]] - positions_um[edge_rows[:, 0]], axis=1)
        elapsed = graph.timepoints()[edge_rows[:, 1]] - graph.timepoints()[edge_rows[:, 0]]
        valid = (elapsed == _NEXT_FRAME) & (displacement <= self.edge_max_um)
        chosen = self._single_parent(np.flatnonzero(valid), edge_rows[:, 1], displacement)
        return graph.edges[chosen]

    @staticmethod
    def _single_parent(
        candidate: Int[np.ndarray, "c"],
        target_rows: Int[np.ndarray, "e"],
        displacement: Float[np.ndarray, "e"],
    ) -> Int[np.ndarray, "k"]:
        """Keep, per target node, only its closest incoming candidate edge — one parent per cell."""
        nearest_first = candidate[np.argsort(displacement[candidate], kind="stable")]
        _, per_target = np.unique(target_rows[nearest_first], return_index=True)
        return np.sort(nearest_first[per_target])

    @staticmethod
    def _prune_isolated(graph: TrackGraph, edges: Int[np.ndarray, "k 2"]) -> TrackGraph:
        """Drop every node no surviving edge references — the unbacked, unlinkable detections."""
        keep: Bool[np.ndarray, "n"] = np.zeros(len(graph.node_ids), dtype=bool)
        keep[graph.rows_of(np.unique(edges.reshape(-1)))] = True
        return TrackGraph(node_ids=graph.node_ids[keep], coordinates=graph.coordinates[keep], edges=edges)
