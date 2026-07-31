"""A global integer-program linker: min-cost selection of edges over the whole video at once.

The nearest-neighbour and division-aware linkers decide each frame pair in isolation, greedily. This linker
poses the whole video as one optimisation — the family the leading public solutions use — and lets a solver
choose the edge set that minimises a global cost under biological constraints. Every candidate edge (a pair
one frame apart within the physical gate) carries a reward for being selected, discounted by the distance it
spans, so the solver links as much as it can and, where it must choose, prefers the shorter edge. Biology
enters as constraints: at most one parent per cell, at most two children, so a division — one parent, two
daughters — is representable where a one-to-one assignment forbids it. Every detection is kept (a small node
reward), so the linker decides edges, not which cells exist.

Candidate edges are built with a KD-tree so only within-gate pairs enter the program — the number of
variables stays linear in the detections, not quadratic, which is what lets the solve run per video.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from jaxtyping import Float, Int
from motile import Solver, TrackGraph
from motile.constraints import MaxChildren, MaxParents
from motile.costs import EdgeSelectedCost, NodeSelectedCost
from scipy.spatial import KDTree

from core.data.tracks import TrackGraph as CellTrackGraph
from core.geometry import Spacing

_FRAME = "t"
_DISTANCE = "distance"
# A node is kept for a nudge below zero, so every detection survives and only the edges are the solver's to
# decide; without a reason to select a node it would drop the isolated ones.
_KEEP_NODE_REWARD = -1.0
# Selecting an edge is rewarded far more than the longest in-gate edge costs, so linking always beats not
# linking; the distance term then only breaks ties, letting the constraints pick which links survive.
_LINK_REWARD = 1.0e6


@dataclass(frozen=True)
class ILPLinker:
    """Global min-cost edge selection over the whole video, division-aware, solved as an integer program."""

    spacing: Spacing
    max_distance_um: float
    division: bool = True

    def link(self, detections: CellTrackGraph) -> CellTrackGraph:
        """Select the globally cheapest edge set over all frames, then return the detections wired by it."""
        solver = Solver(TrackGraph(self._candidate_graph(detections), frame_attribute=_FRAME))
        solver.add_cost(NodeSelectedCost(constant=_KEEP_NODE_REWARD))
        solver.add_cost(EdgeSelectedCost(attribute=_DISTANCE, weight=1.0, constant=-_LINK_REWARD))
        solver.add_constraint(MaxParents(1))
        solver.add_constraint(MaxChildren(2 if self.division else 1))
        selected = solver.get_selected_subgraph(solver.solve())
        edges = [list(edge) for edge in selected.edges]
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return CellTrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _candidate_graph(self, detections: CellTrackGraph) -> "nx.DiGraph[int]":
        """The detections as a DiGraph, a distance-weighted candidate edge on each within-gate consecutive pair."""
        positions_um = self.spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        graph: nx.DiGraph[int] = nx.DiGraph()
        for row, node_id in enumerate(detections.node_ids.tolist()):
            graph.add_node(node_id, **{_FRAME: int(timepoints[row])})
        for timepoint in np.unique(timepoints)[:-1]:
            self._add_candidate_edges(graph, detections.node_ids.tolist(), positions_um, timepoints, int(timepoint))
        return graph

    def _add_candidate_edges(
        self,
        graph: "nx.DiGraph[int]",
        node_ids: list[int],
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
        timepoint: int,
    ) -> None:
        """Add a distance-weighted edge for each source→target pair one frame apart and within the gate."""
        sources = np.flatnonzero(timepoints == timepoint)
        targets = np.flatnonzero(timepoints == timepoint + 1)
        if not len(sources) or not len(targets):
            return
        source_tree, target_tree = KDTree(positions_um[sources]), KDTree(positions_um[targets])
        neighbours = source_tree.query_ball_tree(target_tree, r=self.max_distance_um)
        for source_index, target_indices in enumerate(neighbours):
            for target_index in target_indices:
                source, target = sources[source_index], targets[target_index]
                distance = float(np.linalg.norm(positions_um[source] - positions_um[target]))
                graph.add_edge(node_ids[source], node_ids[target], **{_DISTANCE: distance})
