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
from scipy.spatial import KDTree

from celltrack.affinity import EdgeAffinity
from core.data.tracks import TrackGraph as CellTrackGraph
from core.geometry import Spacing

_FRAME = "t"
_COST = "cost"
# A node is kept for a nudge below zero, so every detection survives and only the edges are the solver's to
# decide; without a reason to select a node it would drop the isolated ones.
_KEEP_NODE_REWARD = -1.0
# With no learned association to drive selection, an edge needs a flat reward far above the longest in-gate
# distance so linking always beats not linking and the distance term only breaks ties. When an affinity IS
# wired this reward is dropped to zero: the association probability itself drives selection (frontier
# `edge_weight = -prob`), so an edge is worth taking only where its learned score beats its distance.
_LINK_REWARD = 1.0e6


@dataclass(frozen=True)
class ILPLinker:
    """Global min-cost edge selection over the whole video, division-aware, solved as an integer program.

    An optional `affinity` turns the objective from distance-only into the frontier's prob-driven one: the
    edge cost becomes `distance - affinity_bonus * P(s->t)` and the flat link reward is removed, so the
    globally optimal selection is the one the learned associations most support. Global optimisation over the
    whole video with those probabilities beats the per-frame greedy linker (proxy 0.8989 -> 0.9125).
    """

    spacing: Spacing
    max_distance_um: float
    division: bool = True
    affinity: EdgeAffinity | None = None
    affinity_bonus: float = 0.0

    def link(self, detections: CellTrackGraph) -> CellTrackGraph:
        """Select the globally cheapest edge set over all frames, then return the detections wired by it."""
        # motile pulls in the ilpy/SCIP stack, which the SCIP-free submission kernel does not bundle. Importing it
        # lazily keeps `celltrack.linkers.linkers` (and so every kernel) importable without motile — only an actual ILP
        # solve needs it, which the kernel never runs (it selects the assignment/flow linkers).
        from motile import Solver, TrackGraph  # noqa: PLC0415
        from motile.constraints import MaxChildren, MaxParents  # noqa: PLC0415
        from motile.costs import EdgeSelectedCost, NodeSelectedCost  # noqa: PLC0415

        link_reward = 0.0 if self.affinity is not None else _LINK_REWARD
        solver = Solver(TrackGraph(self._candidate_graph(detections), frame_attribute=_FRAME))
        solver.add_cost(NodeSelectedCost(constant=_KEEP_NODE_REWARD))
        solver.add_cost(EdgeSelectedCost(attribute=_COST, weight=1.0, constant=-link_reward))
        solver.add_constraint(MaxParents(1))
        solver.add_constraint(MaxChildren(2 if self.division else 1))
        selected = solver.get_selected_subgraph(solver.solve())
        edges = [list(edge) for edge in selected.edges]
        stacked = np.array(edges, dtype=np.int64) if edges else np.empty((0, 2), dtype=np.int64)
        return CellTrackGraph(node_ids=detections.node_ids, coordinates=detections.coordinates, edges=stacked)

    def _candidate_graph(self, detections: CellTrackGraph) -> "nx.DiGraph[int]":
        """The detections as a DiGraph, a cost-weighted candidate edge on each within-gate consecutive pair."""
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
        """Add a cost-weighted edge (`distance - bonus*P`) for each source→target pair one frame apart, in gate."""
        sources = np.flatnonzero(timepoints == timepoint)
        targets = np.flatnonzero(timepoints == timepoint + 1)
        if not len(sources) or not len(targets):
            return
        probability = self.affinity.probabilities(timepoint) if self.affinity is not None else None
        source_tree, target_tree = KDTree(positions_um[sources]), KDTree(positions_um[targets])
        neighbours = source_tree.query_ball_tree(target_tree, r=self.max_distance_um)
        for source_index, target_indices in enumerate(neighbours):
            for target_index in target_indices:
                source, target = sources[source_index], targets[target_index]
                distance = float(np.linalg.norm(positions_um[source] - positions_um[target]))
                learned = float(probability[source_index, target_index]) if probability is not None else 0.0
                graph.add_edge(node_ids[source], node_ids[target], **{_COST: distance - self.affinity_bonus * learned})
