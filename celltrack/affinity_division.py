"""Recovering a division from the edge transformer's second-choice target, not geometry alone.

`DivisionRecovery` gates a candidate fork on distance: an unparented cell joins a nearby single-child
parent. On real detections that misses the divisions that matter — at a true mitosis the second daughter
often sits just past the tight parent gate the frontier needs to keep its precision, so the gate that
rejects the false forks rejects the real one too. The learned edge head breaks that deadlock: a dividing
parent is one whose *two* highest-probability targets are both real children, so the association score
separates the true second daughter from the crowd the distance gate cannot.

This stage reads a source→target probability matrix per frame gap (`EdgeAffinity`, the same head the linker
blends into its cost) and proposes a fork only when a single-child parent's kept child is its top target
(the linker and the head agree the primary link is real) and its *second* target is an orphan the head
still scores above a floor. Distance stays a guard, not the selector: a parent gate and a sister gate reject
a high-probability pairing that is physically impossible, but within them the head chooses. A per-gap and a
global cap bound the budget, tightest-probability pairs first, so a floor set for recall cannot flood the
dense frames with speculative forks.

Run before `ShortTrackFilter`, whose division-preserving carve-out can only protect a fork that exists by
the time it sees the graph.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Int

from celltrack.motion_linking import EdgeAffinity
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_SINGLE_CHILD = 1
_UNPARENTED = 0
_TOP_TARGET = 0
_RUNNER_UP = 1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Gap:
    """One frame gap's node context: the t+1 target rows, which of them are unparented, and all positions."""

    targets: Int[np.ndarray, "t"]
    orphan: Bool[np.ndarray, "t"]
    positions_um: Float[np.ndarray, "n 3"]


@dataclass(frozen=True)
class AffinityDivisionRecovery:
    """Add the second daughter the edge head scores highest among a confident parent's orphans."""

    spacing: Spacing
    affinity: EdgeAffinity
    min_second_prob: float
    parent_gate_um: float
    sister_gate_um: float
    max_added_fraction: float
    # The parent's kept child must be its top target at this probability or above, so a fork is only proposed
    # off a link the head itself is confident in — a speculative primary link does not get a second daughter.
    min_kept_prob: float = 0.0

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with edge-head-confirmed division edges added."""
        recovered = self._division_edges(graph)
        if not len(recovered):
            return graph
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([graph.edges, recovered]),
        )

    def _division_edges(self, graph: TrackGraph) -> Int[np.ndarray, "d 2"]:
        """Every accepted fork as node-id pairs, ranked by second-daughter probability, globally capped."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        global_cap = int(self.max_added_fraction * len(graph.edges))
        proposals: list[tuple[float, int, int]] = []
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probability = self.affinity.probabilities(timepoint)
            if probability is None:
                continue
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            proposals.extend(self._gap_proposals(sources, targets, probability, adjacency, positions_um))
        accepted = [(parent, child) for _, parent, child in sorted(proposals, reverse=True)[:global_cap]]
        if not accepted:
            return np.empty((0, 2), dtype=np.int64)
        return np.array([(int(graph.node_ids[p]), int(graph.node_ids[c])) for p, c in accepted], dtype=np.int64)

    def _gap_proposals(
        self,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        probability: Float[np.ndarray, "s t"],
        adjacency: Adjacency,
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[tuple[float, int, int]]:
        """Candidate forks across one gap: a confident single-child parent's best orphan second daughter."""
        gap = _Gap(
            targets=targets,
            orphan=np.array([adjacency.in_degrees[row] == _UNPARENTED for row in targets], dtype=bool),
            positions_um=positions_um,
        )
        found: list[tuple[float, int, int]] = []
        for source_index, parent in enumerate(sources.tolist()):
            if adjacency.out_degrees[parent] != _SINGLE_CHILD:
                continue
            kept = adjacency.successors[parent][_TOP_TARGET]
            probabilities = probability[source_index]
            order = np.argsort(-probabilities)
            if targets[order[_TOP_TARGET]] != kept or probabilities[order[_TOP_TARGET]] < self.min_kept_prob:
                continue
            candidate = self._runner_up(order, probabilities, parent, kept, gap)
            if candidate is not None:
                found.append(candidate)
        return found

    def _runner_up(
        self,
        order: Int[np.ndarray, "t"],
        probabilities: Float[np.ndarray, "t"],
        parent: int,
        kept: int,
        gap: _Gap,
    ) -> tuple[float, int, int] | None:
        """The parent's second-choice target, if it is an unparented cell clearing the floor and both gates.

        The literal "a division is a parent whose two top targets are both children" reading: the kept child
        is the top target (checked by the caller) and this is the runner-up, so a fork is proposed only when
        the head's first *two* choices agree — not any target that merely clears a probability floor.
        """
        if len(order) <= _RUNNER_UP:
            return None
        runner_up = order[_RUNNER_UP]
        probability = float(probabilities[runner_up])
        child = int(gap.targets[runner_up])
        if probability < self.min_second_prob or not gap.orphan[runner_up] or child == kept:
            return None
        if self._distance(gap.positions_um, parent, child) > self.parent_gate_um:
            return None
        if self._distance(gap.positions_um, kept, child) > self.sister_gate_um:
            return None
        return (probability, parent, child)

    @staticmethod
    def _distance(positions_um: Float[np.ndarray, "n 3"], a: int, b: int) -> float:
        """Euclidean distance in micrometres between two node rows."""
        return float(np.linalg.norm(positions_um[a] - positions_um[b]))
