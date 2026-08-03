"""Bridging a one-frame detection dropout — reusing an existing node, or inserting a confirmed synthetic one.

A cell the detector misses for a single frame breaks its track in two: the track before the gap ends, and
the track after it starts, with the metric charging both the lost incoming and outgoing edge. This pass
reconnects them across a one-frame gap — an END (a node with no successor) at time t is matched to a START
(no predecessor) at t+2 when they are close enough to be the same cell, and the missing middle is supplied
between them. Two edges are added, end->middle and middle->start, recovering both dropped edges at once.

The middle is supplied one of two ways. The precision-safe half REUSES an existing isolated detection at
t+1 sitting near the end-start midpoint — geometry only, no invention. The frontier's real lever is the
other half: where no detection sits near the gap, INSERT a *synthetic* node at the bridged midpoint — a
recovered detection — gated on DeepCenter. A wide gap (span past a threshold, where a missed cell is
plausible rather than certain) is inserted only when the centre prior confirms a cell at that voxel; a short
gap is close-motion continuation and inserted directly. Synthetic insertion runs only when a `confirmer` is
mounted (no confirmer -> reuse-only, the original behaviour). Matching is one-to-one per frame (a Hungarian
assignment within the distance gate); a global cap bounds bridges, a second bounds inserted nodes.
"""

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from celltrack.center_prior import CenterConfirmer
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_UNLINKED = 0
_GAP_SPAN = 2  # end at t, start at t+2: exactly one missing frame between them
_BEYOND_GATE = 1.0e9


class SyntheticGap(BaseModel):
    """When no detection sits near a gap, insert a synthetic midpoint node — a DeepCenter-gated recovery.

    A gap whose end-start span is at least `min_span_um` is wide enough that a real cell plausibly dropped
    out; that insertion is kept only if the mounted centre prior confirms a cell at the midpoint voxel. A
    shorter gap is close-motion continuation and inserted without a second opinion. `max_added_fraction`
    bounds the inserted nodes as a share of the graph's node count — synthetic detections are new mass, so
    they carry their own cap independent of the reuse-bridge cap.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_span_um: float = Field(8.5, gt=0)
    max_added_fraction: float = Field(0.05, ge=0)


@dataclass(frozen=True)
class _GapProposal:
    """One gated end->start match: its span (cost) and the reusable middle row, or None to synthesise one."""

    cost: float
    end: int
    start: int
    middle: int | None


@dataclass(frozen=True)
class _SyntheticInserter:
    """The resolved synthetic-insertion policy — a mounted confirmer and the marginal span it gates past.

    Present only when a `SyntheticGap` policy and a `CenterConfirmer` are both configured, so its fields are
    non-optional: the acceptance pass either has a fully-resolved inserter or None (reuse-only), never a half.
    """

    confirmer: CenterConfirmer
    min_span_um: float


@dataclass(frozen=True)
class _AcceptContext:
    """The per-graph state the greedy acceptance reads: node geometry, the two caps, and the synthetic policy."""

    node_ids: Int[np.ndarray, "n"]
    positions_voxel: Int[np.ndarray, "n 3"]
    timepoints: Int[np.ndarray, "n"]
    edge_cap: int
    node_cap: int
    inserter: _SyntheticInserter | None


@dataclass
class _Closure:
    """Accepted bridges accumulating across the greedy pass — reuse edges plus any synthetic nodes and edges."""

    next_id: int
    used: set[int] = field(default_factory=set)
    edges: list[list[int]] = field(default_factory=list)
    new_ids: list[int] = field(default_factory=list)
    new_coordinates: list[list[int]] = field(default_factory=list)
    bridges: int = 0
    synthetic: int = 0


@dataclass(frozen=True)
class GapCloser:
    """Reconnect a one-frame gap through a reused isolated node, or a DeepCenter-confirmed synthetic midpoint."""

    spacing: Spacing
    gate_um: float
    reuse_um: float
    max_added_fraction: float
    # An optional centre-prior confirmer. It gates two things: which isolated nodes may be reused, and — when
    # `synthetic` is also configured — whether a wide gap with no reusable node earns an inserted node.
    confirmer: CenterConfirmer | None = None
    # Synthetic-insertion policy; None keeps the reuse-only behaviour even with a confirmer present.
    synthetic: SyntheticGap | None = None

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with reuse-bridge edges (and, when configured, confirmed synthetic nodes) added."""
        return self._build(graph, self._closure(graph))

    def _closure(self, graph: TrackGraph) -> _Closure:
        """The accepted bridges and synthetic insertions for this graph, within both caps, tightest first."""
        positions_um = self.spacing.to_micrometres(graph.positions())
        context = self._context(graph)
        adjacency = Adjacency.of(graph)
        ends = np.flatnonzero(adjacency.out_degrees == _UNLINKED)
        starts = np.flatnonzero(adjacency.in_degrees == _UNLINKED)
        isolated = np.flatnonzero((adjacency.in_degrees == _UNLINKED) & (adjacency.out_degrees == _UNLINKED))
        isolated = self._confirmed_midpoints(isolated, context.positions_voxel, context.timepoints)
        proposals = self._proposals(ends, starts, isolated, positions_um, context.timepoints)
        proposals.sort(key=lambda proposal: proposal.cost)
        return self._accept(proposals, context)

    def _context(self, graph: TrackGraph) -> _AcceptContext:
        """This graph's acceptance state: node geometry, the reuse-bridge and synthetic-node caps, and the policy."""
        node_cap = int(self.synthetic.max_added_fraction * len(graph.node_ids)) if self.synthetic is not None else 0
        inserter = None
        if self.synthetic is not None and self.confirmer is not None:
            inserter = _SyntheticInserter(confirmer=self.confirmer, min_span_um=self.synthetic.min_span_um)
        return _AcceptContext(
            node_ids=graph.node_ids,
            positions_voxel=graph.positions(),
            timepoints=graph.timepoints(),
            edge_cap=int(self.max_added_fraction * len(graph.edges)),
            node_cap=node_cap,
            inserter=inserter,
        )

    def _build(self, graph: TrackGraph, closure: _Closure) -> TrackGraph:
        """The graph with the accepted synthetic nodes and bridge edges added, or itself when nothing was taken."""
        if not closure.edges:
            return graph
        node_ids = np.concatenate([graph.node_ids, np.asarray(closure.new_ids, dtype=graph.node_ids.dtype)])
        coordinates = np.concatenate(
            [graph.coordinates, np.asarray(closure.new_coordinates, dtype=graph.coordinates.dtype).reshape(-1, 4)]
        )
        edges = np.concatenate([graph.edges, np.asarray(closure.edges, dtype=graph.edges.dtype)])
        return TrackGraph(node_ids=node_ids, coordinates=coordinates, edges=edges)

    def _proposals(
        self,
        ends: Int[np.ndarray, "e"],
        starts: Int[np.ndarray, "s"],
        isolated: Int[np.ndarray, "i"],
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
    ) -> list[_GapProposal]:
        """Every gated end->start match, each carrying its reusable middle if one is in reach, else None."""
        proposals: list[_GapProposal] = []
        for timepoint in np.unique(timepoints):
            frame_ends = ends[timepoints[ends] == timepoint]
            frame_starts = starts[timepoints[starts] == timepoint + _GAP_SPAN]
            middles = isolated[timepoints[isolated] == timepoint + 1]
            if not len(frame_ends) or not len(frame_starts):
                continue
            proposals.extend(self._match_frame(frame_ends, frame_starts, middles, positions_um))
        return proposals

    def _match_frame(
        self,
        frame_ends: Int[np.ndarray, "e"],
        frame_starts: Int[np.ndarray, "s"],
        middles: Int[np.ndarray, "m"],
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[_GapProposal]:
        """One frame's one-to-one end->start assignment, each match carrying its reusable middle or None."""
        cost = cdist(positions_um[frame_ends], positions_um[frame_starts])
        rows, columns = linear_sum_assignment(np.where(cost <= self.gate_um, cost, _BEYOND_GATE))
        matched: list[_GapProposal] = []
        for row, column in zip(rows, columns, strict=True):
            if cost[row, column] > self.gate_um:
                continue
            end, start = int(frame_ends[row]), int(frame_starts[column])
            middle = self._reusable_midpoint(end, start, middles, positions_um)
            matched.append(_GapProposal(cost=float(cost[row, column]), end=end, start=start, middle=middle))
        return matched

    def _confirmed_midpoints(
        self, isolated: Int[np.ndarray, "i"], positions_voxel: Int[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> Int[np.ndarray, "k"]:
        """The isolated rows the centre prior confirms as real centres — all of them when no veto is configured."""
        if self.confirmer is None:
            return isolated
        kept = [row for row in isolated.tolist() if self.confirmer.confirm(int(timepoints[row]), positions_voxel[row])]
        return np.asarray(kept, dtype=isolated.dtype)

    def _reusable_midpoint(
        self, end: int, start: int, middles: Int[np.ndarray, "m"], positions_um: Float[np.ndarray, "n 3"]
    ) -> int | None:
        """The isolated node nearest the end-start midpoint, if one sits within the reuse radius, else None."""
        if not len(middles):
            return None
        midpoint = (positions_um[end] + positions_um[start]) / 2.0
        distances = np.linalg.norm(positions_um[middles] - midpoint, axis=1)
        nearest = int(np.argmin(distances))
        return int(middles[nearest]) if distances[nearest] <= self.reuse_um else None

    def _accept(self, proposals: list[_GapProposal], context: _AcceptContext) -> _Closure:
        """Greedily take the tightest bridges — reuse first, synthetic where allowed — under each cap."""
        node_ids = context.node_ids
        closure = _Closure(next_id=int(node_ids.max()) + 1 if len(node_ids) else 0)
        for proposal in proposals:
            self._consider(proposal, closure, context)
        return closure

    def _consider(self, proposal: _GapProposal, closure: _Closure, context: _AcceptContext) -> None:
        """Route one proposal to the reuse or synthetic path, skipping it if either endpoint is already used."""
        if closure.used & {proposal.end, proposal.start}:
            return
        if proposal.middle is not None:
            self._consider_reuse(proposal, proposal.middle, closure, context)
            return
        self._consider_synthetic(proposal, closure, context)

    @staticmethod
    def _consider_reuse(proposal: _GapProposal, middle: int, closure: _Closure, context: _AcceptContext) -> None:
        """Take a reuse bridge through an isolated node, unless its middle is spoken for or the cap is spent."""
        if middle in closure.used or closure.bridges >= context.edge_cap:
            return
        ids = context.node_ids
        closure.used |= {proposal.end, middle, proposal.start}
        closure.edges += [[int(ids[proposal.end]), int(ids[middle])], [int(ids[middle]), int(ids[proposal.start])]]
        closure.bridges += 1

    def _consider_synthetic(self, proposal: _GapProposal, closure: _Closure, context: _AcceptContext) -> None:
        """Insert a synthetic midpoint when the policy is mounted, the node cap allows it, and DeepCenter confirms."""
        inserter = context.inserter
        if inserter is None or closure.synthetic >= context.node_cap:
            return
        coordinate = self._synthetic_coordinate(proposal, context, inserter)
        if coordinate is None:
            return
        middle_id = closure.next_id
        closure.next_id += 1
        closure.new_ids.append(middle_id)
        closure.new_coordinates.append(coordinate)
        ids = context.node_ids
        closure.edges += [[int(ids[proposal.end]), middle_id], [middle_id, int(ids[proposal.start])]]
        closure.used |= {proposal.end, proposal.start}
        closure.synthetic += 1

    @staticmethod
    def _synthetic_coordinate(
        proposal: _GapProposal, context: _AcceptContext, inserter: _SyntheticInserter
    ) -> list[int] | None:
        """The `(t, z, y, x)` of a synthetic midpoint, or None when a wide gap is not centre-confirmed."""
        pair = context.positions_voxel[[proposal.end, proposal.start]]
        midpoint = np.rint(pair.mean(axis=0)).astype(np.int64)
        timepoint = int(context.timepoints[proposal.end]) + 1
        if proposal.cost >= inserter.min_span_um and not inserter.confirmer.confirm(timepoint, midpoint):
            return None
        return [timepoint, int(midpoint[0]), int(midpoint[1]), int(midpoint[2])]
