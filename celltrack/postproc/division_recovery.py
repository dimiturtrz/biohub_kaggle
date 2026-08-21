"""Recovering the cell divisions a one-to-one assignment cannot represent — a post-link pass.

`MotionHungarianLinker` solves a bipartite matching, so every cell receives at most one successor. At a
division that is not a tuning miss but a structural one: one daughter keeps the edge, the other is left
unparented, and the fork the metric scores is never expressed at all. The competition score adds
`0.1 * division_jaccard` on top of the edge Jaccard, so a pipeline that cannot fork forfeits that entire
term however good its detector is — and the dropped daughter edge costs the edge Jaccard as well.

This pass runs after linking and proposes the missing second daughter: an unparented cell is joined to a
nearby cell in the previous frame that already has exactly one child. It reproduces the public frontier's
`add_safe_divisions_postlink`, whose precision comes from three geometry gates, not one — on real detections
a loose single gate invents far more false forks than true ones, and a false fork is scored twice over
(against the division term it reached for and against the much larger edge term):

- **parent gate** — how far a daughter may sit from its mother (`4.7 um` in the frontier).
- **sister gate** — how far the two daughters may sit from each other (`7.2 um`); the discriminating one, since
  a real mitosis leaves its products adjacent while an ordinary cell that merely lost its link has no sister
  nearby.
- **existing-child gate** — how far the mother's *already-linked* daughter sits from her (`7.8 um`). This
  rejects a mother whose surviving link is itself a long mis-link, the dominant false case in dense frames:
  if the linker already mis-joined this mother across the tissue, her "division" is spurious.

A per-frame cap and a global cap bound how many forks the pass may invent. Candidates are ranked by
`parent_distance + 0.15 * sister_distance`, so the tightest, most division-like pairs win the limited budget.

Run this before `ShortTrackFilter`, whose division-preserving carve-out can only protect forks that exist
by the time it sees the graph. The frontier further confirms each candidate with a DeepCenter centre-prior
veto; that second-opinion model is a separate stage this geometry-only pass does not include.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, Field
from scipy.spatial import KDTree

from celltrack.detectors.center_prior import CenterConfirmer
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_SINGLE_CHILD = 1
_UNPARENTED = 0
_SISTER_WEIGHT = 0.15

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DivisionRecovery:
    """Join unparented cells to a nearby single-child parent, gated on parent, sister and existing-child distance."""

    spacing: Spacing
    parent_gate_um: float
    sister_gate_um: float
    max_added_fraction: float
    # The mother's already-linked daughter must sit within this of her, or the mother is a mis-link, not a
    # divider. Defaults to no gate so a bare (parent, sister, fraction) construction keeps the old behaviour.
    existing_child_gate_um: float = float("inf")
    # Per-frame cap as a fraction of that frame's single-child parents, applied before the global cap.
    frame_fraction_cap: float = 1.0
    # An optional centre-prior veto: a recovered daughter is kept only if DeepCenter confirms a centre at its
    # voxel. Defaults to no veto, so the geometry-only construction keeps its behaviour and existing tests hold.
    confirmer: CenterConfirmer | None = None

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with recovered division edges added."""
        recovered = self._division_edges(graph)
        logger.info("geometric division recovery: %d single-child forks recovered", len(recovered))
        if not len(recovered):
            return graph
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([graph.edges, recovered]),
        )

    def _division_edges(self, graph: TrackGraph) -> Int[np.ndarray, "d 2"]:
        """Every accepted fork as node-id pairs, capped per frame then globally, tightest pairs first."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        confirmed = self._confirmed_rows(graph.positions(), timepoints)
        global_cap = int(self.max_added_fraction * len(graph.edges))
        accepted: list[tuple[int, int]] = []
        for timepoint in np.unique(timepoints)[:-1]:
            if len(accepted) >= global_cap:
                break
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            proposals = self._proposals_between(sources, targets, positions_um, adjacency, confirmed)
            frame_cap = max(1, round(self._frame_source_count(sources, adjacency) * self.frame_fraction_cap))
            for _, parent, child in sorted(proposals)[: min(frame_cap, global_cap - len(accepted))]:
                accepted.append((int(graph.node_ids[parent]), int(graph.node_ids[child])))
        if not accepted:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(accepted, dtype=np.int64)

    @staticmethod
    def _frame_source_count(sources: Int[np.ndarray, "s"], adjacency: Adjacency) -> int:
        """How many of this frame's cells are single-child parents — the base of the per-frame cap."""
        return sum(1 for row in sources.tolist() if adjacency.out_degrees[row] == _SINGLE_CHILD)

    def _proposals_between(
        self,
        source_rows: Int[np.ndarray, "s"],
        target_rows: Int[np.ndarray, "t"],
        positions_um: Float[np.ndarray, "n 3"],
        adjacency: Adjacency,
        confirmed: set[int] | None,
    ) -> list[tuple[float, int, int]]:
        """Candidate forks across one frame gap, as `(score, parent row, daughter row)`, all gates applied."""
        parents = [
            row
            for row in source_rows.tolist()
            if adjacency.out_degrees[row] == _SINGLE_CHILD
            and self._distance(positions_um, row, adjacency.successors[row][0]) <= self.existing_child_gate_um
        ]
        orphans = [
            row
            for row in target_rows.tolist()
            if adjacency.in_degrees[row] == _UNPARENTED and (confirmed is None or row in confirmed)
        ]
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
        """Pick each orphan's lowest-score sister-gated parent, letting no parent gain more than one daughter."""
        claimed: set[int] = set()
        found: list[tuple[float, int, int]] = []
        for orphan, reachable in zip(orphans, within_reach, strict=True):
            eligible = [
                (self._score(positions_um, parents[index], sisters[index], orphan), index)
                for index in reachable
                if parents[index] not in claimed
                and self._distance(positions_um, sisters[index], orphan) <= self.sister_gate_um
            ]
            if not eligible:
                continue
            _, index = min(eligible)
            claimed.add(parents[index])
            found.append((self._score(positions_um, parents[index], sisters[index], orphan), parents[index], orphan))
        return found

    def _score(self, positions_um: Float[np.ndarray, "n 3"], parent: int, sister: int, orphan: int) -> float:
        """Rank key `parent_distance + 0.15 * sister_distance` — the frontier's tightest-pair-first ordering."""
        return self._distance(positions_um, parent, orphan) + _SISTER_WEIGHT * self._distance(
            positions_um, sister, orphan
        )

    def _confirmed_rows(
        self, positions_voxel: Int[np.ndarray, "n 3"], timepoints: Int[np.ndarray, "n"]
    ) -> set[int] | None:
        """The rows the centre prior confirms as real centres, or None when no veto is configured (all pass)."""
        if self.confirmer is None:
            return None
        return {
            row for row in range(len(timepoints)) if self.confirmer.confirm(int(timepoints[row]), positions_voxel[row])
        }

    @staticmethod
    def _distance(positions_um: Float[np.ndarray, "n 3"], a: int, b: int) -> float:
        """Euclidean distance in micrometres between two node rows."""
        return float(np.linalg.norm(positions_um[a] - positions_um[b]))


class DivisionRecoveryConfig(BaseModel):
    """The geometry-only single-child division repair as a nested config, composed like a linker or bridge.

    A DIFFERENT mechanism from `AffinityDivisionConfig`, which can only place a fork when BOTH daughters are
    detected (the failing case). This joins an unparented cell to an already-linked single-child parent, so it
    recovers the fork the metric scores WITHOUT the second daughter's own detection. Gates default to the
    frontier's `add_safe_divisions_postlink` constants (parent 4.7, sister 7.2, existing-child 7.8 um), whose
    three-gate precision is what keeps a false fork — scored twice, against the division term and the edge term
    — from costing more than the true forks it recovers. Present (non-None) turns the stage on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parent_gate_um: float = Field(4.7, gt=0)
    sister_gate_um: float = Field(7.2, gt=0)
    existing_child_gate_um: float = Field(7.8, gt=0)
    max_added_fraction: float = Field(0.05, ge=0)
    frame_fraction_cap: float = Field(1.0, ge=0)

    def build(self, spacing: Spacing) -> DivisionRecovery:
        """The division-recovery stage at this video's spacing (geometry only, no centre-prior veto)."""
        return DivisionRecovery(
            spacing=spacing,
            parent_gate_um=self.parent_gate_um,
            sister_gate_um=self.sister_gate_um,
            max_added_fraction=self.max_added_fraction,
            existing_child_gate_um=self.existing_child_gate_um,
            frame_fraction_cap=self.frame_fraction_cap,
        )
