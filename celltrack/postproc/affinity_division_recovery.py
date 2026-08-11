"""Recovering a division from the edge transformer's second-choice target, ranked by the split's geometry.

`DivisionRecovery` gates a candidate fork on distance alone; this stage keeps that geometry as a guard but
reads the learned edge head for CANDIDACY: a dividing parent is one whose kept child is the head's top target
and whose runner-up is an unparented cell. That candidacy is the frontier's (orphan-only, gated), and it is
sound. What the head cannot do is RANK. On dense tissue the softmax-over-sources normalisation inflates every
second choice, so >1200 parents carry an orphan target at P >= 0.25 while the one genuinely recoverable
division's second daughter sits at P = 0.124: a probability ranking sorts the true fork to the bottom and the
cap then evicts it (measured: 1 TP / 807 FP uncapped, division Jaccard 0 capped).

The ranking is therefore a strategy (`ForkRanking`), not a hard-coded sort key, and probability is only one of
its implementations — kept as the default so no configured pipeline shifts silently:

* `ProbabilityRanking` — the head's second-daughter probability, descending. Today's behaviour.
* `GeometryRanking` — the public frontier's `parent_dist + 0.15 * sister_dist`, ascending, which ships at
  public 0.915 with no probability floor and no learned verifier at all.
* `SplitSymmetryRanking` — the cue the frontier does not use: true daughters separate onto OPPOSITE sides of
  the mother (our division 232 has its two daughters 13.5 um apart), where a false sister in crowded tissue
  sits at a random angle. Ranking a fork by how far the candidate is from the mother, penalised by how badly
  the two daughters fail to balance around her.

`min_second_prob` survives as an admission FLOOR, never as the sort key. A per-video budget bounds the
speculation from both ends — a fraction of the graph's edges and an absolute ceiling, whichever is smaller.

Run before `ShortTrackFilter`, whose division-preserving carve-out can only protect a fork that exists by
the time it sees the graph.

THAT ORDERING HAS A SECOND-ORDER EFFECT WORTH STATING, because it is not what the stage is FOR. A candidate
daughter is an "orphan" by in-degree alone — nothing requires it to have no OUT edges — so it is typically the
FIRST NODE OF A TRACK that begins mid-movie. Adding the fork edge merges that whole track into the parent's
connected component, and `ShortTrackFilter` keeps a component containing a division ENTIRELY. So a short
fragment hanging off the orphan is rescued past the length rule as a side effect of the fork, and its edges
become countable.

That may be a gain (a true short track the blunt length rule would have dropped, which is the same argument
`keep_boundary_tracks` and `ShortTrackRescue` are built on) or a cost (a false fragment smuggled through), and
the two are indistinguishable in the total score. It is UNMEASURED. The quantity that separates them is
`nodes_kept(with) - nodes_kept(without) - forks_emitted`: a fork adds exactly one edge, so any surplus is
rescue rather than division. `reuse` documents the same merge effect deliberately; this one arrives with the
ordering rather than by design.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict

from celltrack.affinity import EdgeAffinity
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_SINGLE_CHILD = 1
_UNPARENTED = 0
_TOP_TARGET = 0
_RUNNER_UP = 1
_DEGENERATE = 0.0

logger = logging.getLogger(__name__)

# Divisions per node-observation, measured on OUR corpus: 151 divisions over 133,318 annotated nodes across
# 199 videos. Deliberately not a published figure — 2D nuclei report ~0.33% and developmental 3D ~1.1%, three
# to ten times higher, because a division rate is a property of the OBSERVATION WINDOW as much as the biology
# and ours is ~100 frames against a longer cell cycle. Importing one would over-budget every movie.
_DIVISION_RATE = 0.00113


@dataclass(frozen=True)
class _Gap:
    """One frame gap's node context: the t+1 target rows, which of them are unparented, and all positions."""

    targets: Int[np.ndarray, "t"]
    orphan: Bool[np.ndarray, "t"]
    positions_um: Float[np.ndarray, "n 3"]


@dataclass(frozen=True)
class ForkCandidate:
    """One proposed fork and every quantity a ranking may read: the head's score and the split's geometry.

    The three nodes are graph ROWS; the micrometre positions are the whole video's, so a candidate carries no
    copied coordinates. Public because a `ForkRanking` is written against this vocabulary.
    """

    parent: int
    kept: int
    child: int
    probability: float
    positions_um: Float[np.ndarray, "n 3"]

    def parent_distance_um(self) -> float:
        """Mother to the proposed new daughter."""
        return self._distance(self.parent, self.child)

    def sister_distance_um(self) -> float:
        """The two daughters' separation — large at a real division, which is why it is a gate, not a score."""
        return self._distance(self.kept, self.child)

    def existing_child_distance_um(self) -> float:
        """Mother to the child she already keeps — a parent whose current link is long is a poor divider."""
        return self._distance(self.parent, self.kept)

    def daughter_angle_cosine(self) -> float:
        """Cosine between the two daughter displacements from the mother: -1 opposite, 0 a random neighbour."""
        kept, child = self._displacements()
        scale = float(np.linalg.norm(kept) * np.linalg.norm(child))
        if scale == _DEGENERATE:
            return _DEGENERATE
        return float(kept @ child) / scale

    def split_imbalance(self) -> float:
        """How far the daughters' centre of mass sits from the mother, as a fraction of how far they travelled.

        `|u + v| / (|u| + |v|)` for the two displacements. It is `daughter_angle_cosine` in half-angle form —
        for equal-length displacements it is exactly `cos(theta / 2)` — generalised to daughters that moved
        unequally, and it is dimensionless: a division that splits twice as far scores the same. 0 is a clean
        split about the mother, 1 is two cells leaving in the same direction.
        """
        kept, child = self._displacements()
        travelled = float(np.linalg.norm(kept) + np.linalg.norm(child))
        if travelled == _DEGENERATE:
            return _DEGENERATE
        return float(np.linalg.norm(kept + child)) / travelled

    def _displacements(self) -> tuple[Float[np.ndarray, "3"], Float[np.ndarray, "3"]]:
        """Both daughters' displacement from the mother."""
        origin = self.positions_um[self.parent]
        return self.positions_um[self.kept] - origin, self.positions_um[self.child] - origin

    def _distance(self, a: int, b: int) -> float:
        """Euclidean distance in micrometres between two node rows."""
        return float(np.linalg.norm(self.positions_um[a] - self.positions_um[b]))


class ForkRanking(Protocol):
    """Orders competing forks under the budget. Lower cost wins, so every ranking speaks one direction."""

    def cost(self, candidate: ForkCandidate) -> float:
        """This fork's rank key — the smaller, the sooner it is admitted."""
        ...


@dataclass(frozen=True)
class ProbabilityRanking:
    """The edge head's second-daughter probability, highest first.

    Measured NOT to discriminate on dense tissue (the true fork's P = 0.124 sits below >1200 false ones), so
    it is the default only because it is what configured pipelines already ran.
    """

    def cost(self, candidate: ForkCandidate) -> float:
        """Negated probability, so the head's most confident second daughter sorts first."""
        return -candidate.probability


@dataclass(frozen=True)
class GeometryRanking:
    """The public frontier's ranking: `parent_dist + w * sister_dist`, ascending — no probability at all."""

    sister_weight: float

    def cost(self, candidate: ForkCandidate) -> float:
        """The frontier's score: tight to the mother first, tie-broken towards a tight sister pair."""
        return candidate.parent_distance_um() + self.sister_weight * candidate.sister_distance_um()


@dataclass(frozen=True)
class SplitSymmetryRanking:
    """Distance from the mother plus the daughters' failure to balance around her, both dimensionless.

    Two terms, no fitted weight. Each is normalised by the constraint that already decides its own
    admissibility — the parent distance by its gate, the imbalance by the distance the daughters travelled —
    so both land in [0, 1] over the admitted set and an equal weighting is the only choice that is not tuned.
    The sister distance appears in neither: it is a physical bound (a gate), and as a SCORE it has the wrong
    sign, because a true mitosis pushes its daughters apart while a false sister sits close by in the crowd.
    """

    parent_gate_um: float

    def cost(self, candidate: ForkCandidate) -> float:
        """Relative reach from the mother plus centroid imbalance — most opposite and closest sorts first."""
        return candidate.parent_distance_um() / self.parent_gate_um + candidate.split_imbalance()


@dataclass(frozen=True)
class AffinityDivisionRecovery:
    """Add the second daughter the edge head proposes and the chosen ranking prefers, under gates and a budget."""

    spacing: Spacing
    affinity: EdgeAffinity
    ranking: ForkRanking
    min_second_prob: float
    parent_gate_um: float
    sister_gate_um: float
    existing_child_gate_um: float
    max_added_forks: int | None
    # The parent's kept child must be its top target at this probability or above, so a fork is only proposed
    # off a link the head itself is confident in — a speculative primary link does not get a second daughter.
    min_kept_prob: float = 0.0

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with edge-head-proposed, geometry-ranked division edges added."""
        recovered = self._division_edges(graph)
        if not len(recovered):
            return graph
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([graph.edges, recovered]),
        )

    def budget(self, node_count: int) -> int:
        """How many forks this video may gain — DERIVED from the measured division rate unless one is set.

        The fraction scales with the movie's SIZE and the old absolute ceiling was a bare constant; neither is
        a statement about biology. What the phenomenon actually says is a RATE: 151 divisions over 133,318
        annotated nodes across 199 videos = 0.113% per node-observation, measured on our own corpus rather
        than imported (published anchors run 0.33-1.1%, three to ten times higher, because their observation
        windows are long relative to a cell cycle and ours is ~100 frames — a literature prior would
        over-budget us by that factor).

        So the expected number of TRUE divisions in a movie is its node count times that rate — ~83 for the
        dense movie's 73,697 detections, ~12 for a sparse one's 10,703 — and the budget is set to it. Matching
        the budget to the phenomenon needs no tuned multiplier: emitting far more forks than there are
        divisions to find can only trade edge jaccard for speculation, and emitting far fewer leaves the term
        unclaimed. `max_added_forks` remains an explicit override so an ARM can bracket that derivation.
        """
        return self.max_added_forks if self.max_added_forks is not None else round(node_count * _DIVISION_RATE)

    def _division_edges(self, graph: TrackGraph) -> Int[np.ndarray, "d 2"]:
        """Every accepted fork as node-id pairs, ordered by the ranking's cost and cut at the budget."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        cap = self.budget(len(graph.node_ids))
        proposals: list[ForkCandidate] = []
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probability = self.affinity.probabilities(timepoint)
            if probability is None:
                continue
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            proposals.extend(self._gap_proposals(sources, targets, probability, adjacency, positions_um))
        accepted = self._admitted(proposals, cap)
        logger.info("division recovery: %d proposals, %d forks emitted (budget %d)", len(proposals), len(accepted), cap)
        if not accepted:
            return np.empty((0, 2), dtype=np.int64)
        pairs = [(int(graph.node_ids[fork.parent]), int(graph.node_ids[fork.child])) for fork in accepted]
        return np.array(pairs, dtype=np.int64)

    def _admitted(self, proposals: list[ForkCandidate], cap: int) -> list[ForkCandidate]:
        """The cheapest forks the budget allows, each proposed daughter claimed by at most one mother.

        Two neighbouring parents can name the same orphan as their runner-up; admitting both would give that
        cell two parents, which is not a division but a broken graph. The ranking decides who claims it.
        """
        claimed: set[int] = set()
        admitted: list[ForkCandidate] = []
        for candidate in sorted(proposals, key=self._rank_key):
            if len(admitted) == cap:
                break
            if candidate.child not in claimed:
                claimed.add(candidate.child)
                admitted.append(candidate)
        return admitted

    def _rank_key(self, candidate: ForkCandidate) -> tuple[float, int, int]:
        """The ranking's cost, with the node rows breaking ties so the accepted set never depends on gap order."""
        return (self.ranking.cost(candidate), candidate.parent, candidate.child)

    def _gap_proposals(
        self,
        sources: Int[np.ndarray, "s"],
        targets: Int[np.ndarray, "t"],
        probability: Float[np.ndarray, "s t"],
        adjacency: Adjacency,
        positions_um: Float[np.ndarray, "n 3"],
    ) -> list[ForkCandidate]:
        """Candidate forks across one gap: a confident single-child parent's orphan runner-up target."""
        gap = _Gap(
            targets=targets,
            orphan=np.array([adjacency.in_degrees[row] == _UNPARENTED for row in targets], dtype=bool),
            positions_um=positions_um,
        )
        found: list[ForkCandidate] = []
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
    ) -> ForkCandidate | None:
        """The parent's second-choice target, if it is an unparented cell clearing the floor and every gate.

        The literal "a division is a parent whose two top targets are both children" reading: the kept child
        is the top target (checked by the caller) and this is the runner-up, so a fork is proposed only when
        the head's first *two* choices agree — not any target that merely clears a probability floor. The
        floor is admission only; which of the admitted forks survives the budget is the ranking's decision.
        """
        if len(order) <= _RUNNER_UP:
            return None
        runner_up = order[_RUNNER_UP]
        child = int(gap.targets[runner_up])
        if float(probabilities[runner_up]) < self.min_second_prob or not gap.orphan[runner_up] or child == kept:
            return None
        candidate = ForkCandidate(
            parent=parent,
            kept=kept,
            child=child,
            probability=float(probabilities[runner_up]),
            positions_um=gap.positions_um,
        )
        return candidate if self._within_gates(candidate) else None

    def _within_gates(self, candidate: ForkCandidate) -> bool:
        """Whether the proposed triangle is physically a division: the mother, her new daughter, her old one."""
        return (
            candidate.parent_distance_um() <= self.parent_gate_um
            and candidate.sister_distance_um() <= self.sister_gate_um
            and candidate.existing_child_distance_um() <= self.existing_child_gate_um
        )


FORK_RANKINGS: dict[str, Callable[["AffinityDivisionConfig"], ForkRanking]] = {
    "probability": lambda _: ProbabilityRanking(),
    "geometry": lambda config: GeometryRanking(sister_weight=config.sister_weight),
    "symmetry": lambda config: SplitSymmetryRanking(parent_gate_um=config.parent_gate_um),
}


class AffinityDivisionConfig(BaseModel):
    """The ranking, the gates and the budget for `AffinityDivisionRecovery`, resolved once a video's affinity is in.

    Defaults are TODAY'S behaviour, deliberately: the probability ranking, our own gate bracket (parent 4.7um,
    sister 7.2um — the frontier's measured 4.66/8.5 within our noise floor), no existing-child gate, and no
    absolute fork ceiling beyond the fraction. An arm that wants the frontier's or the symmetry ranking asks
    for it (`--set division.ranking=symmetry`), so nothing about a configured pipeline moves under it.
    Pydantic (like the other nested stage configs) so a `--set division.x=v` override reaches it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ranking: str = "probability"
    min_second_prob: float = 0.5
    parent_gate_um: float = 4.7
    sister_gate_um: float = 7.2
    # The frontier disqualifies a parent whose existing link is already long (7.65um) before looking at any
    # candidate; infinity is that gate OFF, which is what this stage has always run.
    existing_child_gate_um: float = math.inf
    # The frontier's sister weight, read from seven byte-identical forks at public 0.915. Only `geometry` reads it.
    sister_weight: float = 0.15
    # An absolute ceiling beside the fraction (`budget` takes the smaller). Larger than any video's edge count,
    # so the default leaves the fraction alone; a bet on divisions sets it to the number of forks it will pay for.
    # None DERIVES the ceiling from the measured division rate (see `budget`); a number overrides it, which is
    # what an arm bracketing that derivation sets.
    max_added_forks: int | None = None
    min_kept_prob: float = 0.5

    def build(self, spacing: Spacing, affinity: EdgeAffinity) -> AffinityDivisionRecovery:
        """The recovery stage wired with this ranking, these gates and the video's edge-head probabilities."""
        if self.ranking not in FORK_RANKINGS:
            raise ValueError(f"unknown division ranking {self.ranking!r}, expected one of {sorted(FORK_RANKINGS)}")
        return AffinityDivisionRecovery(
            spacing=spacing,
            affinity=affinity,
            ranking=FORK_RANKINGS[self.ranking](self),
            min_second_prob=self.min_second_prob,
            parent_gate_um=self.parent_gate_um,
            sister_gate_um=self.sister_gate_um,
            existing_child_gate_um=self.existing_child_gate_um,
            max_added_forks=self.max_added_forks,
            min_kept_prob=self.min_kept_prob,
        )
