"""Recovering a division from the edge transformer's second-choice target, ranked by the split's geometry.

`DivisionRecovery` gates a candidate fork on distance alone; this stage keeps that geometry as a guard but
reads the learned edge head for CANDIDACY: a dividing parent is one whose kept child is the head's top target
and whose runner-up is an unparented cell. That candidacy is the frontier's (orphan-only, gated), and it is
sound. What the head cannot do is RANK. On dense tissue the softmax-over-sources normalisation inflates every
second choice, so >1200 parents carry an orphan target at P >= 0.25 while the one genuinely recoverable
division's second daughter sits at P = 0.124: a probability ranking sorts the true fork to the bottom and the
cap then evicts it (measured: 1 TP / 807 FP uncapped, division Jaccard 0 capped).

The ranking is therefore a STRATEGY, not a hard-coded sort key — the `ForkRanking` family (probability,
geometry, split-symmetry, and the composable surviving-daughter decorator) lives in `fork_ranking`, beside the
`ForkCandidate` vocabulary it reads. This stage owns candidacy, the physical gates, admission and the budget.

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

MEASURED since: at the derived budget the whole stage costs the edge term 0.0006 across the four proxy movies
for 157 forks — under a third of one mislink — so the merge is not what the division arms are buying. That
leaves the effect real but small, and `SurvivingDaughterRanking` is the handle on it either way: the fragments
it smuggles are exactly the daughters that fail to persist.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict

from celltrack.affinity import EdgeAffinity
from celltrack.postproc.fork_ranking import (
    _NO_SUCCESSOR,
    _UNCLAIMED,
    DivergingDaughterRanking,
    ForkCandidate,
    ForkRanking,
    GeometryRanking,
    ProbabilityRanking,
    SplitSymmetryRanking,
    SurvivingDaughterRanking,
    _Gap,
)
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing

_SINGLE_CHILD = 1
_UNPARENTED = 0
_TOP_TARGET = 0
_RUNNER_UP = 1

logger = logging.getLogger(__name__)

# Divisions per node-observation, measured on OUR corpus: 151 divisions over 133,318 annotated nodes across
# 199 videos. Deliberately not a published figure — 2D nuclei report ~0.33% and developmental 3D ~1.1%, three
# to ten times higher, because a division rate is a property of the OBSERVATION WINDOW as much as the biology
# and ours is ~100 frames against a longer cell cycle. Importing one would over-budget every movie.
_DIVISION_RATE = 0.00113
# Two daughters, each within the parent gate of one mother, are at most twice that apart.
_SISTERS_PER_PARENT = 2.0
# The public 0.926 "divsub" kernel's constants (add_safe_divisions_postlink), reproduced verbatim so the
# `kernel_faithful` replica matches its emissions rather than our own gate bracket. Gates are the kernel's
# env-overridden values (parent 8.0 / sister 11.0 / existing-child 10.0 um); the two frac caps are its
# per-frame and global fork ceilings; the sister weight and C3 divergence are its score and post-mitotic
# threshold. These live only on the kernel-faithful path — the configured path keeps its own derived gates.
_KERNEL_PARENT_GATE_UM = 8.0
_KERNEL_SISTER_GATE_UM = 11.0
_KERNEL_EXISTING_CHILD_GATE_UM = 10.0
_KERNEL_C3_DIVERGENCE_UM = 2.25
_KERNEL_SISTER_WEIGHT = 0.15
_KERNEL_PER_FRAME_FRAC = 0.0076
_KERNEL_GLOBAL_EDGE_FRAC = 0.00375
_ONE_FRAME = 1
# The kernel floors both frac caps at one, so a small frame or a short movie can still admit a single fork.
_MIN_CAP = 1


@dataclass(frozen=True)
class _DivisionEdits:
    """What an accepted set of forks does to the edge list: the pairs to add, and the claims they displace.

    An orphan-pool fork only ever adds, so `removed` is empty and this is the old return value. A displacing
    fork cuts the incumbent's link in the same breath, which is what keeps the edge count fixed — the whole
    reason its cost is bounded where an additive fork's is not.
    """

    added: Int[np.ndarray, "d 2"]
    removed: Int[np.ndarray, "r 2"]

    @classmethod
    def of(cls, accepted: list["ForkCandidate"], graph: TrackGraph) -> "_DivisionEdits":
        """Translate accepted forks from graph rows into the node-id pairs the edge list speaks."""
        ids = graph.node_ids
        added = [(int(ids[fork.parent]), int(ids[fork.child])) for fork in accepted]
        removed = [
            (int(ids[fork.displaced]), int(ids[fork.child])) for fork in accepted if fork.displaced != _UNCLAIMED
        ]
        return cls(added=cls._pairs(added), removed=cls._pairs(removed))

    def keeping(self, edges: Int[np.ndarray, "e 2"]) -> Int[np.ndarray, "k 2"]:
        """The edges that survive this edit — every one whose (source, target) pair is not displaced."""
        if not len(self.removed):
            return edges
        displaced = {(int(source), int(target)) for source, target in self.removed.tolist()}
        keep = np.array([(int(source), int(target)) not in displaced for source, target in edges.tolist()])
        return edges[keep]

    @staticmethod
    def _pairs(rows: list[tuple[int, int]]) -> Int[np.ndarray, "p 2"]:
        return np.array(rows, dtype=np.int64) if rows else np.empty((0, 2), dtype=np.int64)


@dataclass(frozen=True)
class _FrameProposals:
    """One gap's candidate forks together with how many single-child parents that frame held.

    The count is the kernel's per-frame cap denominator (`round(single_child * frac)`); it is not the number
    of proposals, since a single-child parent may propose no daughter. Only the kernel-faithful two-cap
    admission reads it — the configured path flattens `candidates` across frames and caps globally.
    """

    candidates: list["ForkCandidate"]
    single_child_parents: int


@dataclass(frozen=True)
class RunnerUpCandidacy:
    """Today's candidacy: the head's runner-up target, proposed only when its top target IS the kept child.

    The literal "a division is a parent whose two top targets are both children" reading — head-gated, so a
    speculative primary link earns no second daughter, and the one orphan proposed is the head's own second
    choice. `min_kept_prob` floors the confidence of the kept link, `min_second_prob` the runner-up's; both
    are admission floors, never the sort key (that is the ranking's).
    """

    min_kept_prob: float = 0.0
    min_second_prob: float = 0.0

    def candidates(
        self, parent: int, kept: int, probabilities: Float[np.ndarray, "t"], gap: _Gap
    ) -> list[ForkCandidate]:
        order = np.argsort(-probabilities)
        if gap.targets[order[_TOP_TARGET]] != kept or probabilities[order[_TOP_TARGET]] < self.min_kept_prob:
            return []
        if len(order) <= _RUNNER_UP:
            return []
        runner_up = order[_RUNNER_UP]
        child = int(gap.targets[runner_up])
        if float(probabilities[runner_up]) < self.min_second_prob or not gap.orphan[runner_up] or child == kept:
            return []
        return [ForkCandidate.at_gap(parent, kept, child, float(probabilities[runner_up]), gap)]


@dataclass(frozen=True)
class GeometricCandidacy:
    """The public frontier's candidacy: EVERY unparented target of a single-child parent, head opinion aside.

    `DivisionAwareLinker` joins any orphan geometrically near a single-child mother, no learned verifier at
    candidacy time; the gates then keep only physical triangles and the ranking + budget pick among them.
    Where `RunnerUpCandidacy` proposes only the daughter the head ranks second, this proposes them all — so a
    true sister the dense-tissue softmax-over-sources buried below a false one is still reachable, which is
    exactly the case the module docstring measures the probability RANKING cannot recover once it is proposed.
    """

    def candidates(
        self, parent: int, kept: int, probabilities: Float[np.ndarray, "t"], gap: _Gap
    ) -> list[ForkCandidate]:
        return [
            ForkCandidate.at_gap(parent, kept, int(target), float(probabilities[index]), gap)
            for index, target in enumerate(gap.targets.tolist())
            if gap.orphan[index] and target != kept
        ]


@dataclass(frozen=True)
class StealCandidacy:
    """The pool the orphan candidacies cannot see: targets another predicted track ALREADY parents.

    Both orphan candidacies gate on `gap.orphan`, so a division whose second daughter the linker already
    claimed is outside the stage at every setting of its gates, ranking and budget — measured at 10 of 25
    missed divisions, 40% of the whole term. Reaching them means DISPLACING the incumbent: cut her edge, add
    the mother's. That is edge-NEUTRAL, which is what makes it worth proposing at all; every additive fork
    arm could only ever add false edges, and the uncapped ones duly traded edge jaccard for recall.

    The pool is large, so the incumbent's own score is the admission floor: propose only where the mother
    outscores the track that holds the daughter by `min_steal_margin`. The physical gates and the C3
    divergence veto then apply unchanged — a displacing fork is still a fork, and still has to look like a
    division.
    """

    min_steal_margin: float = 0.0

    def candidates(
        self, parent: int, kept: int, probabilities: Float[np.ndarray, "t"], gap: _Gap
    ) -> list[ForkCandidate]:
        return [
            ForkCandidate.at_gap(parent, kept, int(target), float(probabilities[index]), gap).taking(
                int(gap.claimed_by[index])
            )
            for index, target in enumerate(gap.targets.tolist())
            if not gap.orphan[index]
            and target != kept
            and int(gap.claimed_by[index]) != parent
            and float(probabilities[index]) - float(gap.claim_probability[index]) > self.min_steal_margin
        ]


# Which of a single-child parent's t+1 targets get PROPOSED as its second daughter, before the gates. Candidacy
# is orthogonal to ranking: it decides the set the ranking then orders. The two choices differ only in whether the
# learned head is consulted — `RunnerUpCandidacy` reads it, `GeometricCandidacy` does not — and both hand the
# survivors to the same gates and the same budget. A closed union, not a `Protocol`: two candidacies exist and a
# third extends this list explicitly (one Protocol subject already lives here — `ForkRanking`).
ForkCandidacy = RunnerUpCandidacy | GeometricCandidacy | StealCandidacy


@dataclass(frozen=True)
class AffinityDivisionRecovery:
    """Add the second daughter the edge head proposes and the chosen ranking prefers, under gates and a budget."""

    spacing: Spacing
    affinity: EdgeAffinity
    ranking: ForkRanking
    parent_gate_um: float
    sister_gate_um: float
    existing_child_gate_um: float
    max_added_forks: int | None
    # Which orphan targets are proposed off a single-child parent — the head's runner-up (default, head-gated)
    # or every unparented target (frontier-geometric). The admission floors live on the candidacy that reads
    # them, so a candidacy that does not consult the head carries no floor it would ignore.
    candidacy: ForkCandidacy = RunnerUpCandidacy()
    # The frontier "divsub" safe-division gates, each OFF by default so the shipped recovery is untouched:
    #  * mutual-nearest — the new daughter and the kept child are each other's nearest among the gap's orphans,
    #    so a fork is refused unless the two proposed sisters are one another's closest cell (no interloper);
    #  * C3 divergence — both daughters continue at t+2 AND their separation grows by at least `c3_divergence_um`,
    #    the post-mitotic signature that turns hundreds of speculative forks into the few real divisions.
    require_mutual_nearest: bool = False
    require_c3_divergence: bool = False
    c3_divergence_um: float = 2.25
    # A global fork ceiling as a fraction of the graph's edges (the frontier's ~0.375%), folded into `budget`
    # beside the division-rate derivation — the smaller wins. None leaves the node-rate budget alone.
    max_fork_edge_fraction: float | None = None
    # The master switch for a byte-faithful replica of the public 0.926 kernel's `add_safe_divisions_postlink`.
    # OFF (default) is the shipped stage, untouched. ON turns on ALL FOUR of the kernel's behaviours at once,
    # each of which our configured stage either omits or does differently, regardless of the flags above:
    #  * C1 — the parent must be MID-track (in-degree >= 1); the kernel refuses a fork off a track START. This
    #    only ever REMOVES forks, but it removes ones the configured stage emits (its fixtures fork off
    #    track-start parents), so it cannot be unconditional — it is gated here.
    #  * C2 — mutual-nearest is ONE-directional: the new daughter must be the kept child's nearest ORPHAN
    #    (pool = orphans only), not our symmetric nearest-of-each-other over orphans-plus-kept-child.
    #  * C3 — both daughters must continue at EXACTLY t+2 AND each be a SINGLE-successor node; the kernel reads
    #    `_succ1` (None unless out-degree is one) and checks the successor frame, where our default C3 takes the
    #    first of possibly-many successors with no frame check.
    #  * two-cap — a per-frame ceiling (`round(single_child_parents * 0.0076)`) AND a global one
    #    (`round(edges * 0.00375)`), admitted per-frame-then-global, replacing our single global node-rate budget.
    kernel_faithful: bool = False

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph with edge-head-proposed, geometry-ranked division edges added."""
        edits = self._division_edges(graph)
        if not len(edits.added):
            return graph
        return TrackGraph(
            node_ids=graph.node_ids,
            coordinates=graph.coordinates,
            edges=np.concatenate([edits.keeping(graph.edges), edits.added]),
        )

    def budget(self, node_count: int, edge_count: int | None = None) -> int:
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
        derived = self.max_added_forks if self.max_added_forks is not None else round(node_count * _DIVISION_RATE)
        if self.max_fork_edge_fraction is None or edge_count is None:
            return derived
        return min(derived, round(edge_count * self.max_fork_edge_fraction))

    def _division_edges(self, graph: TrackGraph) -> "_DivisionEdits":
        """Every accepted fork as node-id pairs, ordered by the ranking's cost and cut at the budget."""
        adjacency = Adjacency.of(graph)
        positions_um = self.spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        frames_ahead = self._frames_ahead(adjacency, timepoints)
        successor = self._successors(adjacency, timepoints)
        frames: list[_FrameProposals] = []
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probability = self.affinity.probabilities(timepoint)
            if probability is None:
                continue
            sources = np.flatnonzero(timepoints == timepoint)
            targets = np.flatnonzero(timepoints == timepoint + 1)
            claimed_by = self._claiming_rows(targets, adjacency)
            gap = _Gap(
                targets=targets,
                orphan=claimed_by == _UNCLAIMED,
                positions_um=positions_um,
                frames_ahead=frames_ahead,
                successor=successor,
                claimed_by=claimed_by,
                claim_probability=self._claim_probabilities(claimed_by, sources, probability),
            )
            candidates = self._gap_proposals(sources, probability, adjacency, gap)
            frames.append(_FrameProposals(candidates, self._single_child_count(sources, adjacency)))
        accepted, cap = self._admit(frames, graph)
        proposed = sum(len(frame.candidates) for frame in frames)
        logger.info("division recovery: %d proposals, %d forks emitted (budget %d)", proposed, len(accepted), cap)
        return _DivisionEdits.of(accepted, graph)

    @staticmethod
    def _claiming_rows(targets: Int[np.ndarray, "t"], adjacency: Adjacency) -> Int[np.ndarray, "t"]:
        """The source row already parenting each target, `_UNCLAIMED` where none does.

        The linker admits one parent per node, so a claimed target has exactly one predecessor and this is a
        lookup rather than a choice. `orphan` is derived from it, which keeps the two facts from drifting.
        """
        claiming = [adjacency.predecessors.get(int(row), ()) for row in targets]
        return np.array([rows[0] if rows else _UNCLAIMED for rows in claiming], dtype=np.int64)

    @staticmethod
    def _claim_probabilities(
        claimed_by: Int[np.ndarray, "t"],
        sources: Int[np.ndarray, "s"],
        probability: Float[np.ndarray, "s t"],
    ) -> Float[np.ndarray, "t"]:
        """How strongly the head scored each target's STANDING link, zero where the target is unclaimed.

        A claim whose parent sits outside this gap's sources — the head scores only the rows it was given —
        prices as zero, which lets any proposal outbid it. That is the permissive direction, and the gates
        and the divergence veto still have to pass.
        """
        index_of = {int(row): index for index, row in enumerate(sources.tolist())}
        scored = np.zeros(len(claimed_by), dtype=np.float64)
        for target_index, parent_row in enumerate(claimed_by.tolist()):
            source_index = index_of.get(parent_row)
            if source_index is not None and source_index < len(probability):
                scored[target_index] = float(probability[source_index, target_index])
        return scored

    def _successors(self, adjacency: Adjacency, timepoints: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
        """The daughters' t+2 rows, read the kernel's way under `kernel_faithful` and ours otherwise."""
        if self.kernel_faithful:
            return self._single_successors(adjacency, timepoints)
        return self._first_successors(adjacency, len(timepoints))

    def _single_child_count(self, sources: Int[np.ndarray, "s"], adjacency: Adjacency) -> int:
        """How many of a frame's sources are single-child parents — the kernel's per-frame cap denominator."""
        if not self.kernel_faithful:
            return 0
        return sum(1 for source in sources.tolist() if adjacency.out_degrees[source] == _SINGLE_CHILD)

    @staticmethod
    def _frames_ahead(adjacency: Adjacency, timepoints: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
        """For every node, the number of frames its track still spans, itself included.

        One backward sweep over the frames: a node's future is one more than its successor's, and a node with
        no successor ends its track at 1. Computed once per graph rather than walked per candidate, so the
        whole ranking stays linear in the graph however many forks are proposed.
        """
        frames = np.ones(len(timepoints), dtype=np.int64)
        for timepoint in np.unique(timepoints)[::-1].tolist():
            for node in np.flatnonzero(timepoints == timepoint).tolist():
                successors = adjacency.successors[node]
                if len(successors):
                    frames[node] = 1 + int(frames[list(successors)].max())
        return frames

    @staticmethod
    def _first_successors(adjacency: Adjacency, node_count: int) -> Int[np.ndarray, "n"]:
        """Each node's first successor row, or `_NO_SUCCESSOR` where its track ends — the daughter's t+2 node.

        The linker admits one child per node, so a daughter's continuation is a single row; a node whose track
        ends at this frame has none, and the C3 divergence gate reads that as "no t+2, cannot have diverged".
        """
        successor = np.full(node_count, _NO_SUCCESSOR, dtype=np.int64)
        for node in range(node_count):
            children = adjacency.successors[node]
            if len(children):
                successor[node] = int(children[_TOP_TARGET])
        return successor

    @staticmethod
    def _single_successors(adjacency: Adjacency, timepoints: Int[np.ndarray, "n"]) -> Int[np.ndarray, "n"]:
        """The kernel's `_succ1` with its frame check: a node's successor row only if it has EXACTLY one and it
        sits one frame ahead — `_NO_SUCCESSOR` otherwise.

        The kernel reads C3 divergence off `_succ1` (None unless out-degree is one) and then rejects unless that
        successor is at t+2 — one frame past the daughter, which sits at t+1. Baking both requirements into the
        successor row means `ForkCandidate.divergence_um` returns None for a multi-successor or gap-spanning
        daughter without any extra lookup, exactly as the geometry-only path already treats a track that ends.
        """
        successor = np.full(len(timepoints), _NO_SUCCESSOR, dtype=np.int64)
        for node in range(len(timepoints)):
            children = adjacency.successors[node]
            if len(children) == _SINGLE_CHILD:
                child = int(children[_TOP_TARGET])
                if int(timepoints[child]) == int(timepoints[node]) + _ONE_FRAME:
                    successor[node] = child
        return successor

    def _admit(self, frames: list[_FrameProposals], graph: TrackGraph) -> tuple[list[ForkCandidate], int]:
        """The accepted forks and the cap that bounded them — kernel two-cap under the flag, our budget else."""
        if self.kernel_faithful:
            return self._admitted_two_cap(frames, len(graph.edges))
        cap = self.budget(len(graph.node_ids), len(graph.edges))
        flat = [candidate for frame in frames for candidate in frame.candidates]
        return self._admitted(flat, cap), cap

    def _admitted_two_cap(self, frames: list[_FrameProposals], edge_count: int) -> tuple[list[ForkCandidate], int]:
        """The kernel's per-frame-then-global admission: sort each frame by score, cap it, cap the total.

        Each frame admits at most `round(single_child_parents * 0.0076)` forks (min one, as the kernel floors
        both caps), and the whole video at most `round(edges * 0.00375)`. A daughter is claimed once across the
        whole video, so two frames cannot name the same orphan. The global cap is a hard ceiling — once it is
        reached no later frame can add, which is what the kernel's break-on-`global_cap` does across frames.
        """
        global_cap = max(_MIN_CAP, round(edge_count * _KERNEL_GLOBAL_EDGE_FRAC))
        claimed: set[int] = set()
        admitted: list[ForkCandidate] = []
        for frame in frames:
            frame_cap = max(_MIN_CAP, round(frame.single_child_parents * _KERNEL_PER_FRAME_FRAC))
            added_this_frame = 0
            for candidate in sorted(frame.candidates, key=self._rank_key):
                if len(admitted) >= global_cap:
                    return admitted, global_cap
                if added_this_frame >= frame_cap:
                    break
                if candidate.child in claimed:
                    continue
                claimed.add(candidate.child)
                admitted.append(candidate)
                added_this_frame += 1
        return admitted, global_cap

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
        probability: Float[np.ndarray, "s t"],
        adjacency: Adjacency,
        gap: _Gap,
    ) -> list[ForkCandidate]:
        """Candidate forks across one gap: the candidacy's proposed daughters off each single-child parent, gated."""
        found: list[ForkCandidate] = []
        for source_index, parent in enumerate(sources.tolist()):
            if source_index >= len(probability):
                # A synthetic node inserted downstream (GapCloser) carries no affinity row — it is appended
                # after every detection, so it cannot be an affinity-ranked division parent. Skip it.
                continue
            if adjacency.out_degrees[parent] != _SINGLE_CHILD:
                continue
            if self.kernel_faithful and adjacency.in_degrees[parent] == _UNPARENTED:
                continue  # C1: the kernel forks only off a MID-track parent, never a track start.
            kept = int(adjacency.successors[parent][_TOP_TARGET])
            found.extend(
                candidate
                for candidate in self.candidacy.candidates(parent, kept, probability[source_index], gap)
                if self._within_gates(candidate) and self._safe_division(candidate, gap)
            )
        return found

    def _safe_division(self, candidate: ForkCandidate, gap: _Gap) -> bool:
        """The frontier divsub gates: mutual-nearest daughters and C3 divergence, each a no-op when its flag is off.

        Both fail CLOSED when required — a candidate that cannot be shown to satisfy the requested signature is
        rejected — so turning a gate on only ever removes forks, never invents them.
        """
        return self._daughters_mutual_nearest(candidate, gap) and self._daughters_diverge(candidate)

    def _daughters_mutual_nearest(self, candidate: ForkCandidate, gap: _Gap) -> bool:
        """The kept child and the proposed daughter are each other's nearest — one-directional under the kernel.

        The kernel's C2 is ONE-directional: the proposed daughter must be the kept child's nearest ORPHAN (pool
        = orphans only), no reciprocal check. Our own gate is SYMMETRIC over orphans-plus-kept-child — each must
        be the other's nearest. Both reject a fork that grafts an orphan belonging to some nearer lineage; the
        kernel path exists so the replica matches its emissions rather than our stricter variant.
        """
        if self.kernel_faithful:
            orphans = gap.targets[gap.orphan]
            return self._nearest_in(candidate.kept, orphans, gap.positions_um) == candidate.child
        if not self.require_mutual_nearest:
            return True
        pool = np.append(gap.targets[gap.orphan], candidate.kept)
        return (
            self._nearest_in(candidate.kept, pool, gap.positions_um) == candidate.child
            and self._nearest_in(candidate.child, pool, gap.positions_um) == candidate.kept
        )

    @staticmethod
    def _nearest_in(row: int, pool: Int[np.ndarray, "p"], positions_um: Float[np.ndarray, "n 3"]) -> int:
        """The pool member (other than `row`) closest to `row`, or `_NO_SUCCESSOR` if the pool holds only `row`."""
        others = pool[pool != row]
        if not len(others):
            return _NO_SUCCESSOR
        distances = np.linalg.norm(positions_um[others] - positions_um[row], axis=1)
        return int(others[int(np.argmin(distances))])

    def _daughters_diverge(self, candidate: ForkCandidate) -> bool:
        """Both daughters continue at t+2 and their separation there exceeds the fork's by `c3_divergence_um`.

        The growth test is identical either way; what differs is the successor rows it reads — the kernel-faithful
        path uses single-successor, one-frame-ahead rows (`_single_successors`) so a multi-successor or
        gap-spanning daughter has no t+2 node, where the configured C3 takes the first of any successors.
        """
        if not (self.kernel_faithful or self.require_c3_divergence):
            return True
        divergence = candidate.divergence_um()
        return divergence is not None and divergence >= self.c3_divergence_um

    def _within_gates(self, candidate: ForkCandidate) -> bool:
        """Whether the proposed triangle is physically a division: the mother, her new daughter, her old one."""
        return (
            candidate.parent_distance_um() <= self.parent_gate_um
            and candidate.sister_distance_um() <= self.sister_gate_um
            and candidate.existing_child_distance_um() <= self.existing_child_gate_um
        )


FORK_RANKINGS: dict[str, Callable[["AffinityDivisionConfig", float], ForkRanking]] = {
    "probability": lambda _, __: ProbabilityRanking(),
    "geometry": lambda config, _: GeometryRanking(sister_weight=config.sister_weight),
    "symmetry": lambda config, gate_um: SplitSymmetryRanking(parent_gate_um=config.parent_gate(gate_um)),
}

FORK_CANDIDACIES: dict[str, Callable[["AffinityDivisionConfig"], ForkCandidacy]] = {
    "runner_up": lambda config: RunnerUpCandidacy(
        min_kept_prob=config.min_kept_prob, min_second_prob=config.min_second_prob
    ),
    "geometric": lambda _: GeometricCandidacy(),
    "steal": lambda config: StealCandidacy(min_steal_margin=config.min_steal_margin),
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

    # LB-ARBITRATED, so it is the default: symmetry-ranked recovery scored 0.899 against the 0.895 base, three
    # times over (fork budgets of ~1072, ~400 and the derived ~157 all read 0.899). `probability` is measured NOT
    # to discriminate — the one recoverable division's parent probability sits below more than 1200 false ones —
    # so leaving it as the default meant every diagnostic that switched divisions on without an override got a
    # stage measured to find nothing, while the submission got one measured to pay.
    ranking: str = "symmetry"
    # Head-gated by default (the runner-up target of a confident single-child parent), which is what every LB
    # arbitration of this stage ran under. `geometric` proposes every unparented target instead — the public
    # frontier's `DivisionAwareLinker` candidacy — so a true sister the dense softmax ranks below a false one is
    # still proposed for the gates and ranking to judge. The floors below only bind the head-gated candidacy.
    candidacy: str = "runner_up"
    # How far the proposing mother must OUTSCORE the track that currently holds a daughter, for the `steal`
    # candidacy alone. Zero admits any strict improvement, which is the permissive end of a knob whose whole
    # job is to shrink a pool of every claimed target in the movie down to the few worth displacing.
    min_steal_margin: float = 0.0
    # OFF. A probability floor is a SECOND bound on speculation, and the budget is already the first — derived
    # from the measured division rate, so it admits exactly as many forks as there are divisions to find. The
    # one measurement we have of the floor is that it EXCLUDED the true case: at 0.5 the one recoverable
    # division is never even PROPOSED, because under this pipeline the true divider's kept-child probability
    # sits below it. A bound that duplicates another and is measured to reject the positive is not a guard.
    min_second_prob: float = 0.0
    # None DERIVES both gates from the linker's own, which is the only argued value in reach. A daughter is
    # one frame's travel from her mother, and the linker gate IS our statement of the furthest a cell travels
    # between consecutive frames (10um, the maximum observed annotated displacement of 9.96um) — every longer
    # step is already refused as a link, so a wider parent gate would admit a daughter the linker itself calls
    # impossible. The sister gate then follows with no second choice: two daughters each within `parent` of the
    # mother can be at most `2 * parent` apart, so it is the triangle inequality, not a threshold.
    # The frontier's 4.7 / 7.2 remain settable, and a submission that widened them by hand to 7.0 / 14.0 is what
    # this replaces: 14 was already exactly twice 7, so only ONE number was ever really being chosen.
    parent_gate_um: float | None = None
    sister_gate_um: float | None = None
    # The frontier disqualifies a parent whose existing link is already long (7.65um) before looking at any
    # candidate; infinity is that gate OFF, which is what this stage has always run.
    # None is the gate OFF, which is what this stage has always run. It used to say that with `math.inf`, and
    # that sentinel could not survive the run's own provenance: pydantic serialises infinity to JSON `null`, so
    # a config carrying it dumped and failed to read back (`float` rejects None) — invisible until
    # `TrackerConfig.shipped()` began carrying a division block and the round-trip test finally saw one.
    # A sentinel the serialisation format cannot express is not a default, it is a latent bug.
    existing_child_gate_um: float | None = None
    # The frontier's sister weight, read from seven byte-identical forks at public 0.915. Only `geometry` reads it.
    sister_weight: float = 0.15
    # An absolute ceiling beside the fraction (`budget` takes the smaller). Larger than any video's edge count,
    # so the default leaves the fraction alone; a bet on divisions sets it to the number of forks it will pay for.
    # None DERIVES the ceiling from the measured division rate (see `budget`); a number overrides it, which is
    # what an arm bracketing that derivation sets.
    max_added_forks: int | None = None
    min_kept_prob: float = 0.0
    # Whether the chosen ranking is penalised by a daughter that does not survive as a track. Off keeps today's
    # behaviour; it is a decorator rather than a fourth ranking because persistence is orthogonal to geometry —
    # any ranking can carry it, and none should have to restate another's terms to do so.
    require_persistence: bool = False
    # Whether the chosen ranking is also penalised by a pair that does not move APART after the fork. The same
    # C3 signature `require_c3_divergence` vetoes on, read as a SCORE — a gate cannot order the set it admits,
    # and the divisions still missed are ones where a fork is emitted at the right mother and the wrong
    # daughter wins the sort. Independent of the gate: either, both or neither.
    prefer_divergence: bool = False
    # The frontier "divsub" safe-division gates, OFF by default so the shipped 0.900 recovery is unchanged. Turn
    # both on for the frontier arm: `--set division.require_mutual_nearest=true division.require_c3_divergence=true`.
    # `c3_divergence_um` is the minimum growth in the two daughters' separation from t+1 to t+2 (2.25um, the public
    # kernel's post-mitotic threshold); `max_fork_edge_fraction` is the global fork ceiling as a fraction of edges
    # (~0.00375), folded into the budget beside the division-rate derivation. None leaves the node-rate budget alone.
    require_mutual_nearest: bool = False
    require_c3_divergence: bool = False
    c3_divergence_um: float = 2.25
    max_fork_edge_fraction: float | None = None
    # OFF is the shipped stage. ON builds a byte-faithful replica of the public 0.926 kernel's safe-division
    # postlink instead: the frontier's geometric candidacy and `parent_dist + 0.15*sister_dist` score, the
    # kernel's gate bracket (parent 8.0 / sister 11.0 / existing-child 10.0 um), and all four kernel behaviours
    # (C1 mid-track parent, C2 one-directional mutual-nearest, C3 single-successor t+2 divergence, the per-frame
    # + global two-cap). It ignores the ranking/candidacy/gate/budget fields above, which describe our own
    # configured stage; both remain reachable, toggled by this one flag, so neither is ever the revert-to end state.
    kernel_faithful: bool = False

    def parent_gate(self, gate_um: float) -> float:
        """The mother-to-daughter gate — the linker's own unless one is set, since a fork IS a one-frame step."""
        return self.parent_gate_um if self.parent_gate_um is not None else gate_um

    def sister_gate(self, gate_um: float) -> float:
        """The daughter-to-daughter gate — twice the parent gate unless set, by the triangle inequality."""
        return (
            self.sister_gate_um if self.sister_gate_um is not None else _SISTERS_PER_PARENT * self.parent_gate(gate_um)
        )

    def existing_child_gate(self) -> float:
        """The mother's-own-link gate — unbounded unless one is set, the gate this stage has always run without."""
        return self.existing_child_gate_um if self.existing_child_gate_um is not None else math.inf

    def _decorated(self, ranking: ForkRanking, min_track_length: int) -> ForkRanking:
        """The configured ranking wrapped in whichever implication terms are switched on, in either order."""
        if self.require_persistence:
            ranking = SurvivingDaughterRanking(ranking, min_track_length)
        return DivergingDaughterRanking(ranking) if self.prefer_divergence else ranking

    def build(
        self, spacing: Spacing, affinity: EdgeAffinity, min_track_length: int, gate_um: float
    ) -> AffinityDivisionRecovery:
        """The recovery stage wired with this ranking, these gates and the video's edge-head probabilities.

        `min_track_length` is the tracker's own — the length rule a daughter's track must reach to survive
        `ShortTrackFilter`. The persistence term normalises by it rather than restating it, so the two stages
        cannot disagree about how long a track has to be.
        """
        if self.kernel_faithful:
            return self._build_kernel_faithful(spacing, affinity)
        if self.ranking not in FORK_RANKINGS:
            raise ValueError(f"unknown division ranking {self.ranking!r}, expected one of {sorted(FORK_RANKINGS)}")
        if self.candidacy not in FORK_CANDIDACIES:
            raise ValueError(
                f"unknown division candidacy {self.candidacy!r}, expected one of {sorted(FORK_CANDIDACIES)}"
            )
        return AffinityDivisionRecovery(
            spacing=spacing,
            affinity=affinity,
            ranking=self._decorated(FORK_RANKINGS[self.ranking](self, gate_um), min_track_length),
            candidacy=FORK_CANDIDACIES[self.candidacy](self),
            parent_gate_um=self.parent_gate(gate_um),
            sister_gate_um=self.sister_gate(gate_um),
            existing_child_gate_um=self.existing_child_gate(),
            max_added_forks=self.max_added_forks,
            require_mutual_nearest=self.require_mutual_nearest,
            require_c3_divergence=self.require_c3_divergence,
            c3_divergence_um=self.c3_divergence_um,
            max_fork_edge_fraction=self.max_fork_edge_fraction,
            kernel_faithful=False,
        )

    @staticmethod
    def _build_kernel_faithful(spacing: Spacing, affinity: EdgeAffinity) -> AffinityDivisionRecovery:
        """A byte-faithful replica of the public 0.926 kernel's `add_safe_divisions_postlink`.

        Every value is the kernel's own, so nothing here reads the configured stage's ranking, gates or budget:
        the geometric candidacy (every near orphan), the `parent_dist + 0.15*sister_dist` score, the kernel gate
        bracket, and `kernel_faithful=True` — which turns on C1, C2 one-directional, C3 single-successor t+2 and
        the two-cap admission inside the stage regardless of the `require_*` flags.
        """
        return AffinityDivisionRecovery(
            spacing=spacing,
            affinity=affinity,
            ranking=GeometryRanking(sister_weight=_KERNEL_SISTER_WEIGHT),
            candidacy=GeometricCandidacy(),
            parent_gate_um=_KERNEL_PARENT_GATE_UM,
            sister_gate_um=_KERNEL_SISTER_GATE_UM,
            existing_child_gate_um=_KERNEL_EXISTING_CHILD_GATE_UM,
            max_added_forks=None,
            c3_divergence_um=_KERNEL_C3_DIVERGENCE_UM,
            kernel_faithful=True,
        )
