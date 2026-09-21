"""Where the real divisions rank among the forks the linker ALREADY emitted — the price of a fork-KEEP gate.

`celltrack.eval.oracle_fork_arms` measured the ceiling: culling every fork the annotation does not divide at,
while keeping the branch the annotation agrees with, moves the official ruler from 0.8463 to **0.9726**. Blind
culling — every fork cut to its likeliest branch, no annotation consulted — moves it to 0.8497. The entire
0.126 therefore sits in DISCRIMINATION, and nothing in that gap is bought by the cut itself.

Two facts size the discriminator's job, and they pull in opposite directions. Only **254** of the champion's
emitted forks are CHARGED (the rest fork in unannotated tissue, where the metric counts nothing), so the cost
of a permissive gate is small. But the real divisions number **11**, so a gate that keeps the top-N forks by
any score is paying for a 0.15% base rate. The division term is a RATIO — `k / (11 + m)` for `k` divisions kept
and `m` charged false forks admitted — so it cannot be bought by deleting false forks alone: the blind cull
removes all 254 and still scores 0, because it keeps none of the 11.

`celltrack.postproc.fork_ranking` already holds five ranking strategies, built and measured for the opposite
task — PROPOSING forks to recover missed divisions, where the ceiling turned out to be +0.001. The cues are the
same; the direction is not. This module re-prices them on the task the ceiling is large for.

It does not survive the re-pricing. On the champion's 7185 emitted forks the best combination of the existing
cues reaches division Jaccard 0.10 against the oracle's 1.00, worth **+0.008** of score — under the noise
floor. The second grade here is the sharper one: of the 245 forks the annotation resolves to exactly one
branch, keeping the likeliest branch is right 58.8% of the time, and the best annotation-free cue (nearest)
reaches 60.0%. Chance on a two-branch fork is 50%. The `survival` cue that E39's blip argument predicts should
win is WORSE than affinity (55.5%), so the surplus branch at a fork the annotation resolves is not a blip.

Both grades land in the same place, and it is the place the confusor keystone already describes: the wrong
branch of a fork is PAIRWISE INDISTINGUISHABLE from the right one. The oracle's 0.126 is annotation-bound, not
post-processing-bound, and this module is kept as the instrument that says so rather than as a step toward a
gate — `BranchAccuracy.of` prices any future model-side branch scorer against 0.5878 in one CPU run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from celltrack.eval.official_ruler import DIVISION_WEIGHT
from celltrack.postproc.fork_ranking import (
    DivergingDaughterRanking,
    ForkCandidate,
    ForkRanking,
    GeometryRanking,
    ProbabilityRanking,
    SplitSymmetryRanking,
    SurvivingDaughterRanking,
)

MIN_SUCCESSORS = 2
FRAMES_OVER_PROBABILITY = 1000.0
CHAMPION_DIVISION_JACCARD = 0.0228
SISTER_WEIGHT = 0.5
PARENT_GATE_UM = 10.0
SURVIVING_TRACK_FRAMES = 6
SPATIAL_DIMENSIONS = 3


@dataclass(frozen=True)
class LabelledFork:
    """One emitted fork, with the two labels the metric cares about: is it real, and does it cost anything."""

    candidate: ForkCandidate
    is_true: bool
    is_charged: bool


@dataclass(frozen=True)
class RankReport:
    """One ranking's verdict: where the true forks landed, and how much charged false mass outranks them."""

    name: str
    total: int
    true_ranks: tuple[int, ...]
    charged_above: tuple[int, ...]

    @classmethod
    def of(cls, name: str, ranking: ForkRanking, forks: Sequence[LabelledFork]) -> RankReport:
        """Order the emitted forks by one ranking and record where the true ones landed.

        `charged_above` is per true fork: how many CHARGED false forks a gate must admit to reach it. That is
        the quantity the division ratio is written in, not the raw rank — a fork in unannotated tissue is free.
        """
        ordered = sorted(range(len(forks)), key=lambda index: ranking.cost(forks[index].candidate))
        true_ranks: list[int] = []
        charged_above: list[int] = []
        charged_seen = 0
        for rank, index in enumerate(ordered):
            fork = forks[index]
            if fork.is_true:
                true_ranks.append(rank)
                charged_above.append(charged_seen)
            elif fork.is_charged:
                charged_seen += 1
        return cls(name=name, total=len(forks), true_ranks=tuple(true_ranks), charged_above=tuple(charged_above))

    @staticmethod
    def division_jaccard(kept: int, admitted_false: int, total_true: int) -> float:
        """`k / (T + m)` — what the official division term becomes under a gate's own arithmetic."""
        denominator = total_true + admitted_false
        return kept / denominator if denominator > 0 else float("nan")

    def kept_at(self, budget: int) -> int:
        """How many true forks survive a gate that keeps the best `budget` forks of the whole emitted set."""
        return sum(1 for rank in self.true_ranks if rank < budget)

    def worst_rank(self) -> int:
        """The rank of the last true fork — the budget a gate needs to keep every division."""
        return max(self.true_ranks, default=-1)

    def best_budget(self, total_true: int) -> tuple[int, float]:
        """The keep-budget maximising the division Jaccard, and the value it reaches.

        Only a budget landing just past a true fork can improve anything, so the search runs over those.
        """
        best = (0, 0.0)
        for index, (rank, charged) in enumerate(zip(self.true_ranks, self.charged_above, strict=True), start=1):
            value = self.division_jaccard(index, charged, total_true)
            if value > best[1]:
                best = (rank + 1, value)
        return best


@dataclass(frozen=True)
class Branch:
    """One successor of a predicted fork, with the cues a keep-one rule can read and the label it is graded on."""

    probability: float
    frames_ahead: int
    distance_um: float
    is_gt_edge: bool


@dataclass(frozen=True)
class Branching:
    """One fork's successors. Only a fork the annotation resolves can grade a keep-one rule."""

    branches: tuple[Branch, ...]

    def truth_index(self) -> int | None:
        """The single branch the annotation keeps, or `None` when it keeps none or several.

        A fork with no matched branch cannot grade anything, and one with two matched branches is a real
        division, where cutting back to one successor is wrong whichever branch survives.
        """
        matched = [index for index, branch in enumerate(self.branches) if branch.is_gt_edge]
        return matched[0] if len(matched) == 1 else None


@dataclass(frozen=True)
class BranchAccuracy:
    """What one keep-one rule costs in annotated edges against the oracle that always keeps the right branch."""

    name: str
    decided: int
    correct: int

    @classmethod
    def of(cls, name: str, score: Callable[[Branch], float], branchings: Sequence[Branching]) -> BranchAccuracy:
        """Grade a keep-one rule: on every fork the annotation resolves, is the rule's best branch that one."""
        decided = 0
        correct = 0
        for branching in branchings:
            truth = branching.truth_index()
            if truth is None:
                continue
            decided += 1
            best = max(range(len(branching.branches)), key=lambda index: score(branching.branches[index]))
            correct += int(best == truth)
        return cls(name=name, decided=decided, correct=correct)

    def lost_edges(self) -> int:
        """Annotated edges the rule deletes — the whole distance between it and the oracle cull."""
        return self.decided - self.correct

    def accuracy(self) -> float:
        return self.correct / self.decided if self.decided > 0 else float("nan")


@dataclass(frozen=True)
class MovieGraph:
    """One movie's linked graph as plain rows, matched against its annotation.

    Everything the two grades need and nothing tracksdata-shaped, so the fork building is testable from dicts.
    Node ids are the graph's own; `index_of` maps them to the dense rows `positions_um` is indexed by, which is
    the indexing `ForkCandidate` speaks.
    """

    index_of: Mapping[int, int]
    times: Mapping[int, int]
    positions_um: np.ndarray
    successors: Mapping[int, Sequence[int]]
    probability: Mapping[tuple[int, int], float]
    gt_edges: frozenset[tuple[int, int]]
    annotated_of: Mapping[int, int]
    gt_children: Mapping[int, frozenset[int]]
    gt_out_degree: Mapping[int, int]

    def frames_ahead(self) -> Mapping[int, int]:
        """Frames each node's track survives from itself onward: its longest chain of successors, plus one.

        Walking nodes latest-first means every successor is already resolved when its parent is reached.
        """
        ahead: dict[int, int] = {}
        for node in sorted(self.index_of, key=lambda candidate: -self.times[candidate]):
            children = self.successors.get(node, ())
            ahead[node] = 1 + max((ahead[child] for child in children), default=0)
        return ahead

    def forks(self) -> tuple[list[LabelledFork], list[Branching]]:
        """Every emitted fork twice over: as division candidates to rank, and as a branching to choose within.

        A fork is CHARGED when its parent matched an annotated node — the metric's `out_valid` half — so a fork
        in unannotated tissue costs nothing and must not be counted against a gate that admits it.
        """
        ahead = self.frames_ahead()
        labelled: list[LabelledFork] = []
        branchings: list[Branching] = []
        for node in self.index_of:
            ordered = self._branches_of(node)
            if len(ordered) < MIN_SUCCESSORS:
                continue
            annotated = self.annotated_of.get(node)
            daughters = self.gt_children.get(annotated, frozenset()) if annotated is not None else frozenset()
            divides = annotated is not None and self.gt_out_degree.get(annotated, 0) >= MIN_SUCCESSORS
            branchings.append(self._branching(node, ordered, ahead))
            for child in ordered[1:]:
                child_annotated = self.annotated_of.get(child)
                is_true = divides and child_annotated is not None and child_annotated in daughters
                labelled.append(
                    LabelledFork(
                        candidate=self._candidate(node, ordered, child, ahead),
                        is_true=is_true,
                        is_charged=annotated is not None,
                    )
                )
        return labelled, branchings

    def _branches_of(self, node: int) -> list[int]:
        """The node's successors, likeliest first — the order a keep-one rule cuts back to."""
        children = list(self.successors.get(node, ()))
        return sorted(children, key=lambda child: self.probability.get((node, child), 0.0), reverse=True)

    def _distance_um(self, source: int, target: int) -> float:
        step = self.positions_um[self.index_of[target]] - self.positions_um[self.index_of[source]]
        return float(np.linalg.norm(step))

    def _branching(self, node: int, ordered: Sequence[int], ahead: Mapping[int, int]) -> Branching:
        return Branching(
            branches=tuple(
                Branch(
                    probability=self.probability.get((node, child), 0.0),
                    frames_ahead=ahead[child],
                    distance_um=self._distance_um(node, child),
                    is_gt_edge=(node, child) in self.gt_edges,
                )
                for child in ordered
            )
        )

    def _candidate(self, node: int, ordered: Sequence[int], child: int, ahead: Mapping[int, int]) -> ForkCandidate:
        kept = ordered[0]
        kept_next = self.successors.get(kept, ())
        child_next = self.successors.get(child, ())
        return ForkCandidate(
            parent=self.index_of[node],
            kept=self.index_of[kept],
            child=self.index_of[child],
            probability=self.probability.get((node, child), 0.0),
            positions_um=self.positions_um,
            kept_frames=ahead[kept],
            child_frames=ahead[child],
            kept_successor=self.index_of[kept_next[0]] if kept_next else -1,
            child_successor=self.index_of[child_next[0]] if child_next else -1,
        )


@dataclass(frozen=True)
class ForkGrading:
    """The champion's forks under every cue: one grade for keeping real divisions, one for choosing a branch."""

    forks: tuple[LabelledFork, ...]
    branchings: tuple[Branching, ...]

    def total_true(self) -> int:
        """Emitted forks whose second branch is an annotated daughter of an annotated divider."""
        return sum(1 for fork in self.forks if fork.is_true)

    def charged_false(self) -> int:
        """False forks the metric actually charges for — the ones at a matched node."""
        return sum(1 for fork in self.forks if fork.is_charged and not fork.is_true)

    def resolved(self) -> int:
        """Forks the annotation resolves to one branch — the only ones a keep-one rule can be graded on."""
        return sum(1 for branching in self.branchings if branching.truth_index() is not None)

    def gradings(self) -> list[BranchAccuracy]:
        """Every keep-one rule's branch accuracy, in the order the rules are declared."""
        return [BranchAccuracy.of(name, rule, self.branchings) for name, rule in self._branch_rules()]

    def reports(self) -> list[RankReport]:
        """Every fork ranking's verdict, in the order the rankings are declared."""
        return [RankReport.of(name, ranking, self.forks) for name, ranking in self._rankings()]

    @staticmethod
    def _branch_rules() -> Iterable[tuple[str, Callable[[Branch], float]]]:
        """The keep-one rules worth grading, all readable off the linked graph with no model and no annotation.

        `survival` is the one E39 argues for: a fork's spurious branch should be a BLIP that dies within a
        frame or two while the real continuation runs on. `survival+probability` breaks its ties by affinity,
        which is the only cue `probability` alone has to offer.
        """
        yield "probability", lambda branch: branch.probability
        yield "survival", lambda branch: float(branch.frames_ahead)
        yield (
            "survival+probability",
            lambda branch: FRAMES_OVER_PROBABILITY * branch.frames_ahead + branch.probability,
        )
        yield "nearest", lambda branch: -branch.distance_um
        yield (
            "nearest+survival",
            lambda branch: FRAMES_OVER_PROBABILITY * branch.frames_ahead - branch.distance_um,
        )

    @staticmethod
    def _rankings() -> Iterable[tuple[str, ForkRanking]]:
        """Every `fork_ranking` strategy and the compositions of them, re-aimed at emitted forks."""
        geometry = GeometryRanking(sister_weight=SISTER_WEIGHT)
        symmetry = SplitSymmetryRanking(parent_gate_um=PARENT_GATE_UM)
        surviving = SurvivingDaughterRanking(base=symmetry, min_track_length=SURVIVING_TRACK_FRAMES)
        yield "probability", ProbabilityRanking()
        yield "geometry", geometry
        yield "symmetry", symmetry
        yield "symmetry+surviving", surviving
        yield "symmetry+diverging", DivergingDaughterRanking(base=symmetry)
        yield "geometry+diverging", DivergingDaughterRanking(base=geometry)
        yield "symmetry+surviving+diverging", DivergingDaughterRanking(base=surviving)


@dataclass(frozen=True)
class MatchedPair:
    """A predicted tracksdata graph already matched to its annotation, read down into a `MovieGraph`.

    The graphs stay `object` here: tracksdata's own signatures do not satisfy a structural protocol, and this
    is the one place in the module that touches them.
    """

    predicted: object
    truth: object
    keys: object
    scale: object

    def movie_graph(self) -> MovieGraph:
        rows = [int(node) for node in self.predicted.node_ids()]  # type: ignore[attr-defined]
        index_of = {node: index for index, node in enumerate(rows)}
        positions, times = self._geometry(rows, index_of)
        probability, gt_edges = self._edges()
        return MovieGraph(
            index_of=index_of,
            times=times,
            positions_um=positions,
            successors={node: [int(child) for child in self.predicted.successors(node)] for node in rows},  # type: ignore[attr-defined]
            probability=probability,
            gt_edges=gt_edges,
            annotated_of=self._annotated_of(),
            gt_children=self._gt_children(),
            gt_out_degree=self._gt_out_degree(),
        )

    def _geometry(self, rows: list[int], index_of: Mapping[int, int]) -> tuple[np.ndarray, dict[int, int]]:
        node_id = self.keys.NODE_ID  # type: ignore[attr-defined]
        attrs = self.predicted.node_attrs(attr_keys=[node_id, "t", "z", "y", "x"])  # type: ignore[attr-defined]
        spacing = np.asarray(self.scale, dtype=float)[-SPATIAL_DIMENSIONS:] if self.scale is not None else np.ones(3)
        order = [index_of[int(node)] for node in attrs[node_id]]
        positions = np.zeros((len(rows), SPATIAL_DIMENSIONS), dtype=float)
        positions[order] = np.stack([attrs["z"], attrs["y"], attrs["x"]], axis=1) * spacing
        times = {int(node): int(time) for node, time in zip(attrs[node_id], attrs["t"], strict=True)}
        return positions, times

    def _edges(self) -> tuple[dict[tuple[int, int], float], frozenset[tuple[int, int]]]:
        edges = self.predicted.edge_attrs()  # type: ignore[attr-defined]
        rows = list(
            zip(
                edges[self.keys.EDGE_SOURCE],  # type: ignore[attr-defined]
                edges[self.keys.EDGE_TARGET],  # type: ignore[attr-defined]
                edges["edge_prob"],
                edges[self.keys.MATCHED_EDGE_MASK],  # type: ignore[attr-defined]
                strict=True,
            )
        )
        probability = {(int(source), int(target)): float(prob) for source, target, prob, _ in rows}
        gt_edges = frozenset((int(source), int(target)) for source, target, _, mask in rows if bool(mask))
        return probability, gt_edges

    def _annotated_of(self) -> dict[int, int]:
        node_id = self.keys.NODE_ID  # type: ignore[attr-defined]
        matched_id = self.keys.MATCHED_NODE_ID  # type: ignore[attr-defined]
        matched = self.predicted.node_attrs(attr_keys=[node_id, matched_id])  # type: ignore[attr-defined]
        return {
            int(node): int(gt) for node, gt in zip(matched[node_id], matched[matched_id], strict=True) if int(gt) >= 0
        }

    def _gt_children(self) -> dict[int, frozenset[int]]:
        return {
            int(node): frozenset(int(child) for child in self.truth.successors(int(node)))  # type: ignore[attr-defined]
            for node in self.truth.node_ids()  # type: ignore[attr-defined]
        }

    def _gt_out_degree(self) -> dict[int, int]:
        rows = self.truth.node_ids()  # type: ignore[attr-defined]
        degrees = self.truth.out_degree(rows)  # type: ignore[attr-defined]
        return {int(node): int(degree) for node, degree in zip(rows, degrees, strict=True)}


def main() -> None:
    import argparse  # noqa: PLC0415
    import logging  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    import tracksdata as td  # noqa: PLC0415
    from biohub_tracking.io import open_dataset  # type: ignore[missing-import]  # noqa: PLC0415

    from celltrack.eval.official_ruler import (  # noqa: PLC0415
        CHAMPION_PREDICTIONS,
        COMPETITION,
        MAX_MATCH_DISTANCE_UM,
        TEST_MOVIES,
    )
    from core.paths import DataRoot  # noqa: PLC0415

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = DataRoot.from_config(Path("paths.yaml"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, default=data.processed(COMPETITION) / CHAMPION_PREDICTIONS)
    arguments = parser.parse_args()
    ground_truth = data.raw(COMPETITION) / "train"

    forks: list[LabelledFork] = []
    branchings: list[Branching] = []
    for stem in TEST_MOVIES:
        geff = arguments.pred_dir / f"{stem}.geff"
        if not geff.exists():
            log.info("%s: MISSING -- skipped", stem)
            continue
        loaded = td.graph.IndexedRXGraph.from_geff(geff)
        predicted = loaded[0] if isinstance(loaded, tuple) else loaded
        dataset = open_dataset(ground_truth / stem, require_tracks=True)
        predicted.match(
            dataset.tracks,
            matching=td.metrics.DistanceMatching(max_distance=MAX_MATCH_DISTANCE_UM, scale=dataset.scale),
        )
        pair = MatchedPair(predicted=predicted, truth=dataset.tracks, keys=td.DEFAULT_ATTR_KEYS, scale=dataset.scale)
        movie_forks, movie_branchings = pair.movie_graph().forks()
        forks.extend(movie_forks)
        branchings.extend(movie_branchings)
        log.info("%s: forks so far %d", stem, len(forks))

    grading = ForkGrading(forks=tuple(forks), branchings=tuple(branchings))
    log.info(
        "emitted forks=%d true=%d charged-false=%d | annotation resolves %d of them to one branch",
        len(forks),
        grading.total_true(),
        grading.charged_false(),
        grading.resolved(),
    )
    for graded in grading.gradings():
        log.info(
            "%-24s branch_accuracy=%.4f annotated_edges_lost=%4d",
            graded.name,
            graded.accuracy(),
            graded.lost_edges(),
        )
    for report in grading.reports():
        budget, value = report.best_budget(grading.total_true())
        log.info(
            "%-30s worst_true_rank=%5d best_budget=%5d div_jaccard=%.4f score_delta=%+.4f",
            report.name,
            report.worst_rank(),
            budget,
            value,
            DIVISION_WEIGHT * (value - CHAMPION_DIVISION_JACCARD),
        )


if __name__ == "__main__":
    main()
