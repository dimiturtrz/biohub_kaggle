"""Scoring cell divisions, the small second term of the competition score.

When a cell visibly splits is partly a matter of opinion, so a division is not scored by hitting one
timepoint exactly. Each ground-truth division opens a local window — grandparent, dividing parent,
children, grandchildren — and the prediction is matched against that window on its own. A predicted fork
recovers the division when it sits on the parent side and reaches two of the ground-truth daughter
lineages down two *different* branches of its own.

The rest of the rules exist to stop a fork claiming credit it cannot support: a branch whose evidence
comes from a different lineage fragment of the ground truth, or one whose children are shared with
another parent, proves nothing about this division. Forks and divisions are finally paired one-to-one, so
neither can be counted twice.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from jaxtyping import Int

from core.data.tracks import Adjacency, TrackGraph
from core.metrics.matching import UNMATCHED, DistanceMatcher

_MIN_DAUGHTERS = 2
_UNPARENTED = 0
_SINGLE_CHILD = 1


@dataclass(frozen=True)
class DivisionScoring:
    """One prediction against one annotation, with both graphs' links indexed and a matcher to hand."""

    prediction: TrackGraph
    predicted: Adjacency
    truth: TrackGraph
    annotated: Adjacency
    matcher: DistanceMatcher

    @classmethod
    def of(cls, prediction: TrackGraph, truth: TrackGraph, matcher: DistanceMatcher) -> "DivisionScoring":
        """Index both graphs once, so every division window is scored against the same structure."""
        return cls(
            prediction=prediction,
            predicted=Adjacency.of(prediction),
            truth=truth,
            annotated=Adjacency.of(truth),
            matcher=matcher,
        )

    def counts(self) -> "DivisionCounts":
        """Tally recovered, missed and spurious divisions."""
        evaluable, invalid = self._fork_verdicts()
        candidates: dict[int, set[int]] = {}
        considered: set[int] = set()
        forks = set(self.predicted.dividing_rows().tolist())
        for divider in self.annotated.dividing_rows().tolist():
            local, looked_at = self._local_candidates(divider, forks)
            considered |= looked_at
            candidates[divider] = local - invalid

        pairing = self._maximum_matching(candidates)
        recovered = set(pairing.values())
        return DivisionCounts(
            tp=len(pairing),
            fp=len((considered | evaluable | invalid) - recovered),
            fn=len(candidates) - len(pairing),
        )

    def reach(self) -> "DivisionReach":
        """Why each annotated division was missed — the decomposition a recall method needs before it is built.

        `counts` says how many divisions were lost; it cannot say at which stage. The division Jaccard is
        dominated by its false negatives (measured 2026-09-20: tp=1, fp=10, fn=24), so the term only moves on
        recall — and recall is bought in a different place depending on whether the nodes are absent, or
        present with no fork emitted on them, or forked in a shape the annotation refuses.
        """
        evaluable, invalid = self._fork_verdicts()
        forks = set(self.predicted.dividing_rows().tolist())
        candidates = {
            divider: self._local_candidates(divider, forks)[0] - invalid
            for divider in self.annotated.dividing_rows().tolist()
        }
        pairing = self._maximum_matching(candidates)
        tally = dict.fromkeys(DivisionStage, 0)
        unproposable = 0
        no_kept_child = 0
        for divider in candidates:
            stage = self._stage_of(divider, forks, set(pairing))
            tally[stage] += 1
            if stage is DivisionStage.NO_FORK and self._daughters_all_claimed(divider):
                unproposable += 1
            if stage is DivisionStage.NO_FORK and self._no_kept_child(divider):
                no_kept_child += 1
        return DivisionReach(
            recovered=tally[DivisionStage.RECOVERED],
            nodes_missing=tally[DivisionStage.NODES_MISSING],
            no_fork=tally[DivisionStage.NO_FORK],
            fork_rejected=tally[DivisionStage.FORK_REJECTED],
            unproposable=unproposable,
            no_kept_child=no_kept_child,
            spurious=len((evaluable | invalid) - set(pairing.values())),
        )

    def _no_kept_child(self, divider: int) -> bool:
        """Whether no predicted parent-side node carries exactly one child, so nothing can be forked off.

        `AffinityDivisionRecovery` adds a second daughter beside a kept one: it skips any parent whose
        out-degree is not one. A division whose parent track the linker ended (out-degree zero) is therefore
        invisible to the stage even when both daughters sit beside it as orphans — the fix for those is a
        candidacy that starts two daughters off a childless parent, which the stage cannot currently express.
        """
        parents, _ = self._window_support(divider)
        return not any(int(self.predicted.out_degrees[row]) == _SINGLE_CHILD for row in parents)

    def _daughters_all_claimed(self, divider: int) -> bool:
        """Whether every predicted node standing in for this division's daughters already has a parent.

        `AffinityDivisionRecovery` draws each proposed second daughter from the orphan pool — both its
        candidacies gate on `gap.orphan` — so a division whose daughter lineages the linker has already
        claimed lies outside that stage at every setting of its gates, ranking, candidacy and budget. Only
        the lineage HEADS count: the stage proposes the node one frame past the parent, so an orphan
        grandchild under a claimed daughter is not a proposal the stage could ever make.
        """
        matched = self._matched_window(divider)
        heads = [self._rows_matching(matched, {child}) for child in self.annotated.successors[divider]]
        return not any(any(int(self.predicted.in_degrees[row]) == _UNPARENTED for row in head) for head in heads)

    def _stage_of(self, divider: int, forks: set[int], recovered: set[int]) -> "DivisionStage":
        """How far one annotated division got before it was lost."""
        if divider in recovered:
            return DivisionStage.RECOVERED
        parents, daughters = self._window_support(divider)
        if not self._supported(parents, daughters):
            return DivisionStage.NODES_MISSING
        return DivisionStage.NO_FORK if not self._nearby_forks(parents, forks) else DivisionStage.FORK_REJECTED

    def _local_candidates(self, divider: int, forks: set[int]) -> tuple[set[int], set[int]]:
        """Predicted forks that could recover this division, and every fork that was even looked at."""
        parents, daughters = self._window_support(divider)
        if not self._supported(parents, daughters):
            return set(), set()
        local = self._nearby_forks(parents, forks)
        return {fork for fork in local if self._local_topology_holds(fork, parents, daughters)}, local

    def _matched_window(self, divider: int) -> Int[np.ndarray, "n"]:
        """Each predicted node's annotated partner inside this division's window, or `UNMATCHED`."""
        window, window_rows = self._division_window(divider)
        inside = self.matcher.match(self.prediction, window).gt_rows
        found = inside != UNMATCHED
        return np.where(found, window_rows[np.where(found, inside, 0)], UNMATCHED)

    def _window_support(self, divider: int) -> tuple[set[int], list[set[int]]]:
        """The predicted nodes standing in for this division's parent side, and for each daughter lineage."""
        matched = self._matched_window(divider)
        children = self.annotated.successors[divider]
        if len(children) < _MIN_DAUGHTERS:
            return set(), []
        parents = self._rows_matching(matched, {divider, *self.annotated.predecessors[divider]})
        daughters = [self._rows_matching(matched, {child, *self.annotated.successors[child]}) for child in children]
        return parents, daughters

    def _nearby_forks(self, parents: set[int], forks: set[int]) -> set[int]:
        """The predicted forks sitting on this division's parent side."""
        reachable = parents | {successor for parent in parents for successor in self.predicted.successors[parent]}
        return reachable & forks

    @staticmethod
    def _supported(parents: set[int], daughters: list[set[int]]) -> bool:
        """Whether the prediction even has the nodes a division needs — a parent and two daughter lineages."""
        return bool(parents) and sum(bool(lineage) for lineage in daughters) >= _MIN_DAUGHTERS

    def _fork_verdicts(self) -> tuple[set[int], set[int]]:
        """Forks the annotation can judge at all, and forks whose own branch evidence rules them out."""
        matched = self.matcher.match(self.prediction, self.truth).gt_rows
        components = self.annotated.components()
        forks = self.predicted.dividing_rows().tolist()
        evaluable = {
            fork for fork in forks if matched[fork] != UNMATCHED and self.annotated.out_degrees[matched[fork]] >= 1
        }
        invalid: set[int] = set()
        for fork in forks:
            evidence: list[int] = []
            for child in self.predicted.successors[fork]:
                component, malformed = self._branch_component(fork, child, matched, components)
                if malformed:
                    invalid.add(fork)
                    break
                if component is not None:
                    evidence.append(component)
            else:
                if len(set(evidence)) >= _MIN_DAUGHTERS:
                    invalid.add(fork)
        return evaluable, invalid

    def _division_window(self, divider: int) -> tuple[TrackGraph, Int[np.ndarray, "w"]]:
        """The grandparent-to-grandchildren neighbourhood of one annotated division, as a graph of its own."""
        children = self.annotated.successors[divider]
        grandchildren = [grandchild for child in children for grandchild in self.annotated.successors[child]]
        rows = np.unique(np.array([*self.annotated.predecessors[divider], divider, *children, *grandchildren]))
        inside = np.zeros(len(self.truth.node_ids), dtype=bool)
        inside[rows] = True
        window = TrackGraph(
            node_ids=self.truth.node_ids[rows],
            coordinates=self.truth.coordinates[rows],
            edges=self.truth.edges[inside[self.truth.edge_rows()].all(axis=1)],
        )
        return window, rows

    def _local_topology_holds(self, fork: int, parents: set[int], daughters: list[set[int]]) -> bool:
        """Whether the fork is anchored on the parent side and its branches cover two daughter lineages."""
        if not ({fork, *self.predicted.predecessors[fork]} & parents):
            return False
        branches = [{child, *self.predicted.successors[child]} for child in self.predicted.successors[fork]]
        reach = {
            lineage: {index for index, branch in enumerate(branches) if matched & branch}
            for lineage, matched in enumerate(daughters)
        }
        return len(self._maximum_matching(reach)) >= _MIN_DAUGHTERS

    def _branch_component(
        self,
        fork: int,
        child: int,
        matched: Int[np.ndarray, "n"],
        components: Int[np.ndarray, "g"],
    ) -> tuple[int | None, bool]:
        """Which annotated lineage fragment one branch points at, and whether the branch is shared.

        The child's own match wins when it has one, so a mistake further downstream cannot invalidate a
        division that was matched correctly at the branch point.
        """
        if set(self.predicted.predecessors[child]) != {fork}:
            return None, True
        if matched[child] != UNMATCHED:
            return int(components[matched[child]]), False

        grandchildren = self.predicted.successors[child]
        if any(set(self.predicted.predecessors[node]) != {child} for node in grandchildren):
            return None, True
        found = {int(components[matched[node]]) for node in grandchildren if matched[node] != UNMATCHED}
        return (next(iter(found)), False) if len(found) == 1 else (None, False)

    @staticmethod
    def _rows_matching(matched: Int[np.ndarray, "n"], truth_rows: set[int]) -> set[int]:
        """Predicted nodes whose partner is one of these annotated nodes."""
        found = np.flatnonzero(matched != UNMATCHED).tolist()
        return {row for row in found if int(matched[row]) in truth_rows}

    @staticmethod
    def _maximum_matching(edges: dict[int, set[int]]) -> dict[int, int]:
        """Maximum-cardinality bipartite matching, so one fork can serve at most one division."""
        right_to_left: dict[int, int] = {}
        left_to_right: dict[int, int] = {}

        def augment(left: int, seen: set[int]) -> bool:
            for right in edges.get(left, set()):
                if right in seen:
                    continue
                seen.add(right)
                if right not in right_to_left or augment(right_to_left[right], seen):
                    left_to_right[left] = right
                    right_to_left[right] = left
                    return True
            return False

        for left in edges:
            augment(left, set())
        return left_to_right


class DivisionStage(Enum):
    """How far an annotated division got through the prediction before it was lost."""

    RECOVERED = auto()
    NODES_MISSING = auto()
    NO_FORK = auto()
    FORK_REJECTED = auto()


@dataclass(frozen=True)
class DivisionReach:
    """The false negatives split by the stage that lost them — which buys what a recall method should attack.

    `nodes_missing` is a detector cost: the parent or a daughter lineage is simply not in the prediction, so
    no linking or post-processing decision could have recovered the division. `no_fork` and `fork_rejected`
    are selection costs, and far cheaper to attack: the nodes are there, and either nothing forked on them or
    the fork's shape did not satisfy the annotation.
    """

    recovered: int
    nodes_missing: int
    no_fork: int
    fork_rejected: int
    unproposable: int
    no_kept_child: int
    spurious: int

    @classmethod
    def total(cls, reaches: "Iterable[DivisionReach]") -> "DivisionReach":
        """Sum a per-movie decomposition into one, so the split is read over the whole ruler."""
        parts = list(reaches)
        return cls(
            recovered=sum(part.recovered for part in parts),
            nodes_missing=sum(part.nodes_missing for part in parts),
            no_fork=sum(part.no_fork for part in parts),
            fork_rejected=sum(part.fork_rejected for part in parts),
            unproposable=sum(part.unproposable for part in parts),
            no_kept_child=sum(part.no_kept_child for part in parts),
            spurious=sum(part.spurious for part in parts),
        )

    def missed(self) -> int:
        """Every annotated division the prediction lost, at whatever stage."""
        return self.nodes_missing + self.selection_bound()

    def selection_bound(self) -> int:
        """Divisions a perfect decision over the nodes already predicted could still recover."""
        return self.no_fork + self.fork_rejected


@dataclass(frozen=True)
class DivisionCounts:
    """The confusion counts the division Jaccard is built from."""

    tp: int
    fp: int
    fn: int

    def jaccard(self) -> float:
        """`TP / (TP + FP + FN)`, or NaN when neither graph contains a division."""
        total = self.tp + self.fp + self.fn
        return self.tp / total if total > 0 else float("nan")
