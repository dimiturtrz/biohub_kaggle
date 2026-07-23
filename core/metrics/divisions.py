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

from dataclasses import dataclass

import numpy as np
from jaxtyping import Int

from core.data.tracks import Adjacency, TrackGraph
from core.metrics.matching import UNMATCHED, DistanceMatcher

_MIN_DAUGHTERS = 2


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

    def _local_candidates(self, divider: int, forks: set[int]) -> tuple[set[int], set[int]]:
        """Predicted forks that could recover this division, and every fork that was even looked at."""
        window, window_rows = self._division_window(divider)
        inside = self.matcher.match(self.prediction, window).gt_rows
        found = inside != UNMATCHED
        matched = np.where(found, window_rows[np.where(found, inside, 0)], UNMATCHED)

        children = self.annotated.successors[divider]
        if len(children) < _MIN_DAUGHTERS:
            return set(), set()
        parents = self._rows_matching(matched, {divider, *self.annotated.predecessors[divider]})
        daughters = [self._rows_matching(matched, {child, *self.annotated.successors[child]}) for child in children]
        if not parents or sum(bool(lineage) for lineage in daughters) < _MIN_DAUGHTERS:
            return set(), set()

        reachable = parents | {successor for parent in parents for successor in self.predicted.successors[parent]}
        local = reachable & forks
        return {fork for fork in local if self._local_topology_holds(fork, parents, daughters)}, local

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
