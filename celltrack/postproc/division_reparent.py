"""Move a daughter's incoming link from whoever claimed her to the row her annotated parent landed on.

An annotated division is unrecoverable by any post-processing that only ADDS a second successor: both
daughters already have a predecessor, and the one holding the second daughter is a different track
continuing through her. Ten of E61's twenty-five unrecovered divisions are exactly that shape.

Re-parenting is the only move that reaches them, and unlike a score-selected deletion it is edge-FREE by
construction. The official metric charges a predicted link when EITHER endpoint matches an annotated node
that the annotation continues through (`out_valid | in_valid` in `biohub_tracking.metrics`, the same OR as
`core.metrics.edges.ChargedLinks._countable`). A matched daughter is such a node, so the claimant's link
into her is already counted as a false positive: deleting it removes an FP and the new parent→daughter link
adds a true positive. Both halves of the move gain, before the division term is counted at all.

These are the planners only. `celltrack.eval.oracle_fork_arms` drives them from the annotation and prices
them: the re-parent is worth +0.0031, and the fork cull it sits beside is worth +0.1263 with the annotation
against +0.0034 without it. The shippable form is a linker that can choose to end a continuation in favour of
a division, and it is worth building only for as much as those arms measure.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

MIN_DAUGHTERS = 2
ORACLE_EDGE_PROB = 1.0


@dataclass(frozen=True)
class Reparent:
    """One daughter's incoming link, moved to the row her annotated parent landed on."""

    parent: int
    daughter: int
    claimed_by: int | None

    def is_theft(self) -> bool:
        """Whether some other row already held this daughter — the shape a pure ADD cannot reach."""
        return self.claimed_by is not None


@dataclass(frozen=True)
class ReparentPlan:
    """Every move one movie's annotation asks for, and how many divisions they complete."""

    moves: tuple[Reparent, ...]
    completed: int
    unreachable: int

    def thefts(self) -> int:
        return sum(1 for move in self.moves if move.is_theft())

    @classmethod
    def of(
        cls,
        divisions: Iterable[tuple[int, Sequence[int]]],
        predicted_of: Mapping[int, int],
        predecessors: Callable[[int], Sequence[int]],
    ) -> ReparentPlan:
        """The moves that put every matched daughter under her matched parent.

        A division is `completed` when the parent and both daughters all landed on predicted rows and the plan
        reaches every daughter; `unreachable` counts the ones where the detector never produced a row to link,
        which is a detection miss and not this lever's to fix.
        """
        moves: list[Reparent] = []
        completed = 0
        unreachable = 0
        for divider, daughters in divisions:
            parent = predicted_of.get(divider)
            children = [predicted_of[daughter] for daughter in daughters if daughter in predicted_of]
            if parent is None or len(children) < max(len(daughters), MIN_DAUGHTERS):
                unreachable += 1
                continue
            completed += 1
            moves += cls._moves_under(parent, children, predecessors)
        return cls(moves=tuple(moves), completed=completed, unreachable=unreachable)

    @staticmethod
    def _moves_under(
        parent: int, children: Sequence[int], predecessors: Callable[[int], Sequence[int]]
    ) -> list[Reparent]:
        moves: list[Reparent] = []
        for child in children:
            held_by = list(predecessors(child))
            if parent in held_by:
                continue
            moves.append(Reparent(parent=parent, daughter=child, claimed_by=held_by[0] if held_by else None))
        return moves


@dataclass(frozen=True)
class ForkCull:
    """The surplus successors of a predicted fork whose matched annotation does not divide there."""

    parent: int
    dropped: tuple[int, ...]

    @classmethod
    def plan(
        cls,
        forks: Iterable[tuple[int, Sequence[int]]],
        annotated_of: Mapping[int, int],
        divides: Callable[[int], bool],
        rank: Callable[[int, int], float],
    ) -> tuple[ForkCull, ...]:
        """Cut every predicted fork back to one successor unless its annotation divides there too.

        This is the PRECISION half of the division term, and the term's whole mass: on the six test movies the
        champion forks 254 times where the annotation does not, against 5 divisions it misses. The fork's
        surviving branch is the highest-ranked one, so an unannotated real track keeps its link rather than
        losing it to an arbitrary choice.
        """
        culls: list[ForkCull] = []
        for parent, successors in forks:
            annotated = annotated_of.get(parent)
            if annotated is not None and divides(annotated):
                continue
            ordered = sorted(successors, key=lambda child: rank(parent, child), reverse=True)
            if len(ordered) > 1:
                culls.append(cls(parent=parent, dropped=tuple(ordered[1:])))
        return tuple(culls)
