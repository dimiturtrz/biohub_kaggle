"""The fork candidate and the strategies that RANK competing forks — the vocabulary `AffinityDivisionRecovery` scores.

`ForkCandidate` is the vocabulary a ranking is written against: one proposed division and every quantity a
`ForkRanking` may read — the head's score, the split's geometry, each daughter's future. `AffinityDivisionRecovery`
builds these over a frame gap (`_Gap`) and hands them to whichever ranking a pipeline configured.

The ranking is a STRATEGY, not a hard-coded sort key, and probability is only one of its implementations — kept
as the default so no configured pipeline shifts silently:

* `ProbabilityRanking` — the head's second-daughter probability, descending. Today's behaviour. Measured NOT to
  discriminate on dense tissue: the softmax-over-sources normalisation inflates every second choice, so >1200
  parents carry an orphan target at P >= 0.25 while the one genuinely recoverable division's second daughter sits
  at P = 0.124 — a probability ranking sorts the true fork to the bottom and the cap then evicts it.
* `GeometryRanking` — the public frontier's `parent_dist + w * sister_dist`, ascending, which ships at public
  0.915 with no probability floor and no learned verifier at all.
* `SplitSymmetryRanking` — the cue the frontier does not use: true daughters separate onto OPPOSITE sides of the
  mother (our division 232 has its two daughters 13.5 um apart), where a false sister in crowded tissue sits at a
  random angle. Ranking a fork by how far the candidate is from the mother, penalised by how badly the two
  daughters fail to balance around her.

Orthogonal to all three, and composable with any of them, is what a fork IMPLIES rather than how it looks:

* `SurvivingDaughterRanking` — a decorator penalising a proposed daughter that does not go on to survive as a
  track. A real mitosis leaves two lineages that persist; a false fork typically grafts a fragment. The linked
  graph already knows, so this costs no model, no training and no asset.
* `DivergingDaughterRanking` — a decorator penalising a pair that does not move APART after the fork. The same
  C3 signature the champion vetoes on, read as a score: a gate can only admit or reject the set, and the
  divisions we still miss are ones where a fork IS emitted at the right mother and the wrong daughter wins.

Every ranking speaks one direction: lower cost wins. The admission FLOOR (`min_second_prob`) and the budget live
on the recovery stage, never here — a ranking only orders what it is handed.
"""

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from jaxtyping import Bool, Float, Int

_DEGENERATE = 0.0
# What a fork pays on a [0, 1] term it fails completely — a pair that does not move apart, or cannot be shown to.
_FULL_PENALTY = 1.0
_SHORTEST_TRACK = 1
# A daughter whose track ends at the gap has no t+2 node to read a divergence from, and no node is its own
# nearest neighbour — the same out-of-band row stands for both "no successor" and "no mutual match".
_NO_SUCCESSOR = -1
# A target no predicted track parents, and a fork that therefore displaces nobody — the same out-of-band row
# stands for "this daughter is free" and "this candidate takes her from no one".
_UNCLAIMED = -1


@dataclass(frozen=True)
class _Gap:
    """One frame gap's node context: the t+1 target rows, which are unparented, all positions, all futures."""

    targets: Int[np.ndarray, "t"]
    orphan: Bool[np.ndarray, "t"]
    positions_um: Float[np.ndarray, "n 3"]
    frames_ahead: Int[np.ndarray, "n"]
    # Each node's first successor row (`_NO_SUCCESSOR` where its track ends), so a candidate can read where its
    # two daughters sit at t+2 and test whether they moved apart — the C3 post-mitotic divergence signature.
    successor: Int[np.ndarray, "n"]
    # Which source row already parents each target (`_UNCLAIMED` where none does) and how strongly the head
    # scored that standing link. Together they price DISPLACING the current parent: a claimed daughter is
    # reachable only by cutting her existing edge, and only worth cutting where the proposing mother outscores
    # the incumbent. `orphan` is the same fact as `claimed_by == _UNCLAIMED`, kept because it is read per-gap
    # in the hot loop; these two carry the rest of the story the orphan flag throws away.
    claimed_by: Int[np.ndarray, "t"]
    claim_probability: Float[np.ndarray, "t"]


@dataclass(frozen=True)
class ForkCandidate:
    """One proposed fork and every quantity a ranking may read: the head's score, the split's geometry, its future.

    The three nodes are graph ROWS; the micrometre positions are the whole video's, so a candidate carries no
    copied coordinates. Public because a `ForkRanking` is written against this vocabulary.
    """

    parent: int
    kept: int
    child: int
    probability: float
    positions_um: Float[np.ndarray, "n 3"]
    # How many frames each daughter's track survives from this gap onward, the mother's kept child included.
    # The linker admits one child per node, so a daughter's future is a chain and its length is well defined.
    kept_frames: int
    child_frames: int
    # Each daughter's t+2 node row (`_NO_SUCCESSOR` where her track ends at the gap), read from the linked
    # graph so the C3 divergence gate needs no extra lookup. Defaulted so a geometry-only fork still builds.
    kept_successor: int = _NO_SUCCESSOR
    child_successor: int = _NO_SUCCESSOR
    # The source row this fork takes the daughter FROM, `_UNCLAIMED` for the orphan-pool forks that take her
    # from no one. A displacing candidate is edge-NEUTRAL — one edge cut, one added — which is what separates
    # its cost from an additive fork's.
    displaced: int = _UNCLAIMED

    def taking(self, displaced: int) -> "ForkCandidate":
        """The same fork, declared as taking its daughter FROM the given source row rather than from no one."""
        return replace(self, displaced=displaced)

    @classmethod
    def at_gap(cls, parent: int, kept: int, child: int, probability: float, gap: "_Gap") -> "ForkCandidate":
        """One proposed triangle over a gap, its daughters' futures read from the linked graph's frame counts."""
        return cls(
            parent=parent,
            kept=kept,
            child=child,
            probability=probability,
            positions_um=gap.positions_um,
            kept_frames=int(gap.frames_ahead[kept]),
            child_frames=int(gap.frames_ahead[child]),
            kept_successor=int(gap.successor[kept]),
            child_successor=int(gap.successor[child]),
        )

    def divergence_um(self) -> float | None:
        """How much the daughters' separation GROWS from t+1 to t+2 — None if either track ends at the gap.

        The C3 post-mitotic signature: true sisters continue AND move apart, so a real division's separation is
        larger at t+2 than at t+1. Both daughters must have a t+2 node (otherwise there is no growth to read),
        and the value is that later separation minus the one at the fork — positive where the pair diverges.
        """
        if _NO_SUCCESSOR in {self.kept_successor, self.child_successor}:
            return None
        return self._distance(self.kept_successor, self.child_successor) - self.sister_distance_um()

    def divergence_fraction(self) -> float | None:
        """The daughters' separation growth as a fraction of where they end up — None if either track ends.

        `divergence_um` in dimensionless form: the growth over the WIDER of the two separations, so it lands
        in [-1, 1] — 0 for a pair holding its distance, towards 1 for one that starts touching and flies
        apart, negative for one closing in. It needs no fitted scale, which is what lets divergence be a
        SCORE rather than only the threshold the C3 gate applies — a micrometre growth cannot be weighed
        against a dimensionless geometry term without inventing one.
        """
        divergence = self.divergence_um()
        if divergence is None:
            return None
        widest = max(self.sister_distance_um(), self.sister_distance_um() + divergence)
        if widest == _DEGENERATE:
            return _DEGENERATE
        return divergence / widest

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
class DivergingDaughterRanking:
    """Any ranking, penalised by how far the proposed pair falls short of moving APART after the fork.

    The C3 signature the champion vetoes on, read as a score instead of a threshold. The veto is measured to
    be the single largest division lever we have (+0.0123), but as a gate it only says admissible/not — and
    the divisions still missed are ones where a fork IS emitted at the right mother and the wrong daughter
    wins the sort. A gate cannot order the set it admits; this can.

    Dimensionless by construction (`divergence_fraction`), so it lands in [0, 1] like the terms it decorates
    and an equal weighting stays the untuned choice. A pair with no t+2 node on one side has no divergence to
    read and pays the full penalty — the same treatment the gate gives it, since unproven is not evidence.
    """

    base: ForkRanking

    def cost(self, candidate: ForkCandidate) -> float:
        """The base ranking's cost plus how far this pair falls short of a full post-mitotic separation."""
        fraction = candidate.divergence_fraction()
        if fraction is None:
            return self.base.cost(candidate) + _FULL_PENALTY
        return self.base.cost(candidate) + _FULL_PENALTY - max(fraction, _DEGENERATE)


@dataclass(frozen=True)
class SurvivingDaughterRanking:
    """Any ranking, penalised by how far the proposed daughter falls short of surviving as a track.

    A real mitosis leaves two lineages that PERSIST. A false fork typically grafts a fragment onto a track —
    which is also what makes the stage's component-merge side effect indistinguishable from a division in the
    total score. Nothing in the geometry can tell those apart, and the graph already knows: the recovery runs
    on the linked graph, so both daughters' futures are in hand at ranking time. No model, no training, no
    asset — the only cue here that reads what a fork IMPLIES rather than how it looks.

    Normalised, like the terms it decorates, by the constraint that already decides the daughter's own
    admissibility: `ShortTrackFilter`'s length rule. A daughter that survives it contributes 0, one that dies
    at the fork contributes 1, so the term lands in [0, 1] and the equal weighting stays the untuned choice.

    It decorates rather than replaces because persistence is orthogonal to geometry — every existing ranking
    should be able to carry it, and none of them should have to restate the other's terms to do so.
    """

    base: ForkRanking
    min_track_length: int

    def cost(self, candidate: ForkCandidate) -> float:
        """The base ranking's cost plus the daughter's shortfall against the length a track must reach."""
        return self.base.cost(candidate) + self._shortfall(candidate.child_frames)

    def _shortfall(self, frames: int) -> float:
        """How far short of the length rule this daughter's track stops, over the range a daughter can span.

        A daughter always spans at least the frame it appears in, so the reachable range is `1` to the length
        rule and the normaliser is that span — not the rule itself, which would cap the penalty below 1 and
        quietly weight this term under the geometry it is added to. Saturating at the rule is deliberate: a
        long-lived daughter cannot buy its way past a bad split, it only stops being suspicious.
        """
        span = self.min_track_length - _SHORTEST_TRACK
        if span <= _DEGENERATE:
            return _DEGENERATE
        return (self.min_track_length - min(frames, self.min_track_length)) / span
