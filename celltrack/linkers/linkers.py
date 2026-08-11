"""Building a linker from a name and a handful of gate radii — the one place that knows the whole family.

Every linker satisfies the same `Linker` contract (nodes in, edges out), so a caller that wants to *choose*
one at runtime — an eval sweep, a submission config, a CLI flag, a tracked `RunConfig` field — should name
it, not construct it. This is that seam: a validated config carries the union of gate parameters the family
speaks, and `build` dispatches the name to the concrete linker, passing only the radii that one uses. Adding
a linker means adding one row to the table, not another `if` at every call site.

`build` takes the per-video edge `affinity` because the shipped `assignment` linker scores each candidate edge
by a learned `P(s->t)`; the geometry-only linkers ignore it. The config is pydantic (not a plain dataclass)
because it crosses a trust boundary — a `--set linker.name=` override or a loaded config.json is user input;
an unknown name or a non-positive radius is rejected AT construction with a clear error, not deep inside `build`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.boundary_prior import BoundaryPrior
from celltrack.linkers.division_linking import DivisionAwareLinker
from celltrack.linkers.flow_linking import FlowLinker
from celltrack.linkers.ilp_linking import ILPLinker
from celltrack.linkers.linking import Linker, NearestNeighbourLinker
from celltrack.linkers.motion_linking import MotionHungarianLinker
from celltrack.linkers.mutual_bonus import MutualBonus
from celltrack.linkers.ranker_bonus import RankerBonus
from core.geometry import Spacing

logger = logging.getLogger(__name__)

LINKER_NAMES = ("assignment", "nn", "motion", "ilp", "division", "flow")

# The affinity bonus self-balances to this multiple of the gate. `cost = distance - bonus·P` trades a [0,1]
# probability against a micrometre distance, so the bonus carries gate-scale units; at 2·gate a P=0.5 edge
# earns a one-gate-width discount. The swept proxy optimum at gate 10 (=20, bead 88x) IS this ratio — deriving
# it keeps P-weighting constant when the gate moves instead of silently under-weighting P at a wider gate.
_BONUS_GATE_RATIO = 2.0


@dataclass(frozen=True)
class LinkerParts:
    """The optional pieces `build` resolves from a config plus the per-video inputs, handed to a builder as one.

    Each is `None` when the config leaves its knob off, and each is understood by only some of the family, so
    they travel together rather than as three more positional arguments every builder row must thread through.
    """

    agreement: AgreementGate | None = None
    mutual: MutualBonus | None = None
    ranker: RankerBonus | None = None
    boundary: BoundaryPrior | None = None


class LinkerConfig(BaseModel):
    """A linker choice and the gate radii the family draws from, resolved to a concrete linker by `build`.
    Immutable + validated: `name` must be one the family knows, and every radius is positive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "assignment"
    gate_um: float = Field(10.0, gt=0)
    tight_um: float = Field(6.0, gt=0)
    division_um: float = Field(8.0, gt=0)
    # None self-balances to `_BONUS_GATE_RATIO · gate_um` (see the ratio's note); pin a float to override.
    affinity_bonus: float | None = Field(None, ge=0)
    disappearance_cost: float = Field(0.0, ge=0)
    # None = OFF. A floor admits a candidate edge only where the bidirectionally fused probability clears it
    # (`AgreementGate`, which documents what a principled value would be derived from); the cost keeps ranking
    # whatever survives by the sharp forward probability. Setting it makes `build` require a fused affinity.
    agreement_floor: float | None = Field(None, ge=0, le=1)
    # None = OFF, so the shipped cost is byte-identical. Set, it adds the SECOND cost term of
    # `cost = distance - affinity_bonus*P_forward - mutual_bonus*P_mutual` (`MutualBonus`, which documents what a
    # principled value would be derived from): the fused probability priced as evidence the objective reads,
    # rather than replacing the forward term or filtering before it. Setting it makes `build` require a fused
    # affinity, exactly as the floor does.
    mutual_bonus: float | None = Field(None, ge=0)
    # None = OFF, so the shipped cost is byte-identical. Set, it adds the THIRD cost term of
    # `cost = distance - affinity_bonus*P_forward - mutual_bonus*P_mutual - ranker_bonus*P_ranker`
    # (`RankerBonus`): the CC0 local-association re-ranker's probability, which reads 22 facts about the
    # linking state (crowding, the candidate list, the primary graph's degrees, a motion residual) that the
    # pairwise affinity cannot see. Setting it makes `build` require the ranker's scored affinity.
    #
    # SCALE — the public constant does not transfer. Their per-frame Hungarian costs
    # `motion_dist + 0.05*raw_dist - 0.75*(0.85*P_ranker + 0.15*P_edge)`, i.e. 0.6375 um of discount per unit
    # of ranker probability and 0.1125 per unit of edge probability. Ours is `distance - 20*P` (a derived
    # `2*gate_um`), so their ABSOLUTE weights are ~30x weaker against the same micrometre distance and copying
    # 0.6375 here would be an off-by-30 no-op. What does transfer is their RATIO: 85/15 of the learned evidence
    # on the ranker. Held at our derived P-budget of `affinity_bonus + ranker_bonus = 2*gate_um = 20` that
    # reads `ranker_bonus = 17.0` with `affinity_bonus = 3.0` — the probe to run first, and the one that
    # measures the SPLIT rather than confounding it with a change in how strongly probability beats distance.
    # No default is chosen here: the split is a claim about our data, so a sweep decides it, not this file.
    ranker_bonus: float | None = Field(None, ge=0)
    # OFF = the shipped flat track-boundary cost, charged identically everywhere. ON makes it position- and
    # time-aware (`BoundaryPrior`): a track opening at a face of the imaged volume or in the first observed frame
    # is free, one opening mid-volume mid-movie still pays. There is no strength knob — the discount is derived
    # (see `_EXPLAINED_FACTOR`), so this is an on/off claim about the observation window, not a tuning surface.
    # Setting it makes `build` require the video's volume shape.
    boundary_prior: bool = False

    @property
    def effective_bonus(self) -> float:
        """The affinity bonus, self-balancing to 2·gate (the dimensional P-vs-distance coupling) when unset."""
        return self.affinity_bonus if self.affinity_bonus is not None else _BONUS_GATE_RATIO * self.gate_um

    @property
    def needs_mutual(self) -> bool:
        """Whether this config reads the bidirectionally fused affinity — as an admission floor, a cost term, or both.

        The fusion costs a second reversed scoring pass over every gap, so the tracker pays for it only when a
        knob here will actually consume it.
        """
        return self.agreement_floor is not None or self.mutual_bonus is not None

    @property
    def needs_ranker(self) -> bool:
        """Whether this config prices the re-ranker — i.e. whether the caller must score it before linking.

        The ranker's features read a LINK GRAPH (degrees, the selected edge's probability, a motion residual
        off the parent), so scoring it costs a whole extra linking pass over the video. The tracker pays that
        only when the bonus below will actually consume the result.
        """
        return self.ranker_bonus is not None

    @property
    def learned_budget(self) -> float:
        """The total weight this cost puts on learned evidence — the affinity's share plus the re-ranker's.

        The two bonuses are one budget split two ways (see `ranker_bonus`'s note): both discount the same
        micrometre distance by a probability, so what a sweep over the split varies is WHERE the evidence comes
        from, not how strongly evidence beats distance. Naming the sum is what lets the context pass hold that
        second quantity fixed while the split moves.
        """
        return self.effective_bonus + (self.ranker_bonus or 0.0)

    def without_ranker(self) -> "LinkerConfig":
        """This config with the re-ranker's weight moved onto the affinity — the cost the CONTEXT pass links under.

        The re-ranker cannot score until something has linked, so the first pass cannot price it. But merely
        DELETING the term (the previous behaviour) leaves pass 1 spending only the affinity's share of the
        learned-evidence budget, so a sweep over the split silently degrades the very graph the re-ranker's 22
        features are computed from: at an 85/15 split the context links at `affinity_bonus=3`, where a link
        costs more than a track boundary and the flow terminates tracks instead of linking them (bead ic44 —
        8 of 22 features off-distribution there against 3 of 22 at the control).
        The context is an INPUT to the features, not part of the swept cost, so pass 1 spends the FULL budget
        (`learned_budget`) on the evidence it does have. Two splits of one budget then produce the identical
        pass-1 config, and hence the identical context graph — the invariance the sweep needs to mean anything.
        Everything else (gate, floor, mutual bonus) is held, because the context graph only describes "the
        current best link set" if it was produced by the cost being re-ranked.
        """
        return self.model_copy(update={"ranker_bonus": None, "affinity_bonus": self.learned_budget})

    @model_validator(mode="after")
    def _affinity_knobs_are_readable(self) -> "LinkerConfig":
        """Refuse an affinity knob on a linker that cannot read an affinity — it could not move anything.

        Pairing a bonus or an agreement floor with `motion` or `ilp` is not a harmless default: it is an
        instruction the pipeline would accept and silently drop, and a sweep over it would report a flat
        curve that means nothing.
        """
        if self.name in _AFFINITY_READERS:
            return self
        knobs = (
            ("affinity_bonus", self.affinity_bonus),
            ("agreement_floor", self.agreement_floor),
            ("mutual_bonus", self.mutual_bonus),
            ("ranker_bonus", self.ranker_bonus),
        )
        for knob, value in knobs:
            if value is not None:
                message = (
                    f"linker {self.name!r} does not read an edge affinity, so {knob}={value} cannot affect it; "
                    f"choose one of {sorted(_AFFINITY_READERS)} or drop the {knob}"
                )
                raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _boundary_prior_is_readable(self) -> "LinkerConfig":
        """Refuse the boundary prior on a linker that charges no track boundary — same reason as the affinity knobs.

        Only the flow linker prices appearance and disappearance, so only it has a price for the prior to modulate;
        anywhere else the flag would be accepted, silently dropped, and sweep as a flat curve.
        """
        if self.boundary_prior and self.name not in _BOUNDARY_PRIOR_READERS:
            message = (
                f"linker {self.name!r} charges no track-boundary cost, so boundary_prior cannot affect it; "
                f"choose one of {sorted(_BOUNDARY_PRIOR_READERS)} or drop the flag"
            )
            raise ValueError(message)
        return self

    @field_validator("name")
    @classmethod
    def _known_name(cls, name: str) -> str:
        if name not in LINKER_NAMES:
            raise ValueError(f"unknown linker {name!r}; choose from {LINKER_NAMES}")
        return name

    def build(
        self,
        spacing: Spacing,
        affinity: EdgeAffinity | None = None,
        mutual: EdgeAffinity | None = None,
        volume_shape: tuple[int, int, int] | None = None,
        ranker: EdgeAffinity | None = None,
    ) -> Linker:
        """The concrete linker named by this config, wired with the gates it uses (and the affinity if it reads one).

        Ignoring a mounted affinity is a legitimate choice (run `motion` and disregard the learned head), so it
        warns rather than raises — but it warns, because the affinity was already COMPUTED to be discarded.
        `mutual` carries the bidirectionally fused probabilities, read by the agreement floor as an admission
        filter and by the mutual bonus as the cost's second term; it is needed only when one of those is set.
        `volume_shape` is the video's `(z, y, x)` timepoint shape — the one fact about the imaging a detection
        graph does not carry — and only the boundary prior reads it. `ranker` carries the local-association
        re-ranker's per-gap probabilities (`AssociationRanker.affinities`), read by the ranker bonus as the
        cost's third term; like `mutual` it is needed only when its knob is set.
        """
        if affinity is not None and self.name not in _AFFINITY_READERS:
            logger.warning("linker %r ignores the mounted edge affinity — it was computed and discarded", self.name)
        parts = LinkerParts(self._gate(mutual), self._bonus(mutual), self._ranker(ranker), self._boundary(volume_shape))
        return _BUILDERS[self.name](self, spacing, affinity, parts)

    def _gate(self, mutual: EdgeAffinity | None) -> AgreementGate | None:
        """The agreement gate this config asks for — `None` when no floor is set, an error when it cannot be met.

        A floor without a fused affinity is refused for the same reason a bonus on a non-reader is: the pipeline
        would accept the instruction and silently drop it, and the sweep over it would report a flat curve.
        """
        if self.agreement_floor is None:
            return None
        if mutual is None:
            message = (
                f"agreement_floor={self.agreement_floor} needs the bidirectionally fused affinity to gate on; "
                "score the gaps with `bidirectional` and pass the result as `mutual`, or drop the floor"
            )
            raise ValueError(message)
        return AgreementGate(mutual=mutual, floor=self.agreement_floor)

    def _bonus(self, mutual: EdgeAffinity | None) -> MutualBonus | None:
        """The cost's agreement term — `None` when unset, an error when the fused affinity it prices is missing.

        Refused for the same reason the floor is: a weight on a quantity that was never scored is an instruction
        the pipeline would accept and silently drop, and the sweep over it would report a flat curve.
        """
        if self.mutual_bonus is None:
            return None
        if mutual is None:
            message = (
                f"mutual_bonus={self.mutual_bonus} needs the bidirectionally fused affinity to weigh; "
                "score the gaps with `bidirectional` and pass the result as `mutual`, or drop the bonus"
            )
            raise ValueError(message)
        return MutualBonus(mutual=mutual, bonus=self.mutual_bonus)

    def _ranker(self, ranker: EdgeAffinity | None) -> RankerBonus | None:
        """The cost's re-ranker term — `None` when unset, an error when the scored ranker affinity is missing.

        Refused for the same reason the mutual bonus is: a weight on a quantity that was never scored is an
        instruction the pipeline would accept and silently drop, and the sweep over it would report a flat curve.
        """
        if self.ranker_bonus is None:
            return None
        if ranker is None:
            message = (
                f"ranker_bonus={self.ranker_bonus} needs the local-association re-ranker's probabilities to weigh; "
                "score the gaps with `AssociationRanker.affinities` and pass the result as `ranker`, "
                "or drop the bonus"
            )
            raise ValueError(message)
        return RankerBonus(ranker=ranker, bonus=self.ranker_bonus)

    def _boundary(self, volume_shape: tuple[int, int, int] | None) -> BoundaryPrior | None:
        """The track-boundary prior this config asks for — `None` when off, an error when the shape is missing.

        The prior's whole content is where the imaged volume ENDS, so a caller that enables it without passing the
        video's shape is refused rather than quietly served a flat cost: the alternative is a knob that reports a
        flat sweep curve because it never actually ran.
        """
        if not self.boundary_prior:
            return None
        if volume_shape is None:
            message = (
                "boundary_prior needs the video's (z, y, x) volume shape to know where the imaged volume ends; "
                "pass `volume_shape` (e.g. `CellVideo.volume_shape`), or drop the flag"
            )
            raise ValueError(message)
        return BoundaryPrior(volume_shape=volume_shape)


_Builder = Callable[[LinkerConfig, Spacing, EdgeAffinity | None, LinkerParts], Linker]

# Which linkers actually READ a learned affinity. The rest take the argument and drop it, so a mounted
# affinity paired with them is computed (~a third of inference) and thrown away, and an affinity_bonus set
# on them is a knob that cannot move anything. Both were silent until this set made the coupling checkable.
_AFFINITY_READERS = frozenset({"assignment", "flow"})

# Which linkers price a track boundary per node. Only the flow linker charges appearance and disappearance at all,
# so it is the only one with a price the boundary prior can modulate.
_BOUNDARY_PRIOR_READERS = frozenset({"flow"})

_BUILDERS: dict[str, _Builder] = {
    "assignment": lambda config, spacing, affinity, parts: AssignmentLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        disappearance_cost=config.disappearance_cost,
        agreement=parts.agreement,
        mutual=parts.mutual,
        ranker=parts.ranker,
    ),
    "nn": lambda config, spacing, affinity, parts: NearestNeighbourLinker(
        spacing=spacing, max_distance_um=config.gate_um
    ),
    "motion": lambda config, spacing, affinity, parts: MotionHungarianLinker(
        spacing=spacing, tight_gate_um=config.tight_um, loose_gate_um=config.gate_um
    ),
    "ilp": lambda config, spacing, affinity, parts: ILPLinker(spacing=spacing, max_distance_um=config.gate_um),
    "division": lambda config, spacing, affinity, parts: DivisionAwareLinker(
        spacing=spacing, max_distance_um=config.gate_um, division_distance_um=config.division_um
    ),
    # The global min-cost-flow linker reads the same gate, affinity and bonus as `assignment`; its extra lever is
    # the track-boundary cost, which it draws from `disappearance_cost` (the appearance+disappearance charged per
    # track). At `disappearance_cost = 0` it reduces to `assignment` — that is the A/B baseline for the sweep.
    "flow": lambda config, spacing, affinity, parts: FlowLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        boundary_cost=config.disappearance_cost,
        agreement=parts.agreement,
        mutual=parts.mutual,
        ranker=parts.ranker,
        boundary=parts.boundary,
    ),
}
