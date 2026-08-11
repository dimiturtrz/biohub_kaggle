"""Building a linker from a name and a handful of gate radii — the one place that knows the whole family.

Every linker satisfies the same `Linker` contract (nodes in, edges out), so a caller that wants to *choose*
one at runtime — an eval sweep, a submission config, a CLI flag, a tracked `RunConfig` field — should name
it, not construct it. This is that seam: a validated config carries the union of gate parameters the family
speaks, and `build` dispatches the name to the concrete linker, passing only the radii that one uses. Adding
a linker means adding one row to the table, not another `if` at every call site.

`build` takes the per-video edge `affinity` because the shipped `assignment` linker scores each candidate edge
by a learned `P(s->t)`; the purely geometric linkers ignore it. Which linker reads WHICH learned knob is a
per-knob fact, not a per-linker one (`_KNOB_READERS`): `motion` prices the forward affinity and the re-ranker
inside a geometric gate, yet has no admission gate and no mutual term for the fused probability to enter.

The config is pydantic (not a plain dataclass)
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
from celltrack.linkers.boundary_prior import BoundaryFactorSource, BoundaryPrior
from celltrack.linkers.division_linking import DivisionAwareLinker
from celltrack.linkers.evidence_prior import EvidencePrior
from celltrack.linkers.evidence_ramp import EvidenceRamp
from celltrack.linkers.flow_linking import FlowLinker
from celltrack.linkers.ilp_linking import ILPLinker
from celltrack.linkers.linking import Linker, NearestNeighbourLinker
from celltrack.linkers.motion_linking import MotionHungarianLinker
from celltrack.linkers.motion_prediction import MotionPrediction
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
    boundary: BoundaryFactorSource | None = None
    prediction: MotionPrediction | None = None
    ramp: EvidenceRamp | None = None


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
    # None charges a track's START the same as its END — the shipped, symmetric behaviour. Set it to price the
    # two directions apart: beginning mid-movie has ordinary causes (a cell enters the volume, a division makes a
    # daughter, the annotation starts) where ending mid-movie is rarer, so one price calls both equally
    # implausible. Flow-only: it is the sole linker whose network charges the two boundaries on separate arcs.
    appearance_cost: float | None = Field(None, ge=0)
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
    # Price a birth by the EDGE HEAD's own verdict instead of by geometry: the share of a detection's parent
    # probability held by admissible sources. Measured on the dense movie that share is mean 0.78, median 0.92
    # and p5 0.09, so a twentieth of detections carry almost no evidence of any admissible parent — which is
    # what an appearance IS. Budget-preserving (factors mean 1 within a frame), so it moves WHERE the appearance
    # charge falls and never how much of it there is; the flat cost is its own control. Off ships unchanged.
    appearance_evidence: bool = False
    # OFF = the shipped raw-distance cost, byte for byte. ON prices each candidate by the MOTION-PREDICTED
    # distance instead (`MotionPrediction`): how far the target is from where the source was already heading,
    # with the step taken from the previous gap's affinity and damped by the motion linker's argued half step.
    # The GATE is unchanged — it keeps reading the raw separation, because admissibility is physics and a
    # prediction is an estimate. There is no strength knob: the damping is argued once in `motion_linking`, so
    # this is an on/off claim about which geometry the cost is written in, not a tuning surface. Setting it makes
    # `build` require the edge affinity the velocity is derived from.
    motion_distance: bool = False
    # None = OFF, so the shipped cost is byte-identical; `0.0` is the same cost through the ramp's own arithmetic
    # (`ramp ≡ 1`), which is what makes it the control arm of its own sweep. Set, the forward affinity's weight
    # becomes `1 + evidence_ramp * (d / mean_gated_distance - 1)`, clamped at zero (`EvidenceRamp`): a
    # dimensionless dose, normalised by the gate's own admitted pairs, that spends the SAME evidence budget more
    # on far candidates — where the head's AUC holds at 0.95-0.97 while distance's collapses to 0.64-0.71 — and
    # less on near ones, where 96% of in-gate candidates are already the true successor. No default is chosen
    # here: how fast evidence should overtake proximity is a claim about our data, so a sweep decides it.
    evidence_ramp: float | None = Field(None, ge=0)
    # Whether the priced probability is CONDITIONED on the admissible parents rather than on every source in the
    # frame (`AdmissibleAffinity`). The scorer soft-maxes over all ~700 detections of frame t and the gate then
    # forbids all but ~4, so on the dense movie a mean 22% of each target's mass sits on parents the solver may
    # not pick — median leak 8%, but over 90% for the worst twentieth of targets. A uniform deflation would be
    # absorbed by `affinity_bonus`, which scales the same term; the UNEVENNESS cannot be, and it is worst on the
    # crowded targets where the mislinks live. Off ships the cost unchanged.
    admissible_affinity: bool = False

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
        Everything else (gate, floor, mutual bonus, the cost's geometry) is held, because the context graph only
        describes "the current best link set" if it was produced by the cost being re-ranked.
        """
        return self.model_copy(update={"ranker_bonus": None, "affinity_bonus": self.learned_budget})

    @model_validator(mode="after")
    def _affinity_knobs_are_readable(self) -> "LinkerConfig":
        """Refuse each learned-evidence knob on a linker whose cost has no term for THAT knob.

        Pairing a bonus or an agreement floor with a linker that cannot price it is not a harmless default: it
        is an instruction the pipeline would accept and silently drop, and a sweep over it would report a flat
        curve that means nothing. Support is per-knob, not per-linker — `motion` prices a forward affinity and
        the re-ranker's term but has neither an admission gate nor a mutual term — so each knob is checked
        against its OWN reader set and the refusal names that set (`_KNOB_READERS`).
        """
        for knob, value, readers in self._knobs():
            # A knob at its OFF value asks for nothing, so it cannot be dropped silently and needs no reader.
            # Off is `None` for a knob with a magnitude and `False` for one that is a claim on or off.
            if value not in (None, False) and self.name not in readers:
                message = (
                    f"linker {self.name!r} does not read {knob}, so {knob}={value} cannot affect it; "
                    f"choose one of {sorted(readers)} or drop the {knob}"
                )
                raise ValueError(message)
        return self

    def _knobs(self) -> tuple[tuple[str, float | bool | None, frozenset[str]], ...]:
        """Each learned-evidence knob beside its value and the linkers whose cost actually reads it."""
        return tuple((knob, getattr(self, knob), readers) for knob, readers in _KNOB_READERS.items())

    @model_validator(mode="after")
    def _boundary_prior_is_readable(self) -> "LinkerConfig":
        """Refuse the boundary prior on a linker that charges no track boundary — same reason as the affinity knobs.

        Only the flow linker prices appearance and disappearance, so only it has a price for the prior to modulate;
        anywhere else the flag would be accepted, silently dropped, and sweep as a flat curve.
        """
        if self.appearance_evidence and self.boundary_prior:
            message = (
                "boundary_prior and appearance_evidence both price the SAME track-boundary charge, so enabling "
                "both would silently drop one; choose the geometric prior or the head's evidence"
            )
            raise ValueError(message)
        if (self.boundary_prior or self.appearance_evidence) and self.name not in _BOUNDARY_PRIOR_READERS:
            message = (
                f"linker {self.name!r} charges no track-boundary cost, so a boundary prior cannot affect it; "
                f"choose one of {sorted(_BOUNDARY_PRIOR_READERS)} or drop the flag"
            )
            raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _motion_distance_is_readable(self) -> "LinkerConfig":
        """Refuse the motion-predicted distance on a linker that does not price one — same reason as the prior.

        `motion` already competes on a predicted distance unconditionally and the rest read the raw one, so only
        the flow linker has a cost this flag can rewrite; anywhere else it would be accepted, silently dropped,
        and sweep as a flat curve.
        """
        if self.motion_distance and self.name not in _MOTION_DISTANCE_READERS:
            message = (
                f"linker {self.name!r} does not price a motion-predicted distance, so motion_distance cannot "
                f"affect it; choose one of {sorted(_MOTION_DISTANCE_READERS)} or drop the flag"
            )
            raise ValueError(message)
        if self.appearance_cost is not None and self.name not in _APPEARANCE_READERS:
            message = (
                f"linker {self.name!r} charges one price for a track boundary, so appearance_cost cannot affect "
                f"it; choose one of {sorted(_APPEARANCE_READERS)} or drop the appearance_cost"
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
        if affinity is not None and self.name not in _BONUS_READERS:
            logger.warning("linker %r ignores the mounted edge affinity — it was computed and discarded", self.name)
        parts = LinkerParts(
            self._gate(mutual),
            self._bonus(mutual),
            self._ranker(ranker),
            self._boundary(volume_shape, affinity),
            self._prediction(affinity),
            self._ramp(),
        )
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

    def _ramp(self) -> EvidenceRamp | None:
        """The affinity term's distance-dependent weight — `None` when off, and needing nothing else to build.

        Unlike the other parts this consumes no extra scored quantity: the ramp is normalised by the separations
        the gate already computes, so there is no missing input to refuse. A strength of `0` still builds the
        term, because a control arm that runs the ramp's own arithmetic is what proves the sweep is unconfounded.
        """
        return None if self.evidence_ramp is None else EvidenceRamp(strength=self.evidence_ramp)

    def _boundary(
        self, volume_shape: tuple[int, int, int] | None, affinity: EdgeAffinity | None
    ) -> BoundaryFactorSource | None:
        """What prices this config's track boundaries — geometry, the head's evidence, or the flat charge.

        The two sources are alternatives because there is ONE price to modulate, and a config asking for both is
        refused up front rather than served whichever the code happens to check first.
        """
        if self.appearance_evidence:
            if affinity is None:
                message = (
                    "appearance_evidence prices a birth by the edge head's in-gate probability mass, so it "
                    "needs the mounted affinity; mount one, or drop the flag"
                )
                raise ValueError(message)
            return EvidencePrior(affinity=affinity)
        if not self.boundary_prior:
            return None
        if volume_shape is None:
            message = (
                "boundary_prior needs the video's (z, y, x) volume shape to know where the imaged volume ends; "
                "pass `volume_shape` (e.g. `CellVideo.volume_shape`), or drop the flag"
            )
            raise ValueError(message)
        return BoundaryPrior(volume_shape=volume_shape)

    def _prediction(self, affinity: EdgeAffinity | None) -> MotionPrediction | None:
        """The cost's motion-predicted geometry — `None` when off, an error without the affinity it reads.

        The predicted step is the previous gap's affinity-weighted expected displacement, so without an affinity
        every velocity would be exactly zero and the flag would silently reproduce the raw-distance cost — a knob
        that reports a flat sweep curve because it never actually ran, the same failure the other refusals prevent.
        """
        if not self.motion_distance:
            return None
        if affinity is None:
            message = (
                "motion_distance needs the edge affinity its velocity is derived from (the previous gap's "
                "source->target probabilities); pass `affinity`, or drop the flag"
            )
            raise ValueError(message)
        return MotionPrediction(affinity=affinity)


_Builder = Callable[[LinkerConfig, Spacing, EdgeAffinity | None, LinkerParts], Linker]

# Which linkers actually READ a learned affinity — the ones whose cost carries a `- affinity_bonus * P` term.
# The rest take the argument and drop it, so a mounted affinity paired with them is computed (~a third of
# inference) and thrown away, and an affinity_bonus set on them is a knob that cannot move anything. Both were
# silent until this set made the coupling checkable. `motion` belongs here: it has carried `affinity` and
# `affinity_bonus` since it was written (its gate stays pure geometry, the cost inside the gate is blended).
_BONUS_READERS = frozenset({"assignment", "flow", "motion"})

# Which linkers price the re-ranker's term. The same three, and for the same reason: `RankerBonus.discount`
# subtracts from whatever pair cost the linker already formed, so any linker with a per-pair cost can carry it.
_RANKER_READERS = frozenset({"assignment", "flow", "motion"})

# Which linkers can ADMIT candidates by the fused probability, and which price it as a cost term. Narrower than
# the two above: `motion` admits purely by its geometric tight/loose gates and its cost has no mutual term, so
# a floor or a mutual bonus on it would be accepted and silently dropped — the failure this validator exists
# to prevent. Support differs per knob, so the sets do too.
_AGREEMENT_READERS = frozenset({"assignment", "flow"})
_MUTUAL_READERS = frozenset({"assignment", "flow"})

# Which linkers can weight their forward-affinity term by distance. `motion` is absent: it admits through a
# tight/loose PAIR of gates and prices a predicted geometry, so there is no single admitted-candidate set whose
# mean separation the ramp could normalise by — the dimensionless scale is the whole device, so a ramp there
# would need a second, chosen constant. The two linkers with one gate and one raw distance carry it.
_RAMP_READERS = frozenset({"assignment", "flow"})

# The learned-evidence knobs, each against the readers of THAT knob — what `_affinity_knobs_are_readable` walks.
_KNOB_READERS: dict[str, frozenset[str]] = {
    "affinity_bonus": _BONUS_READERS,
    "agreement_floor": _AGREEMENT_READERS,
    "mutual_bonus": _MUTUAL_READERS,
    "ranker_bonus": _RANKER_READERS,
    "evidence_ramp": _RAMP_READERS,
    "admissible_affinity": _RAMP_READERS,
}

# Which linkers price a track boundary per node. Only the flow linker charges appearance and disappearance at all,
# so it is the only one with a price the boundary prior can modulate.
_BOUNDARY_PRIOR_READERS = frozenset({"flow"})

# Which linkers can be told which GEOMETRY to price a pair in. `motion` is absent on purpose: it is the linker
# that already predicts, unconditionally, so the flag would mean nothing there. Only `flow` has a raw-distance
# cost this can rewrite.
_MOTION_DISTANCE_READERS = frozenset({"flow"})

# Only the flow network charges a track's start and end on SEPARATE arcs (source->in, out->sink), so it is the
# only linker that can price them apart. `assignment` has a `disappearance_cost` but spends it on one skip arc.
_APPEARANCE_READERS = frozenset({"flow"})

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
        ramp=parts.ramp,
        admissible=config.admissible_affinity,
    ),
    "nn": lambda config, spacing, affinity, parts: NearestNeighbourLinker(
        spacing=spacing, max_distance_um=config.gate_um
    ),
    # The motion linker gates on raw geometry but competes pairs on a blended cost, so it reads the same
    # affinity, bonus and re-ranker term as `assignment` — it just has no agreement gate and no mutual term.
    "motion": lambda config, spacing, affinity, parts: MotionHungarianLinker(
        spacing=spacing,
        tight_gate_um=config.tight_um,
        loose_gate_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        ranker=parts.ranker,
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
        appearance_cost=config.appearance_cost,
        agreement=parts.agreement,
        mutual=parts.mutual,
        ranker=parts.ranker,
        boundary=parts.boundary,
        prediction=parts.prediction,
        ramp=parts.ramp,
        admissible=config.admissible_affinity,
    ),
}
