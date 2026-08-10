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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from celltrack.affinity import EdgeAffinity
from celltrack.linkers.agreement_gating import AgreementGate
from celltrack.linkers.assignment_linking import AssignmentLinker
from celltrack.linkers.division_linking import DivisionAwareLinker
from celltrack.linkers.flow_linking import FlowLinker
from celltrack.linkers.ilp_linking import ILPLinker
from celltrack.linkers.linking import Linker, NearestNeighbourLinker
from celltrack.linkers.motion_linking import MotionHungarianLinker
from core.geometry import Spacing

logger = logging.getLogger(__name__)

LINKER_NAMES = ("assignment", "nn", "motion", "ilp", "division", "flow")

# The affinity bonus self-balances to this multiple of the gate. `cost = distance - bonus·P` trades a [0,1]
# probability against a micrometre distance, so the bonus carries gate-scale units; at 2·gate a P=0.5 edge
# earns a one-gate-width discount. The swept proxy optimum at gate 10 (=20, bead 88x) IS this ratio — deriving
# it keeps P-weighting constant when the gate moves instead of silently under-weighting P at a wider gate.
_BONUS_GATE_RATIO = 2.0


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

    @property
    def effective_bonus(self) -> float:
        """The affinity bonus, self-balancing to 2·gate (the dimensional P-vs-distance coupling) when unset."""
        return self.affinity_bonus if self.affinity_bonus is not None else _BONUS_GATE_RATIO * self.gate_um

    @model_validator(mode="after")
    def _affinity_knobs_are_readable(self) -> "LinkerConfig":
        """Refuse an affinity knob on a linker that cannot read an affinity — it could not move anything.

        Pairing a bonus or an agreement floor with `motion` or `ilp` is not a harmless default: it is an
        instruction the pipeline would accept and silently drop, and a sweep over it would report a flat
        curve that means nothing.
        """
        if self.name in _AFFINITY_READERS:
            return self
        for knob, value in (("affinity_bonus", self.affinity_bonus), ("agreement_floor", self.agreement_floor)):
            if value is not None:
                message = (
                    f"linker {self.name!r} does not read an edge affinity, so {knob}={value} cannot affect it; "
                    f"choose one of {sorted(_AFFINITY_READERS)} or drop the {knob}"
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
        self, spacing: Spacing, affinity: EdgeAffinity | None = None, mutual: EdgeAffinity | None = None
    ) -> Linker:
        """The concrete linker named by this config, wired with the gates it uses (and the affinity if it reads one).

        Ignoring a mounted affinity is a legitimate choice (run `motion` and disregard the learned head), so it
        warns rather than raises — but it warns, because the affinity was already COMPUTED to be discarded.
        `mutual` carries the bidirectionally fused probabilities the agreement floor reads; it is needed only
        when a floor is set, and the cost never reads it.
        """
        if affinity is not None and self.name not in _AFFINITY_READERS:
            logger.warning("linker %r ignores the mounted edge affinity — it was computed and discarded", self.name)
        return _BUILDERS[self.name](self, spacing, affinity, self._gate(mutual))

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


_Builder = Callable[[LinkerConfig, Spacing, EdgeAffinity | None, AgreementGate | None], Linker]

# Which linkers actually READ a learned affinity. The rest take the argument and drop it, so a mounted
# affinity paired with them is computed (~a third of inference) and thrown away, and an affinity_bonus set
# on them is a knob that cannot move anything. Both were silent until this set made the coupling checkable.
_AFFINITY_READERS = frozenset({"assignment", "flow"})

_BUILDERS: dict[str, _Builder] = {
    "assignment": lambda config, spacing, affinity, gate: AssignmentLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        disappearance_cost=config.disappearance_cost,
        agreement=gate,
    ),
    "nn": lambda config, spacing, affinity, gate: NearestNeighbourLinker(
        spacing=spacing, max_distance_um=config.gate_um
    ),
    "motion": lambda config, spacing, affinity, gate: MotionHungarianLinker(
        spacing=spacing, tight_gate_um=config.tight_um, loose_gate_um=config.gate_um
    ),
    "ilp": lambda config, spacing, affinity, gate: ILPLinker(spacing=spacing, max_distance_um=config.gate_um),
    "division": lambda config, spacing, affinity, gate: DivisionAwareLinker(
        spacing=spacing, max_distance_um=config.gate_um, division_distance_um=config.division_um
    ),
    # The global min-cost-flow linker reads the same gate, affinity and bonus as `assignment`; its extra lever is
    # the track-boundary cost, which it draws from `disappearance_cost` (the appearance+disappearance charged per
    # track). At `disappearance_cost = 0` it reduces to `assignment` — that is the A/B baseline for the sweep.
    "flow": lambda config, spacing, affinity, gate: FlowLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        boundary_cost=config.disappearance_cost,
        agreement=gate,
    ),
}
