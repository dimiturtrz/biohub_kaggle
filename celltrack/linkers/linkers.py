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

    @property
    def effective_bonus(self) -> float:
        """The affinity bonus, self-balancing to 2·gate (the dimensional P-vs-distance coupling) when unset."""
        return self.affinity_bonus if self.affinity_bonus is not None else _BONUS_GATE_RATIO * self.gate_um

    @model_validator(mode="after")
    def _bonus_is_readable(self) -> "LinkerConfig":
        """Refuse an affinity_bonus on a linker that cannot read an affinity — the knob could not move anything.

        Pairing a bonus with `motion` or `ilp` is not a harmless default: it is an instruction the pipeline
        would accept and silently drop, and a sweep over it would report a flat curve that means nothing.
        """
        if self.affinity_bonus is not None and self.name not in _AFFINITY_READERS:
            message = (
                f"linker {self.name!r} does not read an edge affinity, so affinity_bonus="
                f"{self.affinity_bonus} cannot affect it; choose one of {sorted(_AFFINITY_READERS)} or drop the bonus"
            )
            raise ValueError(message)
        return self

    @field_validator("name")
    @classmethod
    def _known_name(cls, name: str) -> str:
        if name not in LINKER_NAMES:
            raise ValueError(f"unknown linker {name!r}; choose from {LINKER_NAMES}")
        return name

    def build(self, spacing: Spacing, affinity: EdgeAffinity | None = None) -> Linker:
        """The concrete linker named by this config, wired with the gates it uses (and the affinity if it reads one).

        Ignoring a mounted affinity is a legitimate choice (run `motion` and disregard the learned head), so it
        warns rather than raises — but it warns, because the affinity was already COMPUTED to be discarded.
        """
        if affinity is not None and self.name not in _AFFINITY_READERS:
            logger.warning("linker %r ignores the mounted edge affinity — it was computed and discarded", self.name)
        return _BUILDERS[self.name](self, spacing, affinity)


_Builder = Callable[[LinkerConfig, Spacing, EdgeAffinity | None], Linker]

# Which linkers actually READ a learned affinity. The rest take the argument and drop it, so a mounted
# affinity paired with them is computed (~a third of inference) and thrown away, and an affinity_bonus set
# on them is a knob that cannot move anything. Both were silent until this set made the coupling checkable.
_AFFINITY_READERS = frozenset({"assignment", "flow"})

_BUILDERS: dict[str, _Builder] = {
    "assignment": lambda config, spacing, affinity: AssignmentLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        disappearance_cost=config.disappearance_cost,
    ),
    "nn": lambda config, spacing, affinity: NearestNeighbourLinker(spacing=spacing, max_distance_um=config.gate_um),
    "motion": lambda config, spacing, affinity: MotionHungarianLinker(
        spacing=spacing, tight_gate_um=config.tight_um, loose_gate_um=config.gate_um
    ),
    "ilp": lambda config, spacing, affinity: ILPLinker(spacing=spacing, max_distance_um=config.gate_um),
    "division": lambda config, spacing, affinity: DivisionAwareLinker(
        spacing=spacing, max_distance_um=config.gate_um, division_distance_um=config.division_um
    ),
    # The global min-cost-flow linker reads the same gate, affinity and bonus as `assignment`; its extra lever is
    # the track-boundary cost, which it draws from `disappearance_cost` (the appearance+disappearance charged per
    # track). At `disappearance_cost = 0` it reduces to `assignment` — that is the A/B baseline for the sweep.
    "flow": lambda config, spacing, affinity: FlowLinker(
        spacing=spacing,
        max_distance_um=config.gate_um,
        affinity=affinity,
        affinity_bonus=config.effective_bonus,
        boundary_cost=config.disappearance_cost,
    ),
}
