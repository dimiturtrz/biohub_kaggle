"""Building a linker from a name and a handful of gate radii — the one place that knows the whole family.

Every linker satisfies the same `Linker` contract (nodes in, edges out), so a caller that wants to *choose*
one at runtime — an eval sweep, a submission config, a CLI flag, a tracked `RunConfig` field — should name
it, not construct it. This is that seam: a validated config carries the union of gate parameters the family
speaks, and `build` dispatches the name to the concrete linker, passing only the radii that one uses. Adding
a linker means adding one row to the table, not another `if` at every call site.

The config is pydantic (not a plain dataclass) because it crosses a trust boundary — a `--set linker.name=`
override or a loaded config.json is user input; an unknown name or a non-positive radius is rejected AT
construction with a clear error, not deep inside `build`.
"""

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from celltrack.division_linking import DivisionAwareLinker
from celltrack.ilp_linking import ILPLinker
from celltrack.linking import Linker, NearestNeighbourLinker
from celltrack.motion_linking import MotionHungarianLinker
from core.geometry import Spacing

LINKER_NAMES = ("nn", "motion", "ilp", "division")


class LinkerConfig(BaseModel):
    """A linker choice and the gate radii the family draws from, resolved to a concrete linker by `build`.
    Immutable + validated: `name` must be one the family knows, and every radius is positive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "nn"
    gate_um: float = Field(10.0, gt=0)
    tight_um: float = Field(6.0, gt=0)
    division_um: float = Field(8.0, gt=0)

    @field_validator("name")
    @classmethod
    def _known_name(cls, name: str) -> str:
        if name not in LINKER_NAMES:
            raise ValueError(f"unknown linker {name!r}; choose from {LINKER_NAMES}")
        return name

    def build(self, spacing: Spacing) -> Linker:
        """The concrete linker named by this config, wired with the gates it uses."""
        return _BUILDERS[self.name](self, spacing)


_BUILDERS: dict[str, Callable[[LinkerConfig, Spacing], Linker]] = {
    "nn": lambda config, spacing: NearestNeighbourLinker(spacing=spacing, max_distance_um=config.gate_um),
    "motion": lambda config, spacing: MotionHungarianLinker(
        spacing=spacing, tight_gate_um=config.tight_um, loose_gate_um=config.gate_um
    ),
    "ilp": lambda config, spacing: ILPLinker(spacing=spacing, max_distance_um=config.gate_um),
    "division": lambda config, spacing: DivisionAwareLinker(
        spacing=spacing, max_distance_um=config.gate_um, division_distance_um=config.division_um
    ),
}
