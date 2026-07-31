"""Building a linker from a name and a handful of gate radii — the one place that knows the whole family.

Every linker satisfies the same `Linker` contract (nodes in, edges out), so a caller that wants to *choose*
one at runtime — an eval sweep, a submission config, a CLI flag — should name it, not construct it. This is
that seam: a config object carries the union of gate parameters the family speaks, and `build` dispatches
the name to the concrete linker, passing only the radii that one uses. Adding a linker means adding one row
to the table, not another `if` at every call site.
"""

from dataclasses import dataclass
from typing import Callable

from celltrack.division_linking import DivisionAwareLinker
from celltrack.ilp_linking import ILPLinker
from celltrack.linking import Linker, NearestNeighbourLinker
from celltrack.motion_linking import MotionHungarianLinker
from core.geometry import Spacing


@dataclass(frozen=True)
class LinkerConfig:
    """A linker choice and the gate radii the family draws from, resolved to a concrete linker by `build`."""

    name: str = "nn"
    gate_um: float = 10.0
    tight_um: float = 6.0
    division_um: float = 8.0

    def build(self, spacing: Spacing) -> Linker:
        """The concrete linker named by this config, wired with the gates it uses."""
        return _BUILDERS[self.name](self, spacing)


_BUILDERS: dict[str, Callable[["LinkerConfig", Spacing], Linker]] = {
    "nn": lambda config, spacing: NearestNeighbourLinker(spacing=spacing, max_distance_um=config.gate_um),
    "motion": lambda config, spacing: MotionHungarianLinker(
        spacing=spacing, tight_gate_um=config.tight_um, loose_gate_um=config.gate_um
    ),
    "ilp": lambda config, spacing: ILPLinker(spacing=spacing, max_distance_um=config.gate_um),
    "division": lambda config, spacing: DivisionAwareLinker(
        spacing=spacing, max_distance_um=config.gate_um, division_distance_um=config.division_um
    ),
}

LINKER_NAMES = tuple(_BUILDERS)
