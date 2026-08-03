"""The post-detection graph-stage family — one validated spec per slot, resolved to a concrete stage.

Everything a pipeline does after it has nodes is the same shape: a graph goes in, a graph with more (or
fewer) edges and cleaner coordinates comes out. `GraphStage` is that contract, and the existing passes —
the linkers, the gap-closer, the division recovery, the short-track filter, the line-fit smoother — already
speak it (a linker through a thin adapter, since it phrases the same graph→graph step as `link`). So a
pipeline is just an *ordered* list of `GraphStage`s, and the order is part of the configuration: gap-close
before divisions before the short-track filter is a different pipeline from the reverse.

This module is to stages what `linkers.py` is to linkers: the one place that knows the whole family. A
`StageSpec` is a validated, discriminated pydantic union — each variant names one slot, carries only the
gates that slot uses, and `build`s the concrete stage. Adding a stage is one variant here, not another
branch at every call site; choosing a pipeline is naming specs in order, and an unknown slot or a
non-positive gate is rejected at construction, not deep inside a run.
"""

from dataclasses import dataclass
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from celltrack.center_prior import CenterPriorVeto, CenterVetoConfig
from celltrack.division_recovery import DivisionRecovery
from celltrack.gap_closer import GapCloser
from celltrack.linefit_smoother import LinefitSmoother
from celltrack.linkers import LinkerConfig
from celltrack.linking import Linker
from celltrack.short_track_filter import ShortTrackFilter
from celltrack.topology_repair import TopologyRepair
from core.data.tracks import TrackGraph
from core.geometry import Spacing


class GraphStage(Protocol):
    """One post-detection pass: a track graph in, a track graph out. Every slot's impl satisfies this."""

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Return the graph after this stage's transformation (edges added/removed, coordinates cleaned)."""
        ...


@dataclass(frozen=True)
class LinkerStage:
    """Adapts a `Linker` (which phrases the step as `link`) to the `GraphStage` contract, so linking is
    just the first stage in the ordered list rather than a special case the pipeline must know about."""

    linker: Linker

    def transform(self, graph: TrackGraph) -> TrackGraph:
        """Wire the graph's nodes into tracks — the linker's `link`, under the stage verb."""
        return self.linker.link(graph)


_Spec = ConfigDict(frozen=True, extra="forbid")


class LinkerSpec(BaseModel):
    """Select a linker for the first stage, delegating its gate radii to the linker family's own config."""

    model_config = _Spec

    kind: Literal["linker"] = "linker"
    linker: LinkerConfig = LinkerConfig()

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        return LinkerStage(self.linker.build(spacing))


class GapSpec(BaseModel):
    """Reconnect one-frame detection dropouts by reusing an existing isolated node near the midpoint."""

    model_config = _Spec

    kind: Literal["gap"] = "gap"
    gate_um: float = Field(5.8, gt=0)
    reuse_um: float = Field(3.2, gt=0)
    max_added_fraction: float = Field(0.05, ge=0)
    center_veto: CenterVetoConfig | None = None

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        confirmer = self.center_veto.resolve(veto) if self.center_veto is not None else None
        return GapCloser(spacing, self.gate_um, self.reuse_um, self.max_added_fraction, confirmer)


class DivisionSpec(BaseModel):
    """Recover the second daughter of a division a one-to-one linker cannot represent, three gates guarding it."""

    model_config = _Spec

    kind: Literal["division"] = "division"
    parent_gate_um: float = Field(4.7, gt=0)
    sister_gate_um: float = Field(7.2, gt=0)
    max_added_fraction: float = Field(0.004, ge=0)
    existing_child_gate_um: float = Field(7.8, gt=0)
    frame_fraction_cap: float = Field(0.008, ge=0)
    center_veto: CenterVetoConfig | None = None

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        return DivisionRecovery(
            spacing,
            self.parent_gate_um,
            self.sister_gate_um,
            self.max_added_fraction,
            self.existing_child_gate_um,
            self.frame_fraction_cap,
            self.center_veto.resolve(veto) if self.center_veto is not None else None,
        )


class ShortTrackSpec(BaseModel):
    """Drop lineage fragments below a minimum length, keeping any that contain a division."""

    model_config = _Spec

    kind: Literal["short_track"] = "short_track"
    min_length: int = Field(6, ge=1)

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        return ShortTrackFilter(self.min_length)


class SmoothSpec(BaseModel):
    """De-jitter interior-node coordinates toward the line between neighbours; topology untouched."""

    model_config = _Spec

    kind: Literal["smooth"] = "smooth"
    strength: float = Field(0.8, ge=0, le=1)

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        return LinefitSmoother(self.strength)


class TopologySpec(BaseModel):
    """Restore consecutive-frame, in-gate, single-parent edges and drop isolated nodes — pure-geometry cleanups."""

    model_config = _Spec

    kind: Literal["topology"] = "topology"
    edge_max_um: float = Field(14.0, gt=0)

    def build(self, spacing: Spacing, veto: CenterPriorVeto | None = None) -> GraphStage:
        return TopologyRepair(spacing, self.edge_max_um)


StageSpec = Annotated[
    LinkerSpec | GapSpec | DivisionSpec | ShortTrackSpec | SmoothSpec | TopologySpec,
    Field(discriminator="kind"),
]
