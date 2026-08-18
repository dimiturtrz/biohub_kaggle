"""Where annotated tracks actually begin and end — the stake in activating the boundary prior.

The shipped flow linker charges one flat price for every track appearance and disappearance. `BoundaryPrior`
would discount the boundaries the observation window already explains: a birth in the first observed frame
(the cell predates the movie) or within one gate step of a volume face (it walked in from outside the field
of view). Activating the prior only CHANGES those explained boundaries — full price to free — while the
interior, mid-movie boundaries stay charged. So the stake of activation is the EXPLAINED FRACTION of real
track boundaries (how many the prior reclassifies as cheap), and its safety is whether the interior
remainder is genuinely spurious rather than real births the prior would wrongly penalise.

This measures both over the annotated tracks, pure CPU and metadata-only — the volume shape reads a zarr
header, not a frame. A birth is a node with no parent (`in_degree == 0`); a death is one with no child
(`out_degree == 0`). A division daughter has a parent (`in_degree == 1` under a two-child mother) and so is
a continuation, not a birth; a dividing mother has children and so is not a death. Both are correctly
outside the census, which is exactly the flow graph's own reading: only an `in_degree == 0` node draws an
appearance arc, only an `out_degree == 0` node a disappearance arc.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from celltrack.eval.proxy import CV_MOVIES, TestMovieProxy
from celltrack.linkers.boundary_prior import BoundaryPrior
from celltrack.operating_point import TrackerConfig
from core.data.tracks import Adjacency, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# A boundary is EXPLAINED when the prior's discount factor drops below this — the factors are 0 (explained)
# or 1 (interior), so any threshold in the open interval separates them; a half keeps the read robust to a
# future graded factor.
_EXPLAINED_BELOW = 0.5


@dataclass(frozen=True)
class MovieBoundaries:
    """One movie's track-boundary census: how many births and deaths the observation window explains.

    `explained` is the activation stake — the share of real boundaries `BoundaryPrior` turns from the flat
    full charge to free. `first_edge` is the temporal half of that (first frame for births, last for deaths);
    the rest of the explained share is spatial (a volume face). `interior` is the remainder the prior still
    charges full — the boundaries it asserts are implausible, which are safe to penalise only if they are
    genuinely spurious.
    """

    stem: str
    births: int
    deaths: int
    birth_explained: float
    birth_first_frame: float
    death_explained: float
    death_last_frame: float

    @classmethod
    def of(cls, stem: str, graph: TrackGraph, spacing: Spacing, volume_shape: tuple[int, int, int]) -> Self:
        """Classify every birth and death of one annotated movie against the shipped gate's boundary prior."""
        adjacency = Adjacency.of(graph)
        positions_um = spacing.to_micrometres(graph.positions())
        timepoints = graph.timepoints()
        margin_um = TrackerConfig.shipped().linker.gate_um
        factors = BoundaryPrior(volume_shape).factors(spacing, margin_um, positions_um, timepoints)

        births = np.flatnonzero(adjacency.in_degrees == 0)
        deaths = np.flatnonzero(adjacency.out_degrees == 0)
        return cls(
            stem=stem,
            births=int(births.size),
            deaths=int(deaths.size),
            birth_explained=cls._share(factors.appearance[births] < _EXPLAINED_BELOW),
            birth_first_frame=cls._share(timepoints[births] == timepoints.min()),
            death_explained=cls._share(factors.disappearance[deaths] < _EXPLAINED_BELOW),
            death_last_frame=cls._share(timepoints[deaths] == timepoints.max()),
        )

    @staticmethod
    def _share(mask: "np.ndarray") -> float:
        """The fraction of a boundary set a mask selects — nan when the set is empty, so an absent movie reads so."""
        return float(mask.mean()) if mask.size else float("nan")

    def report(self) -> None:
        """Log the census, stake first: how much of each boundary set the prior reclassifies as cheap."""
        logger.info(
            "  %-16s births %4d  explained %5.1f%% (first-frame %4.1f%%)  interior %5.1f%%",
            self.stem,
            self.births,
            100 * self.birth_explained,
            100 * self.birth_first_frame,
            100 * (1 - self.birth_explained),
        )
        logger.info(
            "  %-16s deaths %4d  explained %5.1f%% (last-frame  %4.1f%%)  interior %5.1f%%",
            "",
            self.deaths,
            100 * self.death_explained,
            100 * self.death_last_frame,
            100 * (1 - self.death_explained),
        )


def main() -> None:
    """Census the annotated track boundaries — the stake in activating the flow linker's boundary prior."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Where annotated tracks begin and end vs the imaged volume.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    args = parser.parse_args()

    root = DataRoot.from_config(args.config)
    proxy = TestMovieProxy.load(root, CV_MOVIES)
    logger.info("track-boundary census over %d annotated movies (explained = the prior frees it)", len(proxy.paths))
    for path, truth in zip(proxy.paths, proxy.truths, strict=True):
        volume_shape = CellVideo.from_ome_zarr(path).volume_shape
        MovieBoundaries.of(path.stem, truth.graph, proxy.spacing, volume_shape).report()


if __name__ == "__main__":
    main()
