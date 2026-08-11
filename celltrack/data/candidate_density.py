"""How CROWDED a source's choice actually is, per corpus — the number that decides whether the synthetic
corpus can buy anything at all.

The argument for training on fully-labelled pairs is that the candidate set becomes complete: the crowded
near-neighbours enter the `s x u` edge matrix as true negatives instead of sitting outside it as unannotated
detections. That argument has a precondition nobody had measured. If a synthetic frame's cells are simply
FURTHER APART than the dense movie's detections, then the crowding that produces our mislinks is not present
in the corpus at all, complete labelling buys nothing, and the whole idea is dead before it is run.

So this measures one quantity, three ways, in the SAME units the linker decides in: how many candidates of
frame `t + 1` sit inside the linker's gate (`LinkerConfig.gate_um`) of a source at `t`.

- SYNTHETIC — over ground-truth nodes, which for that corpus is every cell there is. Complete by construction.
- REAL / ANNOTATED — over the competition ground truth, which is the candidate set training sees TODAY.
- REAL / DETECTED — over the shipped detector's own detections on a movie, which is the candidate set the
  tracker actually chooses among, and therefore the crowding the mislinks happen in.

The gap between the last two IS the defect being chased; the gap between the first and the last is whether
the synthetic corpus reproduces it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial import KDTree

from celltrack.data.synthetic_pairs import POOLED_BY
from core.data.tracks import TrackGraph
from core.geometry import Spacing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateDensity:
    """One corpus's in-gate candidate counts, pooled over every source of every consecutive-frame pair."""

    @staticmethod
    def synthetic_spacing(sequence_path: Path) -> Spacing:
        """The RAW-grid spacing of a synthetic sequence: its pooled voxel size, un-pooled back in y and x.

        The corpus records `voxel_um_pooled` because its volumes are pooled, but its NODES stay in the raw grid,
        and this module measures over nodes. Dividing the in-plane size by the pooling is what puts the two
        corpora's distances in the same micrometres — the coordinate trap, in its measuring-tape form.
        """
        pooled = np.asarray(np.load(sequence_path)["voxel_um_pooled"], dtype=np.float64)
        z, y, x = (pooled / np.asarray(POOLED_BY, dtype=np.float64)).tolist()
        return Spacing(z=z, y=y, x=x)

    counts: Int[np.ndarray, "s"]
    sources_per_frame: float
    candidates_per_frame: float

    @classmethod
    def of(cls, graphs: list[TrackGraph], spacing: Spacing, gate_um: float) -> "CandidateDensity":
        """Pool every `t -> t + 1` gap of every graph: per source, how many targets sit within the gate."""
        counts: list[Int[np.ndarray, "s"]] = []
        sources, candidates, gaps = 0, 0, 0
        for graph in graphs:
            timepoints = graph.timepoints()
            positions = spacing.to_micrometres(graph.positions())
            for timepoint in np.unique(timepoints).tolist():
                source = positions[timepoints == timepoint]
                target = positions[timepoints == timepoint + 1]
                if not (len(source) and len(target)):
                    continue
                counts.append(cls._in_gate(source, target, gate_um))
                sources, candidates, gaps = sources + len(source), candidates + len(target), gaps + 1
        pooled = np.concatenate(counts) if counts else np.zeros(0, dtype=np.int64)
        divisor = max(gaps, 1)
        return cls(counts=pooled, sources_per_frame=sources / divisor, candidates_per_frame=candidates / divisor)

    @staticmethod
    def _in_gate(
        source: Float[np.ndarray, "s 3"], target: Float[np.ndarray, "u 3"], gate_um: float
    ) -> Int[np.ndarray, "s"]:
        """Per source, how many of the next frame's cells lie within `gate_um` micrometres of it."""
        tree = KDTree(target)
        return np.array([len(hits) for hits in tree.query_ball_point(source, gate_um)], dtype=np.int64)

    def mean(self) -> float:
        """The headline: mean candidates inside the gate per source."""
        return float(self.counts.mean()) if len(self.counts) else float("nan")

    def summary(self) -> dict[str, float]:
        """Mean, median and the crowded tail, plus the raw per-frame counts the mean has to be read against."""
        if not len(self.counts):
            return {"sources": 0.0}
        return {
            "sources": float(len(self.counts)),
            "sources_per_frame": self.sources_per_frame,
            "candidates_per_frame": self.candidates_per_frame,
            "mean_in_gate": self.mean(),
            "median_in_gate": float(np.median(self.counts)),
            "p90_in_gate": float(np.percentile(self.counts, 90)),
            "share_with_a_rival": float((self.counts > 1).mean()),
        }

    def report(self, name: str) -> None:
        """One population's row — the mean the argument turns on, with the spread it has to be read against."""
        summary = self.summary()
        logger.info(
            "%-22s | sources %6.0f | %6.1f sources/frame -> %6.1f candidates/frame | in-gate mean %5.2f "
            "median %4.1f p90 %4.1f | with a rival %5.1f%%",
            name,
            summary["sources"],
            summary["sources_per_frame"],
            summary["candidates_per_frame"],
            summary["mean_in_gate"],
            summary["median_in_gate"],
            summary["p90_in_gate"],
            100.0 * summary["share_with_a_rival"],
        )
