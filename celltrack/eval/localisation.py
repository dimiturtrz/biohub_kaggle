"""How far a matched detection sits from the annotated centre it matched — localisation, not recall.

Recall asks whether a detection landed inside the metric's 7 um matching radius. Linking asks a far tighter
question: the median annotated inter-frame step on the dense movie is ~1.7 um, so an error of a micrometre at
either endpoint is the size of the motion the linker is trying to measure. A step is the difference of two
independently localised positions, so it carries ~sqrt(2) times the per-node error. This module measures that
residual, and cuts it by whether the node is an endpoint of a MISLINKED annotated edge — a uniform error is a
background noise every candidate pays, while an error concentrated on the failures implicates the mechanism.

WHAT THE MEASUREMENT CAN AND CANNOT SEE. Both sides are quantised. Annotated centres are INTEGER
full-resolution voxel indices (the GEFF store keeps `t, z, y, x` as int64 — there is no sub-voxel annotation
to compare against), and detections are read out on the `(1, 4, 4)`-downsampled grid and multiplied back up,
so a y/x offset is a multiple of 0.40625 um and a z offset a multiple of 1.625 um. In z the two grids
COINCIDE — the downsample leaves z alone — so the z column can only ever report a whole number of 1.625 um
steps, and a sub-voxel z error is invisible to it. Only y and x carry information about the 4x read-out
quantisation, and even there the measured offset is the detection's error PLUS the annotation's own rounding.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float, Int

from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis, DenseFateDiagnosis, Fate
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.proxy_eval import ConfigOverride
from celltrack.tracker import TrackerConfig
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_LOWER_QUARTILE = 25.0
_UPPER_QUARTILE = 75.0
_MEDIAN = 50.0
_AXES = ("z", "y", "x")
# Both grids are integer voxel indices, so the half-block comparison is exact up to float division.
_ROUNDING_SLACK = 1e-9
# The metric's own matching radius: inside it a neighbour is a candidate the assignment could have taken
# instead, which is exactly the condition under which a measured offset may be a swap rather than an error.
_CROWDING_RADIUS_UM = 7.0


@dataclass(frozen=True)
class MatchingInverse:
    """For each annotated node, the predicted row matched to it, or `UNMATCHED` — the matching reversed.

    The per-timepoint assignment is one-to-one, so no annotated node is claimed twice and the inverse is well
    defined. It is what turns an annotated EDGE into the pair of detections that stood in for its endpoints.
    """

    rows: Int[np.ndarray, "g"]

    @classmethod
    def of(cls, matching: NodeMatching, truth_count: int) -> Self:
        """Invert a matching over `truth_count` annotated nodes."""
        inverse = np.full(truth_count, UNMATCHED, dtype=np.int64)
        matched = np.flatnonzero(matching.is_matched())
        inverse[matching.gt_rows[matched]] = matched
        return cls(rows=inverse)


@dataclass(frozen=True)
class OffsetSummary:
    """The distribution of a set of detection-to-annotation offsets, in micrometres.

    The 3D magnitude is what compares against the median step; the per-axis columns are reported SIGNED
    (median) as well as absolute, because a systematic shift — a read-out that lands on a block corner rather
    than its centre — is a bias that cancels inside a displacement, while a symmetric spread does not.
    """

    count: int
    median_um: float
    iqr_um: tuple[float, float]
    axis_median_abs_um: tuple[float, ...]
    axis_iqr_um: tuple[tuple[float, float], ...]
    axis_median_signed_um: tuple[float, ...]

    @classmethod
    def of(cls, offsets: Float[np.ndarray, "m 3"]) -> Self:
        """Summarise signed `(z, y, x)` micrometre offsets — magnitude first, then each axis on its own."""
        distances = np.linalg.norm(offsets, axis=1)
        return cls(
            count=len(offsets),
            median_um=cls._median(distances),
            iqr_um=cls._quartiles(distances),
            axis_median_abs_um=tuple(cls._median(np.abs(offsets[:, axis])) for axis in range(len(_AXES))),
            axis_iqr_um=tuple(cls._quartiles(np.abs(offsets[:, axis])) for axis in range(len(_AXES))),
            axis_median_signed_um=tuple(cls._median(offsets[:, axis]) for axis in range(len(_AXES))),
        )

    @staticmethod
    def _median(values: Float[np.ndarray, "m"]) -> float:
        """The median, or NaN on an empty set — an empty cut is reported as absent, never as zero."""
        return float(np.median(values)) if len(values) else float("nan")

    @staticmethod
    def _quartiles(values: Float[np.ndarray, "m"]) -> tuple[float, float]:
        """The 25th and 75th percentiles, or NaNs on an empty set."""
        if not len(values):
            return float("nan"), float("nan")
        low, high = np.percentile(values, [_LOWER_QUARTILE, _UPPER_QUARTILE])
        return float(low), float(high)


@dataclass(frozen=True)
class LocalisationOffsets:
    """Every matched detection's signed offset from the annotated centre it was paired with, in micrometres.

    The rows travel with the offsets so a subset can be taken by PREDICTION ROW — which is how the mislink cut
    is expressed, the mislinked edges being named in the prediction's row space by the inverted matching.
    """

    prediction_rows: Int[np.ndarray, "m"]
    offsets: Float[np.ndarray, "m 3"]

    @classmethod
    def of(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, spacing: Spacing) -> Self:
        """Detection minus annotation, in micrometres, for every predicted node the matcher paired."""
        rows = np.flatnonzero(matching.is_matched())
        predicted_um = spacing.to_micrometres(prediction.positions()[rows])
        truth_um = spacing.to_micrometres(truth.positions()[matching.gt_rows[rows]])
        return cls(prediction_rows=rows, offsets=predicted_um - truth_um)

    def select(self, rows: Int[np.ndarray, "k"]) -> Self:
        """The offsets of a chosen set of prediction rows — the cut, e.g. to the mislinked edges' endpoints."""
        keep = np.isin(self.prediction_rows, rows)
        return type(self)(prediction_rows=self.prediction_rows[keep], offsets=self.offsets[keep])

    def summary(self) -> OffsetSummary:
        """The distribution of these offsets."""
        return OffsetSummary.of(self.offsets)

    def within_read_out_grid(self, spacing: Spacing, downsample: tuple[int, int, int]) -> tuple[float, ...]:
        """Per axis, the share of offsets a READ-OUT ROUNDING alone could produce — the quantisation ceiling.

        A detection is the centre of a `downsample`-sized block of full-resolution voxels, so rounding alone
        can displace it by at most half a block per axis. An axis whose offsets nearly all fall inside that
        half-block is quantisation and nothing else; an axis that spills outside it is a peak sitting somewhere
        the annotation did not put the cell, which no sub-voxel refinement of the grid would recover.
        """
        voxels = np.abs(self.offsets) / spacing.as_array()
        allowed = np.asarray(downsample, dtype=np.float64) / 2.0
        return tuple(float(np.mean(voxels[:, axis] <= allowed[axis] + _ROUNDING_SLACK)) for axis in range(len(_AXES)))


@dataclass(frozen=True)
class MislinkEndpoints:
    """The prediction rows standing in for the endpoints of every MISLINKED annotated edge.

    `DenseFateDiagnosis` says WHICH annotated edges the tracker mislinked; this resolves those edges back to
    the detections that represented them, which is the row space every per-node quantity (an offset, a
    probability) lives in. Endpoints are de-duplicated: a detection that is the source of one mislink and the
    target of another is one localisation, not two.
    """

    rows: Int[np.ndarray, "k"]
    edge_count: int

    @classmethod
    def of(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching) -> Self:
        """Resolve the mislinked annotated edges' endpoints into distinct prediction rows."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        mislinked = (fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE)
        inverse = MatchingInverse.of(matching, len(truth.node_ids)).rows
        endpoints = inverse[truth.edge_rows()[mislinked].reshape(-1)]
        return cls(rows=np.unique(endpoints[endpoints != UNMATCHED]), edge_count=int(mislinked.sum()))


@dataclass(frozen=True)
class CrowdedNodes:
    """The prediction rows whose annotated partner had more than one detection inside the matching radius.

    A measured offset has two possible causes, and only one of them is localisation error: the detection may
    sit off the cell, or the assignment may have paired the annotated cell with a DIFFERENT cell's detection.
    The second is only available where a second detection lies inside the metric's radius — and the mislinks
    live in exactly the crowded tissue where that is common. Comparing the mislink cut against the crowded
    population, rather than against everything, is what keeps a swap from being read as an error.

    Crowding is counted over DETECTIONS, not annotations: the annotation is sparse (a movie's labelled cells
    are rarely within a radius of one another), so an annotation-side count reports nothing, while the
    detections are dense and are the things the assignment actually chooses between.
    """

    rows: Int[np.ndarray, "k"]

    @classmethod
    def of(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, spacing: Spacing, radius_um: float
    ) -> Self:
        """Prediction rows whose matched annotated node had a rival detection within `radius_um`, same frame."""
        predicted_um = spacing.to_micrometres(prediction.positions())
        truth_um = spacing.to_micrometres(truth.positions())
        rivals = np.zeros(len(truth.node_ids), dtype=np.int64)
        truth_times = truth.timepoints()
        predicted_times = prediction.timepoints()
        for timepoint in np.unique(truth_times):
            annotated = np.flatnonzero(truth_times == timepoint)
            detected = np.flatnonzero(predicted_times == timepoint)
            separations = np.linalg.norm(truth_um[annotated][:, None] - predicted_um[detected][None], axis=-1)
            rivals[annotated] = (separations <= radius_um).sum(axis=1)
        rows = np.flatnonzero(matching.is_matched())
        return cls(rows=rows[rivals[matching.gt_rows[rows]] > 1])


@dataclass(frozen=True)
class CandidatePairs:
    """Per annotated edge: the prediction rows of its source, its true successor, and a RIVAL target.

    The edge head is fed the pair's RELATIVE DISPLACEMENT, computed from quantised positions, so two different
    candidates can present it with the same geometric input. Naming a rival beside the truth for the same source
    is what turns that into a measurement — on the mislinks the rival is the target the tracker actually took,
    and on the correct edges it is the nearest detection it could have taken instead.
    """

    source: Int[np.ndarray, "k"]
    true_target: Int[np.ndarray, "k"]
    rival: Int[np.ndarray, "k"]

    @classmethod
    def _of_rows(cls, rows: list[tuple[int, int, int]]) -> Self:
        """The triples as three column arrays — empty columns, not an error, when nothing qualified."""
        if not rows:
            empty = np.empty(0, dtype=np.int64)
            return cls(source=empty, true_target=empty.copy(), rival=empty.copy())
        columns = np.asarray(rows, dtype=np.int64)
        return cls(source=columns[:, 0], true_target=columns[:, 1], rival=columns[:, 2])

    @classmethod
    def mislinked(cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching) -> Self:
        """The mislinked annotated edges, each with the wrong target the tracker chose for its source."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        mislinked = (fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE)
        inverse = MatchingInverse.of(matching, len(truth.node_ids)).rows
        successors = Adjacency.of(prediction).successors
        return cls._of_rows(
            [
                (int(inverse[source]), int(inverse[target]), int(successors[int(inverse[source])][0]))
                for source, target in truth.edge_rows()[mislinked]
            ]
        )

    @classmethod
    def correct(
        cls, prediction: TrackGraph, truth: TrackGraph, matching: NodeMatching, spacing: Spacing, gate_um: float
    ) -> Self:
        """The reproduced annotated edges, each with the nearest OTHER detection in the target frame as rival.

        The control has to be the same shape as the failure: one source, its true successor, and the closest
        thing that could have been taken instead. A source with nothing else inside the linker's gate had no
        rival to confuse and is left out rather than paired with an arbitrary distant node.
        """
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        inverse = MatchingInverse.of(matching, len(truth.node_ids)).rows
        positions = spacing.to_micrometres(prediction.positions())
        timepoints = prediction.timepoints()
        rows: list[tuple[int, int, int]] = []
        for source, target in truth.edge_rows()[fates == Fate.CORRECT]:
            source_row, true_row = int(inverse[source]), int(inverse[target])
            rival = cls._nearest_rival(positions, timepoints, source_row, true_row, gate_um)
            if rival is not None:
                rows.append((source_row, true_row, rival))
        return cls._of_rows(rows)

    @staticmethod
    def _nearest_rival(
        positions: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
        source_row: int,
        true_row: int,
        gate_um: float,
    ) -> int | None:
        """The detection nearest the source in the true target's frame, other than the true target itself."""
        candidates = np.flatnonzero(timepoints == timepoints[true_row])
        candidates = candidates[candidates != true_row]
        if not len(candidates):
            return None
        separations = np.linalg.norm(positions[candidates] - positions[source_row], axis=1)
        nearest = int(np.argmin(separations))
        return int(candidates[nearest]) if separations[nearest] <= gate_um else None


@dataclass(frozen=True)
class DeltaCollisions:
    """How far apart the true and rival candidates are in the QUANTISED displacement the head is fed.

    Positions are read out on the detection lattice, so a displacement is an integer triple on that lattice.
    A difference of zero means the two candidates hand the head an identical geometric input — it cannot
    separate them by geometry however well trained it is — and a difference of one voxel on one axis is the
    smallest distinction the grid can express at all.
    """

    differences: Int[np.ndarray, "k 3"]

    @classmethod
    def of(cls, prediction: TrackGraph, pairs: CandidatePairs, downsample: tuple[int, int, int]) -> Self:
        """The true minus the rival displacement, in detection-lattice voxels, for every candidate pair."""
        lattice = prediction.positions() // np.asarray(downsample, dtype=np.int64)
        return cls(differences=lattice[pairs.true_target] - lattice[pairs.rival])

    def lattice_separations(self) -> Int[np.ndarray, "k"]:
        """The L1 distance between the two displacements, in lattice voxels — 0 is a collision."""
        return np.abs(self.differences).sum(axis=1)

    def identical(self) -> int:
        """How many candidate pairs are geometrically INDISTINGUISHABLE on the lattice."""
        return int((self.lattice_separations() == 0).sum())

    def single_voxel(self) -> int:
        """How many differ by exactly one voxel on exactly one axis — the grid's finest expressible distinction."""
        return int((self.lattice_separations() == 1).sum())


@dataclass(frozen=True)
class RivalDistanceGap:
    """How much closer the rival target is than the true successor, in micrometres — the margin to invert.

    The distance term of `cost = distance - bonus*P` only mis-orders two candidates if the quantisation noise
    is comparable to the GAP between their distances from the source. Measuring the gap turns "quantisation
    could invert them" from an assertion into an arithmetic comparison against the read-out's own half-block.
    """

    gaps_um: Float[np.ndarray, "k"]

    @classmethod
    def of(cls, prediction: TrackGraph, pairs: CandidatePairs, spacing: Spacing) -> Self:
        """`d(source, true) - d(source, rival)`: positive means the rival is the nearer of the two."""
        positions = spacing.to_micrometres(prediction.positions())
        to_true = np.linalg.norm(positions[pairs.true_target] - positions[pairs.source], axis=1)
        to_rival = np.linalg.norm(positions[pairs.rival] - positions[pairs.source], axis=1)
        return cls(gaps_um=to_true - to_rival)

    def rival_nearer(self) -> float:
        """The share where the rival is the closer candidate — the ordering the distance term would prefer."""
        return float(np.mean(self.gaps_um > 0.0)) if len(self.gaps_um) else float("nan")

    def invertible(self, noise_um: float) -> float:
        """The share whose gap is small enough that a localisation error of `noise_um` could reorder them."""
        return float(np.mean(np.abs(self.gaps_um) < noise_um)) if len(self.gaps_um) else float("nan")


@dataclass(frozen=True)
class LocalisationDiagnosis:
    """Running one movie under a chosen operating point and logging every cut of its localisation error.

    The whole thing is one cache-served tracker pass: the offsets, the crowded control, the mislink cut and the
    candidate-displacement comparison are all functions of the same `(prediction, truth, matching)`, so they are
    read off one run rather than re-detected per question.
    """

    prediction: TrackGraph
    truth: TrackGraph
    matching: NodeMatching
    spacing: Spacing
    downsample: tuple[int, int, int]
    gate_um: float

    @classmethod
    def _of(cls, root: DataRoot, device: str, movie: str, config: TrackerConfig) -> Self:
        """One cache-served tracker pass over `movie` at `config`, matched against its annotations."""
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, device, config)
        truth, spacing = proxy.truths[0].graph, proxy.spacing
        prediction = tracker.run(proxy.paths[0].name, proxy.paths[0])
        return cls(
            prediction=prediction,
            truth=truth,
            matching=DistanceMatcher(spacing=spacing).match(prediction, truth),
            spacing=spacing,
            downsample=tracker.detector.recipe.downsample,
            gate_um=config.linker.gate_um,
        )

    def _report(self) -> None:
        """Log every cut: the reference scale, the offset distributions, and the candidate comparison."""
        offsets = LocalisationOffsets.of(self.prediction, self.truth, self.matching, self.spacing)
        endpoints = MislinkEndpoints.of(self.prediction, self.truth, self.matching)
        crowded = CrowdedNodes.of(self.prediction, self.truth, self.matching, self.spacing, _CROWDING_RADIUS_UM)
        steps = self.truth.link_displacements(self.spacing)
        logger.info("spacing (um/voxel) z=%.5f y=%.5f x=%.5f", self.spacing.z, self.spacing.y, self.spacing.x)
        logger.info("annotated step: median=%.3f um over %d links", float(np.median(steps)), len(steps))
        logger.info("mislinked annotated edges=%d  distinct endpoints=%d", endpoints.edge_count, len(endpoints.rows))
        logger.info("crowded (>1 detection within 7 um of the annotated cell) = %d nodes", len(crowded.rows))
        for label, cut in (
            ("all matched", offsets),
            ("crowded control", offsets.select(crowded.rows)),
            ("mislink endpoints", offsets.select(endpoints.rows)),
        ):
            self._offsets(label, cut)
        self._candidates(offsets)

    def _candidates(self, offsets: LocalisationOffsets) -> None:
        """Log the mislinks' true-vs-chosen candidate comparison beside the same one over the correct edges."""
        lattice = "x".join(map(str, self.downsample))
        logger.info("quantised displacement (true successor vs rival), in %s-lattice voxels", lattice)
        # A displacement is the difference of two independently localised positions, so the noise that could
        # reorder two candidates is the MEASURED median per-node offset propagated through that difference.
        noise_um = math.sqrt(2.0) * offsets.summary().median_um
        for label, pairs in (
            ("mislinks vs chosen", CandidatePairs.mislinked(self.prediction, self.truth, self.matching)),
            (
                "correct vs nearest",
                CandidatePairs.correct(self.prediction, self.truth, self.matching, self.spacing, self.gate_um),
            ),
        ):
            self._collisions(label, DeltaCollisions.of(self.prediction, pairs, self.downsample))
            self._gaps(label, RivalDistanceGap.of(self.prediction, pairs, self.spacing), noise_um)

    @staticmethod
    def _gaps(label: str, gaps: RivalDistanceGap, noise_um: float) -> None:
        """Log the true-minus-rival distance gap and how much of it a localisation error could flip."""
        if not len(gaps.gaps_um):
            logger.info("%-22s n=0", label)
            return
        quartiles = np.percentile(gaps.gaps_um, [_LOWER_QUARTILE, _MEDIAN, _UPPER_QUARTILE])
        logger.info(
            "%-22s n=%4d  gap median=%+.3f um IQR=[%+.3f, %+.3f]  rival nearer=%.1f%%  |gap|<%.3f um=%.1f%%",
            label,
            len(gaps.gaps_um),
            quartiles[1],
            quartiles[0],
            quartiles[2],
            100.0 * gaps.rival_nearer(),
            noise_um,
            100.0 * gaps.invertible(noise_um),
        )

    @staticmethod
    def _collisions(label: str, collisions: DeltaCollisions) -> None:
        """Log how often the two candidates' quantised displacements collide, and how far apart they are otherwise."""
        separations = collisions.lattice_separations()
        if not len(separations):
            logger.info("%-22s n=0", label)
            return
        quartiles = np.percentile(separations, [_LOWER_QUARTILE, _MEDIAN, _UPPER_QUARTILE])
        logger.info(
            "%-22s n=%4d  identical delta=%d (%.1f%%)  one-voxel-apart=%d (%.1f%%)  L1 median=%.1f IQR=[%.1f, %.1f]",
            label,
            len(separations),
            collisions.identical(),
            100.0 * collisions.identical() / len(separations),
            collisions.single_voxel(),
            100.0 * collisions.single_voxel() / len(separations),
            quartiles[1],
            quartiles[0],
            quartiles[2],
        )

    def _offsets(self, label: str, cut: LocalisationOffsets) -> None:
        """Log one cut's offset distribution — magnitude, each axis, and the share read-out rounding explains."""
        summary = cut.summary()
        logger.info(
            "%-22s n=%5d  |offset| median=%.3f um  IQR=[%.3f, %.3f]",
            label,
            summary.count,
            summary.median_um,
            *summary.iqr_um,
        )
        for axis, name in enumerate(_AXES):
            logger.info(
                "  %-20s |d%s| median=%.3f um  IQR=[%.3f, %.3f]  signed median=%+.3f um",
                label,
                name,
                summary.axis_median_abs_um[axis],
                *summary.axis_iqr_um[axis],
                summary.axis_median_signed_um[axis],
            )
        shares = cut.within_read_out_grid(self.spacing, self.downsample)
        logger.info(
            "  %-20s share explainable by the %s read-out grid: %s",
            label,
            "x".join(map(str, self.downsample)),
            "  ".join(f"{name}={share:.3f}" for name, share in zip(_AXES, shares, strict=True)),
        )

    @staticmethod
    def _arguments() -> argparse.Namespace:
        """The CLI: which movie, at which operating point, on which device."""
        parser = argparse.ArgumentParser(description="Measure detection localisation error against the annotations.")
        parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to measure")
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override any tracker knob, exactly as `proxy_eval --set` does",
        )
        return parser.parse_args()


def main() -> None:
    """Measure the localisation error of the matched detections, and cut it by the mislinked edges' endpoints."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = LocalisationDiagnosis._arguments()  # noqa: SLF001
    config = TrackerConfig()
    for assignment in args.overrides:
        config = ConfigOverride.apply(config, assignment)
    logger.info("movie=%s  set=%s", args.movie, list(args.overrides))
    diagnosis = LocalisationDiagnosis._of(DataRoot.from_config(args.config), args.device, args.movie, config)  # noqa: SLF001
    diagnosis._report()  # noqa: SLF001


if __name__ == "__main__":
    main()
