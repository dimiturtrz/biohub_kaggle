"""Does a candidate division mother LOOK mitotic in the raw volume — a cheap gate, not a classifier.

Division recovery proposes hundreds of forks on the dense movie to capture three annotated divisions, and the
edge head's probability cannot order them: parents with an orphan runner-up above P=0.25 outnumber the one
recoverable true division (P=0.124) by three orders of magnitude. A discriminator that is NOT the edge
probability is therefore the only way forward, and the literature nominates APPEARANCE — PMC3278834 separates
mitotic from interphase nuclei at >90% with nine static intensity/shape features and no network, and Hirsch et
al. (MICCAI 2022) cut false-positive divisions 7.5-17x with a learned cell-state head whose `parent` class is
exactly the mother-before-split cue.

Before building either, this module asks whether the signal is present in OUR data at all. It measures a small
set of those static features on RAW full-resolution patches around four populations of detections — the
annotated mothers (and their run-up frames), the forks the recovery would propose, ordinary linked cells, and
ANNOTATED non-dividing cells — and reports each distribution.

WHAT THIS CAN AND CANNOT CONCLUDE. There are three annotated divisions. Three positives cannot support an AUC,
a fitted threshold, or any summary statistic of the positive class: the reporting is therefore per-mother, each
one's feature value placed as a PERCENTILE inside the false-candidate distribution, and nothing is averaged
over the three. The annotated cells are ~1.8% of this movie, so the false candidates are overwhelmingly
UNANNOTATED — a feature that separates the mothers from them may be separating "annotated" from "not
annotated" (a brightness the annotator preferred) rather than mitosis from interphase. The annotated
non-dividing population is that control, and a feature only counts as a mitosis cue if it separates the mothers
from BOTH.

ONE FEATURE IS MEASURED AT THE LIMIT OF THE GRID. Detections are read out on the `(1, 4, 4)`-downsampled
lattice, so a centre's y/x index is a multiple of four full-resolution voxels — up to 0.81 um of quantisation
per axis. The intensity-weighted-centroid OFFSET is a distance from exactly that centre, and it comes out at
0.2-1.3 um across every population here: the feature's whole dynamic range is the size of the error in the
point it is measured from. A null on the offset therefore refutes it AS COMPUTED FROM THE DETECTION CENTRE,
not the underlying chromatin asymmetry — the size and shape features carry no such caveat.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Bool, Float, Int, UInt16

from celltrack.affinity import EdgeAffinity
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis, DenseFateDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.eval.proxy_eval import ConfigOverride
from celltrack.operating_point import TrackerConfig
from celltrack.postproc.affinity_division_recovery import AffinityDivisionConfig
from core.data.tracks import Adjacency, TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "intensity_mean",
    "intensity_sd",
    "intensity_max",
    "local_contrast",
    "volume_um3",
    "centroid_offset_um",
    "elongation",
)

# A nucleus here is a few micrometres across, so a 6.5 um half-box holds the cell plus enough surround for a
# background estimate; the core radius is the region a feature is allowed to integrate over, and the shell
# beyond the background radius is what "background" means — all three in micrometres, so the 4x z anisotropy
# is handled by the spacing rather than by a per-axis voxel count.
_PATCH_RADIUS_UM = 6.5
_BACKGROUND_RADIUS_UM = 5.0
_CORE_RADIUS_UM = 4.0
# Half-maximum above background: the standard size proxy when no segmentation exists, and the only threshold
# choice here that is not read off the data itself.
_VOLUME_LEVEL = 0.5
# How many frames before the split the mother is also measured at — the literature's cue is the run-up, and a
# mother one frame before division is still a mother.
_RUN_UP_FRAMES = 2
_BASELINE_CAP = 3000
_BASELINE_SEED = 0
_SINGLE_CHILD = 1
_LOWER_QUARTILE = 25.0
_MEDIAN = 50.0
_UPPER_QUARTILE = 75.0
_HALF = 0.5
_EPSILON = 1e-12
# A mother has to sit outside this share of the false candidates on a feature before that feature is called a
# separator at all — with three positives, anything less is not distinguishable from where a random cell lands.
_TAIL_PERCENTILE = 90.0
_AXIS_COUNT = 3


@dataclass(frozen=True)
class PatchBox:
    """The fixed cube of raw voxels a detection's appearance is measured in, plus its physical masks.

    The box is defined in MICROMETRES and converted once: z voxels are 4x coarser than y/x, so a cube of voxels
    would be an elongated slab in the tissue and every "shape" feature would report the anisotropy instead of
    the cell. The core, background and offset arrays are cut once here and reused for every detection, which is
    what makes measuring thousands of patches a few seconds of work.
    """

    half_voxels: Int[np.ndarray, "3"]
    spacing: Spacing
    core: Bool[np.ndarray, "z y x"]
    background: Bool[np.ndarray, "z y x"]
    offsets_um: Float[np.ndarray, "z y x 3"]

    @classmethod
    def of(cls, spacing: Spacing, patch_radius_um: float = _PATCH_RADIUS_UM) -> Self:
        """The box enclosing `patch_radius_um` in every direction, with its core and background masks."""
        half = np.ceil(spacing.anisotropic_radius(patch_radius_um)).astype(np.int64)
        axes = [
            np.arange(-extent, extent + 1, dtype=np.float64) * step
            for extent, step in zip(half, spacing.as_array(), strict=True)
        ]
        offsets = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        radius = np.linalg.norm(offsets, axis=-1)
        return cls(
            half_voxels=half,
            spacing=spacing,
            core=radius <= _CORE_RADIUS_UM,
            background=radius > _BACKGROUND_RADIUS_UM,
            offsets_um=offsets,
        )

    def voxel_volume_um3(self) -> float:
        """The physical volume of one voxel — what a voxel count is multiplied by to become a size."""
        return float(np.prod(self.spacing.as_array()))

    def cut(self, volume: UInt16[np.ndarray, "z y x"], centre: Int[np.ndarray, "3"]) -> Float[np.ndarray, "z y x"]:
        """The box around `centre`, as float — clipped at the volume edge and filled with the local median.

        A detection near the border would otherwise be dropped, and with three positives no population can
        afford an exclusion rule; filling the outside with the median of what WAS readable keeps the box a
        fixed shape without inventing contrast at the edge.
        """
        shape = self.core.shape
        lower = np.asarray(centre, dtype=np.int64) - self.half_voxels
        limits = np.asarray(volume.shape, dtype=np.int64)
        start = np.clip(lower, 0, limits)
        stop = np.clip(lower + np.asarray(shape, dtype=np.int64), 0, limits)
        inner = volume[tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))].astype(np.float64)
        patch = np.full(shape, float(np.median(inner)) if inner.size else 0.0)
        patch[tuple(slice(int(a - c), int(b - c)) for a, b, c in zip(start, stop, lower, strict=True))] = inner
        return patch


@dataclass(frozen=True)
class AppearanceFeatures:
    """One detection's static appearance descriptor, in `FEATURE_NAMES` order.

    The seven are the cheap subset of PMC3278834's nine that need no segmentation: three raw intensity
    statistics, the peak's height over its own surround, a half-maximum size, the distance between the
    intensity-weighted centroid and the geometric centre (their most distinctive single feature — a condensed
    mitotic chromatin mass is off-centre in the box the tracker centred on it), and an elongation proxy from
    the intensity second-moment eigenvalues, which stands in for sphericity without a surface.
    """

    values: Float[np.ndarray, "f"]

    @classmethod
    def of(cls, patch: Float[np.ndarray, "z y x"], box: PatchBox) -> Self:
        """Measure one raw patch — background from its own shell, everything else inside the core."""
        background = float(np.median(patch[box.background])) if box.background.any() else 0.0
        core = patch[box.core]
        peak = float(core.max()) if core.size else background
        above = np.clip(patch - background, 0.0, None) * box.core
        offset, elongation = cls._moments(above, box)
        threshold = background + _VOLUME_LEVEL * (peak - background)
        return cls(
            values=np.array(
                [
                    float(core.mean()) if core.size else 0.0,
                    float(core.std()) if core.size else 0.0,
                    peak,
                    peak - background,
                    float(((patch > threshold) & box.core).sum()) * box.voxel_volume_um3(),
                    offset,
                    elongation,
                ]
            )
        )

    @staticmethod
    def _moments(weights: Float[np.ndarray, "z y x"], box: PatchBox) -> tuple[float, float]:
        """The weighted centroid's distance from the box centre, and the second-moment elongation ratio.

        Both read the same background-subtracted mass, so a patch holding nothing above its surround reports a
        zero offset and a unit elongation rather than a division by zero.
        """
        mass = float(weights.sum())
        if mass <= _EPSILON:
            return 0.0, 1.0
        flat = weights.reshape(-1)
        coordinates = box.offsets_um.reshape(-1, _AXIS_COUNT)
        centroid = (flat[:, None] * coordinates).sum(axis=0) / mass
        centred = coordinates - centroid
        covariance = (flat[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(axis=0) / mass
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        smallest = float(eigenvalues.min())
        largest = float(eigenvalues.max())
        return float(np.linalg.norm(centroid)), float(np.sqrt(largest / smallest)) if smallest > _EPSILON else 1.0


@dataclass(frozen=True)
class FeatureTable:
    """A population's detections and their appearance features — one row per detection, columns in name order.

    Rows travel with the values so a per-mother line can be pulled back out by prediction row, and so a
    distribution can be interrogated with `percentile_of` rather than compared through a summary statistic.
    """

    rows: Int[np.ndarray, "m"]
    values: Float[np.ndarray, "m f"]

    @classmethod
    def measure(cls, video: CellVideo, graph: TrackGraph, rows: Int[np.ndarray, "m"], box: PatchBox) -> Self:
        """Measure every listed detection, decompressing each timepoint of the store exactly once.

        The store is chunked one timepoint per chunk, so reading a small window costs the same decompress as
        reading the whole frame — grouping the rows by timepoint is the difference between one pass over the
        movie and one pass per cell.
        """
        ordered = np.asarray(rows, dtype=np.int64)
        timepoints = graph.timepoints()[ordered]
        positions = graph.positions()
        values = np.zeros((len(ordered), len(FEATURE_NAMES)), dtype=np.float64)
        for timepoint in np.unique(timepoints).tolist():
            volume = video.frame(int(timepoint))
            for index in np.flatnonzero(timepoints == timepoint).tolist():
                values[index] = AppearanceFeatures.of(box.cut(volume, positions[ordered[index]]), box).values
        return cls(rows=ordered, values=values)

    def column(self, feature: str) -> Float[np.ndarray, "m"]:
        """One feature's values across this population."""
        return self.values[:, FEATURE_NAMES.index(feature)]

    def quartiles(self, feature: str) -> tuple[float, float, float]:
        """The feature's 25th, 50th and 75th percentiles — NaNs on an empty population, never zeros."""
        values = self.column(feature)
        if not len(values):
            return float("nan"), float("nan"), float("nan")
        low, middle, high = np.percentile(values, [_LOWER_QUARTILE, _MEDIAN, _UPPER_QUARTILE])
        return float(low), float(middle), float(high)

    def percentile_of(self, feature: str, value: float) -> float:
        """Where `value` falls inside this population's distribution, as a percentile with ties split.

        This is the readable statement the three positives can support: not "the mothers score higher", which
        averages three points, but "this mother sits above 97% of the false candidates".
        """
        values = self.column(feature)
        if not len(values):
            return float("nan")
        below = float(np.mean(values < value))
        equal = float(np.mean(values == value))
        return 100.0 * (below + _HALF * equal)

    def of_row(self, row: int) -> Float[np.ndarray, "f"]:
        """One detection's feature vector, by prediction row."""
        return self.values[int(np.flatnonzero(self.rows == row)[0])]


@dataclass(frozen=True)
class TrueMother:
    """One annotated division's parent: its annotated identity, its detection, and its run-up detections.

    The three are not interchangeable — one has both daughters detected, one's second daughter is claimed by
    another parent, one's second daughter was never detected — so they are carried and reported individually
    rather than pooled into a positive class.
    """

    truth_id: int
    prediction_row: int
    run_up_rows: tuple[int, ...]

    @classmethod
    def of(cls, truth: TrackGraph, adjacency: Adjacency, inverse: Int[np.ndarray, "g"], truth_row: int) -> Self:
        """Resolve one annotated dividing node, and the frames before its split, into prediction rows."""
        run_up: list[int] = []
        row = truth_row
        for _ in range(_RUN_UP_FRAMES):
            parents = adjacency.predecessors[row]
            if not parents:
                break
            row = parents[0]
            if inverse[row] != UNMATCHED:
                run_up.append(int(inverse[row]))
        return cls(
            truth_id=int(truth.node_ids[truth_row]),
            prediction_row=int(inverse[truth_row]),
            run_up_rows=tuple(run_up),
        )

    def measured_rows(self) -> tuple[int, ...]:
        """The mother's own detection followed by its run-up detections, newest first."""
        return (self.prediction_row, *self.run_up_rows)


@dataclass(frozen=True)
class DivisionPopulations:
    """The four populations the gate compares, all as prediction rows over one tracker pass.

    Every population is measured on the SAME detections the pipeline produced, so a difference between them is
    a difference in appearance and not in how the centre was obtained.
    """

    mothers: tuple[TrueMother, ...]
    false_candidates: Int[np.ndarray, "b"]
    linked_baseline: Int[np.ndarray, "c"]
    annotated_others: Int[np.ndarray, "d"]

    @classmethod
    def of(
        cls,
        prediction: TrackGraph,
        truth: TrackGraph,
        matching: NodeMatching,
        candidates: Int[np.ndarray, "k"],
    ) -> Self:
        """Split the run into annotated mothers, proposed forks, ordinary linked cells and annotated controls."""
        inverse = DenseFateDiagnosis.predicted_of_truth(matching, len(truth.node_ids))
        truth_adjacency = Adjacency.of(truth)
        dividing = truth_adjacency.dividing_rows()
        mothers = tuple(
            TrueMother.of(truth, truth_adjacency, inverse, int(row)) for row in dividing if inverse[row] != UNMATCHED
        )
        claimed = {row for mother in mothers for row in mother.measured_rows()}
        false_candidates = np.array([row for row in candidates.tolist() if row not in claimed], dtype=np.int64)
        return cls(
            mothers=mothers,
            false_candidates=false_candidates,
            linked_baseline=cls._baseline(prediction, claimed | set(false_candidates.tolist())),
            annotated_others=cls._annotated_others(inverse, dividing, claimed),
        )

    @staticmethod
    def _baseline(prediction: TrackGraph, excluded: set[int]) -> Int[np.ndarray, "c"]:
        """A capped random sample of ordinary linked cells — one parent, one child, neither mother nor candidate."""
        adjacency = Adjacency.of(prediction)
        ordinary = np.flatnonzero((adjacency.out_degrees == _SINGLE_CHILD) & (adjacency.in_degrees == _SINGLE_CHILD))
        ordinary = np.array([row for row in ordinary.tolist() if row not in excluded], dtype=np.int64)
        if len(ordinary) <= _BASELINE_CAP:
            return ordinary
        return np.sort(np.random.default_rng(_BASELINE_SEED).choice(ordinary, _BASELINE_CAP, replace=False))

    @staticmethod
    def _annotated_others(
        inverse: Int[np.ndarray, "g"], dividing: Int[np.ndarray, "d"], claimed: set[int]
    ) -> Int[np.ndarray, "d"]:
        """Detections matched to an annotated NON-dividing cell — the annotated-vs-unannotated control.

        Annotated cells are ~1.8% of this movie, so any feature separating the mothers from the (mostly
        unannotated) candidates has to be checked against cells that share the mothers' one obvious confound:
        somebody chose to label them.
        """
        non_dividing = np.setdiff1d(np.flatnonzero(inverse != UNMATCHED), dividing)
        rows = inverse[non_dividing]
        return np.array([row for row in rows.tolist() if row not in claimed], dtype=np.int64)


@dataclass(frozen=True)
class DivisionCandidates:
    """The prediction rows the affinity division recovery would propose as mothers, at a stated floor.

    Taken from the recovery stage itself through its public `transform` — the candidate set is the thing under
    test, so approximating it with a re-implemented gate would measure a different population than the one the
    pipeline would actually fork.
    """

    rows: Int[np.ndarray, "k"]

    @classmethod
    def of(cls, prediction: TrackGraph, spacing: Spacing, affinity: EdgeAffinity, floor: float) -> Self:
        """Every parent of a fork the recovery proposes with its probability floors set to `floor`, uncapped."""
        config = AffinityDivisionConfig(min_second_prob=floor, min_kept_prob=floor, max_added_fraction=1.0)
        forked = config.build(spacing, affinity).transform(prediction)
        added = forked.edges[len(prediction.edges) :]
        if not len(added):
            return cls(rows=np.empty(0, dtype=np.int64))
        return cls(rows=np.unique(prediction.rows_of(added[:, 0])))


@dataclass(frozen=True)
class MitoticAppearanceGate:
    """One tracker pass over the dense movie, its four populations measured on raw patches, and the report."""

    @staticmethod
    def _arguments() -> argparse.Namespace:
        """The CLI: which movie, at which operating point, with which candidate-probability floor."""
        parser = argparse.ArgumentParser(description="Do candidate division mothers look mitotic in the raw image?")
        parser.add_argument("--config", type=Path, default=Path("paths.yaml"), help="paths.yaml locating the data root")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--movie", default=DENSE_MOVIE, help="the movie stem to measure")
        parser.add_argument("--floor", type=float, default=0.25, help="probability floor defining the candidate forks")
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override any tracker knob, exactly as `proxy_eval --set` does",
        )
        return parser.parse_args()

    populations: DivisionPopulations
    tables: dict[str, FeatureTable]

    @classmethod
    def _measure(cls, root: DataRoot, device: str, movie: str, config: TrackerConfig, floor: float) -> Self:
        """Run the tracker, resolve the populations, and measure every detection's raw-patch appearance."""
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, device, config)
        path, truth, spacing = proxy.paths[0], proxy.truths[0].graph, proxy.spacing
        prediction = tracker.run(path.name, path)
        matching = DistanceMatcher(spacing=spacing).match(prediction, truth)
        affinity = tracker.edge_scorer.affinities(path, prediction, device)
        candidates = DivisionCandidates.of(prediction, spacing, affinity, floor).rows
        populations = DivisionPopulations.of(prediction, truth, matching, candidates)
        video = CellVideo.from_ome_zarr(path)
        box = PatchBox.of(spacing)
        logger.info(
            "populations: mothers=%d  false candidates=%d (floor P>=%.2f)  linked baseline=%d  annotated others=%d",
            len(populations.mothers),
            len(populations.false_candidates),
            floor,
            len(populations.linked_baseline),
            len(populations.annotated_others),
        )
        return cls(populations=populations, tables=cls._tables(video, prediction, populations, box))

    @staticmethod
    def _tables(
        video: CellVideo, prediction: TrackGraph, populations: DivisionPopulations, box: PatchBox
    ) -> dict[str, FeatureTable]:
        """Measure each population's patches — the mothers and their run-up frames as one table."""
        mother_rows = np.array(
            [row for mother in populations.mothers for row in mother.measured_rows()], dtype=np.int64
        )
        return {
            name: FeatureTable.measure(video, prediction, rows, box)
            for name, rows in (
                ("mothers+run-up", mother_rows),
                ("false candidates", populations.false_candidates),
                ("linked baseline", populations.linked_baseline),
                ("annotated others", populations.annotated_others),
            )
        }

    def report(self) -> None:
        """Log the per-population distributions, then each mother's percentile inside the two controls."""
        self._distributions()
        self._mothers()
        self._verdict()

    def _distributions(self) -> None:
        """Per feature, every population's median and IQR — the distributions, side by side."""
        for feature in FEATURE_NAMES:
            for name, table in self.tables.items():
                low, middle, high = table.quartiles(feature)
                logger.info(
                    "%-19s %-18s n=%5d  median=%10.4f  IQR=[%10.4f, %10.4f]",
                    feature,
                    name,
                    len(table.rows),
                    middle,
                    low,
                    high,
                )

    def _mothers(self) -> None:
        """Each annotated mother, and each of its run-up frames, placed inside both control distributions."""
        table = self.tables["mothers+run-up"]
        candidates, annotated = self.tables["false candidates"], self.tables["annotated others"]
        for mother in self.populations.mothers:
            for lag, row in enumerate(mother.measured_rows()):
                values = table.of_row(row)
                for index, feature in enumerate(FEATURE_NAMES):
                    logger.info(
                        "mother id=%-6d t-%d row=%-6d %-19s value=%10.4f  pct(false)=%5.1f  pct(annotated)=%5.1f",
                        mother.truth_id,
                        lag,
                        row,
                        feature,
                        values[index],
                        candidates.percentile_of(feature, float(values[index])),
                        annotated.percentile_of(feature, float(values[index])),
                    )

    def _verdict(self) -> None:
        """Name every feature that puts ALL mothers in a tail of BOTH controls — or state that none does."""
        separators = [feature for feature in FEATURE_NAMES if self._separates(feature)]
        logger.info(
            "features putting every mother in a %.0f%% tail of BOTH controls: %s",
            _TAIL_PERCENTILE,
            separators or "NONE",
        )

    def _separates(self, feature: str) -> bool:
        """Whether every mother's own frame sits in the same tail of the candidates AND the annotated control."""
        table = self.tables["mothers+run-up"]
        index = FEATURE_NAMES.index(feature)
        placed = [
            [
                self.tables[control].percentile_of(feature, float(table.of_row(mother.prediction_row)[index]))
                for control in ("false candidates", "annotated others")
            ]
            for mother in self.populations.mothers
        ]
        if not placed:
            return False
        upper = all(min(pair) >= _TAIL_PERCENTILE for pair in placed)
        lower = all(max(pair) <= 100.0 - _TAIL_PERCENTILE for pair in placed)
        return upper or lower


def main() -> None:
    """Measure the appearance of the true and the candidate division mothers, and report both distributions."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = MitoticAppearanceGate._arguments()  # noqa: SLF001
    config = TrackerConfig.shipped()
    for assignment in args.overrides:
        config = ConfigOverride.apply(config, assignment)
    logger.info("movie=%s  floor=%.2f  set=%s", args.movie, args.floor, list(args.overrides))
    root = DataRoot.from_config(args.config)
    MitoticAppearanceGate._measure(root, args.device, args.movie, config, args.floor).report()  # noqa: SLF001


if __name__ == "__main__":
    main()
