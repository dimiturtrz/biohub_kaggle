"""Do cells look different enough to be told apart — asked of the RAW voxels, at full resolution.

THE QUESTION THIS SETTLES, cheaply, before anything is built. The association failure is 38 mislinks on the
dense movie where the head prefers a wrong near neighbour to the true far successor, and three measurements say
the LEARNED features cannot separate those cells: raw feature cosine prefers the wrong one 77% of the time, a
probe on those features reads chance held out, and two independently-initialised heads fail identically. Every
one of those is measured on the `(1, 4, 4)`-DOWNSAMPLED representation, where a nucleus spans two or three
voxels. Nobody has asked whether the information is there in the full-resolution image, which is 16x finer in
plane — and that is the one structural argument for why a new model might see what the current stack cannot.

WHAT IT DOES. For each mislink it pulls a small raw box around three cells — the source at `t`, its TRUE
successor at `t+1`, and the RIVAL the tracker chose instead — and asks whether the source correlates better
with the true successor than with the rival. No model, no training, no learned features: just voxels. The
honest metric is `InGateRanking` (rank the truth among ALL in-gate candidates, chance = mean of 1/candidates);
the two-alternative `prefers_true` is kept for continuity but its null is NOT 0.5, because raw-box correlation
tracks spatial overlap and the rival is nearer, so the nearer cell wins under the no-identity null.

THE CONTROL IS THE POINT. The same measurement runs over CORRECT edges, where the rival is the nearest
alternative the linker could have taken. Without it a number on the mislinks means nothing: if raw appearance
separates neither population the measurement is simply too weak, while separating the correct edges and NOT the
mislinks is the finding that those cells are genuinely ambiguous. Only "separates both" would say the
information is present and the current model is failing to use it.

CORRELATION, NOT DIFFERENCE, because brightness is a known confound here — median intensity separates "is a
recovery candidate" from "is annotated" in this data, which has nothing to do with identity. Mean-subtracting
each box and taking the cosine is the normalised cross-correlation, blind to a constant offset and to gain.

THE CAVEAT THAT BOUNDS IT: boxes are centred on DETECTIONS, and the mislinks' endpoints are measured sitting
2.1x further from their annotated centres than the population. A miscentred box looks different for reasons
that are not identity, so a null here is partly a statement about our localisation rather than about the cells.
The aligned variant exists for exactly that: it re-centres each box on its own brightest voxel first.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float, Int
from scipy.ndimage import rotate

from celltrack.analysis.drift_field import DriftDiagnosis
from celltrack.analysis.localisation import CandidatePairs
from celltrack.data.raw_boxes import BOX_RADIUS_UM, RawBoxes
from celltrack.eval.dense_diagnosis import DENSE_MOVIE
from celltrack.operating_point import TrackerConfig
from core.data.tracks import TrackGraph
from core.data.video import CellVideo
from core.geometry import Spacing
from core.metrics.matching import NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# A source with a single in-gate candidate poses no ranking question — there is nothing to rank against.
_MIN_CONTESTED = 2
_PERCENTILES = (25.0, 50.0, 75.0)


@dataclass(frozen=True)
class IdentitySimilarity:
    """Per triple, how well the source's appearance matches the true successor against the rival.

    `prefers_true` is the share of triples where the source correlates better with its true successor than with
    the wrong cell the tracker took. It is a two-alternative forced choice, and its NULL IS NOT 0.5. The two
    alternatives are not exchangeable: correlation of raw boxes tracks spatial OVERLAP, and on the mislinks the
    rival is nearer by construction (median +6.44 um), so two nearer boxes share more surrounding context and
    correlate higher regardless of identity. Under the true "no identity information" null the nearer cell wins
    MORE than half the time, so the honest baseline sits above 0.5 for the mislinks — which makes a low score
    worse than it looks against 0.5, not better. The number is kept for continuity, but `InGateRanking` is the
    trustworthy metric here because its baseline (mean of 1/candidates) accounts for the candidate count and the
    chance level directly, rather than assuming an exchangeability the geometry breaks.
    """

    true_similarity: Float[np.ndarray, "k"]
    rival_similarity: Float[np.ndarray, "k"]

    @classmethod
    def of(cls, source: RawBoxes, true_target: RawBoxes, rival: RawBoxes) -> Self:
        """Correlate each source against its true and rival box, over the triples where ALL THREE were read.

        The intersection is what keeps a triple a triple: the three columns drop boundary boxes independently,
        so pairing them by position would compare one triple's source against another's target.
        """
        complete = np.intersect1d(np.intersect1d(source.kept, true_target.kept), rival.kept)
        boxes = [column.select(complete) for column in (source, true_target, rival)]
        return cls(
            true_similarity=np.asarray([cls.correlate(a, b) for a, b in zip(boxes[0], boxes[1], strict=True)]),
            rival_similarity=np.asarray([cls.correlate(a, b) for a, b in zip(boxes[0], boxes[2], strict=True)]),
        )

    @staticmethod
    def correlate(first: Float[np.ndarray, "z y x"], second: Float[np.ndarray, "z y x"]) -> float:
        """Normalised cross-correlation of two boxes — blind to a brightness offset and to gain.

        Brightness is a measured confound in this data (it separates annotated cells from recovery candidates,
        which is nothing to do with identity), so the mean comes out before the cosine.
        """
        left, right = first.ravel() - first.mean(), second.ravel() - second.mean()
        scale = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(left @ right / scale) if scale else 0.0

    @staticmethod
    def raw_correlation_scorer(
        source: Float[np.ndarray, "z y x"], candidates: Float[np.ndarray, "c z y x"]
    ) -> Float[np.ndarray, "c"]:
        """The default `BoxScorer` — normalised cross-correlation of the source box against each candidate box."""
        return np.asarray([IdentitySimilarity.correlate(source, box) for box in candidates])

    def prefers_true(self) -> float:
        """Share of triples where appearance points at the TRUE successor; the null is >0.5, not 0.5 (see class)."""
        if not len(self.true_similarity):
            return float("nan")
        return float(np.mean(self.true_similarity > self.rival_similarity))

    def count(self) -> int:
        """How many complete triples the measurement ran on."""
        return len(self.true_similarity)


@dataclass(frozen=True)
class RotationGain:
    """How much appearance matching improves when the successor box is allowed to ROTATE to fit the source.

    A cell that turns between frames looks different to a fixed-orientation correlation even though it is the
    same cell, which would make appearance a worse identity cue than it should be — and would argue for a
    rotation-invariant descriptor in any identity model. This measures it directly on CORRECT edges, where the
    two boxes ARE the same cell: the correlation at the box's own orientation against the best over a set of
    in-plane rotations. The gain is the rotation the fixed correlation is blind to.

    In-plane (about z) only, deliberately: y and x are isotropic at full resolution so a rotation there is a
    clean resample, while z is 4x coarser and a handful of z-planes cannot express an out-of-plane turn without
    inventing structure. So this bounds IN-PLANE rotation, which is the component the imaging can actually see.
    """

    identity: Float[np.ndarray, "k"]
    best_rotated: Float[np.ndarray, "k"]

    @classmethod
    def of(cls, source: RawBoxes, true_target: RawBoxes, angles_deg: tuple[float, ...]) -> Self:
        """Correlate each same-cell pair at the box orientation and at its best in-plane rotation."""
        complete = np.intersect1d(source.kept, true_target.kept)
        source_boxes, target_boxes = source.select(complete), true_target.select(complete)
        identity: list[float] = []
        best: list[float] = []
        for reference, box in zip(source_boxes, target_boxes, strict=True):
            scores = [IdentitySimilarity.correlate(reference, cls._rotate(box, angle)) for angle in angles_deg]
            identity.append(IdentitySimilarity.correlate(reference, box))
            best.append(max(scores))
        return cls(identity=np.asarray(identity), best_rotated=np.asarray(best))

    @staticmethod
    def _rotate(box: Float[np.ndarray, "z y x"], angle_deg: float) -> Float[np.ndarray, "z y x"]:
        """Rotate a box in the y-x plane, resampling — 0 degrees returns it unchanged."""
        if angle_deg == 0.0:
            return box
        return rotate(box, angle_deg, axes=(1, 2), reshape=False, order=1, mode="nearest")

    def median_gain(self) -> float:
        """The typical correlation a rotation recovers — near zero if cells do not turn in plane."""
        return float(np.median(self.best_rotated - self.identity)) if len(self.identity) else float("nan")

    def count(self) -> int:
        """How many same-cell pairs the probe ran on."""
        return len(self.identity)


# A scorer ranks the source box against each candidate box — raw correlation by default, a learned encoder
# cosine when the encoder gate supplies its own. Both return one similarity per candidate, higher = more alike.
BoxScorer = Callable[[Float[np.ndarray, "z y x"], Float[np.ndarray, "c z y x"]], Float[np.ndarray, "c"]]


@dataclass(frozen=True)
class InGateRanking:
    """Where appearance ranks the TRUE successor among every candidate inside the linker's gate.

    A two-way comparison against one rival is not the task. The linker faces 3.8 in-gate candidates per source
    on this movie, and it has to put the right one first — so this ranks the true successor among ALL of them
    by appearance alone. It also fixes a difficulty mismatch in the two-way form: there the correct edges got
    the NEAREST other detection as rival while the mislinks got the hardest possible one, so the two populations
    were not being asked equally hard questions. Ranking within the same gate asks both the same thing.

    `chance` is the mean of 1/candidates rather than a constant, because sources differ in how contested they
    are; comparing a top-1 rate against a flat 0.5 would flatter a source that only had two candidates.
    """

    true_rank: Int[np.ndarray, "k"]
    candidates: Int[np.ndarray, "k"]

    @classmethod
    def of(  # noqa: PLR0913
        cls,
        prediction: TrackGraph,
        spacing: Spacing,
        box_reader: Callable[[Int[np.ndarray, "k"]], RawBoxes],
        pairs: CandidatePairs,
        gate_um: float,
        scorer: BoxScorer = IdentitySimilarity.raw_correlation_scorer,
    ) -> Self:
        """Rank the true successor by appearance among the source's in-gate candidates, per triple.

        Takes the prediction graph, its spacing and a box-reader closure rather than the owning
        `IdentityDiagnosis`, so the ranking depends only on what it reads — the mutual class reference is what a
        diagnosis is for, not what a ranking needs. The box radius is fixed per call, so it is bound into the
        reader rather than threaded as an argument. `scorer` is how the same ranking serves both the raw-appearance
        diagnosis and the learned-encoder gate: swap what turns two boxes into a similarity, keep the choice set.
        """
        positions = spacing.to_micrometres(prediction.positions())
        timepoints = prediction.timepoints()
        ranks: list[int] = []
        counts: list[int] = []
        for source, true_target in zip(pairs.source, pairs.true_target, strict=True):
            candidates = cls._in_gate(positions, timepoints, int(source), gate_um)
            if len(candidates) < _MIN_CONTESTED or int(true_target) not in candidates.tolist():
                continue
            ranked = cls._rank(box_reader, int(source), candidates, scorer)
            if ranked is None:
                continue
            order = candidates[np.argsort(-ranked)]
            ranks.append(int(np.flatnonzero(order == int(true_target))[0]) + 1)
            counts.append(len(candidates))
        return cls(true_rank=np.asarray(ranks, dtype=np.int64), candidates=np.asarray(counts, dtype=np.int64))

    @staticmethod
    def _in_gate(
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
        source: int,
        gate_um: float,
    ) -> Int[np.ndarray, "c"]:
        """Every detection in the next frame within the linker's gate of this source — its real choice set."""
        candidates = np.flatnonzero(timepoints == timepoints[source] + 1)
        within = np.linalg.norm(positions_um[candidates] - positions_um[source], axis=1) <= gate_um
        return candidates[within]

    @staticmethod
    def _rank(
        box_reader: Callable[[Int[np.ndarray, "k"]], RawBoxes],
        source: int,
        candidates: Int[np.ndarray, "c"],
        scorer: BoxScorer,
    ) -> Float[np.ndarray, "c"] | None:
        """Similarity of the source against each candidate under `scorer`, or None if any box left the volume."""
        source_box = box_reader(np.asarray([source]))
        candidate_boxes = box_reader(candidates)
        if not len(source_box.boxes) or len(candidate_boxes.boxes) != len(candidates):
            return None
        return scorer(source_box.boxes[0], candidate_boxes.boxes)

    def top_one(self) -> float:
        """How often appearance puts the true successor FIRST — the linker's actual question."""
        return float(np.mean(self.true_rank == 1)) if len(self.true_rank) else float("nan")

    def chance(self) -> float:
        """Top-1 rate a coin would achieve on these choice sets — the honest baseline."""
        return float(np.mean(1.0 / self.candidates)) if len(self.candidates) else float("nan")

    def count(self) -> int:
        """How many contested sources the ranking ran on."""
        return len(self.true_rank)


@dataclass(frozen=True)
class IdentityDiagnosis:
    """One movie's mislinks and correct edges, scored on raw appearance at full resolution."""

    video: CellVideo
    prediction: TrackGraph
    truth: TrackGraph
    matching: NodeMatching
    spacing: Spacing

    @classmethod
    def _of(cls, root: DataRoot, device: str, movie: str) -> Self:
        """Mount the shipped tracker's own linked graph beside the RAW video it was read from."""
        mounted, matching = DriftDiagnosis._linked(root, device, movie)  # noqa: SLF001
        path = next(video for video in root.videos("train") if video.stem == movie)
        return cls(
            video=CellVideo.from_ome_zarr(path),
            prediction=mounted.nodes,
            truth=mounted.truth,
            matching=matching,
            spacing=mounted.spacing,
        )

    def _report(self, radius_um: float) -> None:
        """Score both populations, at detection centres and re-centred on the brightest voxel."""
        gate_um = TrackerConfig.shipped().linker.gate_um
        populations = (
            ("mislinked", CandidatePairs.mislinked(self.prediction, self.truth, self.matching)),
            ("correct", CandidatePairs.correct(self.prediction, self.truth, self.matching, self.spacing, gate_um)),
        )
        voxels = RawBoxes._size(self.spacing, radius_um)  # noqa: SLF001
        logger.info("box radius %.2f um -> %s voxels (full resolution)", radius_um, voxels)
        for label, pairs in populations:
            boxes = [self._boxes(rows, radius_um) for rows in (pairs.source, pairs.true_target, pairs.rival)]
            for centring, trio in (("detection centre", boxes), ("peak aligned", [box.aligned() for box in boxes])):
                self._log(label, centring, IdentitySimilarity.of(*trio))
            if label == "correct":
                rotation = RotationGain.of(boxes[0], boxes[1], angles_deg=(0.0, -30.0, -15.0, 15.0, 30.0))
                logger.info(
                    "%-10s rotation gain   n=%4d  identity corr %.3f -> best-rotated %.3f  median gain %+.3f",
                    label,
                    rotation.count(),
                    float(np.median(rotation.identity)) if rotation.count() else float("nan"),
                    float(np.median(rotation.best_rotated)) if rotation.count() else float("nan"),
                    rotation.median_gain(),
                )
            read_box = partial(self._boxes, radius_um=radius_um)
            ranking = InGateRanking.of(self.prediction, self.spacing, read_box, pairs, gate_um)
            logger.info(
                "%-10s in-gate rank    n=%4d  TOP-1 %.3f (chance %.3f)  mean candidates %.2f  median rank %.1f",
                label,
                ranking.count(),
                ranking.top_one(),
                ranking.chance(),
                float(np.mean(ranking.candidates)) if ranking.count() else float("nan"),
                float(np.median(ranking.true_rank)) if ranking.count() else float("nan"),
            )

    def _boxes(self, rows: Int[np.ndarray, "k"], radius_um: float) -> RawBoxes:
        """The raw boxes for one column of the candidate triples."""
        return RawBoxes.of(self.video, self.prediction, rows, self.spacing, radius_um)

    @staticmethod
    def _log(label: str, centring: str, similarity: IdentitySimilarity) -> None:
        """One population's verdict, with the similarity distributions behind it."""
        if not similarity.count():
            logger.info("%-10s %-16s n=0", label, centring)
            return
        logger.info(
            "%-10s %-16s n=%4d  prefers TRUE %.3f (null >0.5, rival nearer)  corr true %s  rival %s",
            label,
            centring,
            similarity.count(),
            similarity.prefers_true(),
            "/".join(f"{value:+.3f}" for value in np.percentile(similarity.true_similarity, _PERCENTILES)),
            "/".join(f"{value:+.3f}" for value in np.percentile(similarity.rival_similarity, _PERCENTILES)),
        )


def main() -> None:
    """Ask whether full-resolution appearance separates the cells the learned features cannot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Raw full-resolution appearance identity gate.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE)
    parser.add_argument("--radius", type=float, default=BOX_RADIUS_UM)
    args = parser.parse_args()

    diagnosis = IdentityDiagnosis._of(DataRoot.from_config(args.config), args.device, args.movie)  # noqa: SLF001
    diagnosis._report(args.radius)  # noqa: SLF001


if __name__ == "__main__":
    main()
