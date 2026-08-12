"""The tissue's velocity field, estimated from unassociated detections by displacement voting.

WHY THIS EXISTS. Whole-frame drift is 61-92% of the mean annotated step, and re-fitting Furth's law in a
drift-removed frame collapses the velocity correlation time on the 6bba movies (9.3 -> 3.4 frames on the dense
one). So most of what looks like persistent cell motion is the tissue being carried. A drift that is UNIFORM
shifts every linking candidate equally and cannot disambiguate anything; a drift that VARIES IN SPACE does not
cancel between candidates, and correcting it changes which successor looks nearest. That is the whole reason to
estimate a field rather than a constant.

THE ESTIMATOR NEEDS NO ASSOCIATION, which is what makes it usable at all — the association is the thing we are
trying to fix, so anything requiring it first would be circular. Pair every detection at `t` with every
detection at `t+1` inside the linker's own gate and let each pair VOTE its displacement. Most pairs are wrong,
and that is fine: neighbouring cells carried by the same tissue produce nearly the SAME true displacement, so
the true votes pile up at one point of displacement space while wrong pairings scatter, their partners sitting
at essentially arbitrary relative positions. The pile is the drift. This is particle image velocimetry done on
points rather than pixels, with the vote-for-a-parameter structure of a Hough transform.

THE MODE, NOT THE MEAN. Roughly one vote in four is a true pair here (3.8 in-gate candidates per source), so an
average is three-quarters noise and gets dragged around by it. Scattered votes never accumulate anywhere, so a
peak-finder ignores them by construction.

IT IS BUILT TO REMOVE AS LITTLE AS POSSIBLE. A field flexible enough to follow individual cells would subtract
their own motion and destroy the residual the linker needs — the correction would make the individual movement
dumber rather than clearer. The guard is the kernel width: a wide kernel can only express motion shared over a
wide region, so the WIDEST kernel that still whitens the residual is the least invasive one that does the job,
and that is the selection rule rather than a tuned constant. `sharpness` is the second guard: where no votes
agree, the peak is flat, and the field reports that instead of inventing a vector.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

from celltrack.analysis.localisation import CandidatePairs
from celltrack.analysis.motion_statistics import GATE_UM
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.operating_point import TrackerConfig
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

# Displacement-space bin for the vote histogram. The read-out lattice is 1.625 um isotropic, so a finer bin
# than half of that would split one lattice displacement across neighbouring bins and blunt the very peak the
# estimator looks for; this is that lattice halved.
_VOTE_BIN_UM = 0.8125
# How far around the winning bin the refinement averages — one bin, so the refinement is a sub-bin centroid of
# the peak rather than a second, wider smoothing.
_REFINE_BINS = 1.0
# A vote whose weight falls below this share of the maximum contributes nothing an anchor can see; dropping it
# keeps the per-anchor histogram to the cells actually near that anchor.
_WEIGHT_FLOOR = 1e-3
_PERCENTILES = (10.0, 50.0, 90.0)


@dataclass(frozen=True)
class DisplacementVotes:
    """Every in-gate `t -> t+1` detection pair, as a vote: where it was cast, and what it claims.

    The gate is the LINKER's own `max_distance_um`, deliberately: a pair the linker could never choose is not
    evidence about the motion the linker has to explain, and admitting it would only add background to the
    histogram. Both frames' detections come from the same read-out, so no association is used or implied.
    """

    origins_um: Float[np.ndarray, "v 3"]
    displacements_um: Float[np.ndarray, "v 3"]

    @classmethod
    def of(
        cls,
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
        timepoint: int,
        gate_um: float = GATE_UM,
    ) -> Self:
        """Every candidate displacement across the gap leaving `timepoint`."""
        sources = np.flatnonzero(timepoints == timepoint)
        targets = np.flatnonzero(timepoints == timepoint + 1)
        if not len(sources) or not len(targets):
            return cls(np.zeros((0, 3)), np.zeros((0, 3)))
        offsets = positions_um[targets][None, :, :] - positions_um[sources][:, None, :]
        within = np.linalg.norm(offsets, axis=2) <= gate_um
        source_index, _ = np.nonzero(within)
        return cls(origins_um=positions_um[sources][source_index], displacements_um=offsets[within])

    def count(self) -> int:
        """How many votes were cast."""
        return len(self.displacements_um)


@dataclass(frozen=True)
class FieldSettings:
    """Where the field is evaluated and how far a vote reaches — the knobs that always travel together.

    `sigma_um` is the only real choice here and it is the one the whole method turns on: a wide kernel can only
    express motion shared across a wide region, so it removes the tissue's common component and leaves each
    cell's own motion intact, while a narrow one flexes enough to follow individual cells and subtracts the very
    residual the linker needs. The selection rule is therefore the WIDEST kernel that still explains the
    annotated steps, and it was measured — 5 um over-corrects (explained -0.708), 12-15 um is a plateau
    (+0.448/+0.446). The others are placement and plumbing: the anchor spacing only has to be fine enough not to
    alias the kernel, the gate is the linker's own, and `device` is free (CPU measured 2x FASTER than CUDA at
    this size, since 2.8k votes cannot amortise kernel launches).
    """

    anchors_um: Float[np.ndarray, "a 3"]
    sigma_um: float
    device: str = "cpu"
    gate_um: float = GATE_UM

    @classmethod
    def over(
        cls,
        positions_um: Float[np.ndarray, "n 3"],
        sigma_um: float,
        spacing_um: float,
        device: str = "cpu",
    ) -> Self:
        """Settings whose anchor grid spans the given detections."""
        return cls(anchors_um=DriftField.anchor_grid(positions_um, spacing_um), sigma_um=sigma_um, device=device)


@dataclass(frozen=True)
class DriftField:
    """A drift vector anywhere in the volume, plus how much the votes there actually agreed.

    Held as values on a regular anchor grid and read back by Gaussian interpolation, so the field is continuous
    with no block edges and the GLOBAL drift is simply its volume average — there is no separate global path to
    write or tune. `sharpness` is the winning bin's share of the local vote weight: high where the tissue moves
    coherently, near the background level where it does not, so a caller can refuse a vector rather than trust
    a flat peak.
    """

    anchors_um: Float[np.ndarray, "a 3"]
    vectors_um: Float[np.ndarray, "a 3"]
    sharpness: Float[np.ndarray, "a"]
    sigma_um: float

    @classmethod
    def of(cls, votes: DisplacementVotes, settings: FieldSettings) -> Self:
        """Estimate the field at every anchor at once — the per-anchor histogram is a single scatter-add.

        Every anchor bins the SAME votes and differs only in how it weights them, so the whole field is one
        `index_add_` of a `(anchors, votes)` weight matrix into an `(anchors, bins)` accumulator. Written as a
        loop this was ~1.5 s per gap and made the training corpus an eight-hour precompute; batched it is
        ~8 ms, which is why the field is recomputed rather than cached.
        """
        anchors_um, device = settings.anchors_um, settings.device
        if not votes.count() or not len(anchors_um):
            return cls(anchors_um, np.zeros_like(anchors_um), np.zeros(len(anchors_um)), settings.sigma_um)
        anchors = torch.as_tensor(anchors_um, dtype=torch.float32, device=device)
        origins = torch.as_tensor(votes.origins_um, dtype=torch.float32, device=device)
        displacements = torch.as_tensor(votes.displacements_um, dtype=torch.float32, device=device)

        weights = torch.exp(-0.5 * (torch.cdist(anchors, origins) / settings.sigma_um) ** 2)
        mass = cls._histogram(displacements, weights, settings.gate_um)
        peak = cls._unbin(mass.argmax(dim=1), settings.gate_um)
        offsets = displacements[None, :, :] - peak[:, None, :]
        near = torch.linalg.norm(offsets, dim=2) <= _REFINE_BINS * _VOTE_BIN_UM
        held = weights * near
        totals = held.sum(dim=1, keepdim=True)
        refined = torch.where(totals > 0, held @ displacements / totals.clamp(min=_WEIGHT_FLOOR), peak)
        sharpness = mass.amax(dim=1) / weights.sum(dim=1).clamp(min=_WEIGHT_FLOOR)
        return cls(
            anchors_um=anchors_um,
            vectors_um=refined.cpu().numpy().astype(np.float64),
            sharpness=sharpness.cpu().numpy().astype(np.float64),
            sigma_um=settings.sigma_um,
        )

    @staticmethod
    def _histogram(
        displacements: Float[Tensor, "v 3"], weights: Float[Tensor, "a v"], gate_um: float
    ) -> Float[Tensor, "a b"]:
        """Weighted vote mass per displacement bin, per anchor — the one operation the whole estimate is."""
        per_axis = int(np.rint(2 * gate_um / _VOTE_BIN_UM)) + 1
        indices = (torch.round(displacements / _VOTE_BIN_UM) + per_axis // 2).long().clamp(0, per_axis - 1)
        flat = (indices[:, 0] * per_axis + indices[:, 1]) * per_axis + indices[:, 2]
        mass = torch.zeros(len(weights), per_axis**3, device=weights.device, dtype=weights.dtype)
        return mass.index_add_(1, flat, weights)

    @staticmethod
    def _unbin(flat: Int[Tensor, "a"], gate_um: float) -> Float[Tensor, "a 3"]:
        """The displacement at the centre of each anchor's winning bin — the histogram index inverted."""
        per_axis = int(np.rint(2 * gate_um / _VOTE_BIN_UM)) + 1
        axes = [(flat // per_axis**power) % per_axis for power in (2, 1, 0)]
        return (torch.stack(axes, dim=1) - per_axis // 2).float() * _VOTE_BIN_UM

    @staticmethod
    def anchor_grid(positions_um: Float[np.ndarray, "n 3"], spacing_um: float) -> Float[np.ndarray, "a 3"]:
        """A regular grid spanning the detections, at `spacing_um` — where the field is evaluated."""
        low, high = positions_um.min(axis=0), positions_um.max(axis=0)
        axes = [np.arange(lo, hi + spacing_um, spacing_um) for lo, hi in zip(low, high, strict=True)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    def at(self, points_um: Float[np.ndarray, "p 3"]) -> Float[np.ndarray, "p 3"]:
        """The field at arbitrary points — Gaussian interpolation over the anchors, so it stays continuous.

        Weighted by the SAME sigma the field was estimated with, so a point reads the anchors whose votes it
        would itself have contributed to; a narrower interpolation would invent structure the estimate never
        had, and a wider one would blur what it did.
        """
        if not len(points_um):
            return np.zeros((0, 3))
        separation = np.linalg.norm(points_um[:, None, :] - self.anchors_um[None, :, :], axis=2)
        weights = np.exp(-0.5 * (separation / self.sigma_um) ** 2)
        totals = weights.sum(axis=1, keepdims=True)
        return np.where(totals > 0, weights @ self.vectors_um / np.maximum(totals, _WEIGHT_FLOOR), 0.0)

    def global_drift(self) -> Float[np.ndarray, "3"]:
        """The field's volume average — the uniform component, with no separate machinery behind it."""
        return self.vectors_um.mean(axis=0)


@dataclass(frozen=True)
class SyntheticRecovery:
    """Impose a KNOWN field on real detections and check the estimator gets it back.

    The control is DIFFERENTIAL: whatever real drift the movie already has is present in both arms, so the
    difference between the estimate with and without the imposed field isolates the imposed one. That is what
    makes it a test of the ESTIMATOR rather than a restatement of the data.
    """

    imposed_um: Float[np.ndarray, "a 3"]
    recovered_um: Float[np.ndarray, "a 3"]

    @classmethod
    def of(
        cls,
        positions_um: Float[np.ndarray, "n 3"],
        timepoints: Int[np.ndarray, "n"],
        timepoint: int,
        settings: FieldSettings,
        amplitude_um: float,
    ) -> Self:
        """Shear the `t+1` frame along x in proportion to z, then recover the difference."""
        plain = DriftField.of(DisplacementVotes.of(positions_um, timepoints, timepoint), settings)
        shifted = positions_um.copy()
        moving = timepoints == timepoint + 1
        shifted[moving, 2] += cls._shear(positions_um[moving], positions_um, amplitude_um)
        sheared = DriftField.of(DisplacementVotes.of(shifted, timepoints, timepoint), settings)
        imposed = np.zeros_like(settings.anchors_um)
        imposed[:, 2] = cls._shear(settings.anchors_um, positions_um, amplitude_um)
        return cls(imposed_um=imposed, recovered_um=sheared.vectors_um - plain.vectors_um)

    @staticmethod
    def _shear(points_um: Float[np.ndarray, "k 3"], extent_of: Float[np.ndarray, "n 3"], amplitude_um: float):
        """An x-displacement rising linearly with z across the volume — a field a constant cannot express."""
        low, high = extent_of[:, 0].min(), extent_of[:, 0].max()
        return amplitude_um * (points_um[:, 0] - low) / max(high - low, _WEIGHT_FLOOR)

    def error_um(self) -> Float[np.ndarray, "a"]:
        """Per anchor, how far the recovered field sits from the one imposed."""
        return np.linalg.norm(self.recovered_um - self.imposed_um, axis=1)


@dataclass(frozen=True)
class AnnotatedResidual:
    """How much of the ANNOTATED displacements the field explains — the payoff number, on held-out truth.

    The field is built from detections and never sees an annotation, so scoring it against annotated steps is
    a genuine out-of-sample read. `explained` is the share of squared displacement the field accounts for; the
    residual is what a linker would still have to resolve, and it is the quantity the whole correction is for.

    A NEGATIVE `explained` is meaningful rather than an error: it says the field made the annotated steps
    LONGER, i.e. it removed motion those cells did not share — exactly the over-correction that would make the
    individual movement dumber, and the reason this is reported rather than assumed.
    """

    raw_um: Float[np.ndarray, "e"]
    residual_um: Float[np.ndarray, "e"]

    @classmethod
    def of(cls, truth: TrackGraph, spacing: Spacing, field_at: dict[int, DriftField]) -> Self:
        """Annotated step lengths, before and after subtracting the field at each source."""
        positions = spacing.to_micrometres(truth.positions())
        timepoints = truth.timepoints()
        raw: list[float] = []
        residual: list[float] = []
        for source, target in Adjacency.of(truth).edges:
            gap = int(timepoints[source])
            if gap not in field_at or timepoints[target] != gap + 1:
                continue
            step = positions[target] - positions[source]
            raw.append(float(np.linalg.norm(step)))
            residual.append(float(np.linalg.norm(step - field_at[gap].at(positions[source][None])[0])))
        return cls(raw_um=np.asarray(raw), residual_um=np.asarray(residual))

    def explained(self) -> float:
        """Share of squared annotated displacement the field accounts for; negative means over-correction."""
        if not len(self.raw_um):
            return float("nan")
        return float(1.0 - (self.residual_um**2).sum() / (self.raw_um**2).sum())


@dataclass(frozen=True)
class CandidateReordering:
    """Does drift-corrected geometry prefer the true successor where raw distance preferred the rival?

    THE DECISIVE TEST, and it needs no solver: a mislink is lost because the wrong partner is NEARER (median
    +6.44 um, rival nearer in 97.3%). Subtracting the tissue's own motion asks the question the linker should
    have been asked — how far is this target from where the source was GOING, rather than from where it was.
    A uniform drift cancels between two candidates of the same source and can change nothing; only the part of
    the field that differs over the separation between them can reorder anything, which is why this is run on a
    FIELD and is a real test rather than a restatement.

    `rescued` counts mislinks the correction reorders in our favour. `broken` counts CORRECT edges it reorders
    against us, and it is the half that decides whether this is worth shipping — a correction that rescues ten
    mislinks while breaking twenty reproduced edges is a loss, and only the pair of numbers can say.
    """

    raw_gap_um: Float[np.ndarray, "k"]
    corrected_gap_um: Float[np.ndarray, "k"]

    @classmethod
    def of(
        cls,
        prediction: TrackGraph,
        pairs: CandidatePairs,
        spacing: Spacing,
        fields: dict[int, DriftField],
    ) -> Self:
        """`d(true) - d(rival)` before and after correcting the source's position by the field."""
        positions = spacing.to_micrometres(prediction.positions())
        timepoints = prediction.timepoints()
        raw: list[float] = []
        corrected: list[float] = []
        for source, true_target, rival in zip(pairs.source, pairs.true_target, pairs.rival, strict=True):
            gap = int(timepoints[source])
            if gap not in fields:
                continue
            origin = positions[source]
            predicted = origin + fields[gap].at(origin[None])[0]
            raw.append(cls._gap(origin, positions[true_target], positions[rival]))
            corrected.append(cls._gap(predicted, positions[true_target], positions[rival]))
        return cls(raw_gap_um=np.asarray(raw), corrected_gap_um=np.asarray(corrected))

    @staticmethod
    def _gap(origin: Float[np.ndarray, "3"], true_target: Float[np.ndarray, "3"], rival: Float[np.ndarray, "3"]):
        """How much further the true successor sits than the rival — positive means the rival looks better."""
        return float(np.linalg.norm(true_target - origin) - np.linalg.norm(rival - origin))

    def rescued(self) -> int:
        """Pairs where the rival was nearer and the correction makes the TRUE successor nearer."""
        return int(((self.raw_gap_um > 0.0) & (self.corrected_gap_um < 0.0)).sum())

    def broken(self) -> int:
        """Pairs where the true successor was nearer and the correction makes the RIVAL nearer."""
        return int(((self.raw_gap_um < 0.0) & (self.corrected_gap_um > 0.0)).sum())

    def count(self) -> int:
        """How many pairs the field could be evaluated on."""
        return len(self.raw_gap_um)


@dataclass(frozen=True)
class DriftDiagnosis:
    """One movie's nodes and annotations, mounted once — what every read below is computed from.

    Two mounts, because the two questions need different node sets. `detected` reads the DETECTOR only, which
    is all the field estimator needs and skips the affinity and the solver entirely. `linked` runs the shipped
    tracker, because the reordering test has to correct the geometry the LINKER actually saw — one read-out for
    both, so the two cannot drift apart.
    """

    nodes: TrackGraph
    truth: TrackGraph
    spacing: Spacing
    device: str

    @classmethod
    def _detected(cls, root: DataRoot, device: str, movie: str) -> Self:
        """Detections from the cached logits — no linking, no affinity, seconds rather than minutes."""
        config = TrackerConfig.shipped()
        proxy = TestMovieProxy.load(root, (movie,))
        tracker = DenseDiagnosis.configured(root, device, config)
        nodes = tracker.detector.nodes(proxy.paths[0].name, proxy.paths[0], config.threshold)
        return cls(nodes=nodes, truth=proxy.truths[0].graph, spacing=proxy.spacing, device=device)

    @classmethod
    def _linked(cls, root: DataRoot, device: str, movie: str) -> tuple[Self, NodeMatching]:
        """The shipped tracker's own linked graph, with its annotations matched — what the reorder test needs."""
        config = TrackerConfig.shipped()
        proxy = TestMovieProxy.load(root, (movie,))
        prediction = DenseDiagnosis.configured(root, device, config).run(proxy.paths[0].name, proxy.paths[0])
        mounted = cls(nodes=prediction, truth=proxy.truths[0].graph, spacing=proxy.spacing, device=device)
        return mounted, DistanceMatcher(spacing=proxy.spacing).match(prediction, proxy.truths[0].graph)

    def _positions_um(self) -> Float[np.ndarray, "n 3"]:
        """The nodes in micrometres — the space every field quantity lives in."""
        return self.spacing.to_micrometres(self.nodes.positions())

    def _fields(self, settings: FieldSettings, gaps: list[int]) -> dict[int, DriftField]:
        """One field per frame gap, recomputed rather than cached — it is milliseconds a gap."""
        positions, timepoints = self._positions_um(), self.nodes.timepoints()
        return {gap: DriftField.of(DisplacementVotes.of(positions, timepoints, gap), settings) for gap in gaps}

    def _sweep(self, sigmas: list[float], spacing_um: float, gap_count: int, shear_um: float) -> None:
        """Report, per kernel width, what the field claims and whether held-out annotations agree with it."""
        positions, timepoints = self._positions_um(), self.nodes.timepoints()
        gaps = [int(gap) for gap in np.unique(timepoints)[:gap_count]]
        logger.info("%d nodes, %d gaps", len(positions), len(gaps))
        for sigma in sigmas:
            settings = FieldSettings.over(positions, sigma, spacing_um, self.device)
            fields = self._fields(settings, gaps)
            recovery = SyntheticRecovery.of(positions, timepoints, gaps[0], settings, shear_um)
            self._report_sweep(sigma, fields, recovery, AnnotatedResidual.of(self.truth, self.spacing, fields))

    @staticmethod
    def _report_sweep(
        sigma: float,
        fields: dict[int, DriftField],
        recovery: SyntheticRecovery,
        annotated: AnnotatedResidual,
    ) -> None:
        """One kernel width's line: what it claims, whether it recovers a known field, what truth says."""
        sharp = np.concatenate([field.sharpness for field in fields.values()])
        magnitude = np.concatenate([np.linalg.norm(field.vectors_um, axis=1) for field in fields.values()])
        logger.info(
            "sigma %5.1f um | drift p10/50/90 %s um | sharpness %s | shear recovery median err %.2f um "
            "(imposed up to %.2f) | annotated step %.3f -> %.3f um, explained %+.3f",
            sigma,
            "/".join(f"{value:.2f}" for value in np.percentile(magnitude, _PERCENTILES)),
            "/".join(f"{value:.3f}" for value in np.percentile(sharp, _PERCENTILES)),
            float(np.median(recovery.error_um())),
            float(np.abs(recovery.imposed_um).max()),
            float(np.median(annotated.raw_um)),
            float(np.median(annotated.residual_um)),
            annotated.explained(),
        )

    def _held_out(self, sigmas: list[float], spacing_um: float, gap_count: int) -> None:
        """Score the field on annotated steps after removing those very cells from the votes.

        The sweep's `explained` is already out-of-sample in the sense that no annotation is used to BUILD the
        field — but the annotated cells are themselves detections, so their own displacements are among the
        votes. That is 1229 of 74347, about 1.6%, so the self-reference is small; small is not zero, and the
        number underwrites a training decision. Dropping every detection the matcher paired with an annotation
        removes it entirely, at the cost of a field estimated from 98.4% of the votes.
        """
        positions, timepoints = self._positions_um(), self.nodes.timepoints()
        matched = DistanceMatcher(spacing=self.spacing).match(self.nodes, self.truth).is_matched()
        kept = ~matched
        logger.info("held-out field: %d of %d detections dropped as annotated", int(matched.sum()), len(positions))
        gaps = [int(gap) for gap in np.unique(timepoints)[:gap_count]]
        for sigma in sigmas:
            settings = FieldSettings.over(positions, sigma, spacing_um, self.device)
            votes = {gap: DisplacementVotes.of(positions[kept], timepoints[kept], gap) for gap in gaps}
            fields = {gap: DriftField.of(vote, settings) for gap, vote in votes.items()}
            annotated = AnnotatedResidual.of(self.truth, self.spacing, fields)
            logger.info(
                "sigma %5.1f um | HELD OUT annotated step %.3f -> %.3f um, explained %+.3f",
                sigma,
                float(np.median(annotated.raw_um)),
                float(np.median(annotated.residual_um)),
                annotated.explained(),
            )

    def _reorder(self, matching: NodeMatching, sigma: float, spacing_um: float) -> None:
        """Ask whether the field reorders the linker's mislinked candidates — and what it breaks doing it."""
        settings = FieldSettings.over(self._positions_um(), sigma, spacing_um, self.device)
        fields = self._fields(settings, [int(gap) for gap in np.unique(self.nodes.timepoints())])
        gate_um = TrackerConfig.shipped().linker.gate_um
        logger.info("field over %d gaps at sigma %.1f um, %d anchors", len(fields), sigma, len(settings.anchors_um))
        for label, pairs in (
            ("mislinked", CandidatePairs.mislinked(self.nodes, self.truth, matching)),
            ("correct", CandidatePairs.correct(self.nodes, self.truth, matching, self.spacing, gate_um)),
        ):
            reordering = CandidateReordering.of(self.nodes, pairs, self.spacing, fields)
            logger.info(
                "%-10s n=%4d  rescued %3d  broken %3d  gap median %+.3f -> %+.3f um",
                label,
                reordering.count(),
                reordering.rescued(),
                reordering.broken(),
                float(np.median(reordering.raw_gap_um)) if reordering.count() else float("nan"),
                float(np.median(reordering.corrected_gap_um)) if reordering.count() else float("nan"),
            )


def main() -> None:
    """Estimate the tissue drift field from detections, validate it, and score it on annotated steps."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Displacement-voting estimate of the tissue drift field.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--movie", default=DENSE_MOVIE)
    parser.add_argument("--sigma", type=float, nargs="+", default=[8.0, 12.0, 15.0])
    parser.add_argument("--anchor-spacing", type=float, default=15.0)
    parser.add_argument("--gaps", type=int, default=20)
    parser.add_argument("--shear", type=float, default=3.0)
    parser.add_argument("--reorder", action="store_true", help="run the mislink reordering test instead")
    parser.add_argument("--held-out", action="store_true", help="score with annotated cells cut from the votes")
    args = parser.parse_args()

    root = DataRoot.from_config(args.config)
    if args.reorder:
        mounted, matching = DriftDiagnosis._linked(root, args.device, args.movie)  # noqa: SLF001
        mounted._reorder(matching, args.sigma[0], args.anchor_spacing)  # noqa: SLF001
        return
    diagnosis = DriftDiagnosis._detected(root, args.device, args.movie)  # noqa: SLF001
    if args.held_out:
        diagnosis._held_out(args.sigma, args.anchor_spacing, args.gaps)  # noqa: SLF001
        return
    diagnosis._sweep(args.sigma, args.anchor_spacing, args.gaps, args.shear)  # noqa: SLF001


if __name__ == "__main__":
    main()
