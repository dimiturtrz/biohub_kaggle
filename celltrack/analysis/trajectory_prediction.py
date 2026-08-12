"""Does the source's OWN past motion pick the true successor — the Kalman criterion, at its ceiling.

THE QUESTION. Every mislink loses on both terms of the per-frame cost: the decoy is nearer AND scores higher
affinity than the true successor. Both are local, single-frame cues, and both are wrong here by construction —
the source is a fast mover in crowded tissue, so the true successor left the neighbourhood and a stranger fills
the spot. The one cue neither term reads is the source's own TRAJECTORY: where it was heading. A Kalman-style
predictor advances the source by its recent velocity and asks which candidate lands where the cell was going,
not which is nearest NOW. If the true successor sits nearer the PREDICTED position than the decoy does, motion
history rescues the case; if it does not, no filter can.

THIS MEASURES THE CEILING, deliberately. The velocity is taken from the source's GROUND-TRUTH incoming step,
not from our own reconstructed track — an oracle Kalman. Using our predicted history would be circular (the
history may itself be mislinked) and would confound "the signal is absent" with "our history is wrong". The
oracle answers the prior question: is the trajectory signal THERE at all. A realistic predictor can only do
worse, so an oracle null closes the direction and an oracle win bounds what a built filter could reach.

WHY IT MIGHT FAIL EVEN AT THE CEILING. The motion analysis measured the individual cell's own motion at ~1 um
per frame with a ~3-frame correlation time once tissue drift is removed — near the localisation noise floor.
So a single-step velocity may be too noisy to predict one frame ahead even when it is real. The drift-corrected
variant exists because the tissue's shared motion is the large, coherent part: subtracting it leaves the cell's
own displacement, which is what actually distinguishes one candidate from another (a shared drift moves every
candidate alike and cannot disambiguate).

THE CONTROL is the same predictor over CORRECT edges. If prediction ranks the true successor first there but
not on the mislinks, the mislinks are trajectory-ambiguous specifically; if it fails on both, the predictor is
simply too weak to trust. The raw-distance ordering travels alongside so a "rescue" is counted only where
distance was actually inverted — otherwise prediction would take credit for cases distance already got right.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float, Int

from celltrack.analysis.drift_field import DisplacementVotes, DriftDiagnosis, DriftField, FieldSettings
from celltrack.analysis.localisation import CandidatePairs
from celltrack.eval.dense_diagnosis import DENSE_MOVIE
from celltrack.operating_point import TrackerConfig
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import UNMATCHED, NodeMatching
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_ANCHOR_SPACING_UM = 15.0
_DRIFT_SIGMA_UM = 15.0


@dataclass(frozen=True)
class PredictedStep:
    """For each candidate triple, where the source was HEADING, and how each candidate sits against it.

    The prediction is `source + velocity`, the velocity being the source's ground-truth incoming step (its
    real previous displacement), optionally with the tissue drift at the source removed so only the cell's own
    motion predicts. A triple with no ground-truth predecessor for the source carries no velocity and is left
    out rather than predicted from a zero step, which would just reproduce the raw-distance ordering.
    """

    predicted_gap_um: Float[np.ndarray, "k"]
    raw_gap_um: Float[np.ndarray, "k"]
    speed_um: Float[np.ndarray, "k"]
    """The magnitude of the velocity each prediction advanced by — how far the predicted point moved."""
    alignment: Float[np.ndarray, "k"]
    """cos(velocity, true outgoing step): +1 = the source was already heading at the true successor, so a
    persistently-moving cell; ~0 = the past step says nothing about where it went; <0 = it reversed. This is
    what decides whether the prediction CAN land on the truth — a large, aligned velocity lands on it, a small
    or misaligned one cannot however the cost is written."""

    @classmethod
    def of(
        cls,
        diagnosis: "TrajectoryDiagnosis",
        pairs: CandidatePairs,
        field_by_gap: dict[int, DriftField] | None,
        history: int = 1,
    ) -> Self:
        """Advance each source by its GT velocity and measure `d(true) - d(decoy)` from the predicted point."""
        det_um = diagnosis.spacing.to_micrometres(diagnosis.prediction.positions())
        gt_um = diagnosis.spacing.to_micrometres(diagnosis.truth.positions())
        gt_of_det = diagnosis.matching.gt_rows
        predecessors = Adjacency.of(diagnosis.truth).predecessors
        timepoints = diagnosis.prediction.timepoints()
        predicted: list[float] = []
        raw: list[float] = []
        speed: list[float] = []
        alignment: list[float] = []
        for source, true_target, rival in zip(pairs.source, pairs.true_target, pairs.rival, strict=True):
            velocity = cls._gt_velocity(int(source), gt_of_det, predecessors, gt_um, history)
            if velocity is None:
                continue
            origin = det_um[source]
            if field_by_gap is not None:
                velocity = velocity - field_by_gap[int(timepoints[source])].at(origin[None])[0]
            outgoing = det_um[true_target] - origin
            predicted.append(cls._gap(origin + velocity, det_um[true_target], det_um[rival]))
            raw.append(cls._gap(origin, det_um[true_target], det_um[rival]))
            speed.append(float(np.linalg.norm(velocity)))
            alignment.append(cls._cosine(velocity, outgoing))
        return cls(
            predicted_gap_um=np.asarray(predicted),
            raw_gap_um=np.asarray(raw),
            speed_um=np.asarray(speed),
            alignment=np.asarray(alignment),
        )

    @staticmethod
    def _cosine(first: Float[np.ndarray, "3"], second: Float[np.ndarray, "3"]) -> float:
        """Cosine between two vectors, or 0 when either is degenerate — the persistence of the motion."""
        scale = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(first @ second / scale) if scale else 0.0

    @staticmethod
    def _gt_velocity(
        source_row: int,
        gt_of_det: Int[np.ndarray, "n"],
        predecessors: dict[int, tuple[int, ...]],
        gt_um: Float[np.ndarray, "m 3"],
        history: int = 1,
    ) -> Float[np.ndarray, "3"] | None:
        """The source's per-frame velocity, averaged over up to `history` ground-truth incoming steps.

        A single step is dominated by localisation noise: the individual motion (~1 um) is below the annotation's
        integer-voxel noise (~1.7 um), so one displacement is mostly noise. Averaging the last `history` steps
        beats that down as ~1/sqrt(history) while leaving a persistent velocity intact — the difference between
        a single-step Kalman and a trajectory fit, and what decides whether the past predicts the next step.
        """
        gt_row = int(gt_of_det[source_row])
        if gt_row == UNMATCHED:
            return None
        steps: list[Float[np.ndarray, "3"]] = []
        node = gt_row
        for _ in range(history):
            parents = predecessors.get(node)
            if not parents:
                break
            steps.append(gt_um[node] - gt_um[parents[0]])
            node = parents[0]
        return np.mean(steps, axis=0) if steps else None

    @staticmethod
    def _gap(origin: Float[np.ndarray, "3"], true_target: Float[np.ndarray, "3"], rival: Float[np.ndarray, "3"]):
        """`d(true) - d(rival)` from `origin`; positive means the rival is the nearer of the two."""
        return float(np.linalg.norm(true_target - origin) - np.linalg.norm(rival - origin))

    def rescued(self) -> int:
        """Cases the RAW distance got wrong (rival nearer) that PREDICTION gets right (true nearer)."""
        return int(((self.raw_gap_um > 0.0) & (self.predicted_gap_um < 0.0)).sum())

    def broken(self) -> int:
        """Cases the raw distance got right (true nearer) that prediction reverses."""
        return int(((self.raw_gap_um < 0.0) & (self.predicted_gap_um > 0.0)).sum())

    def prefers_true(self) -> float:
        """Share where the predicted point sits nearer the true successor than the decoy — the whole signal."""
        return float(np.mean(self.predicted_gap_um < 0.0)) if len(self.predicted_gap_um) else float("nan")

    def count(self) -> int:
        """How many triples carried a ground-truth velocity to predict from."""
        return len(self.predicted_gap_um)


@dataclass(frozen=True)
class TrajectoryDiagnosis:
    """One movie's mislinks and correct edges scored by the oracle Kalman criterion."""

    prediction: TrackGraph
    truth: TrackGraph
    matching: NodeMatching
    spacing: Spacing
    fields: dict[int, DriftField]

    @classmethod
    def _of(cls, root: DataRoot, device: str, movie: str) -> Self:
        """Mount the shipped tracker's linked graph, its annotations, and the drift field per gap."""
        mounted, matching = DriftDiagnosis._linked(root, device, movie)  # noqa: SLF001
        positions = mounted.spacing.to_micrometres(mounted.nodes.positions())
        timepoints = mounted.nodes.timepoints()
        settings = FieldSettings.over(positions, _DRIFT_SIGMA_UM, _ANCHOR_SPACING_UM, device)
        fields = {
            int(gap): DriftField.of(DisplacementVotes.of(positions, timepoints, int(gap)), settings)
            for gap in np.unique(timepoints)
        }
        return cls(mounted.nodes, mounted.truth, matching, mounted.spacing, fields)

    def _report(self) -> None:
        """Score both populations, with raw GT velocity and with tissue drift removed from it."""
        gate_um = TrackerConfig.shipped().linker.gate_um
        populations = (
            ("mislinked", CandidatePairs.mislinked(self.prediction, self.truth, self.matching)),
            ("correct", CandidatePairs.correct(self.prediction, self.truth, self.matching, self.spacing, gate_um)),
        )
        for label, pairs in populations:
            for variant, field, history in (
                ("1-step", None, 1),
                ("3-step fit", None, 3),
                ("3-step+drift", self.fields, 3),
            ):
                self._log(label, variant, PredictedStep.of(self, pairs, field, history))

    @staticmethod
    def _log(label: str, variant: str, step: PredictedStep) -> None:
        """One population and velocity variant: how many mislinks the trajectory prediction reorders."""
        if not step.count():
            logger.info("%-10s %-14s n=0", label, variant)
            return
        logger.info(
            "%-10s %-14s n=%4d  prefers TRUE %.3f  rescued %3d  raw gap %+.2f -> predicted %+.2f um  "
            "|v| %.2f um  align cos %+.2f",
            label,
            variant,
            step.count(),
            step.prefers_true(),
            step.rescued(),
            float(np.median(step.raw_gap_um)),
            float(np.median(step.predicted_gap_um)),
            float(np.median(step.speed_um)),
            float(np.median(step.alignment)),
        )


def main() -> None:
    """Measure whether the source's own trajectory rescues the mislinks — the Kalman criterion at its ceiling."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Oracle trajectory-prediction criterion on the dense mislinks.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE)
    args = parser.parse_args()

    TrajectoryDiagnosis._of(DataRoot.from_config(args.config), args.device, args.movie)._report()  # noqa: SLF001


if __name__ == "__main__":
    main()
