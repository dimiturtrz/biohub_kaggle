"""How much of a target's probability mass comes from sources the linker could actually have chosen.

The edge head's affinity is a softmax across SOURCES: for a target, `P[i, j]` is its parent probability
normalised over the candidate parents. Which sources are in that denominator is decided at inference by
`EdgeGap`, which takes EVERY detection of frame `t` — ~743 on the dense movie. Our training corpora supply a
handful: the annotated pairs carry ~12 sources a frame and the detected corpus 2.87 per pair, because both
supply only the sources whose successor the annotation can adjudicate.

So the head is trained to normalise against a few competitors and deployed normalising against hundreds, on
the one axis the deployed scorer reads. This module measures the size of that gap in the currency that
matters — probability MASS, not source counts — by asking what fraction of each target column sits on sources
inside the linker's own gate, i.e. on the parents the assignment could have picked.

It is the gate on whether a wider training matrix is worth building. If nearly all the mass is already
in-gate, then the far sources are numerically irrelevant, a corpus widened to the in-gate NEIGHBOURHOOD
reproduces the deployed denominator cheaply, and the mismatch is fixable. If the tail is heavy, no affordable
corpus reproduces it and the honest response is to change the normalisation rather than to enlarge the matrix.

Nothing here reaches past the public affinity: the matrix is ALREADY normalised over every source, so summing
a column's in-gate entries IS the fraction, with no access to logits and no second forward pass.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from jaxtyping import Float

from celltrack.affinity import EdgeAffinity
from celltrack.eval.dense_diagnosis import DENSE_MOVIE, DenseDiagnosis
from celltrack.eval.proxy import TestMovieProxy
from celltrack.operating_point import TrackerConfig
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_REPORTED_QUANTILES = (5.0, 25.0, 50.0, 75.0, 95.0)


@dataclass(frozen=True)
class SoftmaxPopulation:
    """Per target, the share of its parent-probability mass held by sources within the linker's gate.

    One entry per target column of every gap, so the summary statistics weight targets equally — which is the
    right weighting, because a target is what the softmax normalises and what the linker assigns a parent to.
    """

    in_gate_mass: Float[np.ndarray, " t"]
    competitors: Float[np.ndarray, " t"]
    sources_per_gap: float

    @classmethod
    def of(cls, detections: TrackGraph, affinity: EdgeAffinity, spacing: Spacing, gate_um: float) -> Self:
        """Measure every gap of one video's detections against its own affinity."""
        positions_um = spacing.to_micrometres(detections.positions())
        timepoints = detections.timepoints()
        mass: list[Float[np.ndarray, " t"]] = []
        counts: list[Float[np.ndarray, " t"]] = []
        sources: list[int] = []
        for timepoint in np.unique(timepoints)[:-1].tolist():
            probability = affinity.probabilities(timepoint)
            if probability is None:
                continue
            source_rows = np.flatnonzero(timepoints == timepoint)
            target_rows = np.flatnonzero(timepoints == timepoint + 1)
            in_gate = cls._in_gate(positions_um[source_rows], positions_um[target_rows], gate_um)
            mass.append((probability * in_gate).sum(axis=0))
            counts.append(in_gate.sum(axis=0).astype(np.float64))
            sources.append(len(source_rows))
        return cls(
            in_gate_mass=np.concatenate(mass) if mass else np.empty(0),
            competitors=np.concatenate(counts) if counts else np.empty(0),
            sources_per_gap=float(np.mean(sources)) if sources else 0.0,
        )

    @staticmethod
    def _in_gate(
        sources_um: Float[np.ndarray, "s 3"], targets_um: Float[np.ndarray, "t 3"], gate_um: float
    ) -> Float[np.ndarray, "s t"]:
        """Which (source, target) pairs sit within the gate — the parents the assignment could have chosen."""
        separation = np.linalg.norm(sources_um[:, None, :] - targets_um[None, :, :], axis=-1)
        return (separation <= gate_um).astype(np.float64)

    def report(self, name: str) -> None:
        """Log the mass distribution beside the competitor counts it is spread over."""
        if not len(self.in_gate_mass):
            logger.info("%s: no scored gaps", name)
            return
        quantiles = np.percentile(self.in_gate_mass, _REPORTED_QUANTILES)
        logger.info(
            "%s: %d targets | sources/gap %.0f (the deployed denominator) | in-gate competitors mean %.1f",
            name,
            len(self.in_gate_mass),
            self.sources_per_gap,
            float(self.competitors.mean()),
        )
        logger.info(
            "  in-gate softmax mass: mean %.4f | p5 %.4f p25 %.4f median %.4f p75 %.4f p95 %.4f",
            float(self.in_gate_mass.mean()),
            *(float(value) for value in quantiles),
        )


def main() -> None:
    """Measure the deployed softmax denominator on one movie, at the shipped operating point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="In-gate share of the source-softmax mass.")
    parser.add_argument("--config", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--movie", default=DENSE_MOVIE)
    args = parser.parse_args()

    config = TrackerConfig.shipped()
    root = DataRoot.from_config(args.config)
    proxy = TestMovieProxy.load(root, (args.movie,))
    tracker = DenseDiagnosis.configured(root, args.device, config)
    path = proxy.paths[0]
    detections = tracker.run(path.name, path)
    affinity = tracker.edge_scorer.affinities(path, detections, args.device)

    SoftmaxPopulation.of(detections, affinity, proxy.spacing, config.linker.gate_um).report(args.movie)


if __name__ == "__main__":
    main()
