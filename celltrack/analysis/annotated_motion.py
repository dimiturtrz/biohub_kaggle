"""Motion read off the ANNOTATED centres alone, cut by the fate the tracker gave each edge — no detection in it.

The dense confusor's mislinks are described as fast movers, but that description comes from DETECTED positions,
which carry ~sqrt(2) times the per-node localisation error in every step. These two cuts measure motion on the
annotations themselves — the step length (was the move genuinely far?) and the turn (does the cell continue in
its direction of travel, or double back the way a wrong label would?) — so a claim about the mislinks' motion is
independent of the read-out. Both are functions of `(truth, prediction, matching, spacing)` and split the same
fate array, so they live together and are read off one tracker pass by `LocalisationDiagnosis`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from jaxtyping import Float, Int

from celltrack.eval.dense_diagnosis import DenseFateDiagnosis, Fate
from core.data.tracks import Adjacency, TrackGraph
from core.geometry import Spacing
from core.metrics.matching import NodeMatching


@dataclass(frozen=True)
class AnnotatedStepByFate:
    """How far the ANNOTATION says each edge's cell travelled, cut by whether we reproduced that edge.

    THE ARBITER between two readings of the elevated endpoint error. The mislinked edges are described as fast
    movers, but that description is taken from DETECTED positions — and if both endpoints are mislocalised by
    ~3.5 um in uncorrelated directions, the step between them inherits about sqrt(2) times that, which is the
    same size as the displacement being called fast. The apparent motion could therefore be manufactured by our
    own read-out rather than observed.

    This measures the step between ANNOTATED centres, which no detection touches. If the mislinked edges are
    long here too, the cells genuinely move fast and the endpoint error is a separate problem riding along. If
    they are ordinary here while long in the detections, the fast mover is our artefact, and the association
    failure is a localisation failure wearing a motion costume.
    """

    mislinked_um: Float[np.ndarray, "m"]
    correct_um: Float[np.ndarray, "c"]

    @classmethod
    def of(cls, truth: TrackGraph, prediction: TrackGraph, matching: NodeMatching, spacing: Spacing) -> Self:
        """Annotated per-edge displacement, split by the fate the tracker gave that edge."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        steps = truth.link_displacements(spacing)
        mislinked = (fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE)
        return cls(mislinked_um=steps[mislinked], correct_um=steps[fates == Fate.CORRECT])


@dataclass(frozen=True)
class AnnotatedTurnByFate:
    """The turn at each edge's source — a smooth continuation or an out-and-back reversal, off the annotation.

    Distance, learned features, and raw full-resolution appearance all rank the near rival above the true far
    successor on the mislinks, unanimously. When every physical cue agrees against the label, a fraction of the
    labels being WRONG is the parsimonious reading, and a wrong label carries a signature the cues do not: a
    genuine fast mover CONTINUES in its direction of travel — its incoming and outgoing steps align, cosine near
    +1 — while a swapped or spurious successor sits back where the cell came from, the outgoing step reversing the
    incoming one, cosine near -1. This measures that turn on ANNOTATED centres alone (no detection, no model) for
    the mislinked edges against the reproduced ones. A reversal share concentrated on the mislinks bounds how much
    of the dense ceiling is label noise the tracker cannot reach rather than association it genuinely gets wrong.

    A source with no annotated predecessor has no incoming step and is dropped: the turn is defined only over the
    covered share, reported so a reversal rate on ten edges is not read as a rate over all thirty-seven.
    """

    mislinked_cos: Float[np.ndarray, "m"]
    correct_cos: Float[np.ndarray, "c"]

    @classmethod
    def of(cls, truth: TrackGraph, prediction: TrackGraph, matching: NodeMatching, spacing: Spacing) -> Self:
        """The incoming-to-outgoing step cosine at each mislinked source, and at each reproduced one."""
        fates = DenseFateDiagnosis.of(prediction, truth, matching).fates
        edge_rows = truth.edge_rows()
        predecessors = Adjacency.of(truth).predecessors
        positions = spacing.to_micrometres(truth.positions())
        mislinked = (fates == Fate.MISLINK_CONFLICT) | (fates == Fate.MISLINK_FREE)
        return cls(
            mislinked_cos=cls._cosines(edge_rows[mislinked], predecessors, positions),
            correct_cos=cls._cosines(edge_rows[fates == Fate.CORRECT], predecessors, positions),
        )

    @staticmethod
    def _cosines(
        edges: Int[np.ndarray, "e 2"],
        predecessors: dict[int, tuple[int, ...]],
        positions: Float[np.ndarray, "n 3"],
    ) -> Float[np.ndarray, "k"]:
        """cos(source - predecessor, target - source) for every edge whose source has an annotated predecessor."""
        cosines = []
        for source, target in edges.tolist():
            parents = predecessors.get(int(source), ())
            if not parents:
                continue
            incoming = positions[int(source)] - positions[parents[0]]
            outgoing = positions[int(target)] - positions[int(source)]
            norm = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            if norm > 0.0:
                cosines.append(float(np.dot(incoming, outgoing)) / norm)
        return np.asarray(cosines, dtype=np.float64)
