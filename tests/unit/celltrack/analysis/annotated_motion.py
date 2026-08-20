import numpy as np
import pytest

from celltrack.analysis.annotated_motion import AnnotatedStepByFate, AnnotatedTurnByFate
from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import NodeMatching

_SPACING = Spacing(z=1.625, y=0.40625, x=0.40625)


def _graph(coordinates: list[list[int]], edges: list[list[int]]) -> TrackGraph:
    """A track graph over consecutively-numbered node ids, `t, z, y, x` per node."""
    return TrackGraph(
        node_ids=np.arange(len(coordinates), dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )


def test_annotated_step_by_fate_of() -> None:
    """The step between ANNOTATED centres, split by fate — the quantity no detection touches.

    This is the arbiter for whether a mislinked edge's cell really travelled far, or whether the apparent
    distance was manufactured by two mislocalised endpoints. Measuring it off the annotation is what makes it
    independent of the read-out, so the fixture puts the annotated source and target 40 voxels apart while the
    tracker links elsewhere.
    """
    truth = _graph([[0, 0, 0, 0], [1, 0, 0, 40], [1, 0, 0, 4], [0, 0, 80, 0], [1, 0, 80, 4]], [[0, 1], [3, 4]])
    prediction = _graph([[0, 0, 0, 0], [1, 0, 0, 40], [1, 0, 0, 4], [0, 0, 80, 0], [1, 0, 80, 4]], [[0, 2], [3, 4]])
    matching = NodeMatching(gt_rows=np.array([0, 1, 2, 3, 4], dtype=np.int64))

    steps = AnnotatedStepByFate.of(truth, prediction, matching, _SPACING)

    assert steps.mislinked_um == pytest.approx([40 * 0.40625])
    assert steps.correct_um == pytest.approx([4 * 0.40625])


def test_annotated_turn_by_fate_of() -> None:
    """The incoming-to-outgoing turn cosine, split by fate — the label-swap signature on the tail.

    A mislinked edge whose source continues straight (predecessor, source, successor colinear) turns at cosine
    +1: a genuine fast mover. A reproduced edge whose source doubles back (successor sits toward the predecessor)
    turns at -1: the out-and-back a wrong label leaves. A source with no annotated predecessor has no incoming
    step and is dropped, so the two first-frame sources contribute nothing to either arm.
    """
    coordinates = [
        [0, 0, 0, 0],
        [1, 0, 0, 10],
        [2, 0, 0, 20],
        [2, 0, 0, 25],
        [0, 0, 0, 100],
        [1, 0, 0, 110],
        [2, 0, 0, 105],
    ]
    truth = _graph(coordinates, [[0, 1], [1, 2], [4, 5], [5, 6]])
    prediction = _graph(coordinates, [[0, 1], [1, 3], [4, 5], [5, 6]])
    matching = NodeMatching(gt_rows=np.arange(7, dtype=np.int64))

    turns = AnnotatedTurnByFate.of(truth, prediction, matching, _SPACING)

    assert turns.mislinked_cos == pytest.approx([1.0])
    assert turns.correct_cos == pytest.approx([-1.0])
