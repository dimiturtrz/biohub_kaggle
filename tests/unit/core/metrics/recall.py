import math

import numpy as np

from core.data.tracks import TrackGraph
from core.geometry import Spacing
from core.metrics.matching import DistanceMatcher
from core.metrics.recall import DetectionRecall

MATCHER = DistanceMatcher(spacing=Spacing(z=1.0, y=1.0, x=1.0))


def nodes(coordinates: list[list[int]]) -> TrackGraph:
    return TrackGraph(
        node_ids=np.arange(len(coordinates)),
        coordinates=np.array(coordinates, dtype=np.int64).reshape(-1, 4),
        edges=np.empty((0, 2), dtype=np.int64),
    )


def test_fraction():
    assert DetectionRecall(found=3, total=4).fraction() == 0.75


def test_fraction_is_undefined_without_ground_truth():
    assert math.isnan(DetectionRecall(found=0, total=0).fraction())


def test_of():
    """A detection within the cutoff of a ground-truth cell counts that cell as found."""
    truth = nodes([[0, 0, 0, 0], [1, 0, 10, 0]])
    detections = nodes([[0, 0, 1, 0], [1, 0, 10, 0]])
    assert DetectionRecall.of(detections, truth, MATCHER) == DetectionRecall(found=2, total=2)


def test_of_misses_cells_with_no_nearby_detection():
    """A ground-truth cell with no detection within the cutoff is not found."""
    truth = nodes([[0, 0, 0, 0], [0, 0, 50, 0]])
    detections = nodes([[0, 0, 0, 0]])
    assert DetectionRecall.of(detections, truth, MATCHER) == DetectionRecall(found=1, total=2)
