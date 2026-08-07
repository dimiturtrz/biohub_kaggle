from pathlib import Path

import numpy as np

from celltrack.detectors.detection import BlobDetector, detect_over_frames
from core.data.video import CellVideo
from core.geometry import Spacing

ISOTROPIC = Spacing(z=1.0, y=1.0, x=1.0)
DETECTOR = BlobDetector(spacing=ISOTROPIC, scale_um=2.0)


def test_detect_over_frames():
    """The shared loop splits the budget over frames and stamps each frame's centres into an edgeless graph."""

    class _Stub:
        def response(self, frame: np.ndarray) -> np.ndarray:
            return frame

        def centres(self, response: np.ndarray, keep: int) -> np.ndarray:
            return np.array([[0, 0, int(response[0, 0, 0])]], dtype=np.int64)

    frames = [np.full((1, 1, 1), 5, dtype=np.float64), np.full((1, 1, 1), 7, dtype=np.float64)]
    graph = detect_over_frames(_Stub(), frames, frame_count=2, keep=2)

    assert graph.coordinates.tolist() == [[0, 0, 0, 5], [1, 0, 0, 7]]  # (t, z, y, x), one centre stamped per frame
    assert graph.edges.shape == (0, 2)


def a_blob(shape: tuple[int, int, int], centre: tuple[int, int, int], sigma: float) -> np.ndarray:
    """A single bright Gaussian blob on a dark background."""
    grid = np.indices(shape).astype(np.float64)
    squared = sum((grid[axis] - centre[axis]) ** 2 for axis in range(3))
    return np.exp(-squared / (2 * sigma**2))


def test_response():
    """The scale-normalised LoG response is highest at the centre of a blob of about the cell scale."""
    volume = a_blob((16, 16, 16), centre=(8, 8, 8), sigma=2.0)
    response = DETECTOR.response(volume)
    assert tuple(np.unravel_index(np.argmax(response), response.shape)) == (8, 8, 8)


def test_centres():
    response = np.zeros((16, 16, 16))
    response[4, 4, 4] = 1.0
    response[4, 12, 12] = 2.0
    found = DETECTOR.centres(response, keep=2)
    assert sorted(found.tolist()) == [[4, 4, 4], [4, 12, 12]]


def test_centres_keeps_only_the_requested_count():
    response = np.zeros((16, 16, 16))
    for offset, strength in enumerate([3.0, 2.0, 1.0]):
        response[2, 2 + offset * 5, 2] = strength
    found = DETECTOR.centres(response, keep=1)
    assert found.tolist() == [[2, 2, 2]]


def test_centres_suppresses_a_neighbour_within_a_cell():
    """Two maxima closer than the cell scale cannot both survive — the weaker is suppressed."""
    response = np.zeros((16, 16, 16))
    response[8, 8, 8] = 2.0
    response[8, 8, 9] = 1.0
    assert DETECTOR.centres(response, keep=2).tolist() == [[8, 8, 8]]


def test_detect(video_store: Path):
    """Detection over a video's frames returns an edgeless graph of centres with valid timepoints."""
    video = CellVideo.from_ome_zarr(video_store)
    frames = video.normalised_frames(0.001, 0.999)
    detections = DETECTOR.detect(frames, video.timepoint_count, keep=video.timepoint_count)
    assert detections.edges.shape == (0, 2)
    assert len(detections.node_ids) == len(detections.coordinates)
    assert set(detections.timepoints().tolist()) <= set(range(video.timepoint_count))
