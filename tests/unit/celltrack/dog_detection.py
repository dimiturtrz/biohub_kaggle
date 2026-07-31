import numpy as np

from celltrack.dog_detection import DoGDetector
from core.geometry import Spacing

DETECTOR = DoGDetector(spacing=Spacing(z=1.0, y=1.0, x=1.0), scales_um=(2.0,), device="cpu")


def _blob() -> np.ndarray:
    """A single isotropic Gaussian blob at (8, 16, 16) — one clean cell for the band-pass to find."""
    z, y, x = np.ogrid[:16, :32, :32]
    return np.exp(-((z - 8) ** 2 + (y - 16) ** 2 + (x - 16) ** 2) / (2 * 2.0**2)).astype(np.float32)


def test_response():
    """The Difference-of-Gaussians response has the input shape, is finite, and peaks at the blob centre."""
    response = DETECTOR.response(_blob())
    assert response.shape == (16, 32, 32)
    assert np.isfinite(response).all()
    assert np.unravel_index(int(response.argmax()), response.shape) == (8, 16, 16)


def test_centres():
    """The strongest suppressed maximum of the response is the blob's centre, as (z, y, x) voxel indices."""
    assert DETECTOR.centres(DETECTOR.response(_blob()), keep=1).tolist() == [[8, 16, 16]]


def test_detect():
    """Detection over a two-frame video returns an edgeless (t, z, y, x) centre graph."""
    graph = DETECTOR.detect([_blob(), _blob()], frame_count=2, keep=4)
    assert graph.edges.shape == (0, 2)
    assert graph.coordinates.shape[1] == 4
    assert set(graph.timepoints().tolist()) <= {0, 1}
