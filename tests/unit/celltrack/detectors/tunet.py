"""Unit tests for the temporal-U-Net detection READ-OUT — turning cellness volumes into a graph of centres.

The network itself (forward, recipe, weight round-trip) is tested in `tests/unit/celltrack/models`; here the
subject is the read-out this strategy adds: the per-frame forward into logit/probability volumes and the
threshold-and-suppress peak extraction. The `TemporalUNet3D` backbone is stubbed by the autouse `conftest`
fixture, so these run without the `external/` checkout.
"""

from pathlib import Path

import numpy as np
import torch
import zarr

from celltrack.detectors.tunet import DetectorRecipe, TemporalUNetDetector


def _tiny_detector() -> TemporalUNetDetector:
    """A minimal detector for exercising the read-out without the real backbone size."""
    torch.manual_seed(0)
    return TemporalUNetDetector(out_channels=2, layers=(2, 4))


def _video(tmp_path: Path, scale: list[float]) -> Path:
    """A synthetic OME-zarr: two frames, one bright voxel, with a given `(t, z, y, x)` voxel scale."""
    group = zarr.open_group(str(tmp_path / "v.zarr"), mode="w")
    array = group.create_array("0", shape=(2, 4, 8, 8), dtype="float32")
    array[:] = 0.0
    array[:, 2, 4, 4] = 1.0
    group.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}
    group.attrs["multiscales"] = [{"datasets": [{"coordinateTransformations": [{"scale": scale}]}]}]
    return tmp_path / "v.zarr"


def test_detections(tmp_path: Path):
    """End to end on a synthetic video: a frame with one bright voxel yields an edge-free graph of centres."""
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    video = _video(tmp_path, [1.0, 1.0, 1.0, 1.0])
    graph = _tiny_detector().detections(video, threshold=0.0, recipe=recipe, device="cpu")
    assert graph.coordinates.shape[1] == 4
    assert graph.edges.shape == (0, 2)
    assert set(graph.timepoints().tolist()) <= {0, 1}


def test_probability_volumes(tmp_path: Path):
    """The cacheable forward yields one downsampled probability volume per timepoint, in [0, 1], as host arrays."""
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    volumes = _tiny_detector().probability_volumes(_video(tmp_path, [1.0, 1.0, 1.0, 1.0]), recipe, "cpu")
    assert len(volumes) == 2  # one per timepoint
    assert volumes[0].shape == (4, 8, 8)  # the downsampled grid
    assert volumes[0].min() >= 0.0 and volumes[0].max() <= 1.0  # probabilities


def test_logit_volumes(tmp_path: Path):
    """The blend-space response is the pre-sigmoid logits — `probability_volumes` is exactly their sigmoid."""
    detector = _tiny_detector()
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    video = _video(tmp_path, [1.0, 1.0, 1.0, 1.0])
    logits = detector.logit_volumes(video, recipe, "cpu")
    probs = detector.probability_volumes(video, recipe, "cpu")
    assert len(logits) == 2
    assert np.allclose(probs[0], 1.0 / (1.0 + np.exp(-logits[0])), atol=1e-5)


def test_graph_from_volumes():
    """The cheap read-out lifts a peak above threshold into a stamped, edge-free node — no GPU, no video."""
    volume = np.zeros((2, 4, 4), dtype=np.float32)
    volume[1, 2, 3] = 1.0
    recipe = DetectorRecipe(downsample=(1, 1, 1), pool_kernel_um=1.0, tta=False)
    graph = TemporalUNetDetector.graph_from_volumes([volume], (1.0, 1.0, 1.0), 0.5, recipe, "cpu")
    assert graph.coordinates.tolist() == [[0, 1, 2, 3]]  # (t, z, y, x)
    assert graph.edges.shape == (0, 2)


def test_voxel_scale(tmp_path: Path):
    """The voxel scale is read from the OME metadata, dropping the time axis for the peak read-out."""
    assert _tiny_detector().voxel_scale(_video(tmp_path, [1.0, 2.0, 3.0, 4.0])) == (2.0, 3.0, 4.0)
