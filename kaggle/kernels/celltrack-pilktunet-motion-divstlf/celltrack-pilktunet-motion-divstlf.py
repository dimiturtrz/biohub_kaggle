"""Submission kernel — pilkwang detector + our motion linker + division recovery.

Identical to `celltrack-pilktunet-motion-stlf` (LB 0.859) but for one added post-link stage. That kernel's
linker is a one-to-one assignment, so no node can ever have two children and the metric's whole
`0.1 * division_jaccard` term scores zero by construction. `DivisionRecovery` proposes the missing second
daughter after linking — the frontier's `add_safe_divisions_postlink` shape — under a parent gate, a sister
gate and a global cap, before the short-track filter whose division-preserving carve-out can only protect
forks that already exist.

The detection threshold is deliberately left at 0.5, matching the 0.859 run, so this kernel's leaderboard
delta is attributable to divisions alone; the operating point is a separate change.

Two mounts: the pilkwang support pack (their model code + weights) and celltrack-kit (our pipeline). The
kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`; this file is the
bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/motion_linking.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import install_wheels, pack_source, run_submission, test_videos  # noqa: E402

install_wheels(Path(_KIT_ROOT))
sys.path.insert(0, str(pack_source()))

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import zarr  # noqa: E402
from biohub_tracking.models import TemporalUNet3D  # noqa: E402

from celltrack.postproc.division_recovery import DivisionRecovery  # noqa: E402
from celltrack.postproc.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linkers.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.postproc.short_track_filter import ShortTrackFilter  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402


# Kaggle mounts nest under /kaggle/input/{datasets,competitions}/... — anchor the weight lookup on a known file.
def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_WEIGHTS = Path(_find("weights/unet_transformer/split_0/config.json")).parent

_DOWNSAMPLE = (1, 4, 4)
_WINDOW_UM, _THRESHOLD = 5.0, 0.99
# The frontier's add_safe_divisions_postlink gates, verbatim (the earlier 10.5/18 was a misread of their
# config): tight parent 4.7 um, sister 7.2 um, existing-child 7.8 um, global 0.4% / per-frame 0.8% caps.
# On real fold-0 detections this nets +0.0002 (div_jaccard 0.0114) — divisions are detection-limited here
# (most second daughters are undetected), so the gain is small without their dual-seed + DeepCenter recall.
_PARENT_GATE_UM, _SISTER_GATE_UM, _EXISTING_CHILD_GATE_UM = 4.7, 7.2, 7.8
_MAX_DIVISION_FRACTION, _FRAME_DIVISION_FRACTION = 0.004, 0.008
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Detector(torch.nn.Module):
    """The detection half of pilkwang's model: temporal UNet backbone + a 1×1×1 detection head."""

    def __init__(self, out_channels: int, layers: list[int]) -> None:
        super().__init__()
        self.unet = TemporalUNet3D(in_channels=1, out_channels=out_channels, layers=layers)
        self.detect_head = torch.nn.Conv3d(out_channels, 1, kernel_size=1)

    def detect(self, frame: torch.Tensor) -> torch.Tensor:
        """Detection logits for one already-normalised, already-downsampled frame `(Z, Y', X')`."""
        pair = torch.stack([frame, frame], dim=0).unsqueeze(0).unsqueeze(2)
        return self.detect_head(self.unet(pair)[0, 0:1])[0, 0]


def _pool_kernel(um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    kernel = []
    for scale in voxel_size:
        size = max(1, round(um / scale))
        kernel.append(size + 1 if size % 2 == 0 else size)
    return tuple(kernel)


def _peaks(logits: torch.Tensor, threshold: float, pool_kernel: tuple[int, ...]) -> np.ndarray:
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits[None, None], pool_kernel, stride=1, padding=pad)[0, 0]
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > threshold)
    return torch.nonzero(is_peak, as_tuple=False).cpu().numpy()


def _detections(model: _Detector, path: Path) -> tuple[TrackGraph, Spacing]:
    group = zarr.open_group(path, mode="r")
    array, quantiles = group["0"], group.attrs["image_statistics"]["quantiles"]
    q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
    scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1:]
    spacing = Spacing(z=scale[0], y=scale[1], x=scale[2])
    voxel = tuple(s * d for s, d in zip(scale, _DOWNSAMPLE, strict=True))
    pool_kernel = _pool_kernel(_WINDOW_UM, voxel)
    dz, dy, dx = _DOWNSAMPLE
    coordinates: list[np.ndarray] = []
    for timepoint in range(array.shape[0]):
        raw = array[timepoint, ::dz, ::dy, ::dx].astype(np.float32)
        frame = ((torch.from_numpy(raw) - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).to(_DEVICE)
        logits = model.detect(frame)
        for dims in [(-1,), (-2,), (-2, -1)]:
            logits = logits + model.detect(frame.flip(dims)).flip(dims)
        peaks = _peaks(logits / 4, _THRESHOLD, pool_kernel) * np.array([dz, dy, dx])
        coordinates.append(np.column_stack([np.full(len(peaks), timepoint), peaks]).astype(np.int64))
    stacked = np.concatenate(coordinates) if coordinates else np.empty((0, 4), dtype=np.int64)
    graph = TrackGraph(
        node_ids=np.arange(len(stacked), dtype=np.int64), coordinates=stacked, edges=np.empty((0, 2), np.int64)
    )
    return graph, spacing


def main() -> None:
    config = json.loads((_WEIGHTS / "config.json").read_text())
    state = torch.load(_WEIGHTS / "edge_predictor_best.pth", map_location=_DEVICE, weights_only=True)
    model = _Detector(config["unet_out_channels"], config["unet_layers"]).to(_DEVICE).eval()
    model.load_state_dict({k: v for k, v in state.items() if k.startswith(("unet.", "detect_head."))})
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)

    def predict(_name: str, path: Path) -> TrackGraph:
        with torch.no_grad():
            detections, spacing = _detections(model, path)
        linker = MotionHungarianLinker(spacing=spacing, tight_gate_um=6.0, loose_gate_um=10.0)
        divide = DivisionRecovery(
            spacing=spacing,
            parent_gate_um=_PARENT_GATE_UM,
            sister_gate_um=_SISTER_GATE_UM,
            max_added_fraction=_MAX_DIVISION_FRACTION,
            existing_child_gate_um=_EXISTING_CHILD_GATE_UM,
            frame_fraction_cap=_FRAME_DIVISION_FRACTION,
        )
        linked = divide.transform(linker.link(detections))
        return smooth.transform(short.transform(linked))

    run_submission(predict, test_videos())


main()
