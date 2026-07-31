"""Submission kernel — pilkwang detector + our motion linker: the mix-and-match pipeline.

The detector is pilkwang's public TemporalUNet3D + detection head (`unet.*` / `detect_head.*` from
`edge_predictor_best.pth`, split_0), a pure-torch model needing none of their tracksdata/ilpy/geff stack —
so this kernel installs no wheels. Its detections (quantile-normalised, ×4 Y/X downsample, per-frame
fake-pair forward, flip-TTA, max-pool peaks, coordinates rescaled to original voxels) feed OUR
MotionHungarianLinker + short-track + linefit and write the competition CSV. Fold-0 locally ~0.93.

Two mounts: the pilkwang support pack (their model code + weights) and celltrack-kit (our pipeline).
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

_INPUTS = os.listdir("/kaggle/input")
print("INPUT DIRS:", _INPUTS, flush=True)

_PACK = next(p for p in glob.glob("/kaggle/input/*") if "support-pack" in p)
_KIT = next(p for p in glob.glob("/kaggle/input/*") if glob.glob(f"{p}/**/celltrack/motion_linking.py", recursive=True))
_MARKER = next(m for m in glob.glob(f"{_KIT}/**/celltrack/motion_linking.py", recursive=True))
sys.path.insert(0, os.path.dirname(os.path.dirname(_MARKER)))
sys.path.insert(0, str(Path(_PACK) / "repo" / "src"))

from tracking_cellmot.models import TemporalUNet3D  # noqa: E402

from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linkers import LinkerConfig  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
_WEIGHTS = Path(_PACK) / "weights" / "unet_transformer" / "split_0"
_DOWNSAMPLE = (1, 4, 4)
_WINDOW_UM, _THRESHOLD = 5.0, 0.5
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


def _detections(model: _Detector, path: str) -> tuple[TrackGraph, Spacing]:
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
    tests = sorted(glob.glob(f"{_COMP}/test/*.zarr"))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    config = json.loads((_WEIGHTS / "config.json").read_text())
    state = torch.load(_WEIGHTS / "edge_predictor_best.pth", map_location=_DEVICE, weights_only=True)
    model = _Detector(config["unet_out_channels"], config["unet_layers"]).to(_DEVICE).eval()
    model.load_state_dict({k: v for k, v in state.items() if k.startswith(("unet.", "detect_head."))})
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        with torch.no_grad():
            detections, spacing = _detections(model, path)
        linker = LinkerConfig(name="motion").build(spacing)
        graph = smooth.transform(short.transform(linker.link(detections)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()
