"""Submission kernel — our-trained temporal-U-Net detector + our motion linker.

The detector is the pilkwang architecture (`TemporalUNet3D` + detection head) with weights WE trained on our
fold split (`detector_tunet_ours.pt`), not the published ones. Inference is the shared `celltrack.tunet`
path — ×4 Y/X downsample, per-video quantile-norm, fake-pair forward, flip-TTA, max-pool peaks — feeding OUR
MotionHungarianLinker + short-track + linefit. The support pack is mounted only for the architecture class;
the weights and the whole pipeline are ours.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

print("INPUT DIRS:", os.listdir("/kaggle/input"), flush=True)


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_PACK_SRC = Path(_find("repo/src/biohub_tracking/models/__init__.py")).parents[2]
_MARKER = _find("celltrack/tunet.py")
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
_CHECKPOINT = Path(_find("detector_tunet_ours.pt"))

_WHEELS = sorted(glob.glob(f"{Path(_KIT_ROOT).parent}/**/*.whl", recursive=True))
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

sys.path.insert(0, _KIT_ROOT)
sys.path.insert(0, str(_PACK_SRC))

import torch  # noqa: E402
import zarr  # noqa: E402

from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.tunet import TemporalUNetDetector  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.geometry import Spacing  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_THRESHOLD = 0.5
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _spacing(path: str) -> Spacing:
    group = zarr.open_group(path, mode="r")
    scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1:]
    return Spacing(z=scale[0], y=scale[1], x=scale[2])


def main() -> None:
    tests = sorted(glob.glob(_TEST_GLOB, recursive=True))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    detector, recipe = TemporalUNetDetector.from_checkpoint(_CHECKPOINT, map_location=_DEVICE)
    detector = detector.to(_DEVICE).eval()
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        spacing = _spacing(path)
        detections = detector.detections(Path(path), _THRESHOLD, recipe, _DEVICE)
        linker = MotionHungarianLinker(spacing=spacing, tight_gate_um=6.0, loose_gate_um=10.0)
        graph = smooth.transform(short.transform(linker.link(detections)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()
