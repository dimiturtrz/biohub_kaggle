"""Submission kernel — classical: multi-scale DoG detector + motion linker + short-track + linefit.

No learned weights. Multi-scale Difference-of-Gaussians detection (GPU) to a per-acquisition node budget
(derived from the training split), linked by our MotionHungarianLinker, cleaned by short-track + linefit.
The honest classical ceiling (dog-motion-stlf), mounting only celltrack-kit — no model pack.
"""

import glob
import os
import subprocess
import sys

print("INPUT DIRS:", os.listdir("/kaggle/input"), flush=True)


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_MARKER = _find("celltrack/dog_detection.py")
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
_WHEELS = sorted(glob.glob(f"{os.path.dirname(_KIT_ROOT)}/**/*.whl", recursive=True))
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)
sys.path.insert(0, _KIT_ROOT)

import torch  # noqa: E402

from celltrack.detection import HIGH, LOW  # noqa: E402
from celltrack.dog_detection import DoGDetector  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.motion_linking import MotionHungarianLinker  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.video import CellVideo  # noqa: E402

_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_BUDGET = {"44b6": 36887, "6bba": 16454}
_GATE_UM = 10.0
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    tests = sorted(glob.glob(_TEST_GLOB, recursive=True))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        cell = CellVideo.from_ome_zarr(path)
        detector = DoGDetector(spacing=cell.spacing, device=_DEVICE)
        linker = MotionHungarianLinker(spacing=cell.spacing, tight_gate_um=6.0, loose_gate_um=_GATE_UM)
        name = os.path.basename(path)[: -len(".zarr")]
        detections = detector.detect(cell.normalised_frames(LOW, HIGH), cell.timepoint_count, _BUDGET[name[:4]])
        graph = smooth.transform(short.transform(linker.link(detections)))
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()
