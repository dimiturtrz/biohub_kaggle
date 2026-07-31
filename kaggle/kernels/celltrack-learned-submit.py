"""Learned-detector submission kernel: U-Net detect + NN link + post-proc on the mounted test set.

Thin orchestration only — the whole pipeline is imported from the celltrack-kit dataset (no code duplicated),
its extra dependencies come from the wheels in the same kit (internet is off in a code competition), and the
trained weights ship in the kit too. The forward is `LearnedDetector.responses` (fp16 + channels-last, the
same fast path the local eval takes) and detection is read out with a probability threshold — safe now that
`peaks` collapses each saturated plateau to one centre, and the operating point the trained detector scores
best at (~0.8, ~9k nodes/video, well under the metric's node-count ceiling).
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # ease T4 fragmentation OOM

print("INPUT DIRS:", os.listdir("/kaggle/input"), flush=True)

_KITS = [d for d in glob.glob("/kaggle/input/*") if "competition" not in os.path.basename(d)]
_WHEELS = sorted(w for d in _KITS for w in glob.glob(f"{d}/**/*.whl", recursive=True))
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

_WEIGHTS_NAME = "detector_bce_aug_8k.pt"
_MARKER = next(m for d in _KITS for m in glob.glob(f"{d}/**/celltrack/inference.py", recursive=True))
sys.path.insert(0, os.path.dirname(os.path.dirname(_MARKER)))
_WEIGHTS = next(w for d in _KITS for w in glob.glob(f"{d}/**/{_WEIGHTS_NAME}", recursive=True))
print("SRC ROOT:", os.path.dirname(os.path.dirname(_MARKER)), "WEIGHTS:", _WEIGHTS, flush=True)

import torch  # noqa: E402  (import after the kit's src root is on the path)

from celltrack.detection import CELL_SCALE_UM  # noqa: E402
from celltrack.inference import LearnedDetector  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linking import NearestNeighbourLinker  # noqa: E402
from celltrack.peaks import PeakExtractor  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.training.model import DetectionUNet  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.video import CellVideo  # noqa: E402

_COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
_THRESHOLD, _GATE_UM, _BATCH = 0.8, 10.0, 2  # T4 has ~15 GB; full-frame volumes cap the timepoint batch
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    tests = sorted(glob.glob(f"{_COMP}/test/*.zarr"))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    ckpt = torch.load(_WEIGHTS, map_location=_DEVICE)
    model = DetectionUNet.from_checkpoint(ckpt).to(_DEVICE).eval().to(memory_format=torch.channels_last_3d)
    window = int(ckpt["window"])
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        cell = CellVideo.from_ome_zarr(Path(path))
        peaks = PeakExtractor(spacing=cell.spacing, scale_um=CELL_SCALE_UM, device=_DEVICE)
        linker = NearestNeighbourLinker(spacing=cell.spacing, max_distance_um=_GATE_UM)
        detector = LearnedDetector(model=model, peaks=peaks, window=window, device=_DEVICE, batch=_BATCH)
        graph = detector.detect_above(cell, _THRESHOLD)
        graph = smooth.transform(short.transform(linker.link(graph)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()
