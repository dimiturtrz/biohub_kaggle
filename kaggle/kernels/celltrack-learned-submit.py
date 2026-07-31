"""Learned-detector submission kernel: U-Net detect + NN link + post-proc on the mounted test set.

Thin orchestration only — the pipeline is imported from the celltrack-kit dataset (no code duplicated),
its extra dependencies come from the wheels in the same kit (internet is off in a code competition), and the
trained weights ship in the kit too. Detection is read out with a bounded per-frame budget (keep-N), NOT a
probability threshold: the detector saturates to flat plateaus, and thresholding a plateau yields ~1e5
"peaks" per frame whose N-by-N linker cost matrix exhausts memory. keep-N caps the count and the cost.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

print("INPUT DIRS:", os.listdir("/kaggle/input"), flush=True)

_KITS = [d for d in glob.glob("/kaggle/input/*") if "competition" not in os.path.basename(d)]
_WHEELS = sorted(w for d in _KITS for w in glob.glob(f"{d}/**/*.whl", recursive=True))
print("WHEELS:", [os.path.basename(w) for w in _WHEELS], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *_WHEELS], check=True)

_MARKER = next(m for d in _KITS for m in glob.glob(f"{d}/**/celltrack/inference.py", recursive=True))
sys.path.insert(0, os.path.dirname(os.path.dirname(_MARKER)))
_WEIGHTS = next(w for d in _KITS for w in glob.glob(f"{d}/**/detector_bce_aug.pt", recursive=True))
print("SRC ROOT:", os.path.dirname(os.path.dirname(_MARKER)), "WEIGHTS:", _WEIGHTS, flush=True)

import torch  # noqa: E402  (import after the kit's src root is on the path)

from celltrack.detection import CELL_SCALE_UM, HIGH, LOW  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linking import NearestNeighbourLinker  # noqa: E402
from celltrack.peaks import PeakExtractor  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.training.model import DetectionUNet  # noqa: E402
from core.data.submission import Submission  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.data.video import CellVideo  # noqa: E402

_COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
_KEEP_PER_FRAME, _GATE_UM, _BATCH = 250, 10.0, 4
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def responses(model: DetectionUNet, cell: CellVideo, window: int) -> list[np.ndarray]:
    """Each timepoint's probability volume, forwarded a few frames at a time and offloaded to host RAM."""
    frames = list(cell.normalised_frames(LOW, HIGH))
    half, out = window // 2, []
    for start in range(0, len(frames), _BATCH):
        stacks = [
            np.stack(frames[min(max(t - half, 0), max(len(frames) - window, 0)) :][:window])
            for t in range(start, min(start + _BATCH, len(frames)))
        ]
        chunk = torch.from_numpy(np.stack(stacks)).to(_DEVICE)
        with torch.no_grad(), torch.autocast(device_type=_DEVICE.split(":")[0], enabled=_DEVICE == "cuda"):
            probs = torch.sigmoid(model(chunk))
        out.extend(probs[i].float().cpu().numpy() for i in range(probs.shape[0]))
    return out


def graph_of(centres: list[np.ndarray]) -> TrackGraph:
    """Stack per-timepoint (z, y, x) centres into a (t, z, y, x) edgeless track graph."""
    stamped = [np.column_stack([np.full(len(c), t), c]) for t, c in enumerate(centres)]
    stacked = np.concatenate(stamped).astype(np.int64) if stamped else np.empty((0, 4), np.int64)
    return TrackGraph(
        node_ids=np.arange(len(stacked), dtype=np.int64), coordinates=stacked, edges=np.empty((0, 2), np.int64)
    )


def main() -> None:
    tests = sorted(glob.glob(f"{_COMP}/test/*.zarr"))
    print(f"test videos: {len(tests)} on {_DEVICE}", flush=True)
    ckpt = torch.load(_WEIGHTS, map_location=_DEVICE)
    model = DetectionUNet.from_checkpoint(ckpt).to(_DEVICE).eval()
    window = int(ckpt["window"])
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)
    graphs = {}
    for path in tests:
        cell = CellVideo.from_ome_zarr(Path(path))
        peaks = PeakExtractor(spacing=cell.spacing, scale_um=CELL_SCALE_UM, device=_DEVICE)
        linker = NearestNeighbourLinker(spacing=cell.spacing, max_distance_um=_GATE_UM)
        graph = graph_of([peaks.centres(r, _KEEP_PER_FRAME) for r in responses(model, cell, window)])
        graph = smooth.transform(short.transform(linker.link(graph)))
        name = os.path.basename(path)[: -len(".zarr")]
        graphs[name] = graph
        print(f"  {name}: {len(graph.node_ids)} nodes, {len(graph.edges)} edges", flush=True)
    Submission(graphs=graphs).write_csv("/kaggle/working/submission.csv")
    print("wrote /kaggle/working/submission.csv", flush=True)


main()
