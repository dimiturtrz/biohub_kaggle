"""Submission kernel — aug8k detector + NN linker: U-Net detect + NN link + post-proc on the test set.

Named for its method (`celltrack-<detector>-<linker>`): the aug-recipe width-16 U-Net trained 8k steps
(`detector_bce_aug_8k.pt`) read out by threshold, linked nearest-neighbour. A different detector or linker is
a different kernel with its own method name, not a new version of this one.

Thin orchestration only — the whole pipeline is imported from the celltrack-kit dataset (no code duplicated),
its extra dependencies come from the wheels in the same kit (internet is off in a code competition), and the
trained weights ship in the kit too. The forward is `LearnedDetector.responses` (fp16 + channels-last, the
same fast path the local eval takes) and detection is read out with a probability threshold — safe now that
`peaks` collapses each saturated plateau to one centre, and the operating point the trained detector scores
best at (~0.8, ~9k nodes/video, well under the metric's node-count ceiling).

The kit-mount, wheel-install and submission-write ceremony lives in `celltrack.kernel_runtime`; this file is
the bootstrap plus the one assembly it tests.
"""

import glob
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # ease T4 fragmentation OOM

_MARKER = next(iter(glob.glob("/kaggle/input/**/celltrack/motion_linking.py", recursive=True)))
_KIT_ROOT = os.path.dirname(os.path.dirname(_MARKER))
sys.path.insert(0, _KIT_ROOT)

from celltrack.kernel_runtime import install_wheels, run_submission, test_videos  # noqa: E402

install_wheels(Path(_KIT_ROOT))

import torch  # noqa: E402

from celltrack.detection import CELL_SCALE_UM  # noqa: E402
from celltrack.inference import LearnedDetector  # noqa: E402
from celltrack.linefit_smoother import LinefitSmoother  # noqa: E402
from celltrack.linking import NearestNeighbourLinker  # noqa: E402
from celltrack.peaks import PeakExtractor  # noqa: E402
from celltrack.short_track_filter import ShortTrackFilter  # noqa: E402
from celltrack.training.model import DetectionUNet  # noqa: E402
from core.data.tracks import TrackGraph  # noqa: E402
from core.data.video import CellVideo  # noqa: E402


def _find(pattern: str) -> str:
    return next(iter(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)))


_WEIGHTS = _find("detector_bce_aug_8k.pt")

_THRESHOLD, _GATE_UM, _BATCH = 0.8, 10.0, 2  # T4 has ~15 GB; full-frame volumes cap the timepoint batch
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ckpt = torch.load(_WEIGHTS, map_location=_DEVICE)
    model = DetectionUNet.from_checkpoint(ckpt).to(_DEVICE).eval().to(memory_format=torch.channels_last_3d)
    window = int(ckpt["window"])
    short, smooth = ShortTrackFilter(min_length=3), LinefitSmoother(strength=0.8)

    def predict(_name: str, path: Path) -> TrackGraph:
        cell = CellVideo.from_ome_zarr(path)
        peaks = PeakExtractor(spacing=cell.spacing, scale_um=CELL_SCALE_UM, device=_DEVICE)
        linker = NearestNeighbourLinker(spacing=cell.spacing, max_distance_um=_GATE_UM)
        detector = LearnedDetector(model=model, peaks=peaks, window=window, device=_DEVICE, batch=_BATCH)
        graph = detector.detect_above(cell, _THRESHOLD)
        return smooth.transform(short.transform(linker.link(graph)))

    run_submission(predict, test_videos())


main()
