"""How many ground-truth cells the detected candidate set actually contains, per movie.

An edge head can only link cells that were detected. When a movie scores badly with a large edge-FN and
a small edge-FP, the linker is the obvious suspect and usually the wrong one: if the candidates never
covered the GT cells, no linker on top of them can score, and the number to fix is upstream. This
module reports the denominator that tells those two apart -- GT cells found, over GT cells present.

Candidates are the `<stem>.npz` dumps written by the detection pre-run (coords in downsampled voxel
space plus the voxel size needed to put them in microns); ground truth comes from the competition's own
loader, so the match distance here is the one the official ruler scores with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from scipy.spatial import KDTree

COMPETITION = "biohub_cell_tracking"
MATCH_DISTANCE_UM = 7.0

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverageRow:
    """Per-movie candidate coverage of the ground-truth cells."""

    stem: str
    gt_cells: int
    covered: int
    candidates: int

    @property
    def recall(self) -> float:
        return self.covered / self.gt_cells if self.gt_cells else float("nan")

    @property
    def candidates_per_gt_cell(self) -> float:
        return self.candidates / self.gt_cells if self.gt_cells else float("nan")


@dataclass(frozen=True)
class CandidateCoverage:
    """Matches detected candidates against ground-truth cells at the ruler's own distance."""

    match_distance_um: float = MATCH_DISTANCE_UM

    def count(
        self,
        gt_um: Float[np.ndarray, "m 4"],
        candidate_um: Float[np.ndarray, "n 4"],
    ) -> int:
        """Count GT cells with a candidate within the match distance, matching per frame.

        Frame index travels in column 0 of both arrays and is compared exactly -- a candidate one frame
        away is not a detection of this cell, however close it lands in space.
        """
        found = 0
        for frame in np.unique(gt_um[:, 0]):
            truth = gt_um[gt_um[:, 0] == frame][:, 1:]
            nearby = candidate_um[candidate_um[:, 0] == frame][:, 1:]
            if len(nearby) == 0:
                continue
            distance, _ = KDTree(nearby).query(truth, k=1)
            found += int((distance <= self.match_distance_um).sum())
        return found

    def candidates_um(self, npz_path: Path) -> Float[np.ndarray, "n 4"]:
        """Load a candidate dump and put its coordinates in microns.

        `scale` in the dump is microns per FULL-resolution voxel; the candidates were found on the
        downsampled grid, so their own micron factor is `voxel_size` (== scale * downsample). Using
        `scale` here puts every candidate at a quarter of its true y/x and reads as a dead detector.
        """
        loaded = np.load(npz_path, allow_pickle=True)
        coords = loaded["coords"].astype(np.float64)
        voxel_size = np.asarray(loaded["voxel_size"], dtype=np.float64).reshape(-1)
        return np.column_stack([coords[:, 0], coords[:, 1:] * voxel_size[-3:]])

    def row(
        self,
        stem: str,
        gt_um: Float[np.ndarray, "m 4"],
        npz_path: Path,
    ) -> CoverageRow:
        candidate_um = self.candidates_um(npz_path)
        return CoverageRow(
            stem=stem,
            gt_cells=len(gt_um),
            covered=self.count(gt_um, candidate_um),
            candidates=len(candidate_um),
        )


def main() -> None:
    import argparse  # noqa: PLC0415

    from biohub_tracking.io import open_dataset  # type: ignore[missing-import]  # noqa: PLC0415

    from celltrack.eval.official_ruler import TEST_MOVIES  # noqa: PLC0415
    from core.paths import DataRoot  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True, help="dir of <stem>.npz candidates")
    parser.add_argument("--match-um", type=float, default=MATCH_DISTANCE_UM)
    arguments = parser.parse_args()

    data = DataRoot.from_config(Path("paths.yaml"))
    ground_truth = data.raw(COMPETITION) / "train"
    measure = CandidateCoverage(match_distance_um=arguments.match_um)

    for stem in TEST_MOVIES:
        npz_path = arguments.candidate_dir / f"{stem}.npz"
        if not npz_path.exists():
            log.info("%s: MISSING candidates -- skipped", stem)
            continue
        dataset = open_dataset(ground_truth / stem, require_tracks=True)
        attributes = dataset.tracks.node_attrs(attr_keys=["t", "z", "y", "x"])
        gt_um = np.column_stack([np.asarray(attributes[axis], dtype=np.float64) for axis in ("t", "z", "y", "x")])
        gt_um[:, 1:] *= np.asarray(dataset.scale, dtype=np.float64)[-3:]
        row = measure.row(stem, gt_um, npz_path)
        log.info(
            "%s: gt_cells=%d covered=%d recall=%.4f candidates=%d per_gt=%.1f",
            row.stem,
            row.gt_cells,
            row.covered,
            row.recall,
            row.candidates,
            row.candidates_per_gt_cell,
        )


if __name__ == "__main__":
    main()
