"""Machine-local data locations.

One data root per machine, recorded in the gitignored `paths.yaml`. `CELLTRACK_DATA` overrides it, so
a runner with no dataset can point at an empty directory and the data-gated tests skip instead of
erroring.
"""

import os
from pathlib import Path

import yaml

_ENV_OVERRIDE = "CELLTRACK_DATA"
_COMPETITION = "biohub_cell_tracking"


class DataRoot:
    """The data root. Datasets sit under `raw/` untouched; derived artefacts go to `processed/`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @classmethod
    def from_config(cls, config: Path) -> "DataRoot":
        """Resolve the root: the `CELLTRACK_DATA` env var if set, else the `data:` entry of paths.yaml."""
        override = os.environ.get(_ENV_OVERRIDE)
        if override:
            return cls(Path(override))
        entries: dict[str, str] = yaml.safe_load(config.read_text(encoding="utf-8"))
        return cls(Path(entries["data"]))

    def raw(self, dataset: str) -> Path:
        """Where an as-downloaded dataset lives. Never written to."""
        return self._root / "raw" / dataset

    def processed(self, dataset: str) -> Path:
        """Where derived artefacts for a dataset go. Created on demand."""
        destination = self._root / "processed" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def videos(self, split: str) -> list[Path]:
        """Every OME-Zarr video store in a competition split, in sorted order."""
        return sorted((self.raw(_COMPETITION) / split).glob("*.zarr"))

    def track_store(self, video: Path) -> Path:
        """The GEFF annotation store beside a training video — same stem, `.geff` instead of `.zarr`."""
        return video.with_suffix(".geff")
