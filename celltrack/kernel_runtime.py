"""The boilerplate every submission kernel shares, once — mounting the kit, the packs, and writing the CSV.

A Kaggle kernel is one self-contained script with no internet, so each one repeated the same ceremony:
locate the mounted `celltrack-kit` and pilkwang packs by globbing `/kaggle/input`, pip-install the offline
wheels, add the kit and the pack's source to `sys.path`, glob the test videos, run the pipeline over each and
write the submission CSV. That ceremony is identical across kernels and is collected here so a kernel is only
its bootstrap (find the kit, put it on the path — the few lines that must run before this module can be
imported) plus the assembly it is actually testing.

Module imports are standard-library only, so a kernel can import this *before* installing the wheels the rest
of `celltrack` needs; the one library dependency (`Submission`) is imported lazily, after `install_wheels`.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.data.tracks import TrackGraph

logger = logging.getLogger(__name__)

_INPUT_GLOB = "/kaggle/input/**"
_PACK_CONFIG = "weights/unet_transformer/split_0/config.json"
_TEST_GLOB = "/kaggle/input/**/biohub-cell-tracking-during-development/test/*.zarr"
_ZARR_SUFFIX = ".zarr"


class KernelRuntime:
    """The offline-kernel ceremony as one namespace of stateless steps: mount the kit and packs, run, write CSV."""

    @staticmethod
    def find(pattern: str) -> str:
        """The first mounted path matching `pattern` under `/kaggle/input` — the kit and pack discovery primitive."""
        return next(iter(glob.glob(f"{_INPUT_GLOB}/{pattern}", recursive=True)))

    @staticmethod
    def install_wheels(kit_root: Path) -> list[str]:
        """Offline-install the wheels bundled beside the kit, returning their names — the kernel's dependency mount."""
        wheels = sorted(glob.glob(f"{kit_root.parent}/**/*.whl", recursive=True))
        if wheels:
            install = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *wheels]
            subprocess.run(install, check=True)  # noqa: S603
        logger.info("installed wheels: %s", [os.path.basename(wheel) for wheel in wheels])
        return wheels

    @staticmethod
    def pilkwang_packs() -> tuple[Path, Path]:
        """The two published detector packs — the support-pack-50ep seed and the seed314159 seed — for the blend."""
        configs = glob.glob(f"{_INPUT_GLOB}/{_PACK_CONFIG}", recursive=True)
        pack1 = next(Path(config).parent for config in configs if "support-pack" in config or "50ep" in config)
        pack2 = next(Path(config).parent for config in configs if "seed314159" in config)
        logger.info("packs: %s | %s", pack1, pack2)
        return pack1, pack2

    @staticmethod
    def pack_source() -> Path:
        """The mounted pack's `src/` root, so the kernel imports the `TemporalUNet3D` the weights were trained in."""
        return Path(KernelRuntime.find("repo/src/biohub_tracking/models/__init__.py")).parents[2]

    @staticmethod
    def test_videos() -> list[Path]:
        """The competition's test-movie zarrs, sorted — the videos a submission must cover."""
        return [Path(path) for path in sorted(glob.glob(_TEST_GLOB, recursive=True))]

    @staticmethod
    def run_submission(
        predict: Callable[[str, Path], TrackGraph],
        videos: list[Path],
        out: str = "/kaggle/working/submission.csv",
    ) -> None:
        """Run `predict(name, path)` over each video into a track graph and write the competition submission CSV."""
        from core.data.submission import Submission  # noqa: PLC0415  (lazy: imported after install_wheels)

        graphs = {}
        for path in videos:
            name = path.name[: -len(_ZARR_SUFFIX)]
            graph = predict(name, path)
            graphs[name] = graph
            logger.info("  %s: %d nodes, %d edges", name, len(graph.node_ids), len(graph.edges))
        Submission(graphs=graphs).write_csv(out)
        logger.info("wrote %s", out)
