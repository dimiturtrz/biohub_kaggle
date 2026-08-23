"""Running a submission over several GPUs at once — one worker process per device, sharded by VIDEO.

A submission is ~88% GPU work (per-video stage split: detect 11.4s, affinity 6.9s, link+post 2.3s) and its
videos are independent: nothing a video computes is read by another. So the whole pipeline parallelises at the
coarsest possible grain — give each device its own shard of the video list and let it run the *identical*
tracker over it. That keeps the pipeline itself branch-free (no rank-aware code anywhere below this module)
and makes the result exact rather than approximate: every video is processed by the same code, on one device,
as it would have been serially.

Each worker rebuilds its tracker from `CellTracker.ephemeral(pack1, pack2, device, config)` — picklable inputs
only — because CUDA state cannot be forked; `spawn` is mandatory, and a mounted model cannot cross it. With a
single device there is nothing to spawn, so the dispatcher runs in-process on the same code path, which is what
a local one-GPU run and the CPU tests take.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, replace
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty

import torch
import torch.multiprocessing as torch_mp

from celltrack.detectors.center_prior import CenterPriorScorer
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from core.data.submission import Submission
from core.data.tracks import TrackGraph

logger = logging.getLogger(__name__)

_ZARR_SUFFIX = ".zarr"
# How long the parent waits on the result queue before re-checking that every worker is still alive. It only
# bounds the *detection* latency of a crashed worker, never the run: a live worker just loops again.
_POLL_SECONDS = 10.0
_LIVE_EXIT_CODES = (None, 0)


@dataclass(frozen=True)
class VideoShard:
    """One device's slice of the submission: the videos it owns, already keyed by their competition name."""

    device: str
    videos: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class MultiGpuSubmission:
    """Data-parallel submission over a device list: shard the videos, run the full pipeline per device, merge.

    `devices` is a parameter with an auto-detected default, so the same object serves one local GPU, Kaggle's
    two T4s, and an N-GPU box without the caller knowing which it is.
    """

    @staticmethod
    def available_devices() -> tuple[str, ...]:
        """Every visible CUDA device as an explicit ordinal, falling back to the single unindexed device or CPU."""
        count = torch.cuda.device_count()
        if count > 1:
            return tuple(f"cuda:{ordinal}" for ordinal in range(count))
        return ("cuda",) if count == 1 else ("cpu",)

    packs: tuple[Path, Path]
    config: TrackerConfig
    # Declared after the detector it defaults to, so the factory is the method itself rather than a lambda.
    devices: tuple[str, ...] = field(default_factory=available_devices)
    # The mounted local-association re-ranker artifact, or None to run without it. A PATH rather than a mounted
    # model for the same reason the packs are paths: it has to survive the spawn boundary, and where the artifact
    # lives is the caller's fact (a Kaggle dataset mount here, the local data root there).
    ranker_pack: Path | None = None
    # The DeepCenter centre-prior pack, or None to run without confirmed recovery. A PATH for the same
    # survives-the-spawn reason as the packs above — the worker mounts it on its own device.
    center_pack: Path | None = None

    def shards(self, videos: list[Path]) -> tuple[VideoShard, ...]:
        """Deal the videos round-robin across the devices — interleaved, so no one worker gets every long movie.

        A contiguous split would hand a whole block of the sorted list to one device; the movies differ several-
        fold in frame count, so the block that happens to hold the dense ones would set the wall-clock alone.
        """
        stride = len(self.devices)
        dealt = (VideoShard(device, self._keyed(videos[offset::stride])) for offset, device in enumerate(self.devices))
        return tuple(shard for shard in dealt if shard.videos)

    def run(self, videos: list[Path], out: Path) -> None:
        """Track every video on its shard's device and write the one merged submission CSV in the given order."""
        shards = self.shards(videos)
        logger.info("dispatching %d videos over %s", len(videos), [shard.device for shard in shards])
        graphs = self.shard_graphs(shards[0]) if len(shards) == 1 else self._spawned_graphs(shards)
        # A missing key raises here rather than writing a submission that silently omits a video.
        ordered = {key: graphs[key] for key in (self._key(video) for video in videos)}
        Submission(graphs=ordered).write_csv(out)
        logger.info("wrote %s", out)

    def shard_graphs(self, shard: VideoShard) -> dict[str, TrackGraph]:
        """Mount the tracker on this shard's device and run the full pipeline over its videos, keyed by name."""
        tracker = CellTracker.ephemeral(self.packs[0], self.packs[1], shard.device, self.config)
        if self.ranker_pack is not None:
            tracker = tracker.with_ranker(self.ranker_pack)
        if self.center_pack is not None:
            tracker = replace(tracker, center_scorer=CenterPriorScorer.from_pack(self.center_pack, shard.device))
        logger.info("%s: %d videos — %s", shard.device, len(shard.videos), [key for key, _ in shard.videos])
        return {key: tracker.run(key, path) for key, path in shard.videos}

    def _spawned_graphs(self, shards: tuple[VideoShard, ...]) -> dict[str, TrackGraph]:
        context = torch_mp.get_context("spawn")
        queue = context.Queue()
        workers: list[BaseProcess] = [context.Process(target=self._worker, args=(shard, queue)) for shard in shards]
        self._start(workers)
        graphs = self._drained(queue, workers, sum(len(shard.videos) for shard in shards))
        for worker in workers:
            worker.join()
        self._raise_on_dead(workers)
        return graphs

    @staticmethod
    def _start(workers: list[BaseProcess]) -> None:
        """Start every worker with the parent's `__main__` file hidden from spawn's child bootstrap.

        `spawn` normally re-executes the parent's main module inside each child so that objects pickled out of
        `__main__` resolve there. Nothing here comes out of `__main__` — the worker and its arguments are all
        `celltrack` types, and spawn already carries the parent's `sys.path` across — while re-executing a
        Kaggle kernel's main module would repeat the kit mount and the offline pip install once per GPU.
        Removing `__file__` for the duration of the starts is what tells `multiprocessing` there is no main
        module to replay; the kernel's own `__main__` guard is the backstop if a host provides no `__file__`.
        """
        namespace = vars(sys.modules["__main__"])
        hidden = namespace.pop("__file__", None)
        try:
            for worker in workers:
                worker.start()
        finally:
            if hidden is not None:
                namespace["__file__"] = hidden

    def _worker(self, shard: VideoShard, queue: Queue[dict[str, TrackGraph]]) -> None:
        """The spawned child: rebuild the tracker from picklable inputs, run the shard, hand the graphs back.

        Nothing is caught — an exception kills the child with a non-zero exit code and a traceback on the
        kernel's log, which the parent turns into a raise. A dropped video is a broken submission, never a
        quietly shorter one.
        """
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        queue.put(self.shard_graphs(shard))

    def _drained(
        self, queue: Queue[dict[str, TrackGraph]], workers: list[BaseProcess], expected: int
    ) -> dict[str, TrackGraph]:
        graphs: dict[str, TrackGraph] = {}
        while len(graphs) < expected:
            try:
                graphs.update(queue.get(timeout=_POLL_SECONDS))
            except Empty:
                self._raise_on_dead(workers)
        return graphs

    def _raise_on_dead(self, workers: list[BaseProcess]) -> None:
        dead = [worker for worker in workers if worker.exitcode not in _LIVE_EXIT_CODES]
        if dead:
            codes = [(worker.name, worker.exitcode) for worker in dead]
            message = f"submission worker failed: {codes} — see its traceback above"
            raise RuntimeError(message)

    def _keyed(self, videos: list[Path]) -> tuple[tuple[str, Path], ...]:
        return tuple((self._key(video), video) for video in videos)

    @staticmethod
    def _key(video: Path) -> str:
        return video.name[: -len(_ZARR_SUFFIX)]
