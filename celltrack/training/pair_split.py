"""The pair sets one joint run reads, and the enumeration that assembles them.

Three populations, and they are three because each answers to a different rule. `train` is the competition
corpus the loop optimises on and the scale an epoch is counted in. `val` is the held-out movies the edge AUC
is measured over. `synthetic` is the fully-labelled corpus a mixing policy may ALSO draw from
(`celltrack.data.synthetic_pairs`) — kept apart from `train` rather than concatenated into it so a run that
mixes still reports its schedule in passes over the REAL pairs and stays comparable to every arm before it.

Assembly lives here rather than in the trainer's `main` because it is the same three-step recipe every time —
resolve the split, read each video's own intensity quantiles out of its OME metadata, enumerate — and because
the synthetic half carries a precondition (`POOLED_BY`) that must fail at startup, not mid-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import zarr

from celltrack.data.detection_pairs import DetectionPairs
from celltrack.data.frame_source import ZarrFrames
from celltrack.data.joint_dataset import PairTarget
from celltrack.data.synthetic_pairs import POOLED_BY, SyntheticPairs
from celltrack.eval.proxy import TEST_MOVIES, VALIDATION_MOVIES, TestMovieProxy
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker
from celltrack.training.joint_config import WARM_PACKS, JointTrainConfig
from celltrack.training.run_tracking import TrainingSplit
from core.data.tracks import AnnotatedTracks
from core.data.video import CellVideo, ImageStatistics
from core.obs import Obs
from core.paths import DataRoot

logger = logging.getLogger(__name__)

_DATASET = "biohub_cell_tracking"

_SYNTHETIC_DATASET = "biohub_synthetic"
_SYNTHETIC_SEQUENCES = "sequences"


@dataclass(frozen=True)
class PairSplit:
    """What a joint run optimises on, what it may ALSO optimise on, and what it is measured over."""

    train: list[PairTarget]
    val: list[PairTarget]
    synthetic: list[PairTarget] = field(default_factory=list)

    def corpus(self) -> list[PairTarget]:
        """The index space a curriculum draws over: the real pairs first, then the synthetic ones."""
        return [*self.train, *self.synthetic]

    @classmethod
    def assemble(
        cls, root: DataRoot, config: JointTrainConfig, proxy: TestMovieProxy, log: logging.Logger, val_videos: int
    ) -> tuple["PairSplit", TrainingSplit]:
        """Every pair the run reads, plus the video-level split that is its provenance."""
        train_paths = TestMovieProxy.training_videos(root.videos("train"))
        split = TrainingSplit(len(train_paths), len(VALIDATION_MOVIES), len(TEST_MOVIES))
        logger.info(
            "split: %d train / %d validation (selected on) / %d test (estimate only)",
            split.train,
            split.validation,
            split.test,
        )
        detected = cls.of_detected(root, config, train_paths, log)
        if detected:
            train_targets = detected
        else:
            train_targets = []
            with Obs.timed(log, f"enumerating GT pairs over {len(train_paths)} train videos"):
                for path in Obs.progress(train_paths, "videos", len(train_paths)):
                    train_targets.extend(cls.of_video(path, AnnotatedTracks.from_geff(root.track_store(path))))
        logger.info("%d train pairs", len(train_targets))

        # The edge AUC reads the SAME held-out movies the checkpoint is selected on — one validation set, and a
        # pair-level forward per annotated frame, so `val_videos` caps how many of them contribute.
        held = list(zip(proxy.paths, proxy.truths, strict=True))[:val_videos]
        val_targets = [target for video, truth in held for target in cls.of_video(video, truth)]
        logger.info("%d val pairs over %d validation movies", len(val_targets), len(held))
        return cls(train_targets, val_targets, cls.of_synthetic(root, config, log)), split

    @staticmethod
    def of_video(video: Path, annotation: AnnotatedTracks) -> list[PairTarget]:
        """Every consecutive-frame GT pair of one video, with its per-video intensity quantiles read from OME."""
        quantiles = cast(ImageStatistics, zarr.open_group(video, mode="r").attrs["image_statistics"])["quantiles"]
        q_low, q_high = float(quantiles["0.001"]), float(quantiles["0.999"])
        return PairTarget.enumerate(annotation.graph, ZarrFrames(video, q_low, q_high))

    @staticmethod
    def of_detected(
        root: DataRoot, config: JointTrainConfig, train_paths: list[Path], log: logging.Logger
    ) -> list[PairTarget]:
        """Pairs whose candidates are the DETECTED neighbourhood — empty unless a run asks, so nothing is mounted.

        These REPLACE the annotated pairs rather than joining them (see `DataCfg.detected_videos`). The
        detector is the SHIPPED one at the shipped threshold: the corpus has to be the population the tracker
        will choose among, so a different detector or operating point would be a different candidate set and
        the whole point would be lost.
        """
        videos = config.data.detected_videos
        if not videos:
            return []
        proc = root.processed(_DATASET)
        operating = TrackerConfig.shipped()
        tracker = CellTracker.from_packs(
            proc / WARM_PACKS["seed1"],
            proc / WARM_PACKS["seed2"],
            proc / "cache/responses",
            config.runtime.device,
            operating,
        )
        targets: list[PairTarget] = []
        chosen = train_paths[:videos]
        with Obs.timed(log, f"detecting over {len(chosen)} train videos for the detected-pair corpus"):
            for path in Obs.progress(chosen, "videos", len(chosen)):
                detections = tracker.detector.nodes(path.stem, path, operating.threshold, None)
                truth = AnnotatedTracks.from_geff(root.track_store(path)).graph
                spacing = CellVideo.from_ome_zarr(path).spacing
                quantiles = cast(ImageStatistics, zarr.open_group(path, mode="r").attrs["image_statistics"])[
                    "quantiles"
                ]
                frames = ZarrFrames(path, float(quantiles["0.001"]), float(quantiles["0.999"]))
                targets.extend(DetectionPairs.of(detections, truth, frames, spacing, operating.linker.gate_um))
        rows = sum(len(target.source_centres) for target in targets)
        # The matrix SHAPE is logged because it is what the step's cost and memory scale with, and guessing at
        # it once already cost a 32GB VRAM spill: ungated, a frame's ~743 detections make an edge matrix ~60x
        # the annotated one. A run whose widths read in the hundreds is not gated, whatever the flag says.
        widths = [len(target.target_centres) for target in targets] or [0]
        logger.info(
            "%d detected pairs, %d supervised rows | targets/gap mean %.1f median %d max %d — the deployed set",
            len(targets),
            rows,
            sum(widths) / len(widths),
            int(np.median(widths)),
            max(widths),
        )
        return targets

    @staticmethod
    def of_synthetic(root: DataRoot, config: JointTrainConfig, log: logging.Logger) -> list[PairTarget]:
        """The fully-labelled pairs, when a policy asks for them — empty otherwise, so nothing is even decoded.

        The pooling precondition is checked HERE, before a single step runs: the corpus's nodes live on the raw
        grid while its volumes ship pooled, so a run at any other downsample is reading a grid the coordinates
        are not on — the mismatch that has already cost one wrong measurement.
        """
        data = config.data
        if not (data.synthetic_fraction or data.paced_curriculum):
            return []
        if data.downsample != POOLED_BY:
            raise ValueError(f"the synthetic corpus ships pooled by {POOLED_BY}; this run reads {data.downsample}")
        sequences = SyntheticPairs.sequences(
            root.synthetic(_SYNTHETIC_DATASET) / _SYNTHETIC_SEQUENCES, data.synthetic_sequences, config.runtime.seed
        )
        with Obs.timed(log, f"enumerating synthetic pairs over {len(sequences)} sequences"):
            targets = SyntheticPairs.enumerate(sequences)
        logger.info("%d synthetic pairs — candidate sets complete by construction", len(targets))
        return targets
