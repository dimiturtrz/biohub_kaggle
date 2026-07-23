"""Partitioning the training videos for local validation.

The two filename prefixes are two acquisitions, and they are not interchangeable: `44b6` holds ~36.9k
cells at 0.99 % annotated, `6bba` ~16.5k at 9.0 %. A partition that ignores them can hand one fold most
of one acquisition and read the resulting score as generalisation.

It does *not* follow that the acquisitions should be held out from each other. The competition's own test
split is two videos of each prefix, so acquisition identity is present on both sides there — training on
one acquisition and validating on the other would measure a harder task than the leaderboard scores, and
mis-estimate it. Validation therefore mirrors the test split: stratified by prefix, partitioned at video
level, with the score reported per prefix as well as pooled, since a pooled number hides which
acquisition a method actually helped.

Membership comes from a stable hash of the video name rather than a shuffle, so a fold is reproducible
from the filenames alone — no seed to carry, no id list to drift out of sync with the data.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_DIGEST_BYTES = 8
_DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class VideoSplit:
    """One fold: the videos to fit on, and the held-out videos that are never fitted on."""

    train: tuple[Path, ...]
    validation: tuple[Path, ...]

    def validation_by_prefix(self) -> dict[str, tuple[Path, ...]]:
        """The held-out videos grouped by acquisition — a pooled score hides which acquisition improved."""
        return AcquisitionFolds.by_prefix(self.validation)


@dataclass(frozen=True)
class AcquisitionFolds:
    """Deterministic k-fold over videos, stratified so each acquisition is spread evenly across folds."""

    folds: int = _DEFAULT_FOLDS

    def split(self, videos: Sequence[Path], index: int) -> VideoSplit:
        """The train/validation partition for one fold, counting folds from zero."""
        if not 0 <= index < self.folds:
            raise ValueError(f"fold {index} is outside the {self.folds} folds available")
        held_out = set(self._fold_members(videos, index))
        return VideoSplit(
            train=tuple(video for video in videos if video not in held_out),
            validation=tuple(video for video in videos if video in held_out),
        )

    def _fold_members(self, videos: Sequence[Path], index: int) -> list[Path]:
        """The videos of one fold — each acquisition dealt round-robin, so every fold holds both."""
        members: list[Path] = []
        for group in self.by_prefix(videos).values():
            ordered = sorted(group, key=self.rank_of)
            members.extend(ordered[index :: self.folds])
        return members

    @staticmethod
    def by_prefix(videos: Sequence[Path]) -> dict[str, tuple[Path, ...]]:
        """Group videos by acquisition, so each one can be dealt across folds and scored on its own."""
        grouped: dict[str, list[Path]] = {}
        for video in videos:
            grouped.setdefault(AcquisitionFolds.prefix_of(video), []).append(video)
        return {prefix: tuple(members) for prefix, members in sorted(grouped.items())}

    @staticmethod
    def prefix_of(video: Path) -> str:
        """The acquisition a video belongs to, which is the part of its name before the underscore."""
        return video.stem.split("_")[0]

    @staticmethod
    def rank_of(video: Path) -> int:
        """A stable position for a video, so folds are reproducible from the filenames alone."""
        digest = hashlib.blake2b(video.stem.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
        return int.from_bytes(digest, "big")
