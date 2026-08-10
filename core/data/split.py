"""Grouping the training videos by acquisition.

The two filename prefixes are two acquisitions, and they are not interchangeable: `44b6` holds ~36.9k
cells at 0.99 % annotated, `6bba` ~16.5k at 9.0 %. Any partition or per-group report that ignores them
can hand one side most of one acquisition and read the resulting score as generalisation.

There are no k-folds here any more. The competition's four TEST movies live inside `train/`, so the
honest held-out set is those movies scored through the shipped pipeline (`celltrack.eval.proxy`), not a
hashed fold of the train directory — a fold-0 validation split both leaked (the published weights we
warm-start from were fitted on ~93 % of it) and misranked (0.837 on the fold eval vs 0.893 through the
real pipeline for the same weights). What survives is the acquisition vocabulary itself: which videos
belong to which acquisition, and a stable, seed-free rank within one.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_DIGEST_BYTES = 8


@dataclass(frozen=True)
class DatasetSplit:
    """A three-way partition: the videos to fit on, to validate each epoch, and to score once at the end."""

    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    test: tuple[Path, ...]


@dataclass(frozen=True)
class StratifiedSplit:
    """Deterministic train/validation/test over videos, each acquisition partitioned the same way.

    A held-out count per acquisition: the smallest useful test set mirrors the competition's own — a couple
    of videos of each acquisition — and validation is smaller still, since per-epoch scoring must stay cheap.
    Membership is by stable filename rank, so the split is reproducible from the filenames with no seed to
    carry. The trainers do not use this: they hold out the four proxy movies instead.
    """

    test_per_prefix: int = 2
    validation_per_prefix: int = 1

    def partition(self, videos: Sequence[Path]) -> DatasetSplit:
        """Split each acquisition into test, then validation, then train, by stable rank — the rest is train."""
        train: list[Path] = []
        validation: list[Path] = []
        test: list[Path] = []
        held = self.test_per_prefix + self.validation_per_prefix
        for group in Acquisitions.by_prefix(videos).values():
            ordered = sorted(group, key=Acquisitions.rank_of)
            test.extend(ordered[: self.test_per_prefix])
            validation.extend(ordered[self.test_per_prefix : held])
            train.extend(ordered[held:])
        return DatasetSplit(train=tuple(train), validation=tuple(validation), test=tuple(test))


@dataclass(frozen=True)
class Acquisitions:
    """The acquisition a video belongs to, and a stable order within one — the vocabulary, not a partition."""

    @staticmethod
    def by_prefix(videos: Sequence[Path]) -> dict[str, tuple[Path, ...]]:
        """Group videos by acquisition, so each one can be reported and sampled on its own."""
        grouped: dict[str, list[Path]] = {}
        for video in videos:
            grouped.setdefault(Acquisitions.prefix_of(video), []).append(video)
        return {prefix: tuple(members) for prefix, members in sorted(grouped.items())}

    @staticmethod
    def prefix_of(video: Path) -> str:
        """The acquisition a video belongs to, which is the part of its name before the underscore."""
        return video.stem.split("_")[0]

    @staticmethod
    def rank_of(video: Path) -> int:
        """A stable position for a video, so any ordering is reproducible from the filenames alone."""
        digest = hashlib.blake2b(video.stem.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
        return int.from_bytes(digest, "big")
