"""One home for mounting a `CellTracker` off the artifacts on disk — the reference packs and a joint checkpoint.

Every offline driver (a proxy sweep, a fate diagnosis, a decorrelation report, the confusor referee, an edge
finetune, the detected-pair corpus) needs the SAME two mounts, and each had rebuilt them inline: the pilkwang
pack triple (primary weights, seed-2 weights, response cache) as three path literals, and a joint checkpoint
as `JointModel.from_checkpoint` -> `DetectorRecipe(downsample=...)` -> `CellTracker.from_joint`. Two copies of
a layout literal are a layout that can drift; the joint pair is worse, because a recipe built from anything
but the checkpoint's own downsample decodes at a grid the transformer never saw.

The mounts take `proc` (the processed-dataset directory under the data root) rather than reading a config, so
the same call serves a local run and the Kaggle kernel's dataset mount.
"""

from __future__ import annotations

from pathlib import Path

from celltrack.models.joint_model import JointModel
from celltrack.models.temporal_unet_detector import DetectorRecipe
from celltrack.operating_point import TrackerConfig
from celltrack.tracker import CellTracker

PILKWANG_PRIMARY = Path("reference/pilkwang/split_0")
PILKWANG_SECONDARY = Path("reference/pilkwang/seed2/weights/unet_transformer/split_0")
RESPONSE_CACHE = Path("cache/responses")


class ReferenceMount:
    """The mounts an offline driver assembles a tracker with — pack pair, the shipped pilkwang pair, or joint."""

    @staticmethod
    def packs(
        proc: Path, primary: Path, secondary: Path, device: str, config: TrackerConfig | None = None
    ) -> CellTracker:
        """A cache-backed two-pack mount at an arbitrary pack pair, both resolved under `proc`."""
        return CellTracker.from_packs(proc / primary, proc / secondary, proc / RESPONSE_CACHE, device, config)

    @staticmethod
    def pilkwang(proc: Path, device: str, config: TrackerConfig | None = None) -> CellTracker:
        """The champion dual-seed pilkwang mount — the pack pair every offline reading of the shipped tracker uses."""
        return ReferenceMount.packs(proc, PILKWANG_PRIMARY, PILKWANG_SECONDARY, device, config)

    @staticmethod
    def joint(checkpoint: Path, proc: Path, device: str, config: TrackerConfig | None = None) -> CellTracker:
        """A jointly-trained checkpoint's own detector + transformer, its recipe read off the checkpoint itself.

        The downsample comes from the loaded model rather than a caller-stated recipe: `from_joint` raises on a
        mismatch, and there is exactly one right answer, so no call site should be in a position to get it wrong.
        """
        model = JointModel.from_checkpoint(checkpoint, device)
        recipe = DetectorRecipe(downsample=model.downsample)
        return CellTracker.from_joint(checkpoint, recipe, device, config=config, responses=proc / RESPONSE_CACHE)
