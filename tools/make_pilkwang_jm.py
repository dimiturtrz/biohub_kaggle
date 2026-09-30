"""Card-free converter: pilkwang seed1 pack -> a JointModel checkpoint for avl8 `--detector-from`.

avl8 reads ONLY the detector from this file (joint_assembly.py:180-193 builds a fresh C=288 head),
so the saved transformer just has to round-trip through JointModel.from_checkpoint. Runs on CPU.
"""

from pathlib import Path

from celltrack.models.edge_transformer import EdgeTransformerScorer
from celltrack.models.joint_model import JointModel
from celltrack.training.joint_checkpoint import JointCheckpoint
from celltrack.training.joint_config import WARM_PACKS
from core.paths import DataRoot

proc = DataRoot.from_config(Path("paths.yaml")).processed("biohub_cell_tracking")
pack = proc / WARM_PACKS["seed1"]
out = proc / "pilkwang_jm.pt"
downsample = (1, 4, 4)

scorer = EdgeTransformerScorer.from_pack(pack, "cpu")
det = scorer.detector
print(f"detector: out_channels={det.out_channels} layers={det.layers} norm={getattr(det, 'norm', '?')}")

model = JointModel(det, scorer.transformer, downsample)
jc = JointCheckpoint(det.out_channels, tuple(det.layers), downsample, "cpu")
jc.save_best(out, model)
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

# round-trip verify
back = JointModel.from_checkpoint(out, "cpu")
assert back.detector.out_channels == det.out_channels, "out_channels mismatch"
assert tuple(back.detector.layers) == tuple(det.layers), "layers mismatch"
assert tuple(back.downsample) == downsample, "downsample mismatch"
print(f"round-trip OK: detector out={back.detector.out_channels} layers={back.detector.layers} ds={back.downsample}")
