"""What a joint run persists: the best model, the resumable snapshot, and the one payload they share.

Two files come out of a run and they are NOT interchangeable. `save-best` writes the best-scoring model, and
when no window beats the initialisation that file IS the initialisation — so an arm that failed to clear its
own bar leaves a checkpoint measuring nothing about the arm. The `.resume.pt` snapshot holds whatever the run
last trained, beaten bar or not, and is therefore the only honest read-out for a matched comparison across
arms. Three arms were compared on the wrong one before this was separated out, and produced byte-identical
held-out numbers that looked like a tie between objectives and were the same untrained model scored thrice.

Both carry the SAME model payload — both heads plus the shape needed to rebuild them — so
`JointModel.from_checkpoint` reads either. The snapshot adds what continuing a run needs and the best file
does not: the optimiser, the projection head, the step reached and the best score so far.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from celltrack.models.joint_model import JointModel

RESUME_SUFFIX = ".resume.pt"

_LOGGER = logging.getLogger(__name__)

_UNEXPECTED_KEYS = "snapshot carries keys %s has no place for: %s"


@dataclass(frozen=True)
class RunProgress:
    """How far a run got and how well — the two numbers a snapshot needs beyond the model itself."""

    done: int
    best: float


class OptimizationState(Protocol):
    """The optimiser-side state a snapshot round-trips — whatever the trainer's optimisation bundle is."""

    def state_dict(self) -> dict[str, object]:
        """Everything needed to continue stepping."""
        ...

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore it."""
        ...


@dataclass(frozen=True)
class JointCheckpoint:
    """Reads and writes a joint run's two files against one model payload."""

    out_channels: int
    layers: tuple[int, ...]
    downsample: tuple[int, int, int]
    device: str

    def payload(self, model: JointModel) -> dict[str, object]:
        """Both heads and the shape needed to rebuild them — the one format every saved file speaks.

        The model reports its own serialisable half (the two state dicts and the shape facts to rebuild the heads);
        the checkpoint adds the trainer-owned build facts it holds itself. The state dicts sit at the top level and
        the rebuild facts under `config`, the split `from_checkpoint` reads back.
        """
        owned = model.serialisable()
        config = {key: owned.pop(key) for key in ("temporal_position", "relative_position", "norm", "head")}
        return {
            **owned,
            "config": {
                "out_channels": self.out_channels,
                "layers": list(self.layers),
                "downsample": list(self.downsample),
                **config,
            },
        }

    def save_best(self, path: Path, model: JointModel) -> None:
        """Persist the best-scoring model — read back by `JointModel.from_checkpoint`."""
        torch.save(self.payload(model), path)

    def save_snapshot(
        self,
        path: Path,
        model: JointModel,
        trained: dict[str, nn.Module],
        optimization: OptimizationState,
        progress: RunProgress,
    ) -> None:
        """Write the full run state so a killed run can continue — and so a beaten arm stays measurable.

        `trained` is the run's auxiliary trained modules by name (the contrastive head, the link objective) — each
        saved under its own key so a form that holds no parameters writes an empty dict and a resume tolerates it.
        """
        torch.save(
            self.payload(model)
            | {name: module.state_dict() for name, module in trained.items()}
            | {
                "optimization": optimization.state_dict(),
                "done": progress.done,
                "best": progress.best,
            },
            path,
        )

    def _restore_head(self, module: nn.Module, state: dict[str, Tensor]) -> None:
        """Load a snapshot that may predate one of the module's own heads, keeping that head's fresh init.

        The same tolerance the warm-start paths already grant (`JointModel.from_checkpoint`,
        `TemporalUNetDetector.from_pack`): the center-point `embed_head` was added after some snapshots were
        written, and a run resuming one of those must keep that head's initialisation rather than refuse the
        snapshot outright. The tolerance is one-directional — a key the snapshot carries and the module has
        no place for means the two are different architectures, and that still raises.
        """
        incompatible = module.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(_UNEXPECTED_KEYS % (type(module).__name__, sorted(incompatible.unexpected_keys)))
        if incompatible.missing_keys:
            _LOGGER.info(
                "resume: %s keeps its own init for %s", type(module).__name__, sorted(incompatible.missing_keys)
            )

    def restore(
        self, path: Path, model: JointModel, trained: dict[str, nn.Module], optimization: OptimizationState
    ) -> RunProgress:
        """Load a snapshot into both heads, the named auxiliary modules and the optimiser, returning progress."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        # `restore` runs AFTER `_prepare` has already compiled the backbone, so the live module may carry
        # `_orig_mod.` keys the snapshot lacks (or vice-versa when a prior run saved compiled and this one is
        # eager). `align_state_to` remaps the snapshot onto whatever wrapping the target now has — the fix is
        # direction-agnostic, unlike a blind strip which only served an eager target.
        align = JointModel._align_state_to  # noqa: SLF001 — the pack's own key-remap, shared with from_checkpoint
        detector, transformer = model.detector, model.transformer
        self._restore_head(detector, align(detector, state["detector_state"]))
        self._restore_head(transformer, align(transformer, state["transformer_state"]))
        # A snapshot written before an auxiliary module existed (the contrastive head, the link objective) or by
        # a parameter-free form carries no entry or an empty one; a run resuming it keeps that module's own init
        # — projecting nothing, or a fresh slack scalar that re-calibrates on its first batch.
        for name, module in trained.items():
            stored = state.get(name)
            if stored:
                module.load_state_dict(stored)
        optimization.load_state_dict(state["optimization"])
        return RunProgress(done=int(state["done"]), best=float(state["best"]))
