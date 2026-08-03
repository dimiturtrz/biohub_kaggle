"""A run as one injectable, *validated* config (pydantic v2) — the typed source of truth for a run.

A run is fully described by one `RunConfig`, serialized to `runs/<name>/config.json` (provenance +
reproducibility). Why pydantic (not plain dataclasses): the fields cross a trust boundary — `--set`
overrides and a loaded config.json are user input. Bounds (`lr>0`, `patience>=1`, `precision` enum, …)
plus `validate_assignment` reject a bad value AT LOAD with a clear error, instead of silently training
on garbage.

Separate from `paths.yaml` (machine-specific *where data lives*); this is run-specific *how to train*.
The nested sub-configs mirror the runtime objects a trainer wires: `data`/`aug` (the input engine),
`optim` (the update rule), `eval` (the in-loop scorer).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# validate on construction AND on every setattr (so `--set` and cfg.x = y are both bounds-checked).
_VALIDATE = ConfigDict(validate_assignment=True, extra="forbid")

# a leaf field's runtime value type — what `--set a.b=val` coerces `val` into (isinstance-branched).
FieldValue = bool | int | float | tuple[object, ...] | list[object] | str | None


class DataCfg(BaseModel):
    """The input engine: which fold, how much of it, and the spatial/temporal window read per sample."""

    model_config = _VALIDATE

    fold: int = Field(0, ge=0)
    subset: int = Field(0, ge=0)  # videos/prefix for the in-loop test split; 0 = full fold
    window_size: int = Field(2, ge=1)  # frames per temporal sample (t, t+1) for edge supervision
    downsample: tuple[int, int, int] = (1, 4, 4)  # (z, y, x) strided read — the pilkwang x4 in-plane
    pool_kernel_um: float = Field(5.0, gt=0)  # local-max peak-readout kernel (their model_config default)


class AugCfg(BaseModel):
    """Intensity + geometric jitter applied to a training frame (0 everywhere = augmentation off)."""

    model_config = _VALIDATE

    brightness: float = Field(0.0, ge=0)  # multiplicative intensity jitter half-range
    offset: float = Field(0.0, ge=0)  # additive intensity jitter half-range
    flip: bool = False  # random in-plane (y, x) flips, mirroring GT centres


class OptimCfg(BaseModel):
    """The update rule + stopping policy: AdamW lr, epoch ceiling, loss weighting, early stop."""

    model_config = _VALIDATE

    lr: float = Field(1e-3, gt=0)  # their train() default (AdamW)
    epochs: int = Field(50, ge=1)  # ceiling; early stopping ends sooner
    batch: int = Field(1, ge=1)  # 1 for our mixed video shapes (no shape-bucketed batching yet)
    det_loss_weight: float = Field(10.0, ge=0)  # joint loss = edge_loss + det_loss_weight * det_loss (their 1e1)
    neg_weight: float = Field(1e-2, gt=0)  # background-voxel down-weight in the detection BCE
    grad_clip: float = Field(1.0, gt=0)
    cosine_lr: bool = False  # decay lr to zero over `epochs` (cosine); off keeps a flat lr
    patience: int = Field(8, ge=1)  # early stop after this many epochs with no best-score gain
    es_min_delta: float = Field(0.0, ge=0)  # min val-gain to count as improvement (0 = any gain)
    # bf16 keeps fp32's exponent range (no fp16 overflow→NaN); mixed precision keeps fp32 master weights,
    # so accuracy is ~unchanged. Native on the 5090. fp32 is the exact faithful recipe.
    precision: Literal["fp32", "bf16"] = "bf16"


class EvalCfg(BaseModel):
    """The in-loop scorer: how often to score the fold subset and at what detection threshold."""

    model_config = _VALIDATE

    every: int = Field(1, ge=1)  # score every N epochs
    subset: int = Field(8, ge=0)  # videos/prefix scored in-loop; 0 = full fold
    threshold: float = Field(0.99, gt=0, lt=1)  # peak-readout probability operating point


class RunConfig(BaseModel):
    """The whole run = master config. `model_dump()` -> config.json = full provenance; one RunConfig
    fully describes a run."""

    model_config = _VALIDATE

    data: DataCfg = Field(default_factory=DataCfg)
    aug: AugCfg = Field(default_factory=AugCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    name: str = "run"
    seed: int = Field(0, ge=0)
    device: str = "cuda"
    out_dir: str | None = None

    @staticmethod
    def _coerce(val: str, current: FieldValue) -> FieldValue:
        """Parse an override string to the current field's type (tuples via literal_eval). pydantic's
        validate_assignment then enforces the bounds when the result is set."""
        if isinstance(current, bool):
            return val.lower() in ("1", "true", "yes", "on")
        if isinstance(current, int):
            return int(val)
        if isinstance(current, float):
            return float(val)
        if isinstance(current, (tuple, list)):
            return ast.literal_eval(val)
        # str / Optional[str] field: "none"/"null"/"" -> None (validate_assignment rejects it for a
        # required str like `name`, accepts it for an Optional like `out_dir`).
        return None if val.lower() in ("none", "null", "") else val

    def apply_overrides(self, items: list[str] | None) -> "RunConfig":
        """Apply `a.b=val` dotted overrides in place (e.g. 'optim.lr=1e-3', 'data.downsample=(1,2,2)').
        Each setattr is validated (validate_assignment) -> an out-of-bounds/typo value raises at once."""
        for item in items or []:
            key, _, val = item.partition("=")
            *parents, leaf = key.strip().split(".")
            target: object = self
            for parent in parents:
                target = getattr(target, parent)
            setattr(target, leaf, RunConfig._coerce(val.strip(), getattr(target, leaf)))
        return self

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
