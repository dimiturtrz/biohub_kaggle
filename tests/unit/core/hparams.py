import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.hparams import RunConfig


def test_defaults_describe_the_joint_recipe():
    cfg = RunConfig()
    assert cfg.data.window_size == 2
    assert cfg.data.downsample == (1, 4, 4)
    assert cfg.data.pool_kernel_um == pytest.approx(5.0)
    assert cfg.optim.lr == pytest.approx(1e-3)  # their train() default
    assert cfg.optim.det_loss_weight == pytest.approx(10.0)  # their 1e1
    assert cfg.optim.precision == "bf16"  # mixed precision, fp32 exponent range — accuracy ~unchanged
    assert cfg.optim.patience == 8
    assert cfg.eval.threshold == pytest.approx(0.99)


def test_apply_overrides():
    cfg = RunConfig().apply_overrides(["optim.lr=1e-3", "optim.epochs=30", "aug.flip=true"])
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.optim.epochs == 30
    assert cfg.aug.flip is True


def test_to_json(tmp_path: Path):
    path = tmp_path / "config.json"
    RunConfig(name="probe").apply_overrides(["optim.epochs=12"]).to_json(path)
    assert json.loads(path.read_text(encoding="utf-8"))["optim"]["epochs"] == 12


def test_from_json(tmp_path: Path):
    original = RunConfig(name="probe").apply_overrides(["data.subset=4"])
    path = tmp_path / "config.json"
    original.to_json(path)
    assert RunConfig.from_json(path) == original


def test_override_of_tuple_uses_literal_eval():
    cfg = RunConfig().apply_overrides(["data.downsample=(1,2,2)"])
    assert cfg.data.downsample == (1, 2, 2)


def test_override_of_optional_str_clears_to_none():
    cfg = RunConfig(out_dir="runs/x").apply_overrides(["out_dir=none"])
    assert cfg.out_dir is None


def test_out_of_bounds_override_is_rejected_at_assignment():
    with pytest.raises(ValidationError):
        RunConfig().apply_overrides(["optim.lr=0"])  # lr must be > 0


def test_enum_field_rejects_unknown_value():
    with pytest.raises(ValidationError):
        RunConfig().apply_overrides(["optim.precision=fp8"])


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        RunConfig(unknown=1)  # extra="forbid"
