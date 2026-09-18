from pathlib import Path
from types import ModuleType

import pytest
import torch

from tools.frontier_train import _SLOW_MATCH, greedy_unique_match, patch_fast_match

MAX_DISTANCE = 0.6


def _reference_loop(min_d: torch.Tensor, min_i: torch.Tensor, n_gt: int, max_distance: float) -> torch.Tensor:
    matched = torch.full((min_d.shape[0],), -1, dtype=torch.long)
    gt_taken = torch.zeros(n_gt, dtype=torch.bool)
    for idx in min_d.argsort():
        if min_d[idx] > max_distance:
            break
        gi = min_i[idx]
        if not gt_taken[gi]:
            matched[idx] = gi
            gt_taken[gi] = True
    return matched


@pytest.mark.parametrize(("n_det", "n_gt"), [(1, 1), (5, 40), (40, 5), (300, 250)])
def test_greedy_unique_match_equals_reference_loop(n_det: int, n_gt: int) -> None:
    generator = torch.Generator().manual_seed(n_det * 1000 + n_gt)
    min_d = torch.rand(n_det, generator=generator)
    min_i = torch.randint(0, n_gt, (n_det,), generator=generator)

    fast = greedy_unique_match(min_d, min_i, n_gt, MAX_DISTANCE)

    assert torch.equal(fast, _reference_loop(min_d, min_i, n_gt, MAX_DISTANCE))


def test_all_out_of_range_matches_nothing() -> None:
    fast = greedy_unique_match(torch.tensor([0.9, 0.8]), torch.tensor([0, 0]), 1, MAX_DISTANCE)

    assert fast.tolist() == [-1, -1]


def _trainer_module(tmp_path: Path, body: str) -> ModuleType:
    source = tmp_path / "fake_trainer.py"
    source.write_text(
        "import torch\n\n\ndef detect_and_match(min_d, min_i, n_gt, max_match_distance, device='cpu'):\n"
        "    matched = torch.full((min_d.shape[0],), -1, dtype=torch.long)\n"
        "    if True:\n        if True:\n" + body + "    return matched\n"
    )
    module = ModuleType("fake_trainer")
    module.__file__ = str(source)
    exec(compile(source.read_text(), str(source), "exec"), module.__dict__)
    return module


def test_patch_swaps_loop_for_vectorised_match(tmp_path: Path) -> None:
    module = _trainer_module(tmp_path, _SLOW_MATCH)
    min_d, min_i = torch.tensor([0.1, 0.2, 0.9]), torch.tensor([0, 0, 1])

    patch_fast_match(module)

    assert "greedy_unique_match" in module.detect_and_match.__code__.co_names
    assert module.detect_and_match(min_d, min_i, 2, MAX_DISTANCE).tolist() == [0, -1, -1]


def test_patch_rejects_changed_upstream_source(tmp_path: Path) -> None:
    module = _trainer_module(tmp_path, "            pass\n")

    with pytest.raises(RuntimeError, match="no longer applies"):
        patch_fast_match(module)
