from pathlib import Path
from types import ModuleType
from typing import override

import pytest
import torch

from tools.frontier_train import (
    _SLOW_MATCH,
    RESUME_NAME,
    RunProgress,
    greedy_unique_match,
    patch_fast_match,
    patch_resume,
    restore_progress,
    save_progress,
)

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


class _Wrapped(torch.nn.Module):
    """The donor's shape: a ``unet`` attribute the DataParallel prefix logic keys off, plus a head beside it."""

    def __init__(self) -> None:
        super().__init__()
        self.unet = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)

    @override
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.unet(image))


def _stepped() -> tuple[_Wrapped, torch.optim.Optimizer]:
    """A model whose optimizer holds non-trivial moments, so a round trip that drops them is visible."""
    model = _Wrapped()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    return model, optimizer


def test_save_progress_round_trips_the_moments_not_only_the_weights(tmp_path: Path) -> None:
    model, optimizer = _stepped()
    saved = save_progress(tmp_path, model, optimizer, done=7, best=0.9335)
    restored = _Wrapped()
    fresh = torch.optim.AdamW(restored.parameters(), lr=0.1)

    progress = restore_progress(tmp_path, restored, fresh, resume=True)

    assert saved == "ep7"
    assert progress == RunProgress(done=7, best=0.9335)
    assert torch.equal(restored.head.weight, model.head.weight)
    assert fresh.state_dict()["state"][0]["exp_avg"].abs().sum() > 0


def test_restore_progress_starts_from_zero_when_resume_is_not_asked_for(tmp_path: Path) -> None:
    model, optimizer = _stepped()
    save_progress(tmp_path, model, optimizer, done=7, best=0.9335)

    assert restore_progress(tmp_path, model, optimizer, resume=False) == RunProgress(done=0, best=0.0)


_FAKE_TRAIN = """def train(output_dir, model, optimizer, n_epochs):
    best_score = 0.0
    save_path = output_dir / "edge_predictor_best.pth"
    pbar = tqdm(range(n_epochs), desc="Training", disable=False)
    seen = []
    for epoch in pbar:
        is_best = True
        best_score = float(epoch)
        marker = "*" if is_best else " "
        train_time = test_time = 0.0
        seen.append(
            f"{epoch}{marker}"
            f"train={train_time:.1f}s test={test_time:.1f}s",
        )
    return seen
"""


def _resume_trainer(tmp_path: Path, source_text: str) -> ModuleType:
    source = tmp_path / "fake_resume_trainer.py"
    source.write_text(source_text)
    module = ModuleType("fake_resume_trainer")
    module.__file__ = str(source)
    module.__dict__["tqdm"] = lambda iterable, **_: iterable
    exec(compile(source_text, str(source), "exec"), module.__dict__)
    return module


def test_patch_resume_snapshots_every_epoch_and_names_it_in_the_epoch_line(tmp_path: Path) -> None:
    module = _resume_trainer(tmp_path, _FAKE_TRAIN)
    model, optimizer = _stepped()

    patch_resume(module, resume=False)
    seen = module.train(tmp_path, model, optimizer, 2)

    assert (tmp_path / RESUME_NAME).exists()
    assert [line.split("|")[-1].strip() for line in seen] == ["snapshot=ep1", "snapshot=ep2"]


def test_patch_resume_picks_the_run_up_at_the_epoch_after_the_snapshot(tmp_path: Path) -> None:
    module = _resume_trainer(tmp_path, _FAKE_TRAIN)
    model, optimizer = _stepped()
    patch_resume(module, resume=False)
    module.train(tmp_path, model, optimizer, 2)

    patch_resume(module, resume=True)

    assert module.train(tmp_path, model, optimizer, 2) == []
    assert module.train(tmp_path, model, optimizer, 4)[0].startswith("2")


def test_patch_resume_rejects_changed_upstream_source(tmp_path: Path) -> None:
    module = _resume_trainer(tmp_path, "def train(output_dir, model, optimizer, n_epochs):\n    return []\n")

    with pytest.raises(RuntimeError, match="no longer applies"):
        patch_resume(module, resume=False)
