"""Run the frontier ``train_unet_transformer.py`` with a strict FULL-model warm start.

Its own ``--unet-weights`` loads non-strictly into the bare UNet, so a full checkpoint (``unet.``-prefixed keys
plus detect head and transformer) silently loads nothing. ``--init-full <ckpt>`` loads every key strictly.

    python tools/frontier_train.py --frontier-repo <repo> --init-full <ckpt> <trainer args>

Its detection→GT matching walks detections in a Python loop with one GPU sync each (~8k syncs per step on dense
frames, GPU ~5% busy); the loop is replaced by :func:`greedy_unique_match`, the same greedy assignment as a
scatter-min.
"""

import argparse
import importlib
import inspect
import logging
import sys
from pathlib import Path
from types import ModuleType

import torch

log = logging.getLogger(__name__)

_SLOW_MATCH = """            order = min_d.argsort()
            gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)
            for idx in order:
                if min_d[idx] > max_match_distance:
                    break
                gi = min_i[idx]
                if not gt_taken[gi]:
                    matched[idx] = gi
                    gt_taken[gi] = True
"""
_FAST_MATCH = "            matched = greedy_unique_match(min_d, min_i, n_gt, max_match_distance)\n"


def greedy_unique_match(min_d: torch.Tensor, min_i: torch.Tensor, n_gt: int, max_distance: float) -> torch.Tensor:
    """Nearest-first greedy: each GT goes to the closest in-range detection whose nearest GT it is, else -1."""
    index = torch.arange(min_d.shape[0], device=min_d.device)
    within = min_d <= max_distance
    closest = torch.full((n_gt,), torch.inf, device=min_d.device, dtype=min_d.dtype).scatter_reduce(
        0, min_i, torch.where(within, min_d, torch.inf), "amin"
    )
    candidate = within & (min_d == closest[min_i])
    first = torch.full((n_gt,), min_d.shape[0], device=min_d.device, dtype=torch.long).scatter_reduce(
        0, min_i, torch.where(candidate, index, min_d.shape[0]), "amin"
    )
    return torch.where(candidate & (first[min_i] == index), min_i, -1)


def patch_fast_match(trainer: ModuleType) -> None:
    source = inspect.getsource(trainer.detect_and_match)
    if _SLOW_MATCH not in source:
        raise RuntimeError("frontier detect_and_match changed; greedy match patch no longer applies")
    trainer.greedy_unique_match = greedy_unique_match
    exec(compile(source.replace(_SLOW_MATCH, _FAST_MATCH), trainer.__file__, "exec"), trainer.__dict__)


def _strict_init_class(base: type, checkpoint: Path) -> type:
    class StrictInit(base):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            log.warning("full-model init loaded strictly from %s", checkpoint)

    return StrictInit


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-repo", type=Path, required=True)
    parser.add_argument("--init-full", type=Path, default=None)
    args, trainer_args = parser.parse_known_args()

    sys.path[:0] = [str(args.frontier_repo / "src"), str(args.frontier_repo / "scripts")]
    trainer = importlib.import_module("train_unet_transformer")
    patch_fast_match(trainer)
    if args.init_full is not None:
        trainer.UNetNodeTransformer = _strict_init_class(trainer.UNetNodeTransformer, args.init_full)
    sys.argv = ["train_unet_transformer.py", *trainer_args]
    trainer.main()


if __name__ == "__main__":
    main()
