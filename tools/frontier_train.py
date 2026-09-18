"""Run the frontier ``train_unet_transformer.py`` with a strict FULL-model warm start.

Its own ``--unet-weights`` loads non-strictly into the bare UNet, so a full checkpoint (``unet.``-prefixed keys
plus detect head and transformer) silently loads nothing. ``--init-full <ckpt>`` loads every key strictly.

    python tools/frontier_train.py --frontier-repo <repo> --init-full <ckpt> <trainer args>
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch

log = logging.getLogger(__name__)


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
    if args.init_full is not None:
        trainer.UNetNodeTransformer = _strict_init_class(trainer.UNetNodeTransformer, args.init_full)
    sys.argv = ["train_unet_transformer.py", *trainer_args]
    trainer.main()


if __name__ == "__main__":
    main()
