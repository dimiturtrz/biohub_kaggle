"""Difference the `BIOHUB_*` environment settings of two kernels.

A donor kernel is a wall of code around a small set of `os.environ` assignments, and the assignments are
where its method lives: every element we have ever harvested was a flag or a constant, never a rewrite. So
the harvest question -- "is there anything in this kernel we do not already run?" -- is answered by
differencing the two SETS of settings, not by reading the notebook. Reading the notebook invents elements
that turn out to be flags we already ship.

    python -m tools.kernel_env_diff ours.ipynb theirs.py

Three sections come back: what they set and we do not (candidate elements), what both set at different
values (constant deltas -- price these against our own numbers before transplanting, a donor constant is
fitted to the donor's distribution), and what we set and they do not.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_ASSIGNMENT = re.compile(r"os\.environ\[['\"](BIOHUB_[A-Z0-9_]+)['\"]\]\s*=\s*['\"]([^'\"]*)['\"]")


def kernel_settings(path: Path) -> dict[str, str]:
    """Every `BIOHUB_*` setting a kernel assigns, notebook or script.

    A notebook's cell sources are JSON-escaped, so they are decoded before matching -- matching the raw
    file instead needs a pattern that tolerates backslashes and silently returns nothing when it does not.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        cells = json.loads(text)["cells"]
        text = "".join("".join(cell.get("source", [])) for cell in cells)
    return dict(_ASSIGNMENT.findall(text))


def log_difference(ours: dict[str, str], theirs: dict[str, str]) -> None:
    """Report the three ways two setting sets can differ, loudly enough to act on."""
    logger.info("ours=%d theirs=%d", len(ours), len(theirs))
    logger.info("=== THEY SET, WE DO NOT (candidate elements) ===")
    for key in sorted(theirs.keys() - ours.keys()):
        logger.info("  %s = %s", key, theirs[key])
    logger.info("=== BOTH SET, DIFFERENT VALUE (constant deltas) ===")
    for key in sorted(theirs.keys() & ours.keys()):
        if theirs[key] != ours[key]:
            logger.info("  %s: OURS=%s THEIRS=%s", key, ours[key], theirs[key])
    logger.info("=== WE SET, THEY DO NOT ===")
    for key in sorted(ours.keys() - theirs.keys()):
        logger.info("  %s = %s", key, ours[key])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ours", type=Path, help="our kernel (.ipynb or .py)")
    parser.add_argument("theirs", type=Path, help="the donor kernel (.ipynb or .py)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ours, theirs = kernel_settings(args.ours), kernel_settings(args.theirs)
    if not ours or not theirs:
        raise SystemExit(
            f"no BIOHUB_* assignments parsed (ours={len(ours)} theirs={len(theirs)}); an empty side makes "
            "every key on the other side look like a candidate element, which is a denominator error, not a "
            "harvest -- check the path and that the file really carries the kernel source"
        )
    log_difference(ours, theirs)


if __name__ == "__main__":
    main()
