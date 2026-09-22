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
_GET_DEFAULT = re.compile(r"os\.environ\.get\(\s*['\"](BIOHUB_[A-Z0-9_]+)['\"]\s*,\s*['\"]([^'\"]*)['\"]\s*\)")


def kernel_settings(path: Path) -> dict[str, str]:
    """Every `BIOHUB_*` setting a kernel EFFECTIVELY runs with, notebook or script.

    A kernel carries its constants in two places, and reading only one of them is a denominator error:
    an explicit `os.environ[KEY] = value` at the top, and the default of the `os.environ.get(KEY,
    default)` that consumes it. A donor that ships a constant purely as a reader default is invisible to
    an assignment-only parse -- which is how a kernel reading 70 keys and assigning 8 came back as
    "8 settings, nothing to harvest". The assignment wins where both exist, because that is what the
    interpreter does.

    A notebook's cell sources are JSON-escaped, so they are decoded before matching -- matching the raw
    file instead needs a pattern that tolerates backslashes and silently returns nothing when it does not.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        cells = json.loads(text)["cells"]
        text = "".join("".join(cell.get("source", [])) for cell in cells)
    return dict(_GET_DEFAULT.findall(text)) | dict(_ASSIGNMENT.findall(text))


def shadowed_settings(path: Path) -> dict[str, tuple[str, str]]:
    """Keys whose pin differs from the fallback the consumer would otherwise use.

    Quoting the fallback as though it were the running value is the plumbing trap in its purest form: the
    champion assigns `SAFE_DIV_MAX_UM = 9.0` while its consumer reads `os.environ.get(..., '4.7')`, and an
    interpretation written off the second number describes a kernel nobody ran.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".ipynb":
        text = "".join("".join(c.get("source", [])) for c in json.loads(text)["cells"])
    fallbacks, pins = dict(_GET_DEFAULT.findall(text)), dict(_ASSIGNMENT.findall(text))
    return {k: (v, fallbacks[k]) for k, v in pins.items() if k in fallbacks and fallbacks[k] != v}


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
    for label, path in (("ours", args.ours), ("theirs", args.theirs)):
        shadowed = shadowed_settings(path)
        logger.info("=== %s: PIN SHADOWS A DIFFERENT FALLBACK (quote the pin) ===", label)
        for key, (pin, fallback) in sorted(shadowed.items()):
            logger.info("  %s: PIN=%s (consumer fallback %s)", key, pin, fallback)
    if not ours or not theirs:
        raise SystemExit(
            f"no BIOHUB_* assignments parsed (ours={len(ours)} theirs={len(theirs)}); an empty side makes "
            "every key on the other side look like a candidate element, which is a denominator error, not a "
            "harvest -- check the path and that the file really carries the kernel source"
        )
    log_difference(ours, theirs)


if __name__ == "__main__":
    main()
