"""Extract the champion kernel notebook's code cells to a reviewable, git-tracked ``.py``.

The champion (LB 0.924, Kaggle submission 55779061) lives as a Kaggle notebook
(``dimiturnt/celltrack-ilp-faithful-0923``). Repo policy gitignores ``.ipynb`` (no notebooks in
git). This script writes the notebook's code verbatim to ``champion_submission.py`` so the actual
submission source is diffable in git without committing the notebook itself.

Not a submission entrypoint — the kernel runs no-internet on Kaggle with every dependency inlined.
This is a review/audit artifact and the loss-insurance copy of the assembly glue (the tracksdata
global-ILP wiring that does not import locally).

Refresh after editing the kernel on Kaggle::

    uv run kaggle kernels pull dimiturnt/celltrack-ilp-faithful-0923 \
        -p kaggle/kernels/celltrack-ilp-faithful-0923
    uv run python kaggle/kernels/celltrack-ilp-faithful-0923/extract_champion.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

_HERE = Path(__file__).parent
_NOTEBOOK = _HERE / "biohub-0-927-lb.ipynb"
_OUTPUT = _HERE / "champion_submission.py"

_HEADER = """\
# Champion submission (LB 0.924, Kaggle submission 55779061) -- extracted VERBATIM from the
# gitignored kernel notebook dimiturnt/celltrack-ilp-faithful-0923 (biohub-0-927-lb.ipynb).
# REVIEW / DIFF / LOSS-INSURANCE ONLY: the Kaggle-run assembly, inlined (no-internet kernel).
# Not a -m entrypoint; the cells run in order inside the notebook. Regenerate with
# extract_champion.py after pulling the kernel. Do not hand-edit -- edit the kernel and re-extract.
"""


def extract() -> None:
    """Concatenate the notebook's code cells into champion_submission.py, asserting each parses."""
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    lines = [_HEADER]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).rstrip()
        if not source:
            continue
        ast.parse(source)  # a cell that does not parse is a corrupt pull, not a review artifact
        lines.append(f"# ==================== notebook cell [{index}] ====================")
        lines.append(source)
        lines.append("")
    _OUTPUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    extract()
