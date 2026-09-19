"""Re-run a donor kernel's own held-out validator sweep with our candidate configs (no submission slot spent).

The donor's `PP_CANDIDATES` dict and its `diagnose_divisions(...)` calls are replaced by a probe spec:

    python kaggle/donor_probe.py --donor-kernel research/frontier_kernels/biohub-reid3 \
        --spec kaggle/probes/reid3_steal.json --out-kernel kaggle/kernels/celltrack-reid3-steal-probe

Spec = {"candidates": {label: {ENV_KEY: value}}, "diagnose": {label: {ENV_KEY: value}}}.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from element_transplant import OWNER

log = logging.getLogger(__name__)

CANDIDATES_BLOCK = re.compile(r"PP_CANDIDATES: dict\[str, dict\] = \{\n.*?\n\}\n", re.DOTALL)
DIAGNOSE_BLOCK = re.compile(r"(if VALIDATOR_ENABLE and val_stems:\n)((?:    diagnose_divisions\(.*\)\n)+)")


def _candidates_source(candidates: dict[str, dict]) -> str:
    rows = "".join(f"    {json.dumps(label)}: {json.dumps(config)},\n" for label, config in candidates.items())
    return "PP_CANDIDATES: dict[str, dict] = {\n" + rows + "}\n"


def _diagnose_source(diagnose: dict[str, dict]) -> str:
    return "".join(
        f"    diagnose_divisions({json.dumps(label)}, {json.dumps(config)})\n" for label, config in diagnose.items()
    )


def rewrite(source: str, spec: dict) -> str:
    assert len(CANDIDATES_BLOCK.findall(source)) == 1, "PP_CANDIDATES block must be unique"
    assert len(DIAGNOSE_BLOCK.findall(source)) == 1, "diagnose block must be unique"
    source = CANDIDATES_BLOCK.sub(lambda _: _candidates_source(spec["candidates"]), source)
    return DIAGNOSE_BLOCK.sub(lambda m: m.group(1) + _diagnose_source(spec["diagnose"]), source)


def stage_probe(donor: Path, spec: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((donor / "kernel-metadata.json").read_text())
    notebook = json.loads((donor / meta["code_file"]).read_text(encoding="utf-8"))
    rewritten = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "PP_CANDIDATES: dict" in source:
            cell["source"] = rewrite(source, spec).splitlines(keepends=True)
            rewritten += 1
    assert rewritten == 1, "expected exactly one cell holding PP_CANDIDATES"
    name = out.name
    meta.pop("id_no", None)
    meta.update(id=f"{OWNER}/{name}", title=name, code_file=f"{name}.ipynb", is_private=True)
    (out / meta["code_file"]).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-kernel", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-kernel", type=Path, required=True)
    args = parser.parse_args()
    stage_probe(args.donor_kernel, json.loads(args.spec.read_text()), args.out_kernel)
    log.info("staged probe %s -> %s", args.spec, args.out_kernel)


if __name__ == "__main__":
    main()
