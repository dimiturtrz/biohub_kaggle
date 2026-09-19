"""Stage a kernel variant with pinned environment knobs (existing keys re-assigned, new keys appended).

The kernel's own config-drift guard is re-pinned alongside each key it covers, so a pinned value the
guard knows about does not trip it:

    python kaggle/env_variant.py --base-kernel kaggle/kernels/celltrack-public-0947 \
        --out-kernel kaggle/kernels/celltrack-public-0947-fast \
        --set BIOHUB_VALIDATOR_ENABLE=0 --set BIOHUB_MOTION_RELINK_TIGHT_UM=5.5
"""

import argparse
import json
import logging
from pathlib import Path

from local_kernel import pin_env

log = logging.getLogger(__name__)

OWNER = "dimiturnt"
ANCHOR = "os.environ['BIOHUB_DIAGNOSTIC_ARM'] = 'harmonic_association_production'\n"


def patch_notebook(notebook: dict, overrides: dict[str, str]) -> dict:
    cells = [c for c in notebook["cells"] if ANCHOR in "".join(c["source"])]
    assert len(cells) == 1, "expected one cell holding the environment block"
    source = pin_env("".join(cells[0]["source"]), overrides)
    appended = "".join(f"os.environ[{k!r}] = {v!r}\n" for k, v in overrides.items() if f"[{k!r}]" not in source)
    cells[0]["source"] = source.replace(ANCHOR, ANCHOR + appended).splitlines(keepends=True)
    return notebook


def stage_kernel(base: Path, out: Path, overrides: dict[str, str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((base / "kernel-metadata.json").read_text())
    notebook = json.loads((base / meta["code_file"]).read_text(encoding="utf-8"))
    name = out.name
    meta.update(id=f"{OWNER}/{name}", title=name, code_file=f"{name}.ipynb")
    (out / meta["code_file"]).write_text(json.dumps(patch_notebook(notebook, overrides), indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-kernel", type=Path, required=True)
    parser.add_argument("--out-kernel", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides")
    args = parser.parse_args()
    overrides = dict(pair.split("=", 1) for pair in args.overrides)
    stage_kernel(args.base_kernel, args.out_kernel, overrides)
    log.info("staged %s (%s)", args.out_kernel, ", ".join(f"{k}={v}" for k, v in overrides.items()))


if __name__ == "__main__":
    main()
