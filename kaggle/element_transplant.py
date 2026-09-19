"""Transplant a public-kernel element (see the element inventory in the conclusion tree) into a base kernel.

Each element is a code block injected before an anchor line plus a call inserted after a call-site line:

    python kaggle/element_transplant.py --element refine_centroids \
        --base-kernel kaggle/kernels/celltrack-public-0947 --out-kernel kaggle/kernels/celltrack-public-0947-refine
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

OWNER = "dimiturnt"


@dataclass(frozen=True)
class Element:
    definitions: str
    definitions_anchor: str
    call_site: str
    call: str


REFINE_CENTROIDS = Element(
    definitions="""
REFINE_CENTROIDS = os.environ.get('BIOHUB_REFINE_CENTROIDS', '1') == '1'


def refine_centroids(vol, coords, win = (1, 3, 3), baseline_percentile = 20.0, max_shift_um = 2.8):
    out = coords.astype(np.float64).copy()
    scale = np.array(VOXEL_SCALE_UM, dtype = np.float64)
    wz, wy, wx = win
    Z, Y, X = vol.shape
    for i, original in enumerate(out.copy()):
        z, y, x = (int(round(v)) for v in original)
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            continue
        weights = np.maximum(patch - np.percentile(patch, baseline_percentile), 0.0)
        total = float(weights.sum())
        if total <= 0:
            continue
        grid = np.meshgrid(np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing = 'ij')
        refined = np.array([float((weights * g).sum() / total) for g in grid])
        if np.sqrt((((refined - original) * scale) ** 2).sum()) <= max_shift_um:
            out[i] = refined
    return out


def refine_all_centroids(nodes_by_id, dataset):
    nodes_by_t = {}
    for node_id, node in nodes_by_id.items():
        nodes_by_t.setdefault(int(node['t']), []).append(node_id)
    moved = 0
    for t, node_ids in nodes_by_t.items():
        vol = read_test_frame(dataset, t, {})
        coords = np.array([[nodes_by_id[n]['z'], nodes_by_id[n]['y'], nodes_by_id[n]['x']] for n in node_ids], dtype = float)
        refined = refine_centroids(vol, coords)
        moved += int(np.any(refined != coords, axis = 1).sum())
        for idx, n in enumerate(node_ids):
            nodes_by_id[n]['z'], nodes_by_id[n]['y'], nodes_by_id[n]['x'] = (float(v) for v in refined[idx])
    print(f'  [{dataset}] refine_centroids moved {moved}/{len(nodes_by_id)} nodes')
    return nodes_by_id

""",
    definitions_anchor="def read_test_frame(",
    call_site="raw_node_count = len(nodes_by_id)\n",
    call="if REFINE_CENTROIDS:\n    nodes_by_id = refine_all_centroids(nodes_by_id, dataset)\n",
)

ELEMENTS = {"refine_centroids": REFINE_CENTROIDS}


def _indent_of(source: str, index: int) -> str:
    line_start = source.rfind("\n", 0, index) + 1
    return source[line_start:index]


def transplant(source: str, element: Element) -> str:
    assert source.count(element.call_site) == 1, "call site must be unique"
    assert source.count(element.definitions_anchor) == 1, "definitions anchor must be unique"
    site = source.index(element.call_site)
    indent = _indent_of(source, site)
    call = "".join(indent + line + "\n" for line in element.call.splitlines())
    source = source.replace(element.call_site, element.call_site + call)
    anchor = source.index(element.definitions_anchor)
    return source[:anchor] + element.definitions.lstrip("\n") + "\n" + source[anchor:]


def stage_kernel(base: Path, out: Path, element_names: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((base / "kernel-metadata.json").read_text())
    notebook = json.loads((base / meta["code_file"]).read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 1, "expected a single-code-cell kernel"
    source = "".join(code_cells[0]["source"])
    for name in element_names:
        source = transplant(source, ELEMENTS[name])
    code_cells[0]["source"] = source.splitlines(keepends=True)
    name = out.name
    meta.update(id=f"{OWNER}/{name}", title=name, code_file=f"{name}.ipynb")
    (out / meta["code_file"]).write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--element", action="append", choices=sorted(ELEMENTS), required=True)
    parser.add_argument("--base-kernel", type=Path, required=True)
    parser.add_argument("--out-kernel", type=Path, required=True)
    args = parser.parse_args()
    stage_kernel(args.base_kernel, args.out_kernel, args.element)
    log.info("transplanted %s -> %s", ",".join(args.element), args.out_kernel)


if __name__ == "__main__":
    main()
