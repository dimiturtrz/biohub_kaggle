"""Fork a public kernel with an additive cytokinesis-rescue post-link cell.

The donor element is step 2 of the V50 post-link layer carried (and discarded) by
`celltrack-public-v1329f`: a physically gated cytokinesis detector that proposes a second
daughter for a linked parent when seven independent constraints hold. The donor ran it
*after* linearizing every fork, which destroys the base tracker's own divisions -- that
packaged surgery scored 0.907 against the 0.939 foundation. Here only the additive half is
transplanted: existing edges, including existing forks, are never touched.

`min_nodes` is the donor's density precondition, fitted to the donor's films. It is not one
of the physical gates; on the 0947 output it admits one movie of four (70290 nodes) and
yields 13 proposals, against 43 with the precondition dropped.

The rescue reads and rewrites `/kaggle/working/submission.csv`, so it composes with any base
kernel that writes one: `--base` names the base kernel directory under `kaggle/kernels/`.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KERNELS = REPO / "kaggle" / "kernels"

RESCUE_CELL = """
# ==============================================================================
# Additive cytokinesis rescue (donor element: V50 post-link layer, step 2 only).
# Existing edges -- including the base tracker's own forks -- are preserved verbatim.
# ==============================================================================
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

MIN_NODES = {min_nodes}

SCALE_ZYX_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float64)
MAX_PARENT_CHILD_UM = 8.5
MIN_SISTER_UM = 8.5
MAX_SISTER_UM = 13.5
BORDER_LO_PX = 20.0
BORDER_HI_PX = 236.0
MIN_DAUGHTER_TRACK = 12
MIN_DIVERGENCE_UM = 1.20
MIN_BIPOLAR_DEG = 140.0

_sub_path = Path('/kaggle/working/submission.csv')
_frame = pd.read_csv(_sub_path)
_added_total = 0
_out_parts = []

for _ds in sorted(_frame['dataset'].unique()):
    _part = _frame[_frame['dataset'] == _ds]
    _nodes = _part[_part['row_type'] == 'node']
    _edges = _part[_part['row_type'] == 'edge']
    _coords = {{
        int(r.node_id): (int(r.t), float(r.z), float(r.y), float(r.x))
        for r in _nodes.itertuples()
    }}
    _out = defaultdict(list)
    _in_deg = defaultdict(int)
    for _r in _edges.itertuples():
        _out[int(_r.source_id)].append(int(_r.target_id))
        _in_deg[int(_r.target_id)] += 1

    def _pos(nid):
        _c = _coords[nid]
        return np.asarray((_c[1], _c[2], _c[3]), dtype=np.float64) * SCALE_ZYX_UM

    def _dist(a, b):
        return float(np.linalg.norm(_pos(a) - _pos(b)))

    def _track_len(nid):
        _cur, _len, _seen = nid, 1, set()
        while _cur in _out and len(_out[_cur]) == 1 and _cur not in _seen:
            _seen.add(_cur)
            _cur = _out[_cur][0]
            _len += 1
        return _len

    _rescued = []
    if len(_coords) >= MIN_NODES:
        _times = defaultdict(list)
        for _nid, _c in _coords.items():
            _times[_c[0]].append(_nid)

        for _t in sorted(_times):
            _child_ids = _times.get(_t + 1, [])
            if not _child_ids:
                continue
            _srcs = [n for n in _times[_t] if len(_out.get(n, [])) == 1]
            _cands = [n for n in _child_ids if _in_deg[n] == 0]
            if not _srcs or not _cands:
                continue
            _tree = cKDTree(np.stack([_pos(c) for c in _cands]))
            for _s in _srcs:
                _c1 = _out[_s][0]
                if _dist(_s, _c1) > MAX_PARENT_CHILD_UM:
                    continue
                _c2 = _cands[int(_tree.query(_pos(_c1))[1])]
                if _dist(_s, _c2) > MAX_PARENT_CHILD_UM:
                    continue
                _sister = _dist(_c1, _c2)
                if _sister > MAX_SISTER_UM or _sister < MIN_SISTER_UM:
                    continue
                _trio = (_coords[_s], _coords[_c1], _coords[_c2])
                if min(c[3] for c in _trio) < BORDER_LO_PX or max(c[3] for c in _trio) > BORDER_HI_PX:
                    continue
                if min(c[2] for c in _trio) < BORDER_LO_PX or max(c[2] for c in _trio) > BORDER_HI_PX:
                    continue
                if min(_track_len(_c1), _track_len(_c2)) < MIN_DAUGHTER_TRACK:
                    continue
                _s1, _s2 = _out.get(_c1, []), _out.get(_c2, [])
                if len(_s1) != 1 or len(_s2) != 1:
                    continue
                if _dist(_s1[0], _s2[0]) - _sister < MIN_DIVERGENCE_UM:
                    continue
                _v1, _v2 = _pos(_c1) - _pos(_s), _pos(_c2) - _pos(_s)
                _cos = float(np.dot(_v1, _v2) / (np.linalg.norm(_v1) * np.linalg.norm(_v2) + 1e-6))
                if float(np.degrees(np.arccos(np.clip(_cos, -1.0, 1.0)))) < MIN_BIPOLAR_DEG:
                    continue
                _rescued.append((_s, _c2))
                _in_deg[_c2] += 1
                _out[_s].append(_c2)

    _added_total += len(_rescued)
    print(f'{{_ds}}: nodes={{len(_coords)}} rescued={{len(_rescued)}}')
    _out_parts.append(_part)
    if _rescued:
        _tmpl = _edges.iloc[0]
        _new = pd.DataFrame([
            {{**{{c: _tmpl[c] for c in _frame.columns}}, 'dataset': _ds, 'row_type': 'edge',
              'source_id': _s, 'target_id': _c2}}
            for _s, _c2 in _rescued
        ])
        _out_parts.append(_new)

_final = pd.concat(_out_parts, ignore_index=True)
_final['id'] = np.arange(len(_final))
_final.to_csv(_sub_path, index=False)
print(f'CYTOKINESIS RESCUE: added {{_added_total}} edges (MIN_NODES={{MIN_NODES}})')
"""


def build(base: str, tag: str, min_nodes: int) -> Path:
    base_dir = KERNELS / base
    metadata = json.loads((base_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    notebook = json.loads((base_dir / metadata["code_file"]).read_text(encoding="utf-8"))

    forked = copy.deepcopy(notebook)
    forked["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": RESCUE_CELL.format(min_nodes=min_nodes).splitlines(keepends=True),
        }
    )

    slug = f"{base}-{tag}"
    metadata["id"] = f"dimiturnt/{slug}"
    metadata["title"] = slug
    metadata["code_file"] = f"{slug}.ipynb"

    out_dir = KERNELS / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{slug}.ipynb").write_text(json.dumps(forked), encoding="utf-8")
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="celltrack-public-0947", help="base kernel directory under kaggle/kernels")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--min-nodes", type=int, required=True)
    args = parser.parse_args()
    print(build(args.base, args.tag, args.min_nodes))


if __name__ == "__main__":
    main()
