"""Decompose GT edges the tracker missed into REACHABLE (a candidate the ILP declined) vs OUT-OF-GATE.

`validator_results.csv` says 59 % of missing edges are "fragmented" — both endpoints detected and matched,
no predicted link. That alone does not say the ILP could have made the link: an edge whose endpoints sit
further apart than the candidate gate was never on the solver's table, so no cost knob can recover it.
This walks the cached validator predictions against the GT graphs and reports the displacement
distribution of the fragmented edges against the gate:

    .venv/Scripts/python kaggle/fragment_audit.py --cache runs/local_kernel/_prediction_cache/<hash>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
SPACING_UM = np.array([1.625, 0.40625, 0.40625])
GATE_UM = 10.0
MATCH_UM = 5.0


def train_dir() -> Path:
    config = yaml.safe_load((REPO / "paths.yaml").read_text())
    return Path(config["data"]) / "raw" / "biohub_cell_tracking" / "train"


def read_graph(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node ids, node positions in µm (t, z, y, x) and the edge id pairs."""
    store = zarr.open(str(path), mode="r")
    ids = np.asarray(store["nodes/ids"])
    axes = [np.asarray(store[f"nodes/props/{axis}/values"]) for axis in ("t", "z", "y", "x")]
    positions = np.stack(axes, axis=1).astype(np.float64)
    positions[:, 1:] *= SPACING_UM
    edges = np.asarray(store["edges/ids"]).reshape(-1, 2)
    return ids, positions, edges


def match_nodes(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Nearest predicted node per GT node within MATCH_UM, frame by frame; -1 where unmatched."""
    matched = np.full(len(truth), -1, dtype=np.int64)
    for frame in np.unique(truth[:, 0]):
        gt_rows = np.flatnonzero(truth[:, 0] == frame)
        pred_rows = np.flatnonzero(predicted[:, 0] == frame)
        if len(pred_rows) == 0:
            continue
        distance, nearest = cKDTree(predicted[pred_rows, 1:]).query(truth[gt_rows, 1:])
        within = distance <= MATCH_UM
        matched[gt_rows[within]] = pred_rows[nearest[within]]
    return matched


def audit_video(truth_path: Path, predicted_path: Path) -> dict:
    gt_ids, gt_pos, gt_edges = read_graph(truth_path)
    _, pred_pos, pred_edges = read_graph(predicted_path)
    index = {node_id: row for row, node_id in enumerate(gt_ids)}
    matched = match_nodes(gt_pos, pred_pos)

    pred_index = {
        node_id: row for row, node_id in enumerate(np.asarray(zarr.open(str(predicted_path), mode="r")["nodes/ids"]))
    }
    linked = {(pred_index[a], pred_index[b]) for a, b in pred_edges if a in pred_index and b in pred_index}

    outgoing, incoming = _orient(pred_pos, pred_edges, pred_index)

    fragmented, present, verdicts = [], [], []
    for source, target in gt_edges:
        rows = (index.get(source, -1), index.get(target, -1))
        if -1 in rows:
            continue
        step = float(np.linalg.norm(gt_pos[rows[0], 1:] - gt_pos[rows[1], 1:]))
        if -1 in (matched[rows[0]], matched[rows[1]]):
            continue
        pair = (matched[rows[0]], matched[rows[1]])
        if pair in linked or pair[::-1] in linked:
            present.append(step)
            continue
        fragmented.append(step)
        if step <= GATE_UM:
            verdicts.append(_verdict(pair, outgoing, incoming))
    return {"fragmented": fragmented, "present": present, "verdicts": verdicts}


def _orient(pred_pos: np.ndarray, pred_edges: np.ndarray, pred_index: dict) -> tuple[dict, dict]:
    """Predicted edges split into successor / predecessor maps over row indices, ordered by time."""
    outgoing: dict[int, list[int]] = {}
    incoming: dict[int, list[int]] = {}
    for a, b in pred_edges:
        if a not in pred_index or b not in pred_index:
            continue
        first, second = pred_index[a], pred_index[b]
        if pred_pos[first, 0] > pred_pos[second, 0]:
            first, second = second, first
        outgoing.setdefault(first, []).append(second)
        incoming.setdefault(second, []).append(first)
    return outgoing, incoming


def _verdict(pair: tuple[int, int], outgoing: dict, incoming: dict) -> str:
    """Why the solver declined an in-gate GT edge, read off the predicted graph alone.

    `disappearance` means the source ended its track with nothing to pay for instead, so the
    disappearance cost is the binding knob. `source_stole` / `target_taken` mean another node won the
    slot, which is a competition failure no disappearance weight can repair.
    """
    source, target = pair
    took = outgoing.get(source, [])
    claimed = incoming.get(target, [])
    if not took and not claimed:
        return "disappearance"
    if took and claimed:
        return "both_reassigned"
    return "source_stole" if took else "target_taken"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="a prediction-cache dir holding validator geffs")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    source = train_dir()
    fragmented: list[float] = []
    present: list[float] = []
    verdicts: list[str] = []
    for predicted in sorted(args.cache.glob("*.geff"))[: args.limit]:
        truth = source / predicted.name
        if not truth.exists():
            continue
        result = audit_video(truth, predicted)
        fragmented += result["fragmented"]
        present += result["present"]
        verdicts += result["verdicts"]

    steps = np.array(fragmented)
    report = {
        "matched_gt_edges": len(fragmented) + len(present),
        "fragmented": len(fragmented),
        "linked": len(present),
        "fragmented_step_um": {
            "median": float(np.median(steps)) if len(steps) else None,
            "p90": float(np.percentile(steps, 90)) if len(steps) else None,
            "max": float(steps.max()) if len(steps) else None,
            "within_gate": int((steps <= GATE_UM).sum()),
        },
        "linked_median_step_um": float(np.median(present)) if present else None,
        "in_gate_verdicts": {name: verdicts.count(name) for name in sorted(set(verdicts))},
    }
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
