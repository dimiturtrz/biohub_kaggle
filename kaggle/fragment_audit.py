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
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
SPACING_UM = np.array([1.625, 0.40625, 0.40625])
GATE_UM = 10.0
MATCH_UM = 7.0  # the official DistanceMatcher cap (core/metrics/matching.py)
_FORBIDDEN = 1e6
DUPLICATE_UM = 1.6  # one (1, 4, 4) voxel in y/x: closer than the decode grid can separate
DISTINCT_UM = 3.0  # above the 2.87 µm GT nearest-neighbour floor, so a genuinely different cell
CONFIDENT_UM = 2.87  # the GT nearest-neighbour floor: inside it, a match cannot be a neighbouring cell


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
    """One predicted node per GT node, frame by frame; -1 where unmatched.

    Mirrors `core.metrics.matching.DistanceMatcher`: an OPTIMAL per-frame assignment under a µm cap,
    not a nearest-neighbour lookup. Greedy matching lets several GT nodes claim the same prediction,
    which invents fragmented edges wherever the detector merged two cells.
    """
    matched = np.full(len(truth), -1, dtype=np.int64)
    for frame in np.unique(truth[:, 0]):
        gt_rows = np.flatnonzero(truth[:, 0] == frame)
        pred_rows = np.flatnonzero(predicted[:, 0] == frame)
        if len(pred_rows) == 0:
            continue
        cost = np.linalg.norm(truth[gt_rows, None, 1:] - predicted[None, pred_rows, 1:], axis=2)
        cost[cost > MATCH_UM] = _FORBIDDEN
        gt_local, pred_local = linear_sum_assignment(cost)
        keep = cost[gt_local, pred_local] < _FORBIDDEN
        matched[gt_rows[gt_local[keep]]] = pred_rows[pred_local[keep]]
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

    fragmented, present, verdicts, gaps, contests = [], [], [], [], []
    residuals: list[float] = []
    linked_residuals: list[float] = []
    nearest: list[float] = []
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
            linked_residuals += [float(np.linalg.norm(gt_pos[row, 1:] - pred_pos[matched[row], 1:])) for row in rows]
            continue
        fragmented.append(step)
        if step <= GATE_UM:
            endpoint_residuals = [float(np.linalg.norm(gt_pos[row, 1:] - pred_pos[matched[row], 1:])) for row in rows]
            residuals += endpoint_residuals
            confident = max(endpoint_residuals) < CONFIDENT_UM
            if not confident:
                nearest += [
                    _nearest_prediction(gt_pos[row], pred_pos)
                    for row, residual in zip(rows, endpoint_residuals, strict=True)
                    if residual >= CONFIDENT_UM
                ]
            verdicts.append((_verdict(pair, outgoing, incoming), confident))
            gap = _rival_gap(pair, outgoing, incoming, pred_pos)
            if gap is not None:
                gaps.append(gap)
            contest = _step_contest(pair, outgoing, incoming, pred_pos)
            if contest is not None and confident:
                contests.append(contest)
    return {
        "fragmented": fragmented,
        "present": present,
        "verdicts": verdicts,
        "gaps": gaps,
        "contests": contests,
        "residuals": residuals,
        "linked_residuals": linked_residuals,
        "nearest": nearest,
    }


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


def _rival_gap(pair: tuple[int, int], outgoing: dict, incoming: dict, pred_pos: np.ndarray) -> float | None:
    """How far the node that WON the slot sits from the node that should have had it, in µm.

    A gap under roughly one voxel diagonal means the winner is a duplicate detection of the true cell,
    so the loss is a decode/NMS failure the association head never had a chance to avoid. A gap of
    several µm means a genuinely different cell won, which is the confusor axis.
    """
    source, target = pair
    rivals = [pred_pos[node, 1:] for node in outgoing.get(source, [])]
    rivals += [pred_pos[node, 1:] for node in incoming.get(target, [])]
    if not rivals:
        return None
    reference = pred_pos[target, 1:]
    return float(min(np.linalg.norm(rival - reference) for rival in rivals))


def _step_contest(
    pair: tuple[int, int], outgoing: dict, incoming: dict, pred_pos: np.ndarray
) -> tuple[float, float] | None:
    """The declined true edge's step vs the step the edge that WON the slot actually took, in µm.

    Prices the ILP distance prior directly: if the winner is systematically SHORTER, the solver is
    over-trusting displacement and a distance reweight is a live knob. If the winner is longer or the
    same, distance does not discriminate these cases and only edge affinity can move them.
    """
    source, target = pair
    rivals = [float(np.linalg.norm(pred_pos[source, 1:] - pred_pos[node, 1:])) for node in outgoing.get(source, [])]
    rivals += [float(np.linalg.norm(pred_pos[node, 1:] - pred_pos[target, 1:])) for node in incoming.get(target, [])]
    if not rivals:
        return None
    return float(np.linalg.norm(pred_pos[source, 1:] - pred_pos[target, 1:])), min(rivals)


def _contest_report(contests: list[tuple[float, float]]) -> dict:
    """Does the winning edge undercut the true edge on displacement, and by how much?"""
    if not contests:
        return {}
    true_step = np.array([entry[0] for entry in contests])
    winner_step = np.array([entry[1] for entry in contests])
    return {
        "n": len(contests),
        "true_median_um": round(float(np.median(true_step)), 3),
        "winner_median_um": round(float(np.median(winner_step)), 3),
        "winner_shorter_frac": round(float((winner_step < true_step).mean()), 4),
        "winner_shorter_by_over_1um": int((true_step - winner_step > 1.0).sum()),
        "winner_longer_by_over_1um": int((winner_step - true_step > 1.0).sum()),
    }


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


def _nearest_prediction(gt_node: np.ndarray, pred_pos: np.ndarray) -> float:
    """Distance to the closest prediction in the node's own frame, ignoring who the assignment gave it to.

    Separates a MISSING detection (nothing near the GT node at all — a recall loss) from a CONTESTED one
    (a prediction is right there, but the per-frame assignment handed it to a different GT node).
    """
    same_frame = pred_pos[pred_pos[:, 0] == gt_node[0]]
    if not len(same_frame):
        return float("inf")
    return float(np.linalg.norm(same_frame[:, 1:] - gt_node[1:], axis=1).min())


def _verdict_report(verdicts: list[tuple[str, bool]]) -> dict:
    """Verdict counts, plus the total, so a confident-match subset reads against its own denominator."""
    names = [name for name, _ in verdicts]
    return {"total": len(names), **{name: names.count(name) for name in sorted(set(names))}}


def _gap_report(gaps: np.ndarray) -> dict:
    """Split the slot-winners into duplicate detections of the true cell and genuinely other cells."""
    if not len(gaps):
        return {}
    return {
        "n": len(gaps),
        "median": round(float(np.median(gaps)), 3),
        "duplicate_under_1_6um": int((gaps <= DUPLICATE_UM).sum()),
        "distinct_over_3um": int((gaps > DISTINCT_UM).sum()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="a prediction-cache dir holding validator geffs")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    source = train_dir()
    fragmented: list[float] = []
    present: list[float] = []
    verdicts: list[tuple[str, bool]] = []
    gaps: list[float] = []
    contests: list[tuple[float, float]] = []
    residuals: list[float] = []
    linked_residuals: list[float] = []
    nearest: list[float] = []
    for predicted in sorted(args.cache.glob("*.geff"))[: args.limit]:
        truth = source / predicted.name
        if not truth.exists():
            continue
        result = audit_video(truth, predicted)
        fragmented += result["fragmented"]
        present += result["present"]
        verdicts += result["verdicts"]
        gaps += result["gaps"]
        contests += result["contests"]
        residuals += result["residuals"]
        linked_residuals += result["linked_residuals"]
        nearest += result["nearest"]

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
        "in_gate_verdicts": _verdict_report(verdicts),
        "confident_match_verdicts": _verdict_report([entry for entry in verdicts if entry[1]]),
        "rival_gap_um": _gap_report(np.array(gaps)),
        "step_contest_um": _contest_report(contests),
        "loose_endpoint_nearest_um": {
            "n": len(nearest),
            "median": round(float(np.median(nearest)), 3) if nearest else None,
            "has_prediction_inside_floor": int((np.array(nearest) < CONFIDENT_UM).sum()) if nearest else 0,
        },
        "match_residual_um": {
            "fragmented_median": round(float(np.median(residuals)), 3) if residuals else None,
            "fragmented_p90": round(float(np.percentile(residuals, 90)), 3) if residuals else None,
            "linked_median": round(float(np.median(linked_residuals)), 3) if linked_residuals else None,
            "linked_p90": round(float(np.percentile(linked_residuals, 90)), 3) if linked_residuals else None,
        },
    }
    log.info(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
