"""Generate evaluator test cases: integer voxel coordinates, anisotropic spacing."""

import json
from pathlib import Path

import numpy as np

SPACING = [1.625, 0.40625, 0.40625]
MAX_DISTANCE = 7.0
OUT = Path(__file__).parents[2] / "tests" / "assets" / "metric_cases.json"


def chain(rng, start_id, n_t, origin, drift):
    nodes, edges = [], []
    pos = np.array(origin, dtype=float)
    for t in range(n_t):
        nodes.append([start_id + t, t, *np.round(pos).astype(int).tolist()])
        pos = pos + drift + rng.normal(0, 2.0, 3)
        if t:
            edges.append([start_id + t - 1, start_id + t])
    return nodes, edges


def random_case(rng, name, n_tracks_gt, n_tracks_pred, n_t=6, jitter=0.0, extent=60):
    gt_nodes, gt_edges, pred_nodes, pred_edges = [], [], [], []
    uid = 0
    origins = [rng.integers(0, extent, 3).astype(float) for _ in range(max(n_tracks_gt, n_tracks_pred))]
    drifts = [rng.normal(0, 1.5, 3) for _ in origins]
    for i in range(n_tracks_gt):
        n, e = chain(rng, uid, n_t, origins[i], drifts[i])
        gt_nodes += n
        gt_edges += e
        uid += n_t
    for i in range(n_tracks_pred):
        shifted = np.array(origins[i]) + rng.normal(0, jitter, 3)
        n, e = chain(rng, uid, n_t, shifted, drifts[i])
        pred_nodes += n
        pred_edges += e
        uid += n_t
    return dict(name=name, gt=dict(nodes=gt_nodes, edges=gt_edges), pred=dict(nodes=pred_nodes, edges=pred_edges))


def structural_cases():
    gt_nodes = [[0, 0, 10, 10, 10], [1, 1, 10, 10, 10], [2, 2, 10, 20, 10], [3, 2, 10, 0, 10]]
    gt_edges = [[0, 1], [1, 2], [1, 3]]
    gt = dict(nodes=gt_nodes, edges=gt_edges)
    perfect_pred = dict(nodes=[[n[0] + 100, *n[1:]] for n in gt_nodes], edges=[[s + 100, t + 100] for s, t in gt_edges])
    cases = [
        dict(name="perfect", gt=gt, pred=perfect_pred),
        dict(name="empty_pred", gt=gt, pred=dict(nodes=[], edges=[])),
        dict(
            name="duplicate_edges",
            gt=gt,
            pred=dict(nodes=perfect_pred["nodes"], edges=perfect_pred["edges"] + perfect_pred["edges"]),
        ),
        dict(
            name="backward_and_skip",
            gt=gt,
            pred=dict(nodes=perfect_pred["nodes"], edges=perfect_pred["edges"] + [[101, 100], [100, 102]]),
        ),
        dict(
            name="over_out_degree",
            gt=gt,
            pred=dict(
                nodes=[*perfect_pred["nodes"], [104, 2, 10, 40, 10], [105, 2, 10, 60, 10]],
                edges=[*perfect_pred["edges"], [101, 104], [101, 105]],
            ),
        ),
        dict(
            name="merge_onto_one_gt_edge",
            gt=gt,
            pred=dict(
                nodes=[*perfect_pred["nodes"], [106, 0, 11, 10, 10], [107, 1, 11, 10, 10]],
                edges=[*perfect_pred["edges"], [106, 107]],
            ),
        ),
        dict(
            name="free_edges_in_unannotated_tissue",
            gt=gt,
            pred=dict(
                nodes=[*perfect_pred["nodes"], [108, 0, 200, 200, 200], [109, 1, 200, 200, 200]],
                edges=[*perfect_pred["edges"], [108, 109]],
            ),
        ),
        dict(name="no_gt_edges", gt=dict(nodes=gt_nodes, edges=[]), pred=perfect_pred),
    ]
    return cases


def lineage(rng, uid, n_t, origin, drift, divide_at):
    """A track that forks at `divide_at`, both daughters continuing to the end."""
    nodes, edges = [], []
    tips = [(uid, np.array(origin, dtype=float))]
    nodes.append([uid, 0, *np.round(tips[0][1]).astype(int).tolist()])
    uid += 1
    for t in range(1, n_t):
        grown = []
        for parent, pos in tips:
            splits = 2 if t == divide_at else 1
            for k in range(splits):
                offset = (
                    drift + rng.normal(0, 2.0, 3) + (np.array([0, 8.0, 0]) * (1 if k else -1) if splits == 2 else 0)
                )
                child_pos = pos + offset
                nodes.append([uid, t, *np.round(child_pos).astype(int).tolist()])
                edges.append([parent, uid])
                grown.append((uid, child_pos))
                uid += 1
        tips = grown
    return nodes, edges, uid


def dividing_case(rng, name, n_gt, n_pred, n_t, jitter, extent):
    gt_nodes, gt_edges, pred_nodes, pred_edges = [], [], [], []
    uid = 0
    origins = [rng.integers(0, extent, 3).astype(float) for _ in range(max(n_gt, n_pred))]
    drifts = [rng.normal(0, 1.0, 3) for _ in origins]
    splits = [int(rng.integers(1, max(2, n_t - 1))) for _ in origins]
    for i in range(n_gt):
        n, e, uid = lineage(rng, uid, n_t, origins[i], drifts[i], splits[i])
        gt_nodes += n
        gt_edges += e
    for i in range(n_pred):
        shifted = np.array(origins[i]) + rng.normal(0, jitter, 3)
        # Sometimes shift the fork by a timepoint, which the window rules are meant to forgive.
        split = splits[i] + int(rng.integers(-1, 2))
        n, e, uid = lineage(rng, uid, n_t, shifted, drifts[i], split)
        pred_nodes += n
        pred_edges += e
    return dict(name=name, gt=dict(nodes=gt_nodes, edges=gt_edges), pred=dict(nodes=pred_nodes, edges=pred_edges))


def main():
    rng = np.random.default_rng(20260723)
    cases = structural_cases()
    for i, (ngt, npred, jit) in enumerate(
        [
            (3, 3, 0.0),
            (3, 3, 4.0),
            (5, 5, 12.0),
            (8, 4, 6.0),
            (4, 8, 6.0),
            (10, 10, 20.0),
            (2, 6, 3.0),
            (6, 2, 3.0),
            (12, 12, 8.0),
            (1, 1, 30.0),
        ]
    ):
        cases.append(random_case(rng, f"random_{i}", ngt, npred, jitter=jit))
    # Crowded frames: every node within the 7 um cutoff of every other, so the assignment is
    # maximally ambiguous and the 1/(1+d) objective can disagree with nearest-neighbour.
    for i in range(60):
        ngt = int(rng.integers(2, 14))
        npred = int(rng.integers(2, 14))
        extent = int(rng.integers(3, 14))
        cases.append(
            random_case(
                rng,
                f"crowded_{i}",
                ngt,
                npred,
                n_t=int(rng.integers(2, 7)),
                jitter=float(rng.uniform(0, 6)),
                extent=extent,
            )
        )
    for i in range(80):
        cases.append(
            dividing_case(
                rng,
                f"dividing_{i}",
                n_gt=int(rng.integers(1, 5)),
                n_pred=int(rng.integers(1, 5)),
                n_t=int(rng.integers(3, 7)),
                jitter=float(rng.uniform(0, 8)),
                extent=int(rng.integers(4, 40)),
            )
        )
    payload = dict(spacing=SPACING, max_distance=MAX_DISTANCE, cases=cases)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
