"""Can a rival's edge DECISIONS be moved onto the champion's own nodes, importing no nodes at all?

E56 measured 39 ground-truth edges the v1329 family recovers and the champion breaks, and showed that the
33 selection failures among them connect nodes the champion ALREADY HAS. That makes a node-free transfer
arithmetically possible -- but E56 proved it through the ground truth, which exists for four movies and
not for the submission at large. This asks the same question the way a submission would have to: align
rival nodes to champion nodes by distance alone, no ground truth anywhere, and count what survives.

An edge transfers only if BOTH its endpoints land on a champion node. Then it is either one the champion
already drew (redundant), or a new one -- and a new one is CLEAN when the champion has left both slots
free, or CONTESTED when the champion already spends that slot on a different partner. Only the clean ones
are a free addition; the contested ones need the discriminator the champion is missing.

    .venv/Scripts/python kaggle/edge_transfer.py --champion <a.csv> --rival <b.csv>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from error_budget import match_bipartite
from fragment_selection_gate import OFFICIAL_UM
from short_component_prune import read_submission

log = logging.getLogger(__name__)

CELL_SEPARATION_UM = 2.87


def align_nodes(champion_table, rival_table, ruler_um):
    """Which rival node IS which champion node -- the scorer's own per-frame matching, no ground truth."""
    champion_order = list(champion_table)
    rival_order = list(rival_table)
    champion_points = np.array([champion_table[node] for node in champion_order], dtype=float)
    rival_points = np.array([rival_table[node] for node in rival_order], dtype=float)
    rival_to_champion, _ = match_bipartite(rival_points, champion_points, ruler_um)
    return {rival_order[row]: champion_order[column] for row, column in rival_to_champion.items()}


def slots(edges):
    """Which node already spends its outgoing slot, and which already has a predecessor."""
    outgoing, incoming = {}, {}
    for source, target in edges:
        outgoing.setdefault(source, set()).add(target)
        incoming.setdefault(target, set()).add(source)
    return outgoing, incoming


def transfer(champion_table, champion_edges, rival_table, rival_edges, ruler_um):
    """Partition every rival edge by whether it can move onto champion nodes, and at what cost."""
    onto = align_nodes(champion_table, rival_table, ruler_um)
    champion_set = set(champion_edges)
    outgoing, incoming = slots(champion_edges)
    tally = dict.fromkeys(("unaligned", "redundant", "clean", "contested"), 0)
    clean_edges = []
    for source, target in rival_edges:
        mapped_source, mapped_target = onto.get(source), onto.get(target)
        if mapped_source is None or mapped_target is None:
            tally["unaligned"] += 1
        elif (mapped_source, mapped_target) in champion_set:
            tally["redundant"] += 1
        elif outgoing.get(mapped_source) or incoming.get(mapped_target):
            tally["contested"] += 1
        else:
            tally["clean"] += 1
            clean_edges.append((mapped_source, mapped_target))
    return tally, clean_edges, len(onto)


def report(dataset, tally, aligned, rival_nodes, champion_nodes, rival_edges):
    log.info(
        "%-18s nodes %6d/%6d aligned %5.1f%% | champion nodes %6d | rival edges %6d",
        dataset,
        aligned,
        rival_nodes,
        100 * aligned / max(rival_nodes, 1),
        champion_nodes,
        rival_edges,
    )
    log.info(
        "  unaligned %5d | redundant %6d (%4.1f%%) | CLEAN %5d | contested %5d",
        tally["unaligned"],
        tally["redundant"],
        100 * tally["redundant"] / max(rival_edges, 1),
        tally["clean"],
        tally["contested"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True, help="the submission whose nodes are kept")
    parser.add_argument("--rival", type=Path, required=True, help="the submission whose edges are borrowed")
    parser.add_argument("--ruler", type=float, default=CELL_SEPARATION_UM, help="node alignment radius in um")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    champion_nodes, champion_edges = read_submission(args.champion)
    rival_nodes, rival_edges = read_submission(args.rival)
    log.info("ruler %.2f um (official node match is %.1f um)", args.ruler, OFFICIAL_UM)

    totals = dict.fromkeys(("unaligned", "redundant", "clean", "contested"), 0)
    for dataset in sorted(set(champion_nodes) & set(rival_nodes)):
        tally, _, aligned = transfer(
            champion_nodes[dataset],
            champion_edges[dataset],
            rival_nodes[dataset],
            rival_edges[dataset],
            args.ruler,
        )
        report(
            dataset,
            tally,
            aligned,
            len(rival_nodes[dataset]),
            len(champion_nodes[dataset]),
            len(rival_edges[dataset]),
        )
        for key, value in tally.items():
            totals[key] += value

    log.info("=== over %d shared datasets", len(set(champion_nodes) & set(rival_nodes)))
    for key, value in totals.items():
        log.info("  %-10s %7d", key, value)
    log.info("CLEAN edges are additions that import ZERO nodes; contested ones need the missing discriminator.")


if __name__ == "__main__":
    main()
