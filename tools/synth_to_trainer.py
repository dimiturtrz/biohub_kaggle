"""Convert CC0 synthetic sequences (pooled npz) into the zarr+geff layout the frontier trainer reads.

Synthetic volumes ship XY-pooled by 4; the trainer pools raw frames by ``--downsample 1,4,4``, so the frames
are XY-repeated back to native shape and pool back to exactly the stored voxels.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
import zarr

XY_POOL = 4
NATIVE_SCALE = (1.0, 1.625, 0.40625, 0.40625)
QUANTILES = (0.0, 0.001, 0.01, 0.1, 0.9, 0.99, 0.999, 1.0)


def _write_image(volumes: np.ndarray, out: Path) -> None:
    frames = volumes.repeat(XY_POOL, axis=2).repeat(XY_POOL, axis=3)
    group = zarr.open_group(out, mode="w")
    group.attrs["multiscales"] = [
        {
            "version": "0.5",
            "axes": [
                {"name": a, "type": k, "unit": u}
                for a, k, u in zip("TZYX", ["time", *["space"] * 3], ["second", *["micrometer"] * 3], strict=True)
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": list(NATIVE_SCALE)}]}],
            "name": "0",
        }
    ]
    values = np.quantile(volumes, QUANTILES)
    group.attrs["image_statistics"] = {"quantiles": {str(q): float(v) for q, v in zip(QUANTILES, values, strict=True)}}
    array = group.create_array("0", shape=frames.shape, dtype=frames.dtype, chunks=(1, *frames.shape[1:]))
    array[:] = frames


def _write_tracks(nodes: np.ndarray, edges: np.ndarray, out: Path) -> None:
    graph = td.graph.IndexedRXGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Int64, 0)
    coords = np.rint(nodes[:, :4]).astype(np.int64)
    ids = graph.bulk_add_nodes([{"t": int(t), "z": int(z), "y": int(y), "x": int(x)} for t, z, y, x in coords])
    graph.bulk_add_edges([{"source_id": ids[s], "target_id": ids[d]} for s, d in edges])
    graph.to_geff(out)


def convert(src: Path, dst_dir: Path) -> str:
    stem = f"synth_{src.stem}"
    payload = np.load(src)
    _write_image(payload["volumes"], dst_dir / f"{stem}.zarr")
    _write_tracks(payload["nodes"], payload["edges"], dst_dir / f"{stem}.geff")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="biohub_synthetic/sequences directory")
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.src.glob("seq_*.npz"))[: args.count]
    with ProcessPoolExecutor(args.workers) as pool:
        stems = list(pool.map(convert, sources, [args.dst] * len(sources)))
    names = stems
    split = {"train": names[args.val_count :], "test": names[: args.val_count]}
    (args.dst / "splits.json").write_text(json.dumps([split]))


if __name__ == "__main__":
    main()
