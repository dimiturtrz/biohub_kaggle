"""Patch a donor kernel's predict script for local re-runs: cache per-video graphs, keep the GPU fed.

The kernel re-materializes its tracking repo on every start and the predict script wipes its output dir, so a
re-run that only changes post-processing would redo all GPU prediction. `apply` rewrites the per-video loop so
that it copies a cached `<stem>.geff` when one exists under LOCAL_PREDICTION_CACHE, keyed by everything the
prediction reads: its CLI args (bar output naming), the BIOHUB_* env vars the script names, and a fingerprint of
each weights file.

The rest removes CPU work that serialized with the GPU (profiled: ILP solve, zarr frame reads and a Python
double loop over the edge-probability matrix left the GPU idle between encode bursts), with identical outputs:
video k's graph build + ILP solve + save run on a worker thread while the GPU predicts video k+1, the next
frames are read ahead on a thread pool, and edge candidates are thresholded + ordered with numpy. Each video
logs PREDICT_TIMING (GPU-side and CPU-side wall seconds).
"""

import os
from pathlib import Path

PREDICT_SCRIPT = Path("scripts") / "predict_unet_transformer.py"
LOOP_HEAD = (
    '    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):\n        ds_path = data_dir / name\n'
)
POST_HEAD = "        graph = build_graph(coords, edges)\n"
SAVE = '        save_graph(graph, output_dir / f"{name}.geff")\n'
PREDICT_DEF = "def predict(\n"

CACHE_HELPER = """
import concurrent.futures as _futures
import time as _time


def _local_prediction_cache(name):
    import hashlib, re
    source = Path(__file__).read_text(encoding="utf-8")
    named = set(re.findall(r"BIOHUB_[A-Z0-9_]+", source))
    argv, skip = [], False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg in ("--method", "--slice"):
            skip = True
            continue
        argv.append(arg)
    weights = [Path(a) for a in argv if a.endswith((".pt", ".pth", ".ckpt"))]
    weights += [Path(os.environ[k]) for k in named if k.endswith("WEIGHTS") and os.environ.get(k)]
    fingerprints = []
    for path in sorted(set(weights)):
        with path.open("rb") as handle:
            fingerprints.append([path.name, path.stat().st_size, hashlib.sha1(handle.read(1 << 20)).hexdigest()])
    key = [argv, sorted((k, os.environ[k]) for k in named if k in os.environ), fingerprints]
    key = json.dumps(key + [_LOCAL_VARIANT] if _LOCAL_VARIANT else key)
    return Path(os.environ["LOCAL_PREDICTION_CACHE"]) / hashlib.sha1(key.encode()).hexdigest()[:16] / f"{name}.geff"


"""
FRAME_PREFETCH = """
_frame_pool, _frame_jobs, _frame_source = _futures.ThreadPoolExecutor(2), {}, [None]
_load_frame_uncached = _load_frame


def _load_frame(zarr_arr, t, target_shape, downsample=(1, 1, 1)):
    if _frame_source[0] is not zarr_arr:
        _frame_source[0] = zarr_arr
        _frame_jobs.clear()
    for ahead in range(t, min(t + 3, zarr_arr.shape[0])):
        if ahead not in _frame_jobs:
            _frame_jobs[ahead] = _frame_pool.submit(_load_frame_uncached, zarr_arr, ahead, target_shape, downsample)
    for stale in [k for k in _frame_jobs if k < t - 1]:
        del _frame_jobs[stale]
    return _frame_jobs[t].result()


"""
DIHEDRAL_HELPER = """
_DIHEDRAL = (
    (lambda x: x.flip((-1,)), lambda y: y.flip((-1,))),
    (lambda x: x.flip((-2,)), lambda y: y.flip((-2,))),
    (lambda x: x.flip((-2, -1)), lambda y: y.flip((-2, -1))),
    (lambda x: torch.rot90(x, 1, dims=(-2, -1)), lambda y: torch.rot90(y, -1, dims=(-2, -1))),
    (lambda x: torch.rot90(x, 3, dims=(-2, -1)), lambda y: torch.rot90(y, -3, dims=(-2, -1))),
    (lambda x: x.transpose(-1, -2), lambda y: y.transpose(-1, -2)),
    (
        lambda x: torch.rot90(x, 1, dims=(-2, -1)).transpose(-1, -2),
        lambda y: torch.rot90(y.transpose(-1, -2), -1, dims=(-2, -1)),
    ),
)


def _dihedral_encode(model, imgs):
    unet, det = model.encode(torch.cat([forward(imgs) for forward, _ in _DIHEDRAL]))
    return [
        ([inverse(d[v : v + 1]) for d in det], inverse(unet[v : v + 1]))
        for v, (_, inverse) in enumerate(_DIHEDRAL)
    ]


"""
TTA_BLOCKS = (
    (
        "            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n",
        "            del imgs_at, det_at, _u_at\n            _nv += 1\n",
        ("model", "det_logits", "_edge_tta", "_unet_acc", "_nv", 12),
    ),
    (
        "                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = ",
        "                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n"
        "                    _secondary_nv += 1\n",
        ("secondary_model", "secondary_det_logits", "_secondary_edge_tta", "_secondary_unet_acc", "_secondary_nv", 20),
    ),
)
BATCHED_TTA = """{pad}for _det_view, _u_view in _dihedral_encode({model}, imgs):
{pad}    for f in range(W):
{pad}        {det}[f] = {det}[f] + _det_view[f]
{pad}    if {flag}:
{pad}        {acc} = {acc} + _u_view
{pad}    {nv} += 1
"""


def batch_tta(source: str) -> str:
    """Encode the 7 non-identity dihedral TTA views in one batched call instead of 7 batch-1 calls."""
    for start, end, (model, det, flag, acc, nv, pad) in TTA_BLOCKS:
        assert source.count(start) == 1 and source.count(end) == 1, f"TTA block anchor not unique: {start!r}"
        lo, hi = source.index(start), source.index(end) + len(end)
        assert source[lo:hi].count(".encode(") == 4, "expected four view-encode calls in the TTA block"  # noqa: PLR2004
        block = BATCHED_TTA.format(pad=" " * pad, model=model, det=det, flag=flag, acc=acc, nv=nv)
        source = source[:lo] + block + source[hi:]
    return source


CANDIDATES = (
    "            candidates = sorted(\n"
    "                [\n"
    "                    (probs[i, j], i, j)\n"
    "                    for i in range(n_src)\n"
    "                    for j in range(n_tgt)\n"
    "                    if probs[i, j] > cfg.threshold\n"
    "                ],\n"
    "                reverse=True,\n"
    "            )\n"
)
CANDIDATES_VECTORIZED = (
    "            _ci, _cj = np.nonzero(probs > cfg.threshold)\n"
    "            _order = np.lexsort((-_cj, -_ci, -probs[_ci, _cj]))\n"
    "            candidates = [(probs[i, j], i, j) for i, j in zip(_ci[_order].tolist(), _cj[_order].tolist())]\n"
)
LOOP_SETUP = "    _post_pool, _pending = _futures.ThreadPoolExecutor(1), []\n"
REUSE = (
    "        _cached = _local_prediction_cache(name)\n"
    "        if _cached.exists():\n"
    "            import shutil\n"
    '            shutil.copytree(_cached, output_dir / f"{name}.geff")\n'
    "            continue\n"
    "        _gpu_start = _time.perf_counter()\n"
)
FINISH_HEAD = (
    "        def _finish(name, coords, edges, _cached, _gpu_s):\n            _post_start = _time.perf_counter()\n"
)
FINISH_TAIL = (
    "            import shutil\n"
    "            shutil.rmtree(_cached, ignore_errors=True)\n"
    '            shutil.copytree(output_dir / f"{name}.geff", _cached)\n'
    '            print(f"PREDICT_TIMING {name} gpu_s={_gpu_s:.1f} '
    'post_s={_time.perf_counter() - _post_start:.1f}", flush=True)\n'
)
SUBMIT = (
    "        _pending.append(_post_pool.submit(_finish, name, coords, edges, _cached, "
    "_time.perf_counter() - _gpu_start))\n"
)
DRAIN = "    for _job in _pending:\n        _job.result()\n    _post_pool.shutdown()\n"


def indent(block: str) -> str:
    return "".join("    " + line if line.strip() else line for line in block.splitlines(keepends=True))


def apply(repo: Path) -> None:
    script = repo / PREDICT_SCRIPT
    source = script.read_text(encoding="utf-8")
    for anchor in (LOOP_HEAD, POST_HEAD, SAVE, PREDICT_DEF, CANDIDATES):
        assert source.count(anchor) == 1, f"predict-script anchor not unique: {anchor!r}"
    loop_start, post_start = source.index(LOOP_HEAD), source.index(POST_HEAD)
    loop_end = source.index(SAVE) + len(SAVE)
    assert loop_start < post_start < loop_end, "predict loop layout changed"
    gpu_part = source[loop_start + len(LOOP_HEAD) : post_start]
    post_part = source[post_start:loop_end]
    loop = LOOP_SETUP + LOOP_HEAD + REUSE + gpu_part + FINISH_HEAD + indent(post_part) + FINISH_TAIL + SUBMIT + DRAIN
    source = source[:loop_start] + loop + source[loop_end:]
    variant = "batched_tta" if os.environ.get("LOCAL_BATCHED_TTA") == "1" else ""
    helpers = f"_LOCAL_VARIANT = {variant!r}\n" + CACHE_HELPER + FRAME_PREFETCH + DIHEDRAL_HELPER
    source = source.replace(PREDICT_DEF, helpers + PREDICT_DEF)
    if variant:
        source = batch_tta(source)
    source = source.replace(CANDIDATES, CANDIDATES_VECTORIZED)
    script.write_text(source, encoding="utf-8")
