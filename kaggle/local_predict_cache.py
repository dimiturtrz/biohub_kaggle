"""Reuse a donor kernel's per-video prediction graphs across local re-runs.

The kernel re-materializes its tracking repo on every start and the predict script wipes its output dir, so a
re-run that only changes post-processing would redo all GPU prediction. `apply` patches the predict loop to copy
a cached `<stem>.geff` when one exists under LOCAL_PREDICTION_CACHE, keyed by everything the prediction reads:
its CLI args (bar output naming), the BIOHUB_* env vars the script names, and a fingerprint of each weights file.
"""

from pathlib import Path

PREDICT_SCRIPT = Path("scripts") / "predict_unet_transformer.py"
LOOP_HEAD = (
    '    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):\n        ds_path = data_dir / name\n'
)
SAVE = '        save_graph(graph, output_dir / f"{name}.geff")\n'
PREDICT_DEF = "def predict(\n"

CACHE_HELPER = """
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
    key = json.dumps([argv, sorted((k, os.environ[k]) for k in named if k in os.environ), fingerprints])
    return Path(os.environ["LOCAL_PREDICTION_CACHE"]) / hashlib.sha1(key.encode()).hexdigest()[:16] / f"{name}.geff"


"""
REUSE = (
    "        _cached = _local_prediction_cache(name)\n"
    "        if _cached.exists():\n"
    "            import shutil\n"
    '            shutil.copytree(_cached, output_dir / f"{name}.geff")\n'
    "            continue\n"
)
STORE = (
    "        import shutil\n"
    "        shutil.rmtree(_cached, ignore_errors=True)\n"
    '        shutil.copytree(output_dir / f"{name}.geff", _cached)\n'
)


def apply(repo: Path) -> None:
    script = repo / PREDICT_SCRIPT
    source = script.read_text(encoding="utf-8")
    for anchor in (LOOP_HEAD, SAVE, PREDICT_DEF):
        assert source.count(anchor) == 1, f"predict-script anchor not unique: {anchor!r}"
    source = source.replace(PREDICT_DEF, CACHE_HELPER + PREDICT_DEF)
    source = source.replace(LOOP_HEAD, LOOP_HEAD + REUSE)
    source = source.replace(SAVE, SAVE + STORE)
    script.write_text(source, encoding="utf-8")
