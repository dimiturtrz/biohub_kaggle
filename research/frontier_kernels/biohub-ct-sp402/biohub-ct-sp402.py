"""Kernel: 402ep support-pack model + ILP on the 4 public test videos -> submission.csv."""
import json
import os
import subprocess
import sys
from pathlib import Path

print("INPUT:", os.listdir("/kaggle/input"), flush=True)
PACK = "/kaggle/input/datasets/yongjilyu/biohub-ct-inference-pack"
COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
WORK = "/kaggle/working"

# 1. install deps from the vendored wheels (offline)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-index", "--find-links",
     f"{PACK}/wheels", "ilpy", "tracksdata", "zarr", "pyscipopt", "polars", "scipy", "networkx"],
    check=True, capture_output=True)

# 2. paths
SP = f"{PACK}/sp402"           # support-pack repo + weights
TEST = f"{COMP}/test"

# 3. run the support-pack predictor on each test video (402ep weights + ILP)
pred_dir = "/tmp/preds"
os.makedirs(pred_dir, exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = f"{SP}/repo/src"

env["PYTHONNOUSERSITE"] = "1"
test_videos = sorted(os.listdir(TEST))
print("test videos:", test_videos, flush=True)
for tv in test_videos:
    vp = f"{TEST}/{tv}"
    if not os.path.isdir(vp):
        continue
    name = tv.replace(".zarr", "")
    cmd = [
        sys.executable, f"{SP}/repo/scripts/predict_unet_transformer.py",
        "--debug-video", vp,
        "--weights", f"{SP}/weights/unet_transformer/split_0/edge_predictor_best.pth",
        "--use-ilp", "--det-threshold", "0.99",
        "--ilp-appearance-weight", "1.0", "--ilp-disappearance-weight", "2.0",
    ]
    print("running", name, flush=True)
    run_env = dict(env)
    run_env["PREDICTIONS_PATH_OVERRIDE"] = f"/tmp/preds_{name}"
    r = subprocess.run(cmd, env=run_env, capture_output=True, text=True, timeout=3600)
    print(name, "rc=", r.returncode, flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)

# 4. convert geff predictions -> submission.csv (Kaggle format)
import numpy as np
import zarr

def read_geff(path):
    zg = zarr.open_group(path, mode="r")
    n = zg["nodes"]
    ids = np.asarray(n["ids"]).astype(np.int64)
    props = n["props"]
    def get(key):
        arr = props[key]
        if "values" in arr:
            arr = arr["values"]
        return np.asarray(arr).astype(np.float64)
    t = get("t"); z = get("z"); y = get("y"); x = get("x")
    e = zg["edges"]
    eid = e["ids"]
    try:
        eid = eid["values"]
    except Exception:
        pass
    eid = np.asarray(eid).astype(np.int64)
    return ids, t, z, y, x, eid

rows = []
nid = 0
for tv in test_videos:
    name = tv.replace(".zarr", "")
    hits = []
    for root, dirs, files in os.walk("/tmp"):
        for d in dirs:
            if d == f"{name}.zarr.geff":
                hits.append(os.path.join(root, d))
    if not hits:
        print("MISSING prediction for", name, flush=True)
        continue
    ids, t, z, y, x, eid = read_geff(hits[0])
    idmap = {}
    for i, gi in enumerate(ids):
        idmap[int(gi)] = nid
        rows.append(f"{nid},{name},node,{nid},{int(t[i])},{z[i]:.1f},{y[i]:.1f},{x[i]:.1f},-1,-1")
        nid += 1
    for s, tg in eid:
        s, tg = int(s), int(tg)
        if s in idmap and tg in idmap:
            rows.append(f"{nid},{name},edge,-1,-1,-1,-1,-1,{idmap[s]},{idmap[tg]}")
            nid += 1
out = f"{WORK}/submission.csv"
with open(out, "w") as f:
    f.write("id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\n")
    f.write("\n".join(rows) + "\n")
print("submission written:", out, len(rows), "rows", flush=True)
