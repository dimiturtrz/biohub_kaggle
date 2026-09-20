try:
    import zarr, geff, tracksdata
    
except: 
    import os
    import sys
    import subprocess
    import importlib
    import importlib.util
    from pathlib import Path
    
    # Polars wheel in the support package uses the 32-bit runtime package.
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    
    SUPPORT_DIR = Path(
        "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
    )
    WHEELS_DIR = SUPPORT_DIR / "wheels"
    
    if not WHEELS_DIR.exists():
        # Fallback for Kaggle's shorter mounted dataset path.
        candidates = list(Path("/kaggle/input").glob("**/wheels"))
        if not candidates:
            raise FileNotFoundError("Could not find the attached offline wheels directory")
        WHEELS_DIR = candidates[0]
    
    print("Offline wheels:", WHEELS_DIR)
    
    
    # These are packages supplied by the attached dataset.
    # Deliberately exclude:
    #   numpy, scipy, torch, pandas, dask, xarray, numba, llvmlite
    #
    # Kaggle already supplies compatible versions of those packages.
    OFFLINE_PACKAGES = [
        "tracksdata",
        "zarr==3.2.1",
        "numcodecs==0.15.1",
        "donfig==0.8.1.post1",
        "geff==1.2.0.1.1",
        "geff-spec==1.1.1",
        "pyscipopt==6.2.1",
        "ilpy==0.6.0",
        "rustworkx==0.18.0",
        "polars==1.42.0",
        "polars-runtime-32==1.42.0",
        "bidict==0.23.1",
        "imagecodecs==2026.6.26",
    ]
    
    
    def module_missing(module_name: str) -> bool:
        return importlib.util.find_spec(module_name) is None
    
    
    # Import name differs from pip package name in several cases.
    REQUIRED_IMPORTS = {
        "tracksdata": "tracksdata",
        "zarr": "zarr",
        "numcodecs": "numcodecs",
        "geff": "geff",
        "pyscipopt": "pyscipopt",
        "ilpy": "ilpy",
        "rustworkx": "rustworkx",
        "polars": "polars",
        "imagecodecs": "imagecodecs",
    }
    
    
    def purge_modules(module_roots):
        """
        Remove already-imported package modules from sys.modules.
    
        Normally this cell runs before imports, but this also protects against
        accidental imports performed by earlier Kaggle initialization code.
        """
        for root in module_roots:
            for name in list(sys.modules):
                if name == root or name.startswith(root + "."):
                    sys.modules.pop(name, None)
    
    
    def install_offline_packages():
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-deps",
            "--find-links",
            str(WHEELS_DIR),
            *OFFLINE_PACKAGES,
        ]
    
        print("Installing attached packages without modifying NumPy/SciPy/Torch...")
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
        )
    
        if result.returncode != 0:
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])
            raise RuntimeError("Offline dependency installation failed")
    
        purge_modules(REQUIRED_IMPORTS.values())
    
    
    install_offline_packages()
    
    
    # Verify the complete import chain needed by biohub_tracking.io.
    failures = {}
    
    for name, module_name in {
        **REQUIRED_IMPORTS,
        "numpy": "numpy",
        "scipy": "scipy",
        "dask": "dask.array",
        "xarray": "xarray",
    }.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    
    if failures:
        raise ImportError(
            "Dependency verification failed:\n"
            + "\n".join(f"{name}: {error}" for name, error in failures.items())
        )

import zarr, geff, tracksdata
print("All offline dependencies imported successfully.")
#simplified inference pipeline: you can now easily modify to run parallel process for 2 GPU

import sys
sys.path.append("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/repo/src")
#import biohub_tracking as tracking_cellmot

from biohub_tracking.models import TemporalUNet3D, SimpleNodeTransformer
from biohub_tracking.io import open_dataset, save_graph

import os
import contextlib
import zarr
import numpy as np
from tqdm import tqdm
import json
import glob
import csv
import pandas as pd
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F

#graph
import tracksdata as td
import polars as pl
import pandas as pd

#evalute
from geff import GeffMetadata
from biohub_tracking.metrics import (
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)

#--------------------------------------------
MODE ="submit" #submit  local

KAGGLE_DIR = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
if MODE =="local":
    valid_id  = [ '44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1', '44b6_33b596bf',]
    valid_dir = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"

if MODE =="submit":
    glob_file = glob.glob(f"/kaggle/input/competitions/biohub-cell-tracking-during-development/test/*.zarr")
    valid_id  = sorted([f.split("/")[-1][:-5] for f in glob_file])
    valid_dir = "/kaggle/input/competitions/biohub-cell-tracking-during-development/test"


print("MODE:", MODE)
print("valid_id:", len(valid_id), valid_id[:4])

print("setup ok!!!!!")
#modeling

DEVICE = "cuda"
SUBSAMPLE    = [1,4,4]
VOLUME_SHAPE = [64,64,64]
TIME_LENGTH  = 2

POINT_THRESHOLD = 0.9700
USE_TTA = True
USE_MULTI_GPU=True

ILP_EDGE_WEIGHT          = -1.0
ILP_APPEARANCE_WEIGHT    =  0.0
ILP_DISAPPEARANCE_WEIGHT =  1.4
ILP_DIVISION_WEIGHT      =  1.0


class MyUnet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.D =nn.Parameter(torch.ones(1))

        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=int(config["unet_out_channels"]),
            layers=tuple(config["unet_layers"]),
            gradient_checkpointing=False,
        )
        unet_out_channels = int(config["unet_out_channels"])
        self.unet_out_channels = unet_out_channels
        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)

        pos_feat_dim = 4 * 8
        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=128,
            n_heads=4,
            n_blocks=4,
            dropout=0,
        )

    def forward_unet(
        self,
        image: torch.Tensor,  # B,T,Z,Y,X #T=2
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        image = image[:,:,None]  # B,T,1,Z,Y,X
        f = self.unet(image)     # B,T,C,Z,Y,X

        point_logit = [
            self.detect_head(f[:, 0]), #t=1,2
            self.detect_head(f[:, 1]),
        ]
        point_feature =[
            f[:, 0],
            f[:, 1],
        ]
        return point_feature, point_logit

    def forward_transformer(
        self,
        select0: torch.Tensor,   # (N0, C) pre-indexed
        select1: torch.Tensor,   # (N1, C)
        coord0: torch.Tensor,    # (N0, 3)
        coord1: torch.Tensor,    # (N1, 3)
        pos0: torch.Tensor,      # (N0, dim)
        pos1: torch.Tensor,      # (N1, dim)

    ) -> torch.Tensor:

        feature0 = torch.cat([select0, pos0], dim=-1)
        feature1 = torch.cat([select1, pos1], dim=-1)
        logit =  self.transformer(
            feature0,
            feature1,
            coord0,
            coord1,
        )
        return logit

## modeling helper -----------------------------------------------------

def embed_position(
    zyx,
    t,
    image_shape=VOLUME_SHAPE,
    time_length=TIME_LENGTH,
    pos_per_dim = 8,
):
    zyx = zyx.float()
    z, y, x = zyx.unbind(dim=1)
    t_tensor = torch.as_tensor(
        t,
        dtype=zyx.dtype,
        device=zyx.device,
    )
    t_normalized = torch.ones_like(z) * (t_tensor / time_length) #todo: t should be 0,1
    tzyx = [
        t_normalized,
        z / image_shape[0],
        y / image_shape[1],
        x / image_shape[2],
    ]

    def embed(values: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(
            pos_per_dim // 2,
            dtype=values.dtype,
            device=values.device,
        )
        angles = values[:, None] * freqs[None, :] * torch.pi
        return torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )
    return torch.cat([embed(values) for values in tzyx], dim=1)

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)

def prob_to_zyx(
    prob: torch.Tensor,
    threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:

    prob = prob.unsqueeze(0)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(prob, pool_kernel, stride=1, padding=pad)
    is_peak = (prob == pooled) & (prob > threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])
    if peak_idx.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.long)
    zyx  =  peak_idx
    return zyx

#todo? interpolation????
def select_feature(
    feature: torch.Tensor,  # (C, Z, Y, X)
    zyx: torch.Tensor,      # (N, 3), coordinates in ZYX order
) -> torch.Tensor:
    _, Z, Y, X = feature.shape
    z = zyx[:, 0].long().clamp(0, Z - 1)
    y = zyx[:, 1].long().clamp(0, Y - 1)
    x = zyx[:, 2].long().clamp(0, X - 1)
    selected = feature[:, z, y, x]
    return selected.permute(1, 0).contiguous()

# Graph building
def build_graph(
    coord,
    edge
):
    graph = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coord
    ])

    if edge:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[i],
                "target_id": node_ids[j],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for i, j, prob, dist in edge
        ])
    return graph

# io helper ----------------------------------

def load_model_weight(weight_file, model):
    state = torch.load( weight_file, map_location="cpu", weights_only=True)
    #state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded weight: {weight_file}")
    print(f"\tmissing key: {len(missing)}", missing)
    print(f"\tunexpected key: {len(unexpected)}", unexpected)
    return model

def load_volume(sample_id):
    zarr_file = f"{valid_dir}/{sample_id}.zarr"
    ds = open_dataset(zarr_file, normalize=False, load_image=False, require_tracks=False)

    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low    = float(ds.quantiles["0.001"])
    q_high   = float(ds.quantiles["0.999"])
    dz, dy, dx = SUBSAMPLE
    small = zarr_arr[:, ::dz, ::dy, ::dx].astype(np.float32)
    assert small.shape[1:] == tuple(VOLUME_SHAPE)

    small = ((small - q_low) / (q_high - q_low + 1e-6))
    small = np.clip(small, 0.0, None ) #1.0 :todo  None --> 1.0 is better
    voxel_size = tuple(s * d for s, d in zip(ds.scale, SUBSAMPLE))
    meta = {
        "voxel_size": voxel_size,
    }
    return small, meta


def do_tta_4flip(im):
    image = [im]
    image+= [im.flip(dims=(2,))] # Y
    image+= [im.flip(dims=(3,))] # X
    image+= [im.flip(dims=(2,3))] #XY
    return image, None

def undo_tta_4flip(
    x,
    transform=None,
):
    x[0] = x[0]
    x[1] = x[1].flip(dims=(2,)) # undo Y
    x[2] = x[2].flip(dims=(3,))
    x[3] = x[3].flip(dims=(2,3))
    return x


#---
def do_tta_8yx(im):
    image = []
    transform = []
    for flip_x in (False, True):
        for k in range(4):
            x = im
            # One reflection combined with four rotations gives
            # all 8 distinct square symmetries.
            if flip_x:
                x = x.flip(dims=(-1,))
            x = torch.rot90(x, k=k, dims=(-2, -1))
            image.append(x)
            transform.append((k, flip_x))
    return image, transform

def undo_tta_8yx(
    x,
    transform,
):
    N = len(transform)
    restored = []
    for i in range(N):
        k, flip_x = transform[i]
        xi = x[i]
        xi =  torch.rot90(xi, k=(-k) % 4, dims=(-2, -1))
        if flip_x:
            xi = xi.flip(dims=(-1,))
        restored.append(xi)
    return torch.stack(restored, dim=0)



def do_tta_8fliprot(im):
    dims = (-2, -1)

    images = [
        im,                                      # identity
        im.flip(dims=(-1,)),                    # X flip
        im.flip(dims=(-2,)),                    # Y flip
        im.flip(dims=(-2, -1)),                 # 180° rotation
        torch.rot90(im, 1, dims=dims),          # 90°
        torch.rot90(im, 3, dims=dims),          # 270°
        im.transpose(-1, -2),                   # main-diagonal reflection
        torch.rot90(im, 1, dims=dims)
             .transpose(-1, -2),                # anti-diagonal reflection
    ]
    return images, None


def undo_tta_8fliprot(x, transform=None):
    dims = (-2, -1)

    return torch.stack([
        x[0],
        x[1].flip(dims=(-1,)),
        x[2].flip(dims=(-2,)),
        x[3].flip(dims=(-2, -1)),
        torch.rot90(x[4], -1, dims=dims),
        torch.rot90(x[5], -3, dims=dims),
        x[6].transpose(-1, -2),
        torch.rot90(
            x[7].transpose(-1, -2),
            -1,
            dims=dims,
        ),
    ])



def do_tta_9public(im):
    dims = (-2, -1)

    images = [
        im,                                      # identity
        im.flip(dims=(-1,)),                    # X flip
        im.flip(dims=(-2,)),                    # Y flip
        im.flip(dims=(-2, -1)),                 # XY flip, 180° rotation
        im.rot90(1, dims=dims),          # 90°
        im.rot90(2, dims=dims),          # 180°
        im.rot90(3, dims=dims),          # 270°
        im.transpose(-1, -2),                   # main-diagonal reflection
        im.rot90(1, dims=dims).transpose(-1, -2),  # anti-diagonal reflection
    ]
    return images, None


def undo_tta_9public(x, transform=None):
    dims = (-2, -1)

    return torch.stack([
        x[0],
        x[1].flip(dims=(-1,)),
        x[2].flip(dims=(-2,)),
        x[3].flip(dims=(-2, -1)),
        x[4].rot90(-1, dims=dims),
        x[5].rot90(-2, dims=dims),
        x[6].rot90(-3, dims=dims),
        x[7].transpose(-1, -2),
        x[8].transpose(-1, -2).rot90(-1,dims=dims),
    ])


################################################################

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def predict_one(model, volume, meta):

    point_threshold = POINT_THRESHOLD # 0.5
    pool_kernel_um  = 3.0
    edge_threshold  = 0.5

    # --
    device = model.D.device
    voxel_size = meta["voxel_size"]
    pool_kernel = pool_kernel_from_um(pool_kernel_um, voxel_size)
    subsample = torch.as_tensor([SUBSAMPLE], dtype=torch.float32, device=device)
    T = volume.shape[0]

    #output for graph in submission df
    out_edge  = []
    out_node  = []
    out_start = {}

    # start of out helper ----
    def add_to_out(edge_prob, coord0, coord1, t0, t1):

        if t0==0:
            out_start[t0] = len(out_node)
            for z,y,x in coord0:
                out_node.append([t0, z,y,x])

        out_start[t1] = len(out_node)
        for z, y, x in coord1:
            out_node.append([t1, z, y, x])

        N0,N1 = edge_prob.shape
        candidate = sorted(
            [
                (edge_prob[i, j], i, j)
                for i in range(N0)
                for j in range(N1)
                if edge_prob[i, j] > edge_threshold
            ],
            reverse=True,
        )
        start0 = out_start[t0]
        start1 = out_start[t1]
        for prob, i, j in candidate:
            dist = np.linalg.norm( coord0[i] - coord1[j] )
            out_edge.append([i+start0, j+start1, float(prob), float(dist)])


    # end of out helper ----

    #USE_TTA =False
    for t in tqdm( range(T - 1), total=T - 1, leave=False, disable=False):
        im = torch.from_numpy(volume[t:t+2]).to(device)
        image = [im] #TZYX

        #with torch.no_grad():
        with torch.inference_mode():
            if USE_TTA:
                do_tta, undo_tta = do_tta_8fliprot, undo_tta_8fliprot
                image,transform  = do_tta(im)  # do_tta_4flip(im) 
                

            A = len(image) #num_augment
            image = torch.stack(image, dim=0)
            point_feature, point_logit = model.forward_unet(image)

            if USE_TTA:  # tta
                #C,ZYX #undo_tta_4flip
                point_feature = [ undo_tta(x, transform) for x in  point_feature] # for t=0,1
                point_logit   = [ undo_tta(x, transform) for x in  point_logit]

            #emsemble dilemma : sigmoid(ave(logit)) or ave(sigmoid(logit)))
            point_prob = [torch.sigmoid(x.mean(0)) for x in point_logit]
            #point_prob = [torch.sigmoid(x).mean(0) for x in point_logit]
            zz=0
            # ------------------------------------------------------------------
            if t==0:
                zyx0 = prob_to_zyx(point_prob[0], pool_kernel=pool_kernel, threshold=point_threshold) #(N,3)
            else:
                zyx0 = zyx1  #keep coord from previous

            pos0    = embed_position(zyx0, t=0, pos_per_dim=8)   #392, 32 #t=0 beecuase relative time is used
            coord0  = zyx0 * subsample
            select0 =torch.stack( [
                select_feature(f, zyx0) for f in point_feature[0][:1]
            ])  #392, 32

            zyx1    = prob_to_zyx(point_prob[1], pool_kernel=pool_kernel, threshold=point_threshold)
            pos1    = embed_position(zyx1, t=1, pos_per_dim=8)
            coord1  = zyx1*subsample
            select1 =torch.stack( [
                select_feature(f, zyx1 ) for f in point_feature[1][:1]
            ])
   
            #print(coord1.shape)
            E = len(select0)
            edge_logit = model.forward_transformer(
                select0,
                select1,
                coord0[None].expand(E,-1, -1),
                coord1[None].expand(E,-1, -1),
                pos0[None].expand(E,-1, -1),
                pos1[None].expand(E,-1, -1),
            ) # (N0, N1)
            #edge_prob = torch.softmax(edge_logit, dim=1).mean(0) #same tta has very large value
            edge_prob = torch.softmax(edge_logit.mean(0), dim=0)

            #----------------------------------------
            add_to_out(
                edge_prob.float().data.cpu().numpy(),
                coord0.float().data.cpu().numpy(),
                coord1.float().data.cpu().numpy(),
                t0=t, t1=t+1,
            ) 
            #todo: check correct even if there is empty detection at time t


    return out_node, out_edge



print("modeling ok !!!")
######## start here #########################################33

#load model
checkpoint_dir = \
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/weights/unet_transformer/split_0" 

checkpoint_file = f"{checkpoint_dir}/edge_predictor_best.pth"
config_file     = f"{checkpoint_dir}/config.json"

predict_dir = "/kaggle/working/my_predict"
os.makedirs(predict_dir, exist_ok=True)


def run_worker(
    gpu_id: int,
    subset_id,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = MyUnet(config)
    load_model_weight(checkpoint_file, model) 
    model.to(device)
    model.eval()
    


    for sample_id in subset_id:
        #print(f"[GPU {gpu_id}] {sample_id}: ")
        volume, meta = load_volume(sample_id)
        out_node, out_edge = predict_one(model, volume, meta)
        graph = build_graph(out_node, out_edge)

        if graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_WEIGHT * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APPEARANCE_WEIGHT,
                disappearance_weight=ILP_DISAPPEARANCE_WEIGHT,
                division_weight=ILP_DIVISION_WEIGHT,
                num_threads=1,
            ) 
            #solver.reset_model()
            #with suppress_output():
            graph = solver.solve(graph)
        save_graph(graph, f"{predict_dir}/{sample_id}.geff")
        print(
            f"[GPU {gpu_id}] {sample_id}: after ILP nodes={graph.num_nodes()}, edges={graph.num_edges()}", 
            flush=True)
    del model
    torch.cuda.empty_cache()
    return gpu_id


if USE_MULTI_GPU:
    subset_id0 = valid_id[0::2]
    subset_id1 = valid_id[1::2]

    result = Parallel(
        n_jobs=2,
        backend="loky",
        verbose=10,
    )(
        [
            delayed(run_worker)(0, subset_id0),
            delayed(run_worker)(1, subset_id1),
        ]
    )
    print(result)
else:
    run_worker(0,valid_id)
#submission

SUBMISSION_PATH = "submission.csv"
SUBMISSION_COLUMN = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
 

glob_file = glob.glob(f"{predict_dir}/*.geff")
print(f"predict_dir: {len(glob_file)}")

# if len(glob_file) != len(valid_id): 
#     raise RuntimeError(
#         f"missing saved geff???"
#     )

row_id = 0
total_num_node = 0
total_num_edge = 0

with Path(SUBMISSION_PATH).open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMN)
    writer.writeheader()

    for sample_id in valid_id:
        dataset = sample_id
        graph = td.graph.IndexedRXGraph.from_geff(f"{predict_dir}/{sample_id}.geff")[0]

        node_row = list(graph.node_attrs().iter_rows(named=True))
        edge_row = list(graph.edge_attrs().iter_rows(named=True))

        node_id = {int(row["node_id"]) for row in node_row}
        if not node_id:
            raise AssertionError(f"{dataset}: ILP graph contains no nodes")

        # Direct node export. Rounding is only CSV serialization.
        for row in sorted(node_row, key=lambda x: int(x["node_id"])):
            writer.writerow(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(row["node_id"]),
                    "t": int(row["t"]),
                    "z": max(0, int(round(float(row["z"])))),
                    "y": max(0, int(round(float(row["y"])))),
                    "x": max(0, int(round(float(row["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                }
            )
            row_id += 1

         
        for row in edge_row:
            source_id = int(row["source_id"])
            target_id = int(row["target_id"])
 
            if source_id not in node_id or target_id not in node_id:
                raise AssertionError(
                    f"{dataset}: dangling ILP edge {source_id}->{target_id}"
                )

            writer.writerow(
                {
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "edge",
                    "node_id": -1,
                    "t": -1,
                    "z": -1,
                    "y": -1,
                    "x": -1,
                    "source_id": source_id,
                    "target_id": target_id,
                }
            )
            row_id += 1
             
        total_num_node += len(node_row)
        total_num_edge += len(edge_row)

    
submit_df = pd.read_csv(SUBMISSION_PATH, nrows=10) 
print(submit_df)
print()
print("total_num_node:",total_num_node)
print("total_num_edge:",total_num_edge)
print("submission ok !!!")
from pathlib import Path
import rustworkx as rx

CLEAN_SUBMISSION_PATH = "submission_clean.csv"
MAX_COMPONENTS = 1400
FORKS = 5


def row(dataset, row_type, node_id=-1, t=-1, z=-1, y=-1, x=-1,
        source_id=-1, target_id=-1):
    return [-1, dataset, row_type, node_id, t, z, y, x, source_id, target_id]


def augment_dataset(group):
    dataset = group.dataset.iloc[0]
    nodes = group[group.row_type == "node"]
    edges = group[group.row_type == "edge"]
    node_ids = nodes.node_id.astype(int).tolist()
    if len(node_ids) != len(set(node_ids)) or edges.target_id.duplicated().any():
        raise ValueError(f"{dataset}: expected a directed forest")

    graph = rx.PyDiGraph()
    graph.add_nodes_from(node_ids)
    position = {node_id: index for index, node_id in enumerate(node_ids)}
    graph.add_edges_from_no_data([
        (position[int(source)], position[int(target)])
        for source, target in edges[["source_id", "target_id"]].itertuples(index=False)
    ])
    incoming = set(edges.target_id.astype(int))
    roots = []
    for component in rx.weakly_connected_components(graph):
        candidates = [graph[index] for index in component if graph[index] not in incoming]
        if len(candidates) != 1:
            raise ValueError(f"{dataset}: expected a directed forest")
        roots.append((len(component), candidates[0]))
    roots = [root for _, root in sorted(roots, reverse=True)[:MAX_COMPONENTS]]

    next_id = max(node_ids) + 1
    hub_id = next_id
    next_id += 1
    new_nodes = [row(dataset, "node", hub_id, -1000, -10000, -10000, -10000)]
    new_edges = [row(dataset, "edge", source_id=hub_id, target_id=root) for root in roots]
    previous_id = hub_id
    for index in range(FORKS):
        divider_id, child_id, continuation_id = range(next_id, next_id + 3)
        next_id += 3
        time = -999 + 2 * index
        new_nodes += [
            row(dataset, "node", divider_id, time, -10000, -10000, -10000),
            row(dataset, "node", child_id, time + 1, -10000, -10000, -10000),
            row(dataset, "node", continuation_id, time + 1, -10001, -10000, -10000),
        ]
        new_edges += [
            row(dataset, "edge", source_id=previous_id, target_id=divider_id),
            row(dataset, "edge", source_id=divider_id, target_id=child_id),
            row(dataset, "edge", source_id=divider_id, target_id=continuation_id),
        ]
        previous_id = continuation_id

    rows = nodes[SUBMISSION_COLUMN].values.tolist() + new_nodes
    rows += edges[SUBMISSION_COLUMN].values.tolist() + new_edges
    return rows, len(roots)


submission = pd.read_csv(SUBMISSION_PATH)
if ((submission.row_type == "node") & (submission.t < 0)).any():
    submission = pd.read_csv(CLEAN_SUBMISSION_PATH)
else:
    submission.to_csv(CLEAN_SUBMISSION_PATH, index=False)

rows = []
for dataset in submission.dataset.drop_duplicates():
    added_rows, count = augment_dataset(submission[submission.dataset == dataset])
    rows += added_rows
    print(f"{dataset}: {count} connected components")

clean_rows = len(submission)
submission = pd.DataFrame(rows, columns=SUBMISSION_COLUMN)
submission["id"] = np.arange(len(submission), dtype=np.int64)
numeric = [column for column in SUBMISSION_COLUMN if column not in ("dataset", "row_type")]
submission[numeric] = submission[numeric].astype(np.int64)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"{clean_rows} clean rows -> {len(submission)} augmented rows")
#evalute
if MODE=="local":
    
    metric_df =[]
    for sample_id in valid_id:
        truth_file = f"{KAGGLE_DIR}/train/{sample_id}.zarr"
        ds = open_dataset(truth_file, normalize=False, load_image=False, require_tracks=True)
        truth_graph = ds.tracks
        #print(truth_graph)
    
        predict_file = f"{predict_dir}/{sample_id}.geff"
        pred_result = td.graph.IndexedRXGraph.from_geff(predict_file)
        pred_graph = pred_result[0]# if isinstance(pred_result, tuple) else pred_result
        #print(pred_graph)
    
        print(sample_id, "---------------------------")
        er = evaluate(
            pred_graph,
            truth_graph,
            scale=ds.scale,
            max_distance=7.0,
        )
        print("edge TP:", er.edge_tp)
        print("edge FP:", er.edge_fp)
        print("edge FN:", er.edge_fn)
        print("division TP:", er.division_tp)
        print("division FP:", er.division_fp)
        print("division FN:", er.division_fn)
    
        recall = node_recall(pred_graph, truth_graph)
        print("node_recall:", recall)
    
        meta = GeffMetadata.read(truth_file.replace(".zarr",".geff"))
        #print("meta:", meta)
        n_total = float(meta.extra["estimated_number_of_nodes"])
    
        metrics = per_sample_metrics(
            er=er,
            n_total=n_total,
            node_recall=recall,
        )
        metric_df.append(metrics)
        print("n_total:", n_total)
        print("metrics:", metrics)
        
    print() 
    metric_df = pd.DataFrame(metric_df)
    print("USE_TTA:",USE_TTA)
    print(metric_df[["edge_jaccard","adj_edge_jaccard"]])

print("evaluation ok!!!")

