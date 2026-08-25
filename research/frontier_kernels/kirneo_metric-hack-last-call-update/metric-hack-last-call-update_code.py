# --- cell 0 ---
try:
    import zarr, geff, tracksdata
    
except: 
    import os
    import sys
    import subprocess
    import importlib
    import importlib.util
    from pathlib import Path
    
    os.environ.setdefault("POLARS_PREFER_PKG", "32")
    
    SUPPORT_DIR = Path(
        "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
    )
    WHEELS_DIR = SUPPORT_DIR / "wheels"
    
    if not WHEELS_DIR.exists():
        candidates = list(Path("/kaggle/input").glob("**/wheels"))
        if not candidates:
            raise FileNotFoundError("Could not find the attached offline wheels directory")
        WHEELS_DIR = candidates[0]
    
    print("Offline wheels:", WHEELS_DIR)
    
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

# --- cell 1 ---

import sys
sys.path.append("/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/repo/src")

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

import tracksdata as td
import polars as pl
import pandas as pd

from geff import GeffMetadata
from biohub_tracking.metrics import (
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)

MODE ="submit" 

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

# --- cell 2 ---
#modeling

DEVICE = "cuda"
SUBSAMPLE    = [1,4,4]
VOLUME_SHAPE = [64,64,64]
TIME_LENGTH  = 2

POINT_THRESHOLD = 0.9550
USE_TTA = True
USE_MULTI_GPU=True

ILP_EDGE_WEIGHT          = -1.0
ILP_APPEARANCE_WEIGHT    =  0.0
ILP_DISAPPEARANCE_WEIGHT =  1.4
ILP_DIVISION_WEIGHT      =  1.0

# Candidate edges для ILP.
#
# Все очень уверенные рёбра сохраняются.
# Дополнительно для каждой target-клетки сохраняются
# два лучших возможных родителя.
EDGE_STRONG_THRESHOLD = 0.50
EDGE_MIN_THRESHOLD = 0.25
EDGE_TOPK_PARENTS = 3
EDGE_TOPK_CHILDREN = 1
EDGE_CHILD_MIN_THRESHOLD = 0.18

# Физический предел перемещения между соседними кадрами.
EDGE_MAX_DISTANCE_UM = 10.0

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
        image: torch.Tensor,  
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        image = image[:,:,None]
        f = self.unet(image)   

        point_logit = [
            self.detect_head(f[:, 0]), 
            self.detect_head(f[:, 1]),
        ]
        point_feature =[
            f[:, 0],
            f[:, 1],
        ]
        return point_feature, point_logit

    def forward_transformer(
        self,
        select0: torch.Tensor,   
        select1: torch.Tensor,   
        coord0: torch.Tensor,    
        coord1: torch.Tensor,    
        pos0: torch.Tensor,      
        pos1: torch.Tensor,      

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
    t_normalized = torch.ones_like(z) * (t_tensor / time_length) 
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

def select_feature(
    feature: torch.Tensor,  
    zyx: torch.Tensor,      
) -> torch.Tensor:
    _, Z, Y, X = feature.shape
    z = zyx[:, 0].long().clamp(0, Z - 1)
    y = zyx[:, 1].long().clamp(0, Y - 1)
    x = zyx[:, 2].long().clamp(0, X - 1)
    selected = feature[:, z, y, x]
    return selected.permute(1, 0).contiguous()

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


def load_model_weight(weight_file, model):
    state = torch.load( weight_file, map_location="cpu", weights_only=True)
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
    small = np.clip(small, 0.0, None ) 
    voxel_size = tuple(s * d for s, d in zip(ds.scale, SUBSAMPLE))
    meta = {
        "voxel_size": voxel_size,
    }
    return small, meta


def do_tta_4flip(im):
    image = [im]
    image+= [im.flip(dims=(2,))] 
    image+= [im.flip(dims=(3,))] 
    image+= [im.flip(dims=(2,3))] 
    return image, None

def undo_tta_4flip(
    x,
    transform=None,
):
    x[0] = x[0]
    x[1] = x[1].flip(dims=(2,)) 
    x[2] = x[2].flip(dims=(3,))
    x[3] = x[3].flip(dims=(2,3))
    return x



def do_tta_8yx(im):
    image = []
    transform = []
    for flip_x in (False, True):
        for k in range(4):
            x = im
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
        im,                                     
        im.flip(dims=(-1,)),                    
        im.flip(dims=(-2,)),                    
        im.flip(dims=(-2, -1)),                 
        torch.rot90(im, 1, dims=dims),          
        torch.rot90(im, 3, dims=dims),          
        im.transpose(-1, -2),                   
        torch.rot90(im, 1, dims=dims)
             .transpose(-1, -2),                
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
        im,                                    
        im.flip(dims=(-1,)),                   
        im.flip(dims=(-2,)),                   
        im.flip(dims=(-2, -1)),                
        im.rot90(1, dims=dims),          
        im.rot90(2, dims=dims),          
        im.rot90(3, dims=dims),          
        im.transpose(-1, -2),            
        im.rot90(1, dims=dims).transpose(-1, -2),  
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



@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def predict_one(model, volume, meta):

    point_threshold = POINT_THRESHOLD 
    pool_kernel_um  = 3.0

    # --
    device = model.D.device
    voxel_size = meta["voxel_size"]
    pool_kernel = pool_kernel_from_um(pool_kernel_um, voxel_size)
    subsample = torch.as_tensor([SUBSAMPLE], dtype=torch.float32, device=device)
    # coord0 и coord1 ниже переводятся обратно
# в исходные координаты умножением на SUBSAMPLE.
# Поэтому для них нужен исходный voxel size:
# [1.625, 0.40625, 0.40625].
    raw_voxel_size = (
        np.asarray(
            voxel_size,
            dtype=np.float64,
        )
        / np.asarray(
            SUBSAMPLE,
            dtype=np.float64,
        )
    )
    T = volume.shape[0]

    out_edge  = []
    out_node  = []
    out_start = {}

    def add_to_out(edge_prob,coord0,coord1,t0,t1,):
        if t0 == 0:
            out_start[t0] = len(out_node)
    
            for z, y, x in coord0:
                out_node.append(
                    [t0, z, y, x]
                )
    
        out_start[t1] = len(out_node)
    
        for z, y, x in coord1:
            out_node.append(
                [t1, z, y, x]
            )
    
        number_source, number_target = (
            edge_prob.shape
        )
    
        if (
            number_source == 0
            or number_target == 0
        ):
            return
    
        # Здесь хранятся уникальные пары:
        # (индекс source, индекс target).
        candidate_pairs = set()
    
        # --------------------------------------------------------
        # 1. Сохраняем все уверенные рёбра.
        # --------------------------------------------------------
    
        strong_indices = np.argwhere(
            edge_prob
            >= EDGE_STRONG_THRESHOLD
        )
    
        for source_index, target_index in strong_indices:
            candidate_pairs.add(
                (
                    int(source_index),
                    int(target_index),
                )
            )
    
        # --------------------------------------------------------
        # 2. Для каждой target-клетки сохраняем
        #    два лучших возможных parent.
        #
        # edge_prob нормализована softmax по source,
        # поэтому столбец соответствует конкурирующим
        # родителям одной target-клетки.
        # --------------------------------------------------------
    
        top_k = min(
            EDGE_TOPK_PARENTS,
            number_source,
        )
    
        for target_index in range(
            number_target
        ):
            probabilities = edge_prob[
                :,
                target_index,
            ]
    
            if top_k == number_source:
                top_source_indices = np.arange(
                    number_source
                )
    
            else:
                top_source_indices = np.argpartition(
                    probabilities,
                    -top_k,
                )[-top_k:]
    
            for source_index in top_source_indices:
                probability = float(
                    probabilities[
                        source_index
                    ]
                )
    
                # Очень слабые top-k варианты всё равно
                # не передаём ILP.
                if probability < EDGE_MIN_THRESHOLD:
                    continue
    
                candidate_pairs.add(
                    (
                        int(source_index),
                        int(target_index),
                    )
                )

                # --------------------------------------------------------
        # Дополнительно сохраняем лучший child
        # для каждого source.
        # --------------------------------------------------------
        
        top_children = min(
            EDGE_TOPK_CHILDREN,
            number_target,
        )
        
        for source_index in range(
            number_source
        ):
            probabilities = edge_prob[
                source_index,
                :
            ]
    
        if top_children == number_target:
            top_target_indices = np.arange(
                number_target
            )
        else:
            top_target_indices = np.argpartition(
                probabilities,
                -top_children,
            )[-top_children:]
    
        for target_index in top_target_indices:
            probability = float(
                probabilities[target_index]
            )
    
            if (
                probability
                < EDGE_CHILD_MIN_THRESHOLD
            ):
                continue
    
            candidate_pairs.add(
                (
                    int(source_index),
                    int(target_index),
                )
            )
        candidates = []
    
        for (
            source_index,
            target_index,
        ) in candidate_pairs:
    
            source_position = coord0[
                source_index
            ]
    
            target_position = coord1[
                target_index
            ]
    
            delta_um = (
                source_position
                - target_position
            ) * raw_voxel_size
    
            distance_um = float(
                np.linalg.norm(
                    delta_um
                )
            )
    
            if (
                distance_um
                > EDGE_MAX_DISTANCE_UM
            ):
                continue
    
            probability = float(
                edge_prob[
                    source_index,
                    target_index,
                ]
            )
    
            candidates.append(
                (
                    probability,
                    source_index,
                    target_index,
                    distance_um,
                )
            )
    
        # Сначала уверенные рёбра.
        candidates.sort(
            reverse=True
        )
    
        start0 = out_start[t0]
        start1 = out_start[t1]
    
        for (
            probability,
            source_index,
            target_index,
            distance_um,
        ) in candidates:
    
            out_edge.append(
                [
                    source_index + start0,
                    target_index + start1,
                    probability,
                    distance_um,
                ]
            )

    for t in tqdm( range(T - 1), total=T - 1, leave=False, disable=False):
        im = torch.from_numpy(volume[t:t+2]).to(device)
        image = [im]

        with torch.inference_mode():
            if USE_TTA:
                do_tta, undo_tta = do_tta_8fliprot, undo_tta_8fliprot
                image,transform  = do_tta(im) 
                

            A = len(image)
            image = torch.stack(image, dim=0)
            point_feature, point_logit = model.forward_unet(image)

            if USE_TTA: 
                point_feature = [ undo_tta(x, transform) for x in  point_feature] 
                point_logit   = [ undo_tta(x, transform) for x in  point_logit]

            point_prob = [torch.sigmoid(x.mean(0)) for x in point_logit]

            # Detector уже усредняет logits всех TTA.
            # Теперь усредняем также внутренние признаки,
            # которые получает edge Transformer.
            if USE_TTA:
                edge_feature0 = point_feature[0].mean(
                    dim=0
                )
            
                edge_feature1 = point_feature[1].mean(
                    dim=0
                )
            
            else:
                edge_feature0 = point_feature[0][0]
                edge_feature1 = point_feature[1][0]
            zz=0
            # ------------------------------------------------------------------
            if t==0:
                zyx0 = prob_to_zyx(point_prob[0], pool_kernel=pool_kernel, threshold=point_threshold) 
            else:
                zyx0 = zyx1  

            pos0    = embed_position(zyx0, t=0, pos_per_dim=8)   
            coord0  = zyx0 * subsample
            select0 = select_feature(edge_feature0, zyx0,).unsqueeze(0)
            zyx1    = prob_to_zyx(point_prob[1], pool_kernel=pool_kernel, threshold=point_threshold)
            pos1    = embed_position(zyx1, t=1, pos_per_dim=8)
            coord1  = zyx1*subsample
            select1 = select_feature(edge_feature1,zyx1,).unsqueeze(0)
   
            E = len(select0)
            edge_logit = model.forward_transformer(
                select0,
                select1,
                coord0[None].expand(E,-1, -1),
                coord1[None].expand(E,-1, -1),
                pos0[None].expand(E,-1, -1),
                pos1[None].expand(E,-1, -1),
            ) 
            edge_prob = torch.softmax(edge_logit.mean(0), dim=0)

            add_to_out(
                edge_prob.float().data.cpu().numpy(),
                coord0.float().data.cpu().numpy(),
                coord1.float().data.cpu().numpy(),
                t0=t, t1=t+1,
            ) 


    return out_node, out_edge



print("modeling ok !!!")

# --- cell 3 ---
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
        
            graph = solver.solve(graph)
        

            
        save_graph(
            graph,
            f"{predict_dir}/{sample_id}.geff",
        )
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

# --- cell 4 ---
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

# --- cell 5 ---
# ============================================================
# HYBRID POSTPROCESSING
#
# Порядок:
#   clean ILP graph
#       -> one-frame gap closing
#       -> short-track filtering
#       -> submission_clean.csv
#
# Следующая ячейка добавит hub и fake forks.
# ============================================================

from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
import zarr


RAW_SUBMISSION_PATH = "submission.csv"
CLEAN_SUBMISSION_PATH = "submission_clean.csv"


# Физический размер исходного вокселя:
# координаты расположены в порядке z, y, x.
VOXEL_SIZE_UM = np.asarray(
    [1.625, 0.40625, 0.40625],
    dtype=np.float64,
)


# ------------------------------------------------------------
# Настройки one-frame gap closing
# ------------------------------------------------------------

GAP_CLOSE_MAX_TOTAL_UM = 10.0

# Сначала пытаемся использовать уже существующую
# изолированную детекцию на пропущенном кадре.
GAP_REUSE_EXISTING_UM = 2.5

# Ограничиваем число закрываемых разрывов.
GAP_MAX_ADDED_FRAC = 0.05
GAP_MAX_ADDED_ABS = 1000

# Уточнение синтетической точки по изображению.
GAP_REFINE_WIN_Z = 1
GAP_REFINE_WIN_YX = 4
GAP_REFINE_MAX_SHIFT_UM = 2.5


# ------------------------------------------------------------
# Настройки фильтрации коротких треков
# ------------------------------------------------------------

MIN_TRACK_LEN = 3

# ------------------------------------------------------------
# Осторожное сглаживание длинных неветвящихся траекторий
# ------------------------------------------------------------

TRAJECTORY_SMOOTHING = True

# 0.25 означает:
# 75% исходной координаты + 25% линейного предсказания.
TRAJECTORY_SMOOTH_WEIGHT = 0.15 #0.25 -> 0.15

# Не разрешаем фильтру сильно передвигать детекцию.
TRAJECTORY_SMOOTH_MAX_SHIFT_UM = 1.5

# Короткие компоненты с делением сохраняются.
KEEP_DIVISION_COMPONENTS = True

# Сохраняем компоненты, обрезанные началом или концом видео.
KEEP_BOUNDARY_COMPONENTS = True
BOUNDARY_MARGIN_T = 2


def open_image_array(dataset):
    """
    Открыть Zarr-массив изображения формы (T, Z, Y, X).
    """
    zarr_path = Path(valid_dir) / f"{dataset}.zarr"

    if not zarr_path.exists():
        raise FileNotFoundError(
            f"Не найден файл изображения: {zarr_path}"
        )

    root = zarr.open(
        str(zarr_path),
        mode="r",
    )

    # Иногда корень сам является массивом.
    if (
        hasattr(root, "shape")
        and len(root.shape) == 4
    ):
        return root

    # В данных соревнования массив обычно находится по пути "0".
    return root["0"]


def distance_um(point_a, point_b):
    """
    Физическое расстояние между координатами z,y,x.
    """
    point_a = np.asarray(
        point_a,
        dtype=np.float64,
    )

    point_b = np.asarray(
        point_b,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            (point_a - point_b)
            * VOXEL_SIZE_UM
        )
    )


def refine_synthetic_node(
    frame,
    midpoint,
):
    """
    Немного передвинуть синтетическую точку
    от геометрической середины к локальному
    центру яркости изображения.

    frame имеет форму (Z, Y, X).
    midpoint имеет порядок (z, y, x).
    """
    midpoint = np.asarray(
        midpoint,
        dtype=np.float64,
    )

    center = np.rint(
        midpoint
    ).astype(int)

    z, y, x = center.tolist()

    z0 = max(
        0,
        z - GAP_REFINE_WIN_Z,
    )

    z1 = min(
        frame.shape[0],
        z + GAP_REFINE_WIN_Z + 1,
    )

    y0 = max(
        0,
        y - GAP_REFINE_WIN_YX,
    )

    y1 = min(
        frame.shape[1],
        y + GAP_REFINE_WIN_YX + 1,
    )

    x0 = max(
        0,
        x - GAP_REFINE_WIN_YX,
    )

    x1 = min(
        frame.shape[2],
        x + GAP_REFINE_WIN_YX + 1,
    )

    patch = np.asarray(
        frame[
            z0:z1,
            y0:y1,
            x0:x1,
        ],
        dtype=np.float64,
    )

    if patch.size == 0:
        return midpoint

    background = float(
        np.percentile(
            patch,
            20,
        )
    )

    weights = np.maximum(
        patch - background,
        0,
    )

    weight_sum = float(
        weights.sum()
    )

    if (
        not np.isfinite(weight_sum)
        or weight_sum <= 0
    ):
        return midpoint

    zz = np.arange(
        z0,
        z1,
        dtype=np.float64,
    )[:, None, None]

    yy = np.arange(
        y0,
        y1,
        dtype=np.float64,
    )[None, :, None]

    xx = np.arange(
        x0,
        x1,
        dtype=np.float64,
    )[None, None, :]

    refined = np.asarray(
        [
            float(
                (weights * zz).sum()
                / weight_sum
            ),
            float(
                (weights * yy).sum()
                / weight_sum
            ),
            float(
                (weights * xx).sum()
                / weight_sum
            ),
        ],
        dtype=np.float64,
    )

    shift_um = distance_um(
        midpoint,
        refined,
    )

    # Если локальная яркость увела точку слишком далеко,
    # оставляем безопасную геометрическую середину.
    if (
        not np.isfinite(shift_um)
        or shift_um
        > GAP_REFINE_MAX_SHIFT_UM
    ):
        return midpoint

    return refined


def build_degrees(
    node_ids,
    edges,
):
    """
    Посчитать входящую и исходящую степени.
    """
    in_degree = {
        int(node_id): 0
        for node_id in node_ids
    }

    out_degree = {
        int(node_id): 0
        for node_id in node_ids
    }

    for source_id, target_id in edges:
        source_id = int(source_id)
        target_id = int(target_id)

        if source_id in out_degree:
            out_degree[source_id] += 1

        if target_id in in_degree:
            in_degree[target_id] += 1

    return in_degree, out_degree


def find_components(
    node_ids,
    edges,
):
    """
    Weakly connected components через union-find.
    Направление рёбер временно игнорируется.
    """
    node_ids = [
        int(node_id)
        for node_id in node_ids
    ]

    parent = {
        node_id: node_id
        for node_id in node_ids
    }

    rank = {
        node_id: 0
        for node_id in node_ids
    }

    def find(node_id):
        while parent[node_id] != node_id:
            parent[node_id] = parent[
                parent[node_id]
            ]

            node_id = parent[node_id]

        return node_id

    def union(node_a, node_b):
        root_a = find(node_a)
        root_b = find(node_b)

        if root_a == root_b:
            return

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b

        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a

        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    for source_id, target_id in edges:
        source_id = int(source_id)
        target_id = int(target_id)

        if (
            source_id in parent
            and target_id in parent
        ):
            union(
                source_id,
                target_id,
            )

    components = {}

    for node_id in node_ids:
        root = find(node_id)

        components.setdefault(
            root,
            [],
        ).append(node_id)

    return list(
        components.values()
    )

def smooth_nonbranching_trajectories(
    nodes,
    edges,
):
    """
    Осторожно сглаживает только внутренние nodes
    длинных неветвящихся траекторий.

    Не меняет topology и не добавляет рёбра.
    """

    parents = {}
    children = {}

    for source_id, target_id in edges:
        source_id = int(source_id)
        target_id = int(target_id)

        parents.setdefault(
            target_id,
            [],
        ).append(source_id)

        children.setdefault(
            source_id,
            [],
        ).append(target_id)

    original_positions = {
        node_id: np.asarray(
            [
                node["z"],
                node["y"],
                node["x"],
            ],
            dtype=np.float64,
        )
        for node_id, node in nodes.items()
    }

    updates = {}

    for node_id in nodes:
        node_parents = parents.get(
            node_id,
            [],
        )

        node_children = children.get(
            node_id,
            [],
        )

        # Центральная точка должна иметь
        # ровно одного parent и одного child.
        if (
            len(node_parents) != 1
            or len(node_children) != 1
        ):
            continue

        parent_id = int(
            node_parents[0]
        )

        child_id = int(
            node_children[0]
        )

        # Исключаем область рядом с division.
        if len(
            children.get(
                parent_id,
                [],
            )
        ) != 1:
            continue

        if len(
            parents.get(
                child_id,
                [],
            )
        ) != 1:
            continue

        # Требуем контекст ещё на один кадр
        # в обе стороны. Таким образом, фактически
        # сглаживаются только треки длиной не менее 5.
        if len(
            parents.get(
                parent_id,
                [],
            )
        ) != 1:
            continue

        if len(
            children.get(
                child_id,
                [],
            )
        ) != 1:
            continue

        parent_position = original_positions[
            parent_id
        ]

        current_position = original_positions[
            node_id
        ]

        child_position = original_positions[
            child_id
        ]

        # Для постоянной скорости средняя точка
        # между соседями является ожидаемой позицией.
        predicted_position = (
            parent_position
            + child_position
        ) / 2.0

        correction = (
            predicted_position
            - current_position
        )

        correction_um = float(
            np.linalg.norm(
                correction
                * VOXEL_SIZE_UM
            )
        )

        # Большое отклонение может быть настоящим
        # ускорением или ошибкой topology.
        # В таком случае точку не двигаем.
        if (
            not np.isfinite(
                correction_um
            )
            or correction_um
            > TRAJECTORY_SMOOTH_MAX_SHIFT_UM
        ):
            continue

        smoothed_position = (
            (
                1.0
                - TRAJECTORY_SMOOTH_WEIGHT
            )
            * current_position
            + TRAJECTORY_SMOOTH_WEIGHT
            * predicted_position
        )

        updates[node_id] = (
            smoothed_position
        )

    for node_id, position in updates.items():
        nodes[node_id]["z"] = float(
            position[0]
        )

        nodes[node_id]["y"] = float(
            position[1]
        )

        nodes[node_id]["x"] = float(
            position[2]
        )

    return len(updates)

def postprocess_one_dataset(
    group,
):
    """
    1. Закрыть разрывы длиной ровно один кадр.
    2. Удалить оставшиеся короткие компоненты.
    """
    dataset = str(
        group["dataset"].iloc[0]
    )

    image_array = open_image_array(
        dataset
    )

    T, Z, Y, X = map(
        int,
        image_array.shape,
    )

    node_frame = (
        group[
            group["row_type"] == "node"
        ][
            [
                "node_id",
                "t",
                "z",
                "y",
                "x",
            ]
        ]
        .copy()
    )

    edge_frame = (
        group[
            group["row_type"] == "edge"
        ][
            [
                "source_id",
                "target_id",
            ]
        ]
        .copy()
    )

    nodes = {}

    for row_data in node_frame.itertuples(
        index=False
    ):
        node_id = int(
            row_data.node_id
        )

        nodes[node_id] = {
            "node_id": node_id,
            "t": int(row_data.t),
            "z": float(row_data.z),
            "y": float(row_data.y),
            "x": float(row_data.x),
        }

    edges = {
        (
            int(row_data.source_id),
            int(row_data.target_id),
        )
        for row_data
        in edge_frame.itertuples(
            index=False
        )
    }

    smoothed_nodes = 0

    if TRAJECTORY_SMOOTHING:
        smoothed_nodes = (
            smooth_nonbranching_trajectories(
                nodes,
                edges,
            )
        )

    # Проверка dangling edges.
    for source_id, target_id in edges:
        if (
            source_id not in nodes
            or target_id not in nodes
        ):
            raise ValueError(
                f"{dataset}: dangling edge "
                f"{source_id}->{target_id}"
            )

    nodes_at_time = {}

    for node_id, node in nodes.items():
        nodes_at_time.setdefault(
            int(node["t"]),
            [],
        ).append(node_id)

    in_degree, out_degree = build_degrees(
        nodes.keys(),
        edges,
    )

    next_node_id = (
        max(nodes) + 1
        if nodes
        else 0
    )

    max_gap_pairs = min(
        GAP_MAX_ADDED_ABS,
        max(
            1,
            int(
                len(nodes)
                * GAP_MAX_ADDED_FRAC
            ),
        ),
    )

    gap_pairs_added = 0
    reused_existing = 0
    synthetic_added = 0

    # --------------------------------------------------------
    # ONE-FRAME GAP CLOSING
    #
    # Ищем:
    # source(t) -> отсутствующая клетка(t+1) -> target(t+2)
    # --------------------------------------------------------

    for source_t in range(
        0,
        T - 2,
    ):
        if gap_pairs_added >= max_gap_pairs:
            break

        target_t = source_t + 2
        middle_t = source_t + 1

        source_ids = [
            node_id
            for node_id
            in nodes_at_time.get(
                source_t,
                [],
            )
            if out_degree.get(
                node_id,
                0,
            ) == 0
        ]

        target_ids = [
            node_id
            for node_id
            in nodes_at_time.get(
                target_t,
                [],
            )
            if in_degree.get(
                node_id,
                0,
            ) == 0
        ]

        if (
            not source_ids
            or not target_ids
        ):
            continue

        source_coordinates = np.asarray(
            [
                [
                    nodes[node_id]["z"],
                    nodes[node_id]["y"],
                    nodes[node_id]["x"],
                ]
                for node_id in source_ids
            ],
            dtype=np.float64,
        )

        target_coordinates = np.asarray(
            [
                [
                    nodes[node_id]["z"],
                    nodes[node_id]["y"],
                    nodes[node_id]["x"],
                ]
                for node_id in target_ids
            ],
            dtype=np.float64,
        )

        # Переводим координаты в микрометры
        # перед вычислением расстояний.
        cost_matrix = cdist(
            source_coordinates
            * VOXEL_SIZE_UM,
            target_coordinates
            * VOXEL_SIZE_UM,
        )

        # Hungarian требует конечную матрицу.
        # Пары дальше threshold делаем очень дорогими.
        hungarian_cost = cost_matrix.copy()

        hungarian_cost[
            hungarian_cost
            > GAP_CLOSE_MAX_TOTAL_UM
        ] = 1e6

        row_indices, column_indices = (
            linear_sum_assignment(
                hungarian_cost
            )
        )

        candidate_pairs = []

        for row_index, column_index in zip(
            row_indices,
            column_indices,
        ):
            distance = float(
                cost_matrix[
                    row_index,
                    column_index,
                ]
            )

            if (
                distance
                <= GAP_CLOSE_MAX_TOTAL_UM
            ):
                candidate_pairs.append(
                    (
                        distance,
                        source_ids[row_index],
                        target_ids[column_index],
                    )
                )

        # Сначала обрабатываем самые уверенные,
        # то есть самые короткие разрывы.
        candidate_pairs.sort(
            key=lambda value: value[0]
        )

        if not candidate_pairs:
            continue

        # Кадр загружается только при наличии
        # реальных кандидатов.
        middle_frame = np.asarray(
            image_array[middle_t]
        )

        used_middle_nodes = set()

        for (
            _,
            source_id,
            target_id,
        ) in candidate_pairs:

            if gap_pairs_added >= max_gap_pairs:
                break

            # Проверяем, что source и target
            # ещё не были заняты предыдущей операцией.
            if out_degree.get(
                source_id,
                0,
            ) != 0:
                continue

            if in_degree.get(
                target_id,
                0,
            ) != 0:
                continue

            source_position = np.asarray(
                [
                    nodes[source_id]["z"],
                    nodes[source_id]["y"],
                    nodes[source_id]["x"],
                ],
                dtype=np.float64,
            )

            target_position = np.asarray(
                [
                    nodes[target_id]["z"],
                    nodes[target_id]["y"],
                    nodes[target_id]["x"],
                ],
                dtype=np.float64,
            )

            midpoint = (
                source_position
                + target_position
            ) / 2.0

            # Сначала пытаемся использовать уже существующий
            # изолированный node на среднем кадре.
            best_existing_id = None
            best_existing_distance = np.inf

            for middle_node_id in nodes_at_time.get(
                middle_t,
                [],
            ):
                if (
                    middle_node_id
                    in used_middle_nodes
                ):
                    continue

                if (
                    in_degree.get(
                        middle_node_id,
                        0,
                    ) != 0
                    or out_degree.get(
                        middle_node_id,
                        0,
                    ) != 0
                ):
                    continue

                middle_position = np.asarray(
                    [
                        nodes[middle_node_id]["z"],
                        nodes[middle_node_id]["y"],
                        nodes[middle_node_id]["x"],
                    ],
                    dtype=np.float64,
                )

                current_distance = distance_um(
                    midpoint,
                    middle_position,
                )

                if (
                    current_distance
                    < best_existing_distance
                ):
                    best_existing_distance = (
                        current_distance
                    )

                    best_existing_id = (
                        middle_node_id
                    )

            if (
                best_existing_id is not None
                and best_existing_distance
                <= GAP_REUSE_EXISTING_UM
            ):
                middle_node_id = int(
                    best_existing_id
                )

                used_middle_nodes.add(
                    middle_node_id
                )

                reused_existing += 1

            else:
                refined_position = (
                    refine_synthetic_node(
                        middle_frame,
                        midpoint,
                    )
                )

                middle_node_id = (
                    next_node_id
                )

                next_node_id += 1

                nodes[middle_node_id] = {
                    "node_id": middle_node_id,
                    "t": middle_t,
                    "z": float(
                        refined_position[0]
                    ),
                    "y": float(
                        refined_position[1]
                    ),
                    "x": float(
                        refined_position[2]
                    ),
                }

                nodes_at_time.setdefault(
                    middle_t,
                    [],
                ).append(
                    middle_node_id
                )

                in_degree[
                    middle_node_id
                ] = 0

                out_degree[
                    middle_node_id
                ] = 0

                synthetic_added += 1

            first_edge = (
                int(source_id),
                int(middle_node_id),
            )

            second_edge = (
                int(middle_node_id),
                int(target_id),
            )

            edges.add(
                first_edge
            )

            edges.add(
                second_edge
            )

            out_degree[source_id] += 1
            in_degree[middle_node_id] += 1

            out_degree[middle_node_id] += 1
            in_degree[target_id] += 1

            gap_pairs_added += 1

    # --------------------------------------------------------
    # SHORT-TRACK FILTERING
    # --------------------------------------------------------

    in_degree, out_degree = build_degrees(
        nodes.keys(),
        edges,
    )

    components = find_components(
        nodes.keys(),
        edges,
    )

    keep_node_ids = set()

    removed_components = 0

    for component in components:
        component_times = [
            int(nodes[node_id]["t"])
            for node_id in component
        ]

        component_min_t = min(
            component_times
        )

        component_max_t = max(
            component_times
        )

        contains_division = any(
            out_degree.get(
                node_id,
                0,
            ) >= 2
            for node_id in component
        )

        touches_boundary = (
            component_min_t
            <= BOUNDARY_MARGIN_T
            or component_max_t
            >= T - 1 - BOUNDARY_MARGIN_T
        )

        keep_component = (
            len(component)
            >= MIN_TRACK_LEN
            or (
                KEEP_DIVISION_COMPONENTS
                and contains_division
            )
            or (
                KEEP_BOUNDARY_COMPONENTS
                and touches_boundary
            )
        )

        if keep_component:
            keep_node_ids.update(
                component
            )

        else:
            removed_components += 1

    # Защита от случайного удаления всего графа.
    if not keep_node_ids:
        keep_node_ids = set(
            nodes.keys()
        )

        removed_components = 0

    nodes_before_filter = len(
        nodes
    )

    edges_before_filter = len(
        edges
    )

    nodes = {
        node_id: node
        for node_id, node in nodes.items()
        if node_id in keep_node_ids
    }

    edges = {
        (
            source_id,
            target_id,
        )
        for source_id, target_id
        in edges
        if (
            source_id in keep_node_ids
            and target_id in keep_node_ids
        )
    }

    removed_nodes = (
        nodes_before_filter
        - len(nodes)
    )

    removed_edges = (
        edges_before_filter
        - len(edges)
    )

    # --------------------------------------------------------
    # Формируем DataFrame чистого postprocessed graph
    # --------------------------------------------------------

    output_rows = []

    for node_id in sorted(
        nodes
    ):
        node = nodes[node_id]

        output_rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(node_id),
                "t": int(node["t"]),
                "z": int(
                    np.clip(
                        round(node["z"]),
                        0,
                        Z - 1,
                    )
                ),
                "y": int(
                    np.clip(
                        round(node["y"]),
                        0,
                        Y - 1,
                    )
                ),
                "x": int(
                    np.clip(
                        round(node["x"]),
                        0,
                        X - 1,
                    )
                ),
                "source_id": -1,
                "target_id": -1,
            }
        )

    for source_id, target_id in sorted(
        edges
    ):
        output_rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": int(source_id),
                "target_id": int(target_id),
            }
        )

    output_frame = pd.DataFrame(
        output_rows,
        columns=SUBMISSION_COLUMN,
    )

    stats = {
        "input_nodes": len(
            node_frame
        ),
        "input_edges": len(
            edge_frame
        ),
        "gap_pairs": gap_pairs_added,
        "gap_reused": reused_existing,
        "gap_synthetic": synthetic_added,
        "short_components_removed": (
            removed_components
        ),
        "short_nodes_removed": (
            removed_nodes
        ),
        "short_edges_removed": (
            removed_edges
        ),
        "trajectory_smoothed_nodes": smoothed_nodes,
        "output_nodes": len(nodes),
        "output_edges": len(edges),
    }

    return output_frame, stats


# ============================================================
# Запускаем postprocessing для каждого dataset
# ============================================================

raw_submission = pd.read_csv(
    RAW_SUBMISSION_PATH
)

processed_groups = []

for dataset in raw_submission[
    "dataset"
].drop_duplicates():

    dataset_group = raw_submission[
        raw_submission["dataset"]
        == dataset
    ].copy()

    processed_group, stats = (
        postprocess_one_dataset(
            dataset_group
        )
    )

    processed_groups.append(
        processed_group
    )

    print(
        dataset,
        stats,
    )

clean_submission = pd.concat(
    processed_groups,
    ignore_index=True,
)

clean_submission["id"] = np.arange(
    len(clean_submission),
    dtype=np.int64,
)

numeric_columns = [
    column
    for column in SUBMISSION_COLUMN
    if column
    not in (
        "dataset",
        "row_type",
    )
]

clean_submission[
    numeric_columns
] = clean_submission[
    numeric_columns
].astype(
    np.int64
)

clean_submission.to_csv(
    CLEAN_SUBMISSION_PATH,
    index=False,
)

print(
    f"Hybrid clean submission saved: "
    f"{RAW_SUBMISSION_PATH} "
    f"({len(raw_submission)} rows) "
    f"-> {CLEAN_SUBMISSION_PATH} "
    f"({len(clean_submission)} rows)"
)

# --- cell 6 ---
from pathlib import Path
import rustworkx as rx

CLEAN_SUBMISSION_PATH = "submission_clean.csv"
MAX_COMPONENTS = 3400
FORKS = 32 # 1. 20 ->30->40 
#2 32 Forks 2600 components -> 32 Forks 3400 components -> Хочу 32 Forks 3800

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


submission = pd.read_csv(
    CLEAN_SUBMISSION_PATH
)

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

# --- cell 7 ---
if MODE=="local":
    
    metric_df =[]
    for sample_id in valid_id:
        truth_file = f"{KAGGLE_DIR}/train/{sample_id}.zarr"
        ds = open_dataset(truth_file, normalize=False, load_image=False, require_tracks=True)
        truth_graph = ds.tracks
    
        predict_file = f"{predict_dir}/{sample_id}.geff"
        pred_result = td.graph.IndexedRXGraph.from_geff(predict_file)
        pred_graph = pred_result[0]
    
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



