'''
Biohub Adaptive Temporal Accord

Research 3D lineage reconstruction with dual temporal models,
edge-feature test-time augmentation with a three-quarter-strength secondary feature blend and consensus association.

Research edition.
'''


import os
BIOHUB_PRESET = 'harmonic_v3_division_wide'
BIOHUB_SCORE_AXIS = 'public 0.939 base + holdout-selected post-process configuration'

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = '1.0'
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "2"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.0"
os.environ["BIOHUB_GAP_DENSITY_ADAPTIVE"] = "1"
os.environ["BIOHUB_GAP_DENSITY_REFERENCE_UM"] = "6.5"
os.environ["BIOHUB_GAP_DENSITY_GAIN"] = "0.040"
os.environ["BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM"] = "0.125"
os.environ["BIOHUB_GAP_DENSITY_NEIGHBORS"] = "3"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "9.0"  


os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "14.0"  



os.environ["BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU"] = "0.6"  
os.environ["BIOHUB_SAFE_DIV_DIVERGE_UM"] = "2.25"  


os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "10.0"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.2"     
os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.88"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.0"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.012"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "120"
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_EXPECTED_EPOCH"] = "2"
os.environ["BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM"] = "8.5"
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_THRESHOLD"] = "0.25"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"
os.environ["BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT"] = "0.15"
os.environ["BIOHUB_BIDIRECTIONAL_FUSION_MODE"] = "harmonic_probability"
os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
os.environ["BIOHUB_DIAGNOSTIC_ARM"] = "harmonic_association_production"
os.environ["BIOHUB_VALIDATOR_N_PER_TYPE"] = "4"
os.environ["BIOHUB_PPSWEEP_SELECT_MARGIN"] = "0.001"
os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"

os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"
os.environ["BIOHUB_DEEPCENTER_TTA"] = "1"
os.environ["BIOHUB_HOCT_VETO"] = "2"
print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)

import json as _guard_json
import math as _guard_math
import os as _guard_os

_EXPECTED_NUMERIC = {
    "BIOHUB_DET_THRESHOLD": 0.965,
    "BIOHUB_ILP_APPEARANCE_WEIGHT": 0.0,
    "BIOHUB_ILP_DISAPPEARANCE_WEIGHT": 2,
    "BIOHUB_GAP_CLOSE_UM": 5.0,
    "BIOHUB_OUTPUT_MIN_TRACK_LEN": 6.0,
    "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT": 0.15,
}

_EXPECTED_TEXT = {
    "BIOHUB_BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
    "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION": "0.90",
}

_drift = {}
for _key, _want in _EXPECTED_NUMERIC.items():
    _raw = _guard_os.environ.get(_key)
    if _raw is None:
        _drift[_key] = "missing"
        continue
    _got = float(_raw)
    if not _guard_math.isclose(_got, _want, rel_tol=0.0, abs_tol=1e-12):
        _drift[_key] = {"expected": _want, "actual": _got}

for _key, _want in _EXPECTED_TEXT.items():
    _got = _guard_os.environ.get(_key)
    if _got != _want:
        _drift[_key] = {"expected": _want, "actual": _got}

if _drift:
    raise RuntimeError(
        "Configuration drift detected: " + _guard_json.dumps(_drift, sort_keys=True)
    )

print("Configuration guard: PASS")
print("Baseline: fixed-90 dual-seed clean pipeline (public LB 0.913)")
print("Single model-level change: harmonic mutual-support association fusion")
print("Reverse-time association weight: 0.200")

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
import sys
import time
from pathlib import Path

import pandas as pd
from IPython.display import display

COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((path for path in COMP_DIR_CANDIDATES if path.exists()), COMP_DIR_CANDIDATES[0])

TEST_DIR = COMP_DIR / "test"

WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
REPO_DIR = WORKING_DIR / "tracking_repo"
SUBMISSION_PATH = WORKING_DIR / "submission.csv"
RUN_STATS_PATH = WORKING_DIR / "run_stats.csv"

METHOD = "unet_transformer"
WEIGHTS_RELATIVE = f"weights/{METHOD}/split_0/edge_predictor_best.pth"
EXPERIMENT_TAG = "selected_101_dual_seed_near_balanced_center_confirmed_synthetic_gap"
TARGET_ARTIFACT_SLUG = os.environ.get("BIOHUB_TARGET_ARTIFACT_SLUG", "biohub-tracking-support-pack-50ep-v1")
PRIMARY_ARTIFACT_MANIFEST = Path(os.environ.get(
    "BIOHUB_PRIMARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1/ARTIFACT_MANIFEST.json",
))
ALLOW_ARTIFACT_FALLBACK = os.environ.get("BIOHUB_ALLOW_ARTIFACT_FALLBACK", "0") != "0"

DET_THRESHOLD = float(os.environ.get("BIOHUB_DET_THRESHOLD", "0.99"))
UNET_BATCH_SIZE = int(os.environ.get("BIOHUB_UNET_BATCH_SIZE", "4"))
USE_ILP = os.environ.get("BIOHUB_USE_ILP", "1") != "0"
ILP_EDGE_WEIGHT = float(os.environ.get("BIOHUB_ILP_EDGE_WEIGHT", "-1.0"))
ILP_APPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_APPEARANCE_WEIGHT", "0.1"))
ILP_DISAPPEARANCE_WEIGHT = float(os.environ.get("BIOHUB_ILP_DISAPPEARANCE_WEIGHT", "0.1"))
ILP_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_ILP_DIVISION_WEIGHT", "1.0"))


SLICE = ""



ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"
RUN_OUTPUT_DIAGNOSTICS = os.environ.get("BIOHUB_RUN_OUTPUT_DIAGNOSTICS", "1") != "0"


OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))
OUTPUT_ENFORCE_NEXT_FRAME = os.environ.get("BIOHUB_OUTPUT_ENFORCE_NEXT_FRAME", "1") != "0"
OUTPUT_SINGLE_PARENT_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR", "1") != "0"
OUTPUT_SINGLE_CHILD_REPAIR = os.environ.get("BIOHUB_OUTPUT_SINGLE_CHILD_REPAIR", "0") != "0"
OUTPUT_PRUNE_ISOLATED = os.environ.get("BIOHUB_OUTPUT_PRUNE_ISOLATED", "1") != "0"
OUTPUT_MOTION_RELINK = os.environ.get("BIOHUB_OUTPUT_MOTION_RELINK", "1") != "0"
MOTION_RELINK_TIGHT_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_TIGHT_UM", "6.0"))
MOTION_RELINK_RELAXED_UM = float(os.environ.get("BIOHUB_MOTION_RELINK_RELAXED_UM", "10.0"))
MOTION_RELINK_VELOCITY_WEIGHT = float(os.environ.get("BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT", "0.5"))
MOTION_RELINK_LEARNED_BONUS = float(os.environ.get("BIOHUB_MOTION_RELINK_LEARNED_BONUS", "0.75"))
MOTION_RELINK_MAX_FRAME_NODES = int(os.environ.get("BIOHUB_MOTION_RELINK_MAX_FRAME_NODES", "2600"))

OUTPUT_DIVISION_GEOMETRY_FILTER = os.environ.get("BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER", "0") != "0"
DIV_PARENT_MAX_UM = float(os.environ.get("BIOHUB_DIV_PARENT_MAX_UM", "10.5"))
DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_DIV_SISTER_MAX_UM", "8.0"))
DIV_DROP_TO_SINGLE_IF_BAD = os.environ.get("BIOHUB_DIV_DROP_TO_SINGLE_IF_BAD", "1") != "0"
OUTPUT_GAP_CLOSE = os.environ.get("BIOHUB_OUTPUT_GAP_CLOSE", "1") != "0"
GAP_CLOSE_MAX_GAP = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_GAP", "1"))
GAP_CLOSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0"))
GAP_DENSITY_ADAPTIVE = os.environ.get("BIOHUB_GAP_DENSITY_ADAPTIVE", "0") != "0"
GAP_DENSITY_REFERENCE_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_REFERENCE_UM", "6.5"))
GAP_DENSITY_GAIN = float(os.environ.get("BIOHUB_GAP_DENSITY_GAIN", "0.040"))
GAP_DENSITY_MAX_STEP_DELTA_UM = float(os.environ.get("BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM", "0.125"))
GAP_DENSITY_NEIGHBORS = int(os.environ.get("BIOHUB_GAP_DENSITY_NEIGHBORS", "3"))
GAP_CLOSE_REUSE_EXISTING = os.environ.get("BIOHUB_GAP_CLOSE_REUSE_EXISTING", "1") != "0"
GAP_CLOSE_REUSE_UM = float(os.environ.get("BIOHUB_GAP_CLOSE_REUSE_UM", "3.2"))
GAP_CLOSE_MAX_ADDED_FRAC = float(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_FRAC", "0.05"))
GAP_CLOSE_MAX_ADDED_ABS = int(os.environ.get("BIOHUB_GAP_CLOSE_MAX_ADDED_ABS", "2000"))
GAP_REFINE_SYNTHETIC = os.environ.get("BIOHUB_GAP_REFINE_SYNTHETIC", "1") != "0"
GAP_REFINE_WIN_Z = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_Z", "1"))
GAP_REFINE_WIN_YX = int(os.environ.get("BIOHUB_GAP_REFINE_WIN_YX", "3"))
GAP_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_GAP_REFINE_MAX_SHIFT_UM", "3.2"))

OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "6"))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"
ADAPTIVE_SHORT_TRACK_RESCUE = os.environ.get("BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE", "0") != "0"
SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC", "0.10"))
SHORT_TRACK_RESCUE_MIN_LEN = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN", "4"))
SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "0.82"))
SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM", "3.25"))
SHORT_TRACK_RESCUE_MAX_NODES_FRAC = float(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC", "0.018"))
SHORT_TRACK_RESCUE_MAX_NODES_ABS = int(os.environ.get("BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS", "180"))

OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))

OUTPUT_GAP2_RECOVERY = os.environ.get("BIOHUB_OUTPUT_GAP2_RECOVERY", "0") != "0"
GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_GAP2_MAX_TOTAL_UM", "10.2"))
GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_GAP2_MAX_STEP_UM", "4.4"))
GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_GAP2_MAX_LINKS_FRAC", "0.0045"))
GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_GAP2_MAX_LINKS_ABS", "180"))
GAP2_REQUIRE_CONTEXT = os.environ.get("BIOHUB_GAP2_REQUIRE_CONTEXT", "1") != "0"
GAP2_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_GAP2_FRAME_FRAC_CAP", "0.006"))

OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_SISTER_SYMMETRY_TAU = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU", "0.0"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))


SAFE_DIV_DIVERGE_UM = float(os.environ.get("BIOHUB_SAFE_DIV_DIVERGE_UM", "2.25"))
SAFE_DIV_REQUIRE_DIVERGENCE = os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_DIVERGENCE", "1") != "0"
SAFE_DIV_REQUIRE_MUTUAL_NN = os.environ.get("BIOHUB_SAFE_DIV_REQUIRE_MUTUAL_NN", "1") != "0"


USE_DEEPCENTER_VETO = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "1") != "0"
REQUIRE_DEEPCENTER_VETO = os.environ.get("BIOHUB_REQUIRE_DEEPCENTER_VETO", "1") != "0"
DEEPCENTER_MANIFEST_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_MANIFEST_DEFAULT",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/ARTIFACT_MANIFEST.json",
)
DEEPCENTER_CHECKPOINT_DEFAULT = os.environ.get(
    "BIOHUB_DEEPCENTER_CHECKPOINT_DEFAULT",
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/full_frame_center/best.pt",
)
DEEPCENTER_RELATIVE = os.environ.get("BIOHUB_DEEPCENTER_RELATIVE", "weights/full_frame_center/best.pt")
DEEPCENTER_GAP_VETO = os.environ.get("BIOHUB_DEEPCENTER_GAP_VETO", "1") != "0"
DEEPCENTER_SAFE_DIV_VETO = os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_VETO", "1") != "0"
DEEPCENTER_GAP_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_THRESHOLD", "0.10"))
DEEPCENTER_EXPECTED_EPOCH = int(os.environ.get("BIOHUB_DEEPCENTER_EXPECTED_EPOCH", "0"))
DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM = float(os.environ.get("BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM", "0"))
DEEPCENTER_SAFE_DIV_THRESHOLD = float(os.environ.get("BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD", "0.12"))
DEEPCENTER_SCORE_WIN_Z = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_Z", "1"))
DEEPCENTER_SCORE_WIN_YX = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_WIN_YX", "2"))
DEEPCENTER_SCORE_CACHE_MAX_FRAMES = int(os.environ.get("BIOHUB_DEEPCENTER_SCORE_CACHE_MAX_FRAMES", "8"))

CONFIG_DISPLAY = {
    "experiment_tag": EXPERIMENT_TAG,
    "method": METHOD,
    "weights": WEIGHTS_RELATIVE,
    "target_artifact_slug": TARGET_ARTIFACT_SLUG,
    "primary_artifact_manifest": str(PRIMARY_ARTIFACT_MANIFEST),
    "allow_artifact_fallback": ALLOW_ARTIFACT_FALLBACK,
    "det_threshold": DET_THRESHOLD,
    "unet_batch_size": UNET_BATCH_SIZE,
    "use_ilp": USE_ILP,
    "ilp_edge_weight": ILP_EDGE_WEIGHT,
    "ilp_appearance_weight": ILP_APPEARANCE_WEIGHT,
    "ilp_disappearance_weight": ILP_DISAPPEARANCE_WEIGHT,
    "ilp_division_weight": ILP_DIVISION_WEIGHT,
    "slice": SLICE,
    "allow_pip_install": ALLOW_PIP_INSTALL,
    "output_edge_max_um": OUTPUT_EDGE_MAX_UM,
    "output_enforce_next_frame": OUTPUT_ENFORCE_NEXT_FRAME,
    "output_single_parent_repair": OUTPUT_SINGLE_PARENT_REPAIR,
    "output_single_child_repair": OUTPUT_SINGLE_CHILD_REPAIR,
    "output_prune_isolated": OUTPUT_PRUNE_ISOLATED,
    "output_motion_relink": OUTPUT_MOTION_RELINK,
    "motion_relink_tight_um": MOTION_RELINK_TIGHT_UM,
    "motion_relink_relaxed_um": MOTION_RELINK_RELAXED_UM,
    "motion_relink_velocity_weight": MOTION_RELINK_VELOCITY_WEIGHT,
    "motion_relink_learned_bonus": MOTION_RELINK_LEARNED_BONUS,
    "motion_relink_max_frame_nodes": MOTION_RELINK_MAX_FRAME_NODES,
    "output_division_geometry_filter": OUTPUT_DIVISION_GEOMETRY_FILTER,
    "div_parent_max_um": DIV_PARENT_MAX_UM,
    "div_sister_max_um": DIV_SISTER_MAX_UM,
    "div_drop_to_single_if_bad": DIV_DROP_TO_SINGLE_IF_BAD,
    "output_gap_close": OUTPUT_GAP_CLOSE,
    "gap_close_max_gap": GAP_CLOSE_MAX_GAP,
    "gap_close_effective_max_gap": min(GAP_CLOSE_MAX_GAP, 1),
    "gap_close_um": GAP_CLOSE_UM,
    "gap_density_adaptive": GAP_DENSITY_ADAPTIVE,
    "gap_density_reference_um": GAP_DENSITY_REFERENCE_UM,
    "gap_density_gain": GAP_DENSITY_GAIN,
    "gap_density_max_step_delta_um": GAP_DENSITY_MAX_STEP_DELTA_UM,
    "gap_density_neighbors": GAP_DENSITY_NEIGHBORS,
    "gap_close_reuse_existing": GAP_CLOSE_REUSE_EXISTING,
    "gap_close_reuse_um": GAP_CLOSE_REUSE_UM,
    "gap_close_max_added_frac": GAP_CLOSE_MAX_ADDED_FRAC,
    "gap_close_max_added_abs": GAP_CLOSE_MAX_ADDED_ABS,
    "gap_refine_synthetic": GAP_REFINE_SYNTHETIC,
    "gap_refine_win_z": GAP_REFINE_WIN_Z,
    "gap_refine_win_yx": GAP_REFINE_WIN_YX,
    "gap_refine_max_shift_um": GAP_REFINE_MAX_SHIFT_UM,
    "output_filter_short_tracks": OUTPUT_FILTER_SHORT_TRACKS,
    "output_min_track_len": OUTPUT_MIN_TRACK_LEN,
    "output_keep_division_components": OUTPUT_KEEP_DIVISION_COMPONENTS,
    "adaptive_short_track_rescue": ADAPTIVE_SHORT_TRACK_RESCUE,
    "short_track_rescue_trigger_removed_frac": SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC,
    "short_track_rescue_min_len": SHORT_TRACK_RESCUE_MIN_LEN,
    "short_track_rescue_min_mean_edge_prob": SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB,
    "short_track_rescue_max_mean_edge_dist_um": SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM,
    "short_track_rescue_max_nodes_frac": SHORT_TRACK_RESCUE_MAX_NODES_FRAC,
    "short_track_rescue_max_nodes_abs": SHORT_TRACK_RESCUE_MAX_NODES_ABS,
    "output_linefit_smooth": OUTPUT_LINEFIT_SMOOTH,
    "output_linefit_weight": OUTPUT_LINEFIT_WEIGHT,
    "output_linefit_window": OUTPUT_LINEFIT_WINDOW,
    "output_gap2_recovery": OUTPUT_GAP2_RECOVERY,
    "gap2_max_total_um": GAP2_MAX_TOTAL_UM,
    "gap2_max_step_um": GAP2_MAX_STEP_UM,
    "gap2_max_links_frac": GAP2_MAX_LINKS_FRAC,
    "gap2_max_links_abs": GAP2_MAX_LINKS_ABS,
    "gap2_require_context": GAP2_REQUIRE_CONTEXT,
    "gap2_frame_frac_cap": GAP2_FRAME_FRAC_CAP,
    "output_safe_divisions": OUTPUT_SAFE_DIVISIONS,
    "safe_div_max_um": SAFE_DIV_MAX_UM,
    "safe_div_sister_max_um": SAFE_DIV_SISTER_MAX_UM,
    "safe_div_existing_child_max_um": SAFE_DIV_EXISTING_CHILD_MAX_UM,
    "safe_div_frame_frac_cap": SAFE_DIV_FRAME_FRAC_CAP,
    "safe_div_global_frac_cap": SAFE_DIV_GLOBAL_FRAC_CAP,
    "use_deepcenter_add_only_gate": USE_DEEPCENTER_VETO,
    "deepcenter_gap_add_gate": DEEPCENTER_GAP_VETO,
    "deepcenter_safe_div_add_gate": DEEPCENTER_SAFE_DIV_VETO,
    "deepcenter_gap_threshold": DEEPCENTER_GAP_THRESHOLD,
    "deepcenter_expected_epoch": DEEPCENTER_EXPECTED_EPOCH,
    "deepcenter_gap_confirm_min_span_um": DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM,
    "deepcenter_safe_div_threshold": DEEPCENTER_SAFE_DIV_THRESHOLD,
    "deepcenter_checkpoint_default": DEEPCENTER_CHECKPOINT_DEFAULT,
}

print("Biohub learned UNet + node-transformer + ILP submission")
print("COMP_DIR:", COMP_DIR, "exists:", COMP_DIR.exists())
print("TEST_DIR:", TEST_DIR, "exists:", TEST_DIR.exists())
print(json.dumps(CONFIG_DISPLAY, indent=2, sort_keys=True))
import re

os.environ.setdefault("POLARS_PREFER_PKG", "32")

PACKAGE_SPECS = {
    "tracksdata": ("tracksdata", "tracksdata"),
    "zarr": ("zarr", "zarr>=3.0.10,<4"),
    "pyscipopt": ("pyscipopt", "pyscipopt"),
    "geff": ("geff", "geff>=1.1.3.1.1"),
    "geff_spec": ("geff_spec", "geff-spec<1.2"),
    "ilpy": ("ilpy", "ilpy>=0.5.1"),
    "polars": ("polars", "polars>=1.36"),
    "blosc2": ("blosc2", "blosc2"),
    "dask": ("dask", "dask"),
    "imagecodecs": ("imagecodecs", "imagecodecs"),
    "skimage": ("skimage", "scikit-image>=0.24"),
    "pyarrow": ("pyarrow", "pyarrow"),
    "rustworkx": ("rustworkx", "rustworkx>=0.17.1"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy>=2"),
    "numcodecs": ("numcodecs", "numcodecs>=0.13,<0.16"),
    "donfig": ("donfig", "donfig>=0.8"),
    "google_crc32c": ("google_crc32c", "google-crc32c>=1.5"),
    "bidict": ("bidict", "bidict>=0.23.1"),
    "psygnal": ("psygnal", "psygnal>=0.14"),
    "rich": ("rich", "rich"),
    "networkx": ("networkx", "networkx>=3.2.1"),
    "pydantic": ("pydantic", "pydantic>=2.11"),
    "pydantic_core": ("pydantic_core", "pydantic-core"),
    "annotated_types": ("annotated_types", "annotated-types"),
    "typing_extensions": ("typing_extensions", "typing-extensions>=4.13"),
    "typing_inspection": ("typing_inspection", "typing-inspection"),
    "markdown_it": ("markdown_it", "markdown-it-py"),
    "pygments": ("pygments", "pygments"),
    "click": ("click", "click"),
    "cloudpickle": ("cloudpickle", "cloudpickle"),
    "fsspec": ("fsspec", "fsspec"),
    "partd": ("partd", "partd"),
    "locket": ("locket", "locket"),
    "toolz": ("toolz", "toolz"),
    "yaml": ("yaml", "pyyaml"),
    "ndindex": ("ndindex", "ndindex"),
    "msgpack": ("msgpack", "msgpack"),
    "numexpr": ("numexpr", "numexpr"),
    "deprecated": ("deprecated", "deprecated"),
    "wrapt": ("wrapt", "wrapt"),
    "imageio": ("imageio", "imageio"),
    "PIL": ("PIL", "pillow"),
    "tifffile": ("tifffile", "tifffile"),
    "lazy_loader": ("lazy_loader", "lazy-loader"),
    "tqdm": ("tqdm", "tqdm"),
}
EXTRA_SPECS_BY_NAME = {
    "tracksdata": ["bidict>=0.23.1", "psygnal>=0.14", "rich"],
    "zarr": ["donfig>=0.8", "google-crc32c>=1.5", "numcodecs>=0.13,<0.16"],
    "geff": ["geff-spec<1.2", "networkx>=3.2.1", "pydantic>=2.11", "numcodecs>=0.13,<0.16"],
    "geff_spec": ["pydantic>=2.11", "annotated-types", "pydantic-core", "typing-inspection"],
    "polars": ["polars-runtime-32"],
    "dask": ["click", "cloudpickle", "fsspec", "partd", "pyyaml", "toolz"],
    "partd": ["locket"],
    "blosc2": ["ndindex", "msgpack", "numexpr"],
    "numcodecs": ["deprecated", "msgpack", "wrapt"],
    "rich": ["markdown-it-py", "pygments"],
    "pydantic": ["annotated-types", "pydantic-core", "typing-extensions>=4.13", "typing-inspection"],
    "skimage": ["imageio", "pillow", "tifffile", "lazy-loader", "networkx"],
}
PIP_DEPENDENCIES = [spec for _, spec in PACKAGE_SPECS.values()]
REQUIRED_MODULES = {name: module for name, (module, _) in PACKAGE_SPECS.items() if module}
FALLBACK_ARTIFACT_SLUGS = ["biohub-tracking-support-pack-v1"]



ALLOW_PIP_INSTALL = os.environ.get("BIOHUB_ALLOW_PIP_INSTALL", "0") != "0"


def module_missing(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is None


def has_model_artifact(path: Path) -> bool:
    has_repo_dir = (path / "repo").exists()
    has_weights_dir = (path / "weights" / METHOD / "split_0" / "edge_predictor_best.pth").exists()
    has_repo_zip = (path / "repo.zip").exists()
    has_weights_zip = (path / "weights.zip").exists()
    return (has_repo_dir and has_weights_dir) or (has_repo_zip and has_weights_zip)


def artifact_manifest(path: Path) -> dict:
    manifest = path / "ARTIFACT_MANIFEST.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text())
    except Exception:
        return {}


def artifact_matches_target(path: Path) -> bool:
    if ALLOW_ARTIFACT_FALLBACK:
        return True
    manifest = artifact_manifest(path)
    artifact_name = str(manifest.get("artifact_name", ""))
    path_text = str(path)
    return TARGET_ARTIFACT_SLUG in {artifact_name, path.name} or TARGET_ARTIFACT_SLUG in path_text


def candidate_roots_for_slug(slug: str) -> list[Path]:
    return [
        Path(f"/kaggle/input/datasets/pilkwang/{slug}"),
        Path(f"/kaggle/input/{slug}"),
        Path(f"/kaggle/input/{slug}/{slug}"),
        Path(f"PublicNotebook/{slug}"),
    ]


def find_artifacts_root() -> Path:
    candidates: list[Path] = []
    for env_name in ["BIOHUB_MODEL_ARTIFACTS", "BIOHUB_ARTIFACTS"]:
        explicit = os.environ.get(env_name, "").strip()
        if explicit:
            candidates.append(Path(explicit))

    candidates.append(PRIMARY_ARTIFACT_MANIFEST.parent)
    candidates.extend(candidate_roots_for_slug(TARGET_ARTIFACT_SLUG))

    if ALLOW_ARTIFACT_FALLBACK:
        for slug in FALLBACK_ARTIFACT_SLUGS:
            candidates.extend(candidate_roots_for_slug(slug))

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if not child.is_dir():
                continue
            child_text = str(child)
            if TARGET_ARTIFACT_SLUG in child_text or ALLOW_ARTIFACT_FALLBACK:
                candidates.append(child)
                candidates.append(child / child.name)
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.append(grandchild)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_model_artifact(candidate) and artifact_matches_target(candidate):
            return candidate
    checked = "\n".join(str(path) for path in candidates[:80])
    raise FileNotFoundError(
        "Could not find the required model artifact. "
        f"Expected slug: {TARGET_ARTIFACT_SLUG}\n"
        "Attach the newly uploaded support dataset, or set BIOHUB_MODEL_ARTIFACTS.\n"
        "To debug with an older artifact, set BIOHUB_ALLOW_ARTIFACT_FALLBACK=1.\n"
        "Checked:\n" + checked
    )


def _has_package_file(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return any(any(path.glob(pattern)) for pattern in patterns)


def find_offline_package_dirs(artifacts: Path) -> list[Path]:
    candidates: list[Path] = [
        artifacts / "wheels",
        artifacts,
        Path("/kaggle/working"),
        Path("/kaggle/working/wheels"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for child in input_root.iterdir():
            if child.is_dir():
                candidates.extend([child / "wheels", child])
                for grandchild in child.iterdir():
                    if grandchild.is_dir():
                        candidates.extend([grandchild / "wheels", grandchild])

    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen:
            continue
        seen.add(candidate)
        if _has_package_file(candidate):
            out.append(candidate)
    return out


def purge_imported_modules(package_names: list[str]) -> None:
    roots = {"tracksdata"}
    for name in package_names:
        if name in PACKAGE_SPECS:
            module = PACKAGE_SPECS[name][0]
            roots.add(module.split(".")[0])
        if name == "polars":
            roots.add("polars")
    for root in roots:
        for module_name in list(sys.modules):
            if module_name == root or module_name.startswith(root + "."):
                sys.modules.pop(module_name, None)


def polars_runtime_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries

        _ = _PySeries
        return hasattr(_pl, "Float16") and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


def packages_requiring_refresh() -> list[str]:
    refresh: list[str] = []
    if not module_missing("polars") and not polars_runtime_ready():
        refresh.append("polars")

    if not module_missing("zarr"):
        try:
            import zarr as _zarr
            version_text = str(getattr(_zarr, "__version__", "0"))
            major = int(version_text.split(".", 1)[0])
            if major < 3:
                refresh.append("zarr")
        except Exception:
            refresh.append("zarr")
    return refresh


def dependency_specs_for(missing: list[str]) -> list[str]:
    specs: list[str] = []
    seen: set[str] = set()

    def add(spec: str) -> None:
        key = spec.lower()
        if key not in seen:
            seen.add(key)
            specs.append(spec)

    for name in missing:
        if name in PACKAGE_SPECS:
            add(PACKAGE_SPECS[name][1])
        for spec in EXTRA_SPECS_BY_NAME.get(name, []):
            add(spec)
    return specs


def import_failures() -> dict[str, str]:
    failures: dict[str, str] = {}
    for name, module_name in REQUIRED_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return failures


def missing_names_from_failures(failures: dict[str, str]) -> list[str]:
    names: list[str] = []
    module_to_name = {module: name for name, module in REQUIRED_MODULES.items()}
    for message in failures.values():
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", message)
        if match:
            module = match.group(1).split(".")[0]
        else:
            match = re.search(r"module ['\"]([^'\"]+)['\"] has no attribute", message)
            if not match:
                continue
            module = match.group(1).split(".")[0]
        name = module_to_name.get(module)
        if name and name not in names:
            names.append(name)
    return names


def install_missing_dependencies(missing: list[str], artifacts: Path) -> None:
    specs = dependency_specs_for(missing)
    force_reinstall = bool({"polars", "zarr"} & set(missing))
    if not specs:
        return

    package_dirs = find_offline_package_dirs(artifacts)
    if package_dirs:
        offline_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
        if force_reinstall:
            offline_cmd.append("--force-reinstall")
        for package_dir in package_dirs:
            offline_cmd.extend(["--find-links", str(package_dir)])
        offline_cmd.extend(specs)
        print("Installing missing packages from offline package dirs:", missing)
        print("Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel.")
        print("Offline package dirs:", [str(path) for path in package_dirs])
        result = subprocess.run(offline_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("Offline dependency install succeeded.")
            return
        print("Offline dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    if ALLOW_PIP_INSTALL:
        online_cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
        if force_reinstall:
            online_cmd.append("--force-reinstall")
        online_cmd.extend(specs)
        print("Installing missing packages from PyPI:", missing)
        result = subprocess.run(online_cmd, text=True, capture_output=True)
        if result.returncode == 0:
            purge_imported_modules(missing)
            print("PyPI dependency install succeeded.")
            return
        print("PyPI dependency install failed. Last pip output:")
        print((result.stdout or "")[-2000:])
        print((result.stderr or "")[-2000:])

    command = "pip install tracksdata zarr>=3.0.10,<4 pyscipopt geff geff-spec ilpy polars blosc2 dask imagecodecs pyarrow rustworkx sqlalchemy donfig numcodecs"
    raise ImportError(
        "Missing required packages or dependency wheels: " + ", ".join(missing) + "\n"
        "Attach the support dataset with offline wheels. If supplying Kaggle dependency input instead, use:\n"
        + command + "\n"
        "Do not quote zarr>=3.0.10,<4 in Kaggle dependency input."
    )


def ensure_dependencies(artifacts: Path) -> None:
    for _ in range(5):
        refresh = packages_requiring_refresh()
        if refresh:
            install_missing_dependencies(refresh, artifacts)
            continue

        missing = [pkg for pkg, module in REQUIRED_MODULES.items() if module_missing(module)]
        if missing:
            install_missing_dependencies(missing, artifacts)
            continue

        failures = import_failures()
        if not failures:
            print("Required graph/Zarr/ILP packages import successfully.")
            return

        missing_from_import = missing_names_from_failures(failures)
        if missing_from_import:
            install_missing_dependencies(missing_from_import, artifacts)
            continue

        raise ImportError(
            "Required packages are present but failed to import. "
            "This may indicate a binary dependency mismatch in the live notebook kernel. "
            "Keep Kaggle dependency input empty and attach the wheels artifact.\n"
            + json.dumps(failures, indent=2)
        )

    failures = import_failures()
    raise ImportError(
        "Dependency recovery did not converge after repeated offline installs. "
        "The attached support artifact may be missing wheels.\n"
        + json.dumps(failures, indent=2)
    )


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_or_extract_tree(src_dir: Path, src_zip: Path, dst: Path) -> None:
    remove_path(dst)
    if src_dir.exists() and src_dir.is_dir():
        shutil.copytree(src_dir, dst)
        return
    if src_zip.exists() and src_zip.is_file():
        dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(dst)
        return
    raise FileNotFoundError(f"Missing source tree or zip: {src_dir} / {src_zip}")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    remove_path(dst)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except Exception:
        shutil.copytree(src, dst)


def materialize_inference_repo(artifacts: Path) -> None:
    copy_or_extract_tree(artifacts / "repo", artifacts / "repo.zip", REPO_DIR)

    weights_src = artifacts / "weights"
    weights_zip = artifacts / "weights.zip"
    weights_dst = REPO_DIR / "weights"
    if weights_src.exists() and weights_src.is_dir():
        link_or_copy_tree(weights_src, weights_dst)
    elif weights_zip.exists() and weights_zip.is_file():
        remove_path(weights_dst)
        weights_dst.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(weights_zip) as zf:
            zf.extractall(weights_dst)
    else:
        raise FileNotFoundError(f"Missing weights tree or zip under {artifacts}")

    required = [
        REPO_DIR / "scripts" / "predict_unet_transformer.py",
        REPO_DIR / WEIGHTS_RELATIVE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Materialized inference repo is incomplete:\n" + "\n".join(missing))
    print("Inference repo:", REPO_DIR)
    print("Weights:", REPO_DIR / WEIGHTS_RELATIVE)


ARTIFACTS = find_artifacts_root()
print("ARTIFACTS:", ARTIFACTS)
print("Has offline wheels:", (ARTIFACTS / "wheels").exists())
manifest_info = artifact_manifest(ARTIFACTS)
if manifest_info:
    print("Artifact name:", manifest_info.get("artifact_name"))
    print("Weight sha256:", manifest_info.get("model", {}).get("weight_sha256"))
    print("Weight path:", manifest_info.get("model", {}).get("weight_path"))
    _expected_primary_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
    _actual_primary_sha256 = str(manifest_info.get("model", {}).get("weight_sha256", ""))
    if _actual_primary_sha256 != _expected_primary_sha256:
        raise RuntimeError(
            "Primary model checksum mismatch: "
            f"expected {_expected_primary_sha256}, got {_actual_primary_sha256 or 'missing'}"
        )

ensure_dependencies(ARTIFACTS)
materialize_inference_repo(ARTIFACTS)




import hashlib as _integrity_hashlib

_support_expected_sha256 = {
    "scripts/augmentations.py": "13db09817bf492f8d0f710a0a4d09776320b262060167055090a303fc6057f4e",
    "scripts/dataspec.py": "e69bf952fb985477ac50ff8598a35020c95d20a035a09b81ab4056e655dd311f",
    "scripts/evaluate.py": "614813cc51c3581c6ccda4bb20725a19da8ecac4a27620654bfca58319cffa3c",
    "scripts/predict_unet_transformer.py": "c44e771ba5980b820f93091e03a303c25dfe8f3232e501f54dc9565731c234b9",
    "scripts/train_unet_transformer.py": "c4f6317736bb3bb1ec8f3f6e9a6d935a463e3f0f1f685481b2d13218d35dc9ea",
    "src/biohub_tracking/__init__.py": "26a18d8da84e40da73281a48ebc3017d847a2e57431ab63e8629d2109e6e8571",
    "src/biohub_tracking/division_metrics.py": "d1cf1e0a43009d02174f1699ce2aa28458a2220ac4b521731d3bcf31cf8c76be",
    "src/biohub_tracking/img_proc.py": "00e8ef0adc8b39f1aaaa547ea6197b906bf9e8c009e339d3e95f8f8dbf31be3f",
    "src/biohub_tracking/io.py": "efae135b088cecaab463d889f16c885ef6da3ad27b0747327d8ddc28d866b7bd",
    "src/biohub_tracking/metrics.py": "31baf45b54c78f68bab4f65dd8f4b38bca702abb644171c6df7c46cdeef55d83",
    "src/biohub_tracking/models/__init__.py": "ab7587ef79856bae50d24b62e5805092d0459ee1c586522b763f9ef70c093e1d",
    "src/biohub_tracking/models/simple_node_transformer.py": "b97209edeb03840e80d903e3e2a8c81c520641c8ef343f6ca2904d0f80db064e",
    "src/biohub_tracking/models/temporal_unet.py": "d809c35d42f504161074ddeaaa7aee5b407e5bca7f9b4e1d5f9b2ff345666cac"
}
_support_expected_manifest_sha256 = "978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029"
_primary_expected_sha256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"
_deepcenter_expected_sha256 = "8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0"  


def _integrity_sha256_file(path: Path) -> str:
    digest = _integrity_hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_support_materialized_paths = {
    path.relative_to(REPO_DIR).as_posix(): path
    for path in REPO_DIR.rglob("*.py")
}
_support_actual_names = set(_support_materialized_paths)
_support_expected_names = set(_support_expected_sha256)
if _support_actual_names != _support_expected_names:
    raise RuntimeError({
        "support_repo_python_files_missing": sorted(
            _support_expected_names - _support_actual_names
        ),
        "support_repo_python_files_extra": sorted(
            _support_actual_names - _support_expected_names
        ),
    })
_support_actual_sha256 = {
    relative: _integrity_sha256_file(_support_materialized_paths[relative])
    for relative in sorted(_support_materialized_paths)
}
if _support_actual_sha256 != _support_expected_sha256:
    raise RuntimeError({
        "support_repo_python_checksum_mismatch": {
            relative: {
                "expected": _support_expected_sha256[relative],
                "actual": _support_actual_sha256[relative],
            }
            for relative in sorted(_support_expected_sha256)
            if _support_actual_sha256[relative]
            != _support_expected_sha256[relative]
        }
    })
_support_manifest_bytes = "".join(
    f"{_support_actual_sha256[relative]}  {relative}\n"
    for relative in sorted(_support_actual_sha256)
).encode("utf-8")
_support_actual_manifest_sha256 = _integrity_hashlib.sha256(
    _support_manifest_bytes
).hexdigest()
if _support_actual_manifest_sha256 != _support_expected_manifest_sha256:
    raise RuntimeError(
        "Support repo manifest checksum mismatch: "
        f"expected {_support_expected_manifest_sha256}, "
        f"got {_support_actual_manifest_sha256}"
    )

_primary_materialized_path = REPO_DIR / WEIGHTS_RELATIVE
_primary_actual_sha256 = _integrity_sha256_file(_primary_materialized_path)
if _primary_actual_sha256 != _primary_expected_sha256:
    raise RuntimeError(
        "Materialized primary model checksum mismatch: "
        f"expected {_primary_expected_sha256}, got {_primary_actual_sha256}"
    )

_deepcenter_candidate_strings = [
    os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", "").strip(),
    "/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1/weights/"
    "full_frame_center/best.pt",
    "/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1/"
    "weights/full_frame_center/best.pt",
]
_deepcenter_candidates = []
for _candidate_string in _deepcenter_candidate_strings:
    if not _candidate_string:
        continue
    _candidate_path = Path(_candidate_string)
    if _candidate_path not in _deepcenter_candidates:
        _deepcenter_candidates.append(_candidate_path)
_deepcenter_materialized_path = next(
    (path for path in _deepcenter_candidates if path.is_file()),
    None,
)
if _deepcenter_materialized_path is None:
    raise FileNotFoundError({
        "missing_deepcenter_checkpoint": [str(path) for path in _deepcenter_candidates]
    })
_deepcenter_actual_sha256 = _integrity_sha256_file(
    _deepcenter_materialized_path
)
if _deepcenter_actual_sha256 != _deepcenter_expected_sha256:
    raise RuntimeError(
        "DeepCenter checkpoint checksum mismatch: "
        f"expected {_deepcenter_expected_sha256}, "
        f"got {_deepcenter_actual_sha256}"
    )
os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(
    _deepcenter_materialized_path
)

print("Support repo Python manifest SHA256:", _support_actual_manifest_sha256)
print("Primary materialized SHA256:", _primary_actual_sha256)
print("DeepCenter materialized SHA256:", _deepcenter_actual_sha256)



import hashlib as _hashlib

_secondary_manifest_explicit = Path(os.environ.get(
    "BIOHUB_SECONDARY_ARTIFACT_MANIFEST",
    "/kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1/ARTIFACT_MANIFEST.json",
))
_secondary_expected_sha256 = "9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f"
_secondary_slug = "biohub-temporal-unet3d-seed314159-v1"


def _find_secondary_artifact_root() -> tuple[Path, dict]:
    candidates = [
        _secondary_manifest_explicit,
        Path(f"/kaggle/input/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
        Path(f"/kaggle/input/datasets/pilkwang/{_secondary_slug}/ARTIFACT_MANIFEST.json"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob("ARTIFACT_MANIFEST.json"))

    seen = set()
    for manifest_path in candidates:
        manifest_path = manifest_path.expanduser()
        if manifest_path in seen or not manifest_path.is_file():
            continue
        seen.add(manifest_path)
        try:
            info = json.loads(manifest_path.read_text())
        except Exception:
            continue
        sha256 = str(info.get("model", {}).get("weight_sha256", ""))
        if sha256 == _secondary_expected_sha256:
            return manifest_path.parent, info
    raise FileNotFoundError(
        "Could not find the independent-seed artifact with weight SHA256 "
        + _secondary_expected_sha256
    )


SECONDARY_ARTIFACTS, secondary_manifest_info = _find_secondary_artifact_root()
SECONDARY_WEIGHTS_ROOT = WORKING_DIR / "secondary_seed_weights"
copy_or_extract_tree(
    SECONDARY_ARTIFACTS / "weights",
    SECONDARY_ARTIFACTS / "weights.zip",
    SECONDARY_WEIGHTS_ROOT,
)
SECONDARY_WEIGHTS_PATH = (
    SECONDARY_WEIGHTS_ROOT
    / "unet_transformer"
    / "split_0"
    / "edge_predictor_best.pth"
)
SECONDARY_CONFIG_PATH = SECONDARY_WEIGHTS_PATH.parent / "config.json"
for _required_secondary_path in (SECONDARY_WEIGHTS_PATH, SECONDARY_CONFIG_PATH):
    if not _required_secondary_path.is_file():
        raise FileNotFoundError(f"Missing secondary model file: {_required_secondary_path}")


def _sha256_file(path: Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_secondary_actual_sha256 = _sha256_file(SECONDARY_WEIGHTS_PATH)
if _secondary_actual_sha256 != _secondary_expected_sha256:
    raise RuntimeError(
        "Secondary model checksum mismatch: "
        f"expected {_secondary_expected_sha256}, got {_secondary_actual_sha256}"
    )

os.environ["BIOHUB_SECONDARY_WEIGHTS"] = str(SECONDARY_WEIGHTS_PATH)
os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"] = "0.15"
print("Secondary artifact:", SECONDARY_ARTIFACTS)
print("Secondary weight:", SECONDARY_WEIGHTS_PATH)
print("Secondary SHA256:", _secondary_actual_sha256)
print("Secondary edge-logit weight:", os.environ["BIOHUB_SECONDARY_EDGE_WEIGHT"])

os.environ["BIOHUB_SECONDARY_DETECTION_WEIGHT"] = "0.80"  
os.environ["BIOHUB_SECONDARY_LINK_MODE"] = "low_margin_consensus"
os.environ["BIOHUB_SECONDARY_MIX_TEMPERATURE"] = "1"
os.environ["BIOHUB_SECONDARY_LOW_MARGIN_MAX"] = "0.35"
os.environ["BIOHUB_DUAL_SEED_EDGE_THRESHOLD"] = "0.48"

_runtime_integrity_receipt = {
    "status": "complete_label_free_runtime_integrity",
    "verified_before_dynamic_source_patch": True,
    "support_repo_python_file_count": len(_support_actual_sha256),
    "support_repo_python_sha256": _support_actual_sha256,
    "support_repo_python_manifest_sha256": _support_actual_manifest_sha256,
    "checkpoint_sha256": {
        "primary": _primary_actual_sha256,
        "secondary": _secondary_actual_sha256,
        "deepcenter": _deepcenter_actual_sha256,
    },
    "materialized_paths": {
        "primary": str(_primary_materialized_path),
        "secondary": str(SECONDARY_WEIGHTS_PATH),
        "deepcenter": str(_deepcenter_materialized_path),
    },
    "ground_truth_accessed": False,
}
_runtime_integrity_receipt_path = (
    WORKING_DIR / "bidirectional_production_runtime_integrity.json"
)
_runtime_integrity_receipt_path.write_text(
    json.dumps(_runtime_integrity_receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Runtime integrity receipt:", _runtime_integrity_receipt_path)


import torch as _torch

if not _torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for this notebook. Enable a Kaggle GPU accelerator and commit again."
    )
print("CUDA device:", _torch.cuda.get_device_name(0))


_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_old = """        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4"""
_new = """        if cfg.det_tta:
            _nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
                _nv += 1
            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
                _, det_rot = model.encode(imgs_rot)
                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
                del imgs_rot, det_rot
                _nv += 1
            imgs_t = imgs.transpose(-1, -2)
            _, det_t = model.encode(imgs_t)
            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
            del imgs_t, det_t
            _nv += 1
            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
            _, det_at = model.encode(imgs_at)
            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
            del imgs_at, det_at
            _nv += 1
            for f in range(W):
                det_logits[f] = det_logits[f] / _nv"""
if _old in _s:
    _ps.write_text(_s.replace(_old, _new))
    print("TTA patch applied (400ep spatial D4-style)")
else:
    print("TTA WARNING: block not found - using default 4-way")


_s = _ps.read_text()
_ensemble_replacements = [
    ('    downsample: tuple[int, ...] = (1, 4, 4),\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:', '    downsample: tuple[int, ...] = (1, 4, 4),\n    secondary_model: UNetNodeTransformer | None = None,\n    secondary_edge_weight: float = 0.0,\n    secondary_detection_weight: float = 0.0,\n    secondary_link_mode: str = "raw",\n    secondary_mix_temperature: float = 1.0,\n    secondary_low_margin_max: float = 0.2,\n) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:'),
    ('            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        del imgs', '            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n\n        secondary_unet_out = None\n        if secondary_model is not None:\n            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):\n                    primary_det = det_logits[f]\n                    secondary_det = secondary_det_logits[f]\n                    primary_mean = primary_det.mean()\n                    secondary_mean = secondary_det.mean()\n                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)\n                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_det_aligned = (\n                        (secondary_det - secondary_mean) * scale_ratio + primary_mean\n                    )\n                    det_logits[f] = (\n                        (1.0 - secondary_detection_weight) * primary_det\n                        + secondary_detection_weight * secondary_det_aligned\n                    )\n\n            del secondary_det_logits\n\n        del imgs'),
    ('            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            raw = edge_logits_pair[0]', '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n                if secondary_unet_out is None:\n                    raise RuntimeError("Secondary model is loaded but its feature map is missing")\n                secondary_feat_src = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx], p_coords_src, p_mask_src,\n                )\n                secondary_feat_tgt = secondary_model._index_features(\n                    secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,\n                )\n                secondary_logits_pair = secondary_model.predict_edges(\n                    secondary_feat_src, secondary_feat_tgt,\n                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                    p_pos_src, p_pos_tgt,\n                    p_mask_src, p_mask_tgt,\n                )\n\n                if secondary_link_mode == "raw":\n                    secondary_for_mix = secondary_logits_pair\n                    blend_weight = secondary_edge_weight\n                elif secondary_link_mode in {\n                    "calibrated", "adaptive", "low_margin_consensus"\n                }:\n                    primary_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    primary_scale = edge_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)\n                    secondary_scale = secondary_logits_pair.float().std(\n                        dim=1, keepdim=True, unbiased=False\n                    ).clamp_min(1e-4)\n                    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)\n                    secondary_for_mix = (\n                        (secondary_logits_pair - secondary_center) * secondary_scale_ratio\n                        + primary_center\n                    )\n                    if secondary_link_mode == "calibrated":\n                        blend_weight = secondary_edge_weight\n                    elif secondary_link_mode == "adaptive":\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]\n                            local_weight = (\n                                secondary_edge_weight + secondary_margin - primary_margin\n                            ).clamp(0.15, 0.75)\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            local_weight = torch.where(\n                                same_parent,\n                                torch.maximum(\n                                    local_weight,\n                                    torch.full_like(local_weight, secondary_edge_weight),\n                                ),\n                                local_weight,\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = secondary_edge_weight\n                    else:\n                        if n_src >= 2:\n                            primary_probs = torch.softmax(edge_logits_pair[0], dim=0)\n                            secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)\n                            primary_top2 = torch.topk(primary_probs, k=2, dim=0)\n                            secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)\n                            primary_margin = primary_top2.values[0] - primary_top2.values[1]\n                            same_parent = primary_top2.indices[0].eq(\n                                secondary_top2.indices[0]\n                            )\n                            uncertainty = (\n                                (secondary_low_margin_max - primary_margin)\n                                / secondary_low_margin_max\n                            ).clamp(0.0, 1.0)\n                            local_weight = secondary_edge_weight * uncertainty\n                            local_weight = torch.where(\n                                same_parent,\n                                local_weight,\n                                torch.zeros_like(local_weight),\n                            )\n                            blend_weight = local_weight.view(1, 1, -1)\n                        else:\n                            blend_weight = 0.0\n                else:\n                    raise ValueError(f"Unsupported secondary link mode: {secondary_link_mode}")\n\n                edge_logits_pair = (\n                    (1.0 - blend_weight) * edge_logits_pair\n                    + blend_weight * secondary_for_mix\n                )\n                if secondary_mix_temperature != 1.0:\n                    mixed_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                    edge_logits_pair = mixed_center + (\n                        edge_logits_pair - mixed_center\n                    ) / secondary_mix_temperature\n\n            raw = edge_logits_pair[0]'),
    ('        del unet_out\n', '        del unet_out\n        if secondary_unet_out is not None:\n            del secondary_unet_out\n'),
    ('    model, window_size, downsample = load_model(weights_path, device)\n    print(', '    model, window_size, downsample = load_model(weights_path, device)\n\n    secondary_model = None\n    secondary_weights_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()\n    secondary_edge_weight = float(os.environ.get("BIOHUB_SECONDARY_EDGE_WEIGHT", "0"))\n    secondary_detection_weight = float(\n        os.environ.get("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0")\n    )\n    secondary_link_mode = os.environ.get("BIOHUB_SECONDARY_LINK_MODE", "raw").strip()\n    secondary_mix_temperature = float(\n        os.environ.get("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1")\n    )\n    secondary_low_margin_max = float(\n        os.environ.get("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.2")\n    )\n    edge_candidate_threshold = float(\n        os.environ.get("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", str(cfg.threshold))\n    )\n    if secondary_weights_text:\n        if not 0.0 < secondary_edge_weight < 1.0:\n            raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")\n        if not 0.0 <= secondary_detection_weight < 1.0:\n            raise ValueError(\n                "BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)"\n            )\n        if secondary_link_mode not in {\n            "raw", "calibrated", "adaptive", "low_margin_consensus"\n        }:\n            raise ValueError(\n                "BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, "\n                "or low_margin_consensus"\n            )\n        if not 0.5 <= secondary_mix_temperature <= 2.0:\n            raise ValueError("BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]")\n        if not 0.0 < edge_candidate_threshold < 1.0:\n            raise ValueError("BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1")\n        if not 0.0 < secondary_low_margin_max <= 1.0:\n            raise ValueError("BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]")\n        secondary_model, secondary_window_size, secondary_downsample = load_model(\n            Path(secondary_weights_text), device,\n        )\n        if secondary_window_size != window_size or secondary_downsample != downsample:\n            raise ValueError(\n                "Primary and secondary models have incompatible inference grids: "\n                f"primary=(window={window_size}, downsample={downsample}), "\n                f"secondary=(window={secondary_window_size}, downsample={secondary_downsample})"\n            )\n        cfg.threshold = edge_candidate_threshold\n        print(\n            f"Secondary model: {secondary_weights_text} | "\n            f"edge weight={secondary_edge_weight:.3f} | "\n            f"detection weight={secondary_detection_weight:.3f} | "\n            f"link mode={secondary_link_mode} | "\n            f"temperature={secondary_mix_temperature:.3f} | "\n            f"low-margin max={secondary_low_margin_max:.3f} | "\n            f"edge threshold={cfg.threshold:.3f}",\n            flush=True,\n        )\n\n    print('),
    ('                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n            )', '                unet_batch_size=unet_batch_size,\n                downsample=downsample,\n                secondary_model=secondary_model,\n                secondary_edge_weight=secondary_edge_weight,\n                secondary_detection_weight=secondary_detection_weight,\n                secondary_link_mode=secondary_link_mode,\n                secondary_mix_temperature=secondary_mix_temperature,\n                secondary_low_margin_max=secondary_low_margin_max,\n            )'),
]
for _patch_index, (_ensemble_old, _ensemble_new) in enumerate(
    _ensemble_replacements, start=1
):
    _ensemble_count = _s.count(_ensemble_old)
    if _ensemble_count != 1:
        raise RuntimeError(
            f'Calibrated dual-seed patch {_patch_index} expected one match, '
            f'found {_ensemble_count}'
        )
    _s = _s.replace(_ensemble_old, _ensemble_new, 1)
compile(_s, str(_ps), 'exec')
_ps.write_text(_s)
print('Calibrated dual-seed runtime patch applied')



os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"] = "0.90"
for _guard_old_log in WORKING_DIR.glob("retention_guard_*.jsonl"):
    _guard_old_log.unlink()

_s = _ps.read_text()
_guard_old = """                    det_logits[f] = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )"""
_guard_new = """                    blended_det = (
                        (1.0 - secondary_detection_weight) * primary_det
                        + secondary_detection_weight * secondary_det_aligned
                    )
                    primary_candidates = len(_detect_cells_pooled(
                        primary_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    blended_candidates = len(_detect_cells_pooled(
                        blended_det[0],
                        int(frame_indices[f]),
                        cfg.det_threshold,
                        pool_k,
                    ))
                    minimum_retention = float(os.environ.get(
                        "BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION",
                        "0.90",
                    ))
                    candidate_retention = (
                        blended_candidates / primary_candidates
                        if primary_candidates
                        else 1.0
                    )
                    use_primary_detection = bool(
                        primary_candidates > 0
                        and candidate_retention < minimum_retention
                    )
                    det_logits[f] = (
                        primary_det if use_primary_detection else blended_det
                    )
                    if int(frame_indices[f]) not in seen_frames:
                        shard = os.environ.get(
                            "BIOHUB_GPU_SHARD", "single"
                        ).replace("/", "_")
                        guard_log = (
                            Path("/kaggle/working")
                            / f"retention_guard_{shard}.jsonl"
                        )
                        guard_record = {
                            "dataset": ds_path.stem,
                            "frame": int(frame_indices[f]),
                            "primary_candidates": int(primary_candidates),
                            "blended_candidates": int(blended_candidates),
                            "retention": float(candidate_retention),
                            "minimum_retention": float(minimum_retention),
                            "use_primary": bool(use_primary_detection),
                        }
                        with guard_log.open("a") as guard_handle:
                            guard_handle.write(
                                json.dumps(guard_record, sort_keys=True)
                                + "\\n"
                            )
                        if use_primary_detection:
                            print(
                                "BIOHUB_RETENTION_GUARD "
                                + json.dumps(guard_record, sort_keys=True),
                                flush=True,
                            )"""
_guard_matches = _s.count(_guard_old)
if _guard_matches != 1:
    raise RuntimeError(
        f"Retention guard expected one blend block, found {_guard_matches}"
    )
_s = _s.replace(_guard_old, _guard_new, 1)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Frozen frame retention guard applied at "
    + os.environ["BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION"]
)



import math as _bidirectional_math

_bidirectional_weight_guard = float(
    os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")
)
if not _bidirectional_math.isclose(
    _bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12
):
    raise ValueError({
        "expected_bidirectional_weight": 0.15,
        "actual_bidirectional_weight": _bidirectional_weight_guard,
    })

_s = _ps.read_text()
_bi_old = '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            if secondary_model is not None:\n'
_bi_new = '            edge_logits_pair = model.predict_edges(\n                unet_feat_src, unet_feat_tgt,\n                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n                p_pos_src, p_pos_tgt,\n                p_mask_src, p_mask_tgt,\n            )  # (1, n_src, n_tgt)\n\n            _bidirectional_weight = float(\n                os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0")\n            )\n            if _bidirectional_weight > 0.0:\n                reverse_logits_native = model.predict_edges(\n                    unet_feat_tgt, unet_feat_src,\n                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,\n                    p_pos_tgt, p_pos_src,\n                    p_mask_tgt, p_mask_src,\n                )  # (1, n_tgt, n_src)\n                reverse_logits_pair = reverse_logits_native.transpose(1, 2)\n\n                forward_center = edge_logits_pair.mean(dim=1, keepdim=True)\n                forward_scale = edge_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)\n                reverse_scale = reverse_logits_pair.float().std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)\n                reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)\n                reverse_aligned = (\n                    (reverse_logits_pair - reverse_center) * reverse_scale_ratio\n                    + forward_center\n                )\n                # Biohub 145: require mutual forward/reverse support in probability space.\n                # The harmonic mean penalizes a candidate when either temporal direction\n                # assigns it very low probability, while calibration preserves the forward\n                # logit scale used by the unchanged downstream candidate threshold and ILP.\n                forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)\n                reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)\n                harmonic_prob = 1.0 / (\n                    (1.0 - _bidirectional_weight) / forward_prob\n                    + _bidirectional_weight / reverse_prob\n                )\n                harmonic_prob = harmonic_prob / harmonic_prob.sum(\n                    dim=1, keepdim=True\n                ).clamp_min(1e-8)\n                harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))\n                harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)\n                harmonic_scale = harmonic_logits.std(\n                    dim=1, keepdim=True, unbiased=False\n                ).clamp_min(1e-4)\n                harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)\n                edge_logits_pair = (\n                    (harmonic_logits - harmonic_center) * harmonic_scale_ratio\n                    + forward_center\n                ).to(reverse_aligned.dtype)\n                del (\n                    reverse_logits_native,\n                    reverse_logits_pair,\n                    reverse_aligned,\n                    forward_prob,\n                    reverse_prob,\n                    harmonic_prob,\n                    harmonic_logits,\n                )\n            if secondary_model is not None:\n'
_bi_count = _s.count(_bi_old)
if _bi_count != 1:
    raise RuntimeError(
        f"Bidirectional edge patch expected one transformed block, found {_bi_count}"
    )
_s = _s.replace(_bi_old, _bi_new, 1)

_coordinate_manifest_old = '    coords = coords.astype(np.int16)\n    return coords, all_edges'
_coordinate_manifest_new = '    coords = coords.astype(np.int16)\n\n    # Label-free, pre-ILP detector-coordinate manifest. This executes inside\n    # predict_video, before build_graph and the ILP call in predict().\n    _coordinate_manifest_arm = os.environ.get(\n        "BIOHUB_DIAGNOSTIC_ARM", ""\n    ).strip()\n    if _coordinate_manifest_arm:\n        import hashlib as _coordinate_hashlib\n\n        _coordinate_shard = os.environ.get(\n            "BIOHUB_GPU_SHARD", "single"\n        ).replace("/", "_")\n        _coordinate_array = np.ascontiguousarray(\n            coords.astype("<i2", copy=False)\n        )\n        _coordinate_frame_counts = [\n            [int(_coordinate_t), int((_coordinate_array[:, 0] == _coordinate_t).sum())]\n            for _coordinate_t in np.unique(_coordinate_array[:, 0])\n        ]\n        _coordinate_record = {\n            "columns": ["t", "z", "y", "x"],\n            "coordinate_sha256": _coordinate_hashlib.sha256(\n                _coordinate_array.tobytes(order="C")\n            ).hexdigest(),\n            "dataset": ds_path.stem,\n            "dtype": "<i2",\n            "frame_counts": _coordinate_frame_counts,\n            "rows": int(len(_coordinate_array)),\n            "stage": "post_detection_pre_graph_pre_ilp",\n        }\n        _coordinate_manifest_path = (\n            Path("/kaggle/working")\n            / f"detector_coordinates_{_coordinate_manifest_arm}_"\n            f"{_coordinate_shard}.jsonl"\n        )\n        with _coordinate_manifest_path.open("a") as _coordinate_handle:\n            _coordinate_handle.write(\n                json.dumps(_coordinate_record, sort_keys=True) + "\\n"\n            )\n\n    return coords, all_edges'
_coordinate_manifest_count = _s.count(_coordinate_manifest_old)
if _coordinate_manifest_count != 1:
    raise RuntimeError(
        "Coordinate-manifest patch expected one pre-return block, found "
        f"{_coordinate_manifest_count}"
    )
_s = _s.replace(
    _coordinate_manifest_old, _coordinate_manifest_new, 1
)
compile(_s, str(_ps), "exec")
_ps.write_text(_s)
print(
    "Bidirectional harmonic-probability association fusion applied | weight=",
    _bidirectional_weight_guard,
)
print("Pre-ILP detector-coordinate manifest hook applied")


_et_s = _ps.read_text()
_et_old = '        if cfg.det_tta:\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                del imgs_flip, det_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                del imgs_rot, det_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            del imgs_t, det_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n'
_et_new = "        if cfg.det_tta:\n            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'\n            _unet_acc = unet_out.clone() if _edge_tta else None\n            _nv = 1\n            for dims in [(-1,), (-2,), (-2, -1)]:\n                imgs_flip = imgs.flip(dims)\n                _u_flip, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                if _edge_tta:\n                    _unet_acc = _unet_acc + _u_flip.flip(dims)\n                del imgs_flip, det_flip, _u_flip\n                _nv += 1\n            for _k in (1, 3):\n                imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                _u_rot, det_rot = model.encode(imgs_rot)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))\n                if _edge_tta:\n                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims=(-2, -1))\n                del imgs_rot, det_rot, _u_rot\n                _nv += 1\n            imgs_t = imgs.transpose(-1, -2)\n            _u_t, det_t = model.encode(imgs_t)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)\n            if _edge_tta:\n                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)\n            del imgs_t, det_t, _u_t\n            _nv += 1\n            imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)\n            _u_at, det_at = model.encode(imgs_at)\n            for f in range(W):\n                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))\n            if _edge_tta:\n                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims=(-2, -1))\n            del imgs_at, det_at, _u_at\n            _nv += 1\n            for f in range(W):\n                det_logits[f] = det_logits[f] / _nv\n            if _edge_tta:\n                if _unet_acc.shape != unet_out.shape:\n                    raise RuntimeError('EDGE-TTA SHAPE MISMATCH: %s vs %s'\n                                       % (tuple(_unet_acc.shape), tuple(unet_out.shape)))\n                _delta = float((_unet_acc / _nv - unet_out).abs().mean())\n                if _delta == 0.0:\n                    raise RuntimeError('EDGE-TTA NO-OP: averaged features bit-identical to the '\n                                       'single-pass features, so the augmented encodes '\n                                       'contributed nothing and this arm would read as a '\n                                       'false null')\n                unet_out = _unet_acc / _nv\n                print('EDGE_TTA_ACTIVE views=', _nv, 'mean_abs_feat_delta=', round(_delta, 6), flush=True)\n                del _unet_acc\n"
if _et_s.count(_et_old) != 1:
    raise RuntimeError('edge-TTA anchor block not unique: %d' % _et_s.count(_et_old))
_et_s = _et_s.replace(_et_old, _et_new, 1)
compile(_et_s, str(_ps), 'exec')
_ps.write_text(_et_s)
if 'EDGE_TTA_ACTIVE' not in _ps.read_text():
    raise RuntimeError('EDGE-TTA PATCH DID NOT PERSIST')
os.environ['BIOHUB_EDGE_FEATURE_TTA'] = '1'
print('EDGE_TTA patch installed and enabled in', _ps)

_secondary_tta_source = _ps.read_text()
_secondary_tta_old = '        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        del secondary_imgs_flip, secondary_det_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    del secondary_imgs_t, secondary_det_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _, secondary_det_at = secondary_model.encode(secondary_imgs_at)\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    del secondary_imgs_at, secondary_det_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n\n                for f in range(W):'
_secondary_tta_new = '        secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)\n            _secondary_edge_tta = os.environ.get(\n                "BIOHUB_SECONDARY_EDGE_FEATURE_TTA", "0"\n            ) != "0"\n            _secondary_unet_acc = (\n                secondary_unet_out.clone() if _secondary_edge_tta else None\n            )\n\n            if secondary_detection_weight > 0.0:\n                if cfg.det_tta:\n                    _secondary_nv = 1\n                    for dims in [(-1,), (-2,), (-2, -1)]:\n                        secondary_imgs_flip = imgs.flip(dims)\n                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(\n                            secondary_imgs_flip\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = (\n                                secondary_det_logits[f] + secondary_det_flip[f].flip(dims)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + _secondary_u_flip.flip(dims)\n                        del secondary_imgs_flip, secondary_det_flip, _secondary_u_flip\n                        _secondary_nv += 1\n                    for _k in (1, 3):\n                        secondary_imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))\n                        _secondary_u_rot, secondary_det_rot = secondary_model.encode(\n                            secondary_imgs_rot\n                        )\n                        for f in range(W):\n                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                                secondary_det_rot[f], -_k, dims=(-2, -1)\n                            )\n                        if _secondary_edge_tta:\n                            _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                                _secondary_u_rot, -_k, dims=(-2, -1)\n                            )\n                        del secondary_imgs_rot, secondary_det_rot, _secondary_u_rot\n                        _secondary_nv += 1\n                    secondary_imgs_t = imgs.transpose(-1, -2)\n                    _secondary_u_t, secondary_det_t = secondary_model.encode(secondary_imgs_t)\n                    for f in range(W):\n                        secondary_det_logits[f] = (\n                            secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + _secondary_u_t.transpose(-1, -2)\n                    del secondary_imgs_t, secondary_det_t, _secondary_u_t\n                    _secondary_nv += 1\n                    secondary_imgs_at = torch.rot90(\n                        imgs, 1, dims=(-2, -1)\n                    ).transpose(-1, -2)\n                    _secondary_u_at, secondary_det_at = secondary_model.encode(\n                        secondary_imgs_at\n                    )\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(\n                            secondary_det_at[f].transpose(-1, -2),\n                            -1,\n                            dims=(-2, -1),\n                        )\n                    if _secondary_edge_tta:\n                        _secondary_unet_acc = _secondary_unet_acc + torch.rot90(\n                            _secondary_u_at.transpose(-1, -2), -1, dims=(-2, -1)\n                        )\n                    del secondary_imgs_at, secondary_det_at, _secondary_u_at\n                    _secondary_nv += 1\n                    for f in range(W):\n                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv\n                    if _secondary_edge_tta:\n                        if _secondary_unet_acc.shape != secondary_unet_out.shape:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_SHAPE_MISMATCH")\n                        _secondary_delta = float(\n                            (_secondary_unet_acc / _secondary_nv - secondary_unet_out).abs().mean()\n                        )\n                        if _secondary_delta == 0.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_NO_OP")\n                        _secondary_edge_tta_weight = float(os.environ.get(\n                            "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT", "1.0"\n                        ))\n                        if not 0.0 < _secondary_edge_tta_weight <= 1.0:\n                            raise RuntimeError("SECONDARY_EDGE_TTA_BAD_WEIGHT")\n                        _secondary_tta_mean = _secondary_unet_acc / _secondary_nv\n                        secondary_unet_out = (\n                            (1.0 - _secondary_edge_tta_weight) * secondary_unet_out\n                            + _secondary_edge_tta_weight * _secondary_tta_mean\n                        )\n                        print(\n                            "SECONDARY_EDGE_TTA_ACTIVE views=",\n                            _secondary_nv,\n                            "weight=",\n                            _secondary_edge_tta_weight,\n                            "mean_abs_feat_delta=",\n                            round(_secondary_delta, 6),\n                            flush=True,\n                        )\n                        del _secondary_unet_acc\n\n                for f in range(W):'
_secondary_tta_count = _secondary_tta_source.count(_secondary_tta_old)
if _secondary_tta_count != 1:
    raise RuntimeError(
        "secondary edge-TTA anchor expected one match, found "
        + str(_secondary_tta_count)
    )
_secondary_tta_source = _secondary_tta_source.replace(
    _secondary_tta_old, _secondary_tta_new, 1
)
compile(_secondary_tta_source, str(_ps), "exec")
_ps.write_text(_secondary_tta_source)
if "SECONDARY_EDGE_TTA_ACTIVE" not in _ps.read_text():
    raise RuntimeError("secondary edge-TTA patch did not persist")
os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA"] = "1"
os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "1.0"
print("secondary edge-feature TTA patch installed and enabled", flush=True)


def list_test_stems() -> list[str]:
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory does not exist: {TEST_DIR}")
    stems = sorted(path.name[:-5] for path in TEST_DIR.iterdir() if path.name.endswith(".zarr"))
    if not stems:
        raise FileNotFoundError(f"No test .zarr files found in {TEST_DIR}")
    return stems


test_stems = list_test_stems()
print(f"Found {len(test_stems)} test videos")
print(test_stems[:10])

splits_path = REPO_DIR / "kaggle_test_splits_50ep.json"
splits_path.parent.mkdir(parents=True, exist_ok=True)
splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": test_stems}], indent=2))

predict_cmd = [
    sys.executable,
    "scripts/predict_unet_transformer.py",
    "--data-dir",
    str(TEST_DIR),
    "--splits",
    str(splits_path.name),
    "--split",
    "0",
    "--weights",
    WEIGHTS_RELATIVE,
    "--unet-batch-size",
    str(UNET_BATCH_SIZE),
    "--det-threshold",
    str(DET_THRESHOLD),
    "--ilp-edge-weight",
    str(ILP_EDGE_WEIGHT),
    "--ilp-appearance-weight",
    str(ILP_APPEARANCE_WEIGHT),
    "--ilp-disappearance-weight",
    str(ILP_DISAPPEARANCE_WEIGHT),
    "--ilp-division-weight",
    str(ILP_DIVISION_WEIGHT),
]
if USE_ILP:
    predict_cmd.append("--use-ilp")
if SLICE:
    predict_cmd.extend(["--slice", SLICE])

def _visible_cuda_tokens(count: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw != "-1":
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if len(tokens) < count:
            raise RuntimeError(
                f"torch reports {count} CUDA devices but CUDA_VISIBLE_DEVICES={raw!r}"
            )
        return tokens[:count]
    return [str(index) for index in range(count)]


def _prediction_dir_for_method(method: str) -> Path:
    matches = sorted((REPO_DIR / "predictions").glob(f"*/{method}/split_0"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one prediction directory for {method!r}, found {matches}"
        )
    return matches[0]


def _wait_for_prediction_shards(
    processes: dict[int, subprocess.Popen],
    commands: dict[int, list[str]],
) -> None:
    while processes:
        failed: tuple[int, int] | None = None
        for shard_index, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del processes[shard_index]
            if return_code != 0:
                failed = (shard_index, return_code)
                break
        if failed is None:
            if processes:
                time.sleep(1.0)
            continue

        failed_index, failed_code = failed
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise subprocess.CalledProcessError(failed_code, commands[failed_index])


def _merge_prediction_shards(worker_count: int) -> Path:
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(test_stems)

    for shard_index in range(worker_count):
        shard_method = f"{METHOD}_gpu{shard_index}"
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(test_stems[shard_index::worker_count])
        shard_paths = sorted(shard_dir.glob("*.geff"))
        found = {path.stem for path in shard_paths}
        if found != expected:
            raise RuntimeError(
                f"GPU shard {shard_index} output mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        overlap = seen & found
        if overlap:
            raise RuntimeError(f"Duplicate datasets across GPU shards: {sorted(overlap)}")
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(
            f"Merged GPU shards do not cover the test set: "
            f"missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}"
        )

    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f"GPU shards used inconsistent prediction roots: {username_roots}")
    import shutil as _shutil

    final_root = next(iter(username_roots)) / METHOD
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_dual_gpu_staging"
    if staging_dir.exists():
        if staging_dir.is_dir():
            _shutil.rmtree(staging_dir)
        else:
            staging_dir.unlink()
    staging_dir.mkdir(parents=True, exist_ok=False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob("*.geff")):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f"Refusing to overwrite duplicate merged output: {destination}")
            _shutil.move(str(source), str(destination))

    merged = {path.stem for path in staging_dir.glob("*.geff")}
    if merged != expected_all:
        raise RuntimeError(
            f"Staged prediction directory failed verification: "
            f"missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}"
        )

    if final_dir.exists():
        if final_dir.is_dir():
            _shutil.rmtree(final_dir)
        else:
            final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    print(f"Merged {len(merged)} prediction graphs into {final_dir}")
    return final_dir


start_time = time.time()
available_gpu_count = _torch.cuda.device_count()
worker_count = min(2, available_gpu_count, len(test_stems))

if worker_count >= 2 and not SLICE:
    cuda_tokens = _visible_cuda_tokens(worker_count)
    processes: dict[int, subprocess.Popen] = {}
    commands: dict[int, list[str]] = {}
    print(f"Launching {worker_count} independent video shards on CUDA devices {cuda_tokens}")
    for shard_index in range(worker_count):
        shard_method = f"{METHOD}_gpu{shard_index}"
        shard_cmd = [
            *predict_cmd,
            "--method",
            shard_method,
            "--slice",
            f"{shard_index}::{worker_count}",
        ]
        shard_env = {**os.environ, "PYTHONPATH": "src"}
        shard_env["CUDA_VISIBLE_DEVICES"] = cuda_tokens[shard_index]
        shard_env["BIOHUB_GPU_SHARD"] = f"{shard_index}/{worker_count}"
        print(
            f"GPU shard {shard_index}: CUDA_VISIBLE_DEVICES={cuda_tokens[shard_index]} | "
            + " ".join(shard_cmd),
            flush=True,
        )
        commands[shard_index] = shard_cmd
        processes[shard_index] = subprocess.Popen(
            shard_cmd,
            cwd=REPO_DIR,
            env=shard_env,
        )
    _wait_for_prediction_shards(processes, commands)
    _merge_prediction_shards(worker_count)
else:
    reason = "SLICE is active" if SLICE else f"only {available_gpu_count} CUDA device(s) available"
    print(f"Using single-process prediction because {reason}.")
    print(" ".join(predict_cmd))
    subprocess.run(
        predict_cmd,
        cwd=REPO_DIR,
        env={**os.environ, "PYTHONPATH": "src"},
        check=True,
    )

predict_seconds = time.time() - start_time
print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")
import tracksdata as td
import numpy as np
import blosc2
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SUBMISSION_COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
CSV_COLUMNS = ["id", *SUBMISSION_COLUMNS]
VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def graph_from_geff(path: Path):
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def point_distance_um(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dz = (a[0] - b[0]) * VOXEL_SCALE_UM[0]
    dy = (a[1] - b[1]) * VOXEL_SCALE_UM[1]
    dx = (a[2] - b[2]) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def node_point(node: dict[str, object]) -> tuple[float, float, float]:
    return (float(node["z"]), float(node["y"]), float(node["x"]))


def edge_sort_key(edge: dict[str, object]) -> tuple[float, float]:
    prob = edge.get("edge_prob")
    prob_value = float(prob) if prob is not None else 0.0
    return prob_value, -float(edge["distance_um"])


def _next_node_id(nodes_by_id: dict[int, dict[str, object]]) -> int:
    return max(nodes_by_id) + 1 if nodes_by_id else 1



def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        raw = chunk_path.read_bytes()
        arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            frame = arr.reshape(frame_shape).copy()
            frame_cache[t] = frame
            return frame
    except Exception:
        pass
    import zarr
    frame = np.asarray(zarr.open(zarr_path / "0", mode="r")[t])
    frame_cache[t] = frame
    return frame


def refine_synthetic_midpoint(
    dataset: str | None,
    t: int,
    midpoint: tuple[float, float, float],
    frame_cache: dict[int, np.ndarray],
    stats: dict[str, int],
) -> tuple[float, float, float]:
    if not GAP_REFINE_SYNTHETIC or dataset is None:
        return midpoint
    try:
        frame = read_test_frame(dataset, t, frame_cache)
        z, y, x = [int(round(v)) for v in midpoint]
        z0 = max(0, z - GAP_REFINE_WIN_Z)
        z1 = min(frame.shape[0], z + GAP_REFINE_WIN_Z + 1)
        y0 = max(0, y - GAP_REFINE_WIN_YX)
        y1 = min(frame.shape[1], y + GAP_REFINE_WIN_YX + 1)
        x0 = max(0, x - GAP_REFINE_WIN_YX)
        x1 = min(frame.shape[2], x + GAP_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0:
            stats["gap_refine_failed"] += 1
            return midpoint
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        refined = (
            float((weights * zz).sum() / total),
            float((weights * yy).sum() / total),
            float((weights * xx).sum() / total),
        )
        if point_distance_um(refined, midpoint) > GAP_REFINE_MAX_SHIFT_UM:
            stats["gap_refine_rejected_shift"] += 1
            return midpoint
        stats["gap_refined_synthetic"] += 1
        return refined
    except Exception:
        stats["gap_refine_failed"] += 1
        return midpoint



def _dc_pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _dc_normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(
        ratio,
        float(getattr(cfg, "norm_clip_lo", -0.5)),
        float(getattr(cfg, "norm_clip_hi", 6.0)),
    ).astype(np.float32)


def _dc_manifest_weight_paths(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        print("Could not read DeepCenter manifest:", manifest_path, type(exc).__name__, exc)
        return []
    root = manifest_path.parent
    sections: list[dict[str, object]] = []
    for section in [
        manifest.get("model", {}),
        manifest.get("models", {}).get("full_frame_center", {}) if isinstance(manifest.get("models", {}), dict) else {},
        manifest.get("full_frame_center", {}),
    ]:
        if isinstance(section, dict):
            sections.append(section)
    candidates: list[Path] = []
    for section in sections:
        for key in ("weight_path", "path"):
            rel = section.get(key)
            if isinstance(rel, str) and rel:
                candidates.append(root / rel)
        for key in ("last_checkpoint", "best_checkpoint"):
            item = section.get(key)
            if isinstance(item, dict):
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    candidates.append(root / rel)
    for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
        candidates.append(root / "weights" / "full_frame_center" / name)
        candidates.append(root / name)
    candidates.append(root / DEEPCENTER_RELATIVE)
    return candidates


def _dc_checkpoint_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT", DEEPCENTER_CHECKPOINT_DEFAULT).strip()
    if explicit:
        candidates.append(Path(explicit))
    manifest_explicit = os.environ.get("BIOHUB_DEEPCENTER_MANIFEST", DEEPCENTER_MANIFEST_DEFAULT).strip()
    if manifest_explicit:
        candidates.extend(_dc_manifest_weight_paths(Path(manifest_explicit)))

    input_root = Path("/kaggle/input")
    preferred_dirs = [
        Path("/kaggle/input/biohub-deepcenter-unet3d-center-prior-v1"),
        Path("/kaggle/input/datasets/pilkwang/biohub-deepcenter-unet3d-center-prior-v1"),
    ]
    for directory in preferred_dirs:
        candidates.extend(_dc_manifest_weight_paths(directory / "ARTIFACT_MANIFEST.json"))
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.append(directory / "weights" / "full_frame_center" / name)
            candidates.append(directory / name)
    if input_root.exists():
        for name in ("checkpoint_last.pt", "best.pt", "last.pt"):
            candidates.extend(sorted(input_root.glob(f"**/full_frame_center/**/{name}")))

    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        path = path.expanduser()
        try:
            key = path.resolve() if path.exists() else path
        except Exception:
            key = path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


try:
    import torch
except Exception as _dc_torch_error:
    torch = None


if torch is not None:
    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)
else:
    _DCConvBlock3d = None
    _DCDeepCenterUNet3D = None

def load_deepcenter_veto_detector() -> dict[str, object] | None:
    if not USE_DEEPCENTER_VETO:
        print("DeepCenter add-only repair gate disabled by configuration.")
        return None
    if torch is None:
        if REQUIRE_DEEPCENTER_VETO:
            raise ImportError("torch is required for DeepCenter add-only repair gate")
        print("DeepCenter add-only repair gate skipped because torch is unavailable.")
        return None
    from types import SimpleNamespace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_errors: list[str] = []
    for checkpoint_path in _dc_checkpoint_candidates():
        if not checkpoint_path.exists():
            continue
        try:
            print("Trying DeepCenter add-only gate checkpoint:", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                raise ValueError("checkpoint has no model_state")
            checkpoint_epoch = int(checkpoint.get("epoch", -1))
            if DEEPCENTER_EXPECTED_EPOCH > 0 and checkpoint_epoch != DEEPCENTER_EXPECTED_EPOCH:
                raise ValueError(
                    f"expected DeepCenter epoch {DEEPCENTER_EXPECTED_EPOCH}, got {checkpoint_epoch}"
                )
            cfg = SimpleNamespace(**checkpoint.get("config", {}))
            model = _DCDeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)
            model.eval()
            print("Loaded DeepCenter add-only gate checkpoint:", checkpoint_path)
            print("DeepCenter checkpoint epoch:", checkpoint.get("epoch"), "best_score:", checkpoint.get("best_score"))
            return {
                "model": model,
                "cfg": cfg,
                "device": device,
                "path": checkpoint_path,
                "torch": torch,
            }
        except Exception as exc:
            load_errors.append(f"{checkpoint_path}: {type(exc).__name__}: {exc}")
            print("Skipping incompatible DeepCenter checkpoint:", checkpoint_path, "|", type(exc).__name__, exc)
    message = "No usable DeepCenter checkpoint found for add-only repair gate."
    if REQUIRE_DEEPCENTER_VETO:
        checked = "\n".join(str(p) for p in _dc_checkpoint_candidates()[:80])
        errors = "\n".join(load_errors[-20:])
        raise FileNotFoundError(message + "\nChecked:\n" + checked + ("\nLoad errors:\n" + errors if errors else ""))
    print(message)
    return None


def _dc_cache_trim(cache: dict[tuple[str, int], np.ndarray]) -> None:
    limit = max(1, int(DEEPCENTER_SCORE_CACHE_MAX_FRAMES))
    while len(cache) > limit:
        cache.pop(next(iter(cache)))


def deepcenter_heatmap_for_frame(
    dataset: str,
    t: int,
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray | None:
    if detector_bundle is None:
        return None
    key = (dataset, int(t))
    cached = heatmap_cache.get(key)
    if cached is not None:
        return cached
    model = detector_bundle["model"]
    cfg = detector_bundle["cfg"]
    device = detector_bundle["device"]
    torch_mod = detector_bundle["torch"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    volume = read_test_frame(dataset, int(t), frame_cache)
    pooled = _dc_pool_frame_xy(volume, pool_factor)
    image = _dc_normalize_dynamic_range(pooled, cfg)
    with torch_mod.no_grad():
        tensor = torch_mod.from_numpy(image[None, None, ...]).to(device=device, dtype=torch_mod.float32)
        logits = model(tensor)
        # BIOHUB_DEEPCENTER_TTA: average the veto model's logits over the same
        # 8-view D4 group the detection and edge paths already use. This was the
        # last stage in the pipeline scored from a single un-augmented view, and
        # the veto it drives is the only gate whose threshold moves the score in
        # both directions (0.15 -> -0.002, 0.35 -> -0.005 around 0.25).
        if os.environ.get("BIOHUB_DEEPCENTER_TTA", "0") != "0":
            acc = logits.clone(); nv = 1
            for dims in [(-1,), (-2,), (-2, -1)]:
                acc = acc + model(tensor.flip(dims)).flip(dims); nv += 1
            if tensor.shape[-1] == tensor.shape[-2]:
                for k in (1, 3):
                    acc = acc + torch_mod.rot90(model(torch_mod.rot90(tensor, k, dims=(-2, -1))), -k, dims=(-2, -1)); nv += 1
                acc = acc + model(tensor.transpose(-1, -2)).transpose(-1, -2); nv += 1
                at = torch_mod.rot90(tensor, 1, dims=(-2, -1)).transpose(-1, -2)
                acc = acc + torch_mod.rot90(model(at).transpose(-1, -2), -1, dims=(-2, -1)); nv += 1
            delta = float((acc / nv - logits).abs().mean())
            if delta == 0.0:
                raise RuntimeError("DEEPCENTER_TTA_NO_OP: averaged veto logits identical to the single view")
            if not getattr(deepcenter_heatmap_for_frame, "_tta_announced", False):
                print("DEEPCENTER_TTA_ACTIVE views=", nv, "mean_abs_logit_delta=", round(delta, 6), flush=True)
                deepcenter_heatmap_for_frame._tta_announced = True
            logits = acc / nv
        heatmap = torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    heatmap_cache[key] = heatmap
    _dc_cache_trim(heatmap_cache)
    return heatmap


def deepcenter_score_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
) -> float | None:
    if not USE_DEEPCENTER_VETO or detector_bundle is None or dataset is None:
        return None
    heatmap = deepcenter_heatmap_for_frame(dataset, int(t), detector_bundle, frame_cache, heatmap_cache)
    if heatmap is None or heatmap.size == 0:
        return None
    cfg = detector_bundle["cfg"]
    pool_factor = int(getattr(cfg, "pool_factor", 4))
    z = int(round(float(point[0])))
    y = int(round(float(point[1]) / max(pool_factor, 1)))
    x = int(round(float(point[2]) / max(pool_factor, 1)))
    z0, z1 = max(0, z - DEEPCENTER_SCORE_WIN_Z), min(heatmap.shape[0], z + DEEPCENTER_SCORE_WIN_Z + 1)
    y0, y1 = max(0, y - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[1], y + DEEPCENTER_SCORE_WIN_YX + 1)
    x0, x1 = max(0, x - DEEPCENTER_SCORE_WIN_YX), min(heatmap.shape[2], x + DEEPCENTER_SCORE_WIN_YX + 1)
    patch = heatmap[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return None
    score = float(np.max(patch))
    return score if np.isfinite(score) else None


def deepcenter_accept_repair_point(
    dataset: str | None,
    t: int,
    point: tuple[float, float, float],
    detector_bundle: dict[str, object] | None,
    frame_cache: dict[int, np.ndarray],
    heatmap_cache: dict[tuple[str, int], np.ndarray],
    stats: dict[str, int],
    prefix: str,
    threshold: float,
) -> bool:
    if not USE_DEEPCENTER_VETO:
        return True
    if detector_bundle is None or dataset is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    stats[f"deepcenter_{prefix}_checked"] += 1
    score = deepcenter_score_point(dataset, int(t), point, detector_bundle, frame_cache, heatmap_cache)
    if score is None:
        stats[f"deepcenter_{prefix}_missing"] += 1
        return True
    if score < float(threshold):
        stats[f"deepcenter_{prefix}_rejected"] += 1
        return False
    stats[f"deepcenter_{prefix}_accepted"] += 1
    return True

def _position_um(node: dict[str, object]) -> np.ndarray:
    return np.array(
        [float(node["z"]) * VOXEL_SCALE_UM[0], float(node["y"]) * VOXEL_SCALE_UM[1], float(node["x"]) * VOXEL_SCALE_UM[2]],
        dtype=np.float64,
    )


def motion_relink_edges(
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
    learned_edge_probs: dict[tuple[int, int], float] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_MOTION_RELINK or not nodes_by_id:
        return []

    learned_edge_probs = learned_edge_probs or {}

    def learned_prob(source_id: int, target_id: int) -> float:
        value = learned_edge_probs.get((source_id, target_id), 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        return float(np.clip(value, 0.0, 1.0))

    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()

    frame_sizes = [len(ids) for ids in ids_by_t.values()]
    if frame_sizes and max(frame_sizes) > MOTION_RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    position_um = {node_id: _position_um(node) for node_id, node in nodes_by_id.items()}
    predecessor_position_um: dict[int, np.ndarray] = {}
    selected_edges: list[dict[str, object]] = []

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        gate_um: float,
    ) -> list[tuple[int, int, float, float, float]]:
        if not source_ids or not target_ids:
            return []
        big = gate_um * 1000.0 + 1.0
        cost = np.full((len(source_ids), len(target_ids)), big, dtype=np.float64)
        raw_dist = np.full_like(cost, np.inf)
        motion_dist = np.full_like(cost, np.inf)
        prob_matrix = np.zeros_like(cost)
        for i, source_id in enumerate(source_ids):
            source_pos = position_um[source_id]
            prev_pos = predecessor_position_um.get(source_id)
            if prev_pos is None:
                predicted = source_pos
            else:
                predicted = source_pos + MOTION_RELINK_VELOCITY_WEIGHT * (source_pos - prev_pos)
            for j, target_id in enumerate(target_ids):
                target_pos = position_um[target_id]
                raw = float(np.linalg.norm(target_pos - source_pos))
                if raw > gate_um:
                    continue
                motion = float(np.linalg.norm(target_pos - predicted))
                prob = learned_prob(source_id, target_id)
                raw_dist[i, j] = raw
                motion_dist[i, j] = motion
                prob_matrix[i, j] = prob
                cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob
        row_ind, col_ind = linear_sum_assignment(cost)
        matches: list[tuple[int, int, float, float, float]] = []
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= big:
                continue
            matches.append((
                source_ids[int(r)],
                target_ids[int(c)],
                float(raw_dist[r, c]),
                float(motion_dist[r, c]),
                float(prob_matrix[r, c]),
            ))
        return matches

    times = sorted(ids_by_t)
    for t in times:
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_matches: list[tuple[int, int, float, float, str, float]] = []
        for pass_name, gate_um in (("tight", MOTION_RELINK_TIGHT_UM), ("relaxed", MOTION_RELINK_RELAXED_UM)):
            pass_sources = [node_id for node_id in source_ids if node_id in unmatched_sources]
            pass_targets = [node_id for node_id in target_ids if node_id in unmatched_targets]
            matches = assign_pass(pass_sources, pass_targets, gate_um)
            for source_id, target_id, raw, motion, prob in matches:
                if source_id not in unmatched_sources or target_id not in unmatched_targets:
                    continue
                unmatched_sources.remove(source_id)
                unmatched_targets.remove(target_id)
                frame_matches.append((source_id, target_id, raw, motion, pass_name, prob))
                if pass_name == "tight":
                    stats["motion_relink_tight_edges"] += 1
                else:
                    stats["motion_relink_relaxed_edges"] += 1
        for source_id, target_id, raw, motion, pass_name, prob in frame_matches:
            selected_edges.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_prob": prob,
                "distance_um": raw,
                "motion_distance_um": motion,
                "motion_relinked": 1,
                "motion_pass": pass_name,
            })
            predecessor_position_um[target_id] = position_um[source_id]
        stats["motion_relink_frames"] += 1

    stats["motion_relink_edges"] = len(selected_edges)
    return selected_edges

def close_single_frame_gaps(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP_CLOSE or GAP_CLOSE_MAX_GAP < 1 or not edges:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    incident = outgoing | incoming

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    isolated_by_t: dict[int, list[int]] = {}
    all_ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        all_ids_by_t.setdefault(t, []).append(node_id)
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)
        if node_id not in incident:
            isolated_by_t.setdefault(t, []).append(node_id)

    max_synthetic = min(
        GAP_CLOSE_MAX_ADDED_ABS,
        max(1, int(round(len(nodes_by_id) * GAP_CLOSE_MAX_ADDED_FRAC))) if GAP_CLOSE_MAX_ADDED_FRAC > 0 else 0,
    )
    next_id = _next_node_id(nodes_by_id)
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
    used_starts: set[int] = set()
    used_isolated: set[int] = set()
    synthetic_added = 0
    new_edges: list[dict[str, object]] = []

    density_cache: dict[int, dict[int, float]] = {}

    def frame_local_spacing(t: int) -> dict[int, float]:
        cached = density_cache.get(t)
        if cached is not None:
            return cached

        frame_ids = all_ids_by_t.get(t, [])
        if len(frame_ids) <= 1:
            result = {
                node_id: GAP_DENSITY_REFERENCE_UM
                for node_id in frame_ids
            }
            density_cache[t] = result
            return result

        positions = np.stack(
            [_position_um(nodes_by_id[node_id]) for node_id in frame_ids]
        )
        tree = cKDTree(positions)
        query_k = min(
            len(frame_ids),
            max(2, GAP_DENSITY_NEIGHBORS + 1),
        )
        distances, _ = tree.query(positions, k=query_k)
        if distances.ndim == 1:
            distances = distances[:, None]

        result: dict[int, float] = {}
        for idx, node_id in enumerate(frame_ids):
            neighbour_distances = distances[idx, 1:]
            neighbour_distances = neighbour_distances[
                np.isfinite(neighbour_distances)
            ]
            spacing = (
                float(np.median(neighbour_distances))
                if neighbour_distances.size
                else GAP_DENSITY_REFERENCE_UM
            )
            result[node_id] = spacing

        density_cache[t] = result
        stats["gap_density_nodes_scored"] += len(result)
        return result

    effective_gap_max = min(GAP_CLOSE_MAX_GAP, 1)
    stats["gap_close_effective_max_gap"] = effective_gap_max
    for gap in range(1, effective_gap_max + 1):
        for t, end_ids in sorted(ends_by_t.items()):
            start_ids = [sid for sid in starts_by_t.get(t + gap + 1, []) if sid not in used_starts]
            if not end_ids or not start_ids:
                continue

            end_points = [node_point(nodes_by_id[eid]) for eid in end_ids]
            start_points = [node_point(nodes_by_id[sid]) for sid in start_ids]
            threshold_um = GAP_CLOSE_UM * (gap + 1)
            d = np.zeros(
                (len(end_ids), len(start_ids)),
                dtype=np.float64,
            )
            adaptive_threshold = np.full_like(d, threshold_um)

            source_spacing = frame_local_spacing(t)
            target_spacing = frame_local_spacing(t + gap + 1)

            for i, ep in enumerate(end_points):
                for j, sp in enumerate(start_points):
                    d[i, j] = point_distance_um(ep, sp)

                    if GAP_DENSITY_ADAPTIVE:
                        local_spacing = 0.5 * (
                            source_spacing.get(
                                end_ids[i],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                            + target_spacing.get(
                                start_ids[j],
                                GAP_DENSITY_REFERENCE_UM,
                            )
                        )
                        step_delta = float(
                            np.clip(
                                GAP_DENSITY_GAIN
                                * (
                                    local_spacing
                                    - GAP_DENSITY_REFERENCE_UM
                                ),
                                -GAP_DENSITY_MAX_STEP_DELTA_UM,
                                GAP_DENSITY_MAX_STEP_DELTA_UM,
                            )
                        )
                        adaptive_threshold[i, j] = (
                            threshold_um + step_delta * (gap + 1)
                        )
                        stats[
                            "gap_density_step_delta_milli_sum"
                        ] += int(round(1000.0 * step_delta))

            base_allowed = d <= threshold_um
            adaptive_allowed = d <= adaptive_threshold

            stats["gap_density_candidates_expanded"] += int(
                (adaptive_allowed & ~base_allowed).sum()
            )
            stats["gap_density_candidates_restricted"] += int(
                (base_allowed & ~adaptive_allowed).sum()
            )
            stats["gap_candidates"] += int(adaptive_allowed.sum())

            if not np.isfinite(d).any():
                continue

            max_threshold = float(np.max(adaptive_threshold))
            big = max_threshold * 1000.0 + 1.0
            cost = np.where(adaptive_allowed, d, big)
            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if not adaptive_allowed[r, c]:
                    continue
                if not base_allowed[r, c]:
                    stats[
                        "gap_density_selected_outside_base"
                    ] += 1
                source_id = end_ids[int(r)]
                target_id = start_ids[int(c)]
                if source_id in outgoing or target_id in used_starts:
                    continue

                source = nodes_by_id[source_id]
                target = nodes_by_id[target_id]
                mid_t = int(source["t"]) + gap
                mid_point = (
                    (float(source["z"]) + float(target["z"])) / 2.0,
                    (float(source["y"]) + float(target["y"])) / 2.0,
                    (float(source["x"]) + float(target["x"])) / 2.0,
                )

                middle_id: int | None = None
                middle_reused = False
                if GAP_CLOSE_REUSE_EXISTING:
                    candidates = [nid for nid in isolated_by_t.get(mid_t, []) if nid not in used_isolated]
                    if candidates:
                        distances = [point_distance_um(node_point(nodes_by_id[nid]), mid_point) for nid in candidates]
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= GAP_CLOSE_REUSE_UM:
                            middle_id = candidates[best_idx]
                            middle_reused = True

                if middle_id is None:
                    if synthetic_added >= max_synthetic:
                        stats["gap_skipped_node_cap"] += 1
                        continue
                    middle_id = next_id
                    next_id += 1
                    refined_point = refine_synthetic_midpoint(dataset, mid_t, mid_point, frame_cache, stats)
                    nodes_by_id[middle_id] = {
                        "node_id": middle_id,
                        "t": mid_t,
                        "z": refined_point[0],
                        "y": refined_point[1],
                        "x": refined_point[2],
                        "gap_synthetic": 1,
                    }
                    synthetic_added += 1
                    stats["gap_inserted_synthetic"] += 1

                middle = nodes_by_id[middle_id]
                gap_span_um = float(d[r, c])
                marginal_gap = gap_span_um >= DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM
                synthetic_middle = int(middle.get("gap_synthetic", 0)) == 1
                requires_center_confirmation = (
                    DEEPCENTER_GAP_VETO and marginal_gap and synthetic_middle
                )
                if DEEPCENTER_GAP_VETO and not marginal_gap:
                    stats["deepcenter_gap_bypassed_strong_motion"] += 1
                elif DEEPCENTER_GAP_VETO and not synthetic_middle:
                    stats["deepcenter_gap_bypassed_observed_node"] += 1
                if requires_center_confirmation and not deepcenter_accept_repair_point(
                    dataset,
                    mid_t,
                    node_point(middle),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "gap",
                    DEEPCENTER_GAP_THRESHOLD,
                ):
                    if int(middle.get("gap_synthetic", 0)) == 1:
                        nodes_by_id.pop(middle_id, None)
                        synthetic_added = max(0, synthetic_added - 1)
                        stats["gap_inserted_synthetic"] = max(0, stats["gap_inserted_synthetic"] - 1)
                    continue
                if middle_reused:
                    used_isolated.add(middle_id)
                    stats["gap_reused_existing"] += 1

                e1 = {
                    "source_id": source_id,
                    "target_id": middle_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(source, middle),
                    "gap_closed": 1,
                }
                e2 = {
                    "source_id": middle_id,
                    "target_id": target_id,
                    "edge_prob": None,
                    "distance_um": edge_distance_um(middle, target),
                    "gap_closed": 1,
                }
                new_edges.extend([e1, e2])
                outgoing.add(source_id)
                incoming.add(middle_id)
                outgoing.add(middle_id)
                incoming.add(target_id)
                used_starts.add(target_id)
                stats["gap_pairs_selected"] += 1
                stats["gap_added_edges"] += 2

    if new_edges:
        edges = [*edges, *new_edges]
    stats["gap_added_nodes"] = stats["gap_inserted_synthetic"]
    return nodes_by_id, edges


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


def _single_predecessor_map(edges: list[dict[str, object]]) -> dict[int, int]:
    by_target: dict[int, list[int]] = {}
    for edge in edges:
        by_target.setdefault(int(edge["target_id"]), []).append(int(edge["source_id"]))
    return {target: sources[0] for target, sources in by_target.items() if len(sources) == 1}


def recover_strict_gap2(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_GAP2_RECOVERY or not edges or not nodes_by_id:
        return nodes_by_id, edges

    outgoing = {int(edge["source_id"]) for edge in edges}
    incoming = {int(edge["target_id"]) for edge in edges}
    predecessor = _single_predecessor_map(edges)
    successor = _single_successor_map(edges)

    ends_by_t: dict[int, list[int]] = {}
    starts_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        t = int(node["t"])
        if node_id not in outgoing:
            ends_by_t.setdefault(t, []).append(node_id)
        if node_id not in incoming:
            starts_by_t.setdefault(t, []).append(node_id)

    cap = min(GAP2_MAX_LINKS_ABS, max(1, int(round(len(edges) * GAP2_MAX_LINKS_FRAC))))
    proposals: list[tuple[float, int, int, int, float]] = []

    def pos_um(node_id: int) -> np.ndarray:
        node = nodes_by_id[node_id]
        return np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64) * np.array(VOXEL_SCALE_UM)

    for t, end_ids in sorted(ends_by_t.items()):
        start_ids = starts_by_t.get(t + 3, [])
        if not end_ids or not start_ids:
            continue
        for end_id in end_ids:
            end_pos = pos_um(end_id)
            for start_id in start_ids:
                start_pos = pos_um(start_id)
                dist = float(np.linalg.norm(start_pos - end_pos))
                if dist > GAP2_MAX_TOTAL_UM or dist / 3.0 > GAP2_MAX_STEP_UM:
                    continue
                step = (start_pos - end_pos) / 3.0
                context_penalty = 0.0
                if GAP2_REQUIRE_CONTEXT:
                    ok_context = False
                    prev_id = predecessor.get(end_id)
                    if prev_id is not None:
                        prev_step = end_pos - pos_um(prev_id)
                        prev_norm = float(np.linalg.norm(prev_step))
                        step_norm = float(np.linalg.norm(step))
                        if prev_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(prev_step, step) / (prev_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(prev_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    next_id = successor.get(start_id)
                    if next_id is not None:
                        next_step = pos_um(next_id) - start_pos
                        next_norm = float(np.linalg.norm(next_step))
                        step_norm = float(np.linalg.norm(step))
                        if next_norm <= 0.01 or step_norm <= 0.01:
                            ok_context = True
                        else:
                            cos = float(np.dot(next_step, step) / (next_norm * step_norm + 1e-9))
                            if cos > -0.25 and np.linalg.norm(next_step - step) <= 6.0:
                                ok_context = True
                            context_penalty += max(0.0, 0.25 - cos)
                    if not ok_context:
                        continue
                proposals.append((dist + 2.0 * context_penalty, end_id, start_id, t, dist))

    proposals.sort(key=lambda item: item[0])
    stats["gap2_candidates"] = len(proposals)
    if not proposals:
        return nodes_by_id, edges

    selected: list[tuple[float, int, int, int, float]] = []
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    per_frame_count: dict[int, int] = {}
    for proposal in proposals:
        if len(selected) >= cap:
            stats["gap2_skipped_cap"] += 1
            break
        _, end_id, start_id, t, _ = proposal
        if end_id in used_ends or start_id in used_starts:
            continue
        frame_cap = max(1, int(round(len(ends_by_t.get(t, [])) * GAP2_FRAME_FRAC_CAP)))
        if per_frame_count.get(t, 0) >= frame_cap:
            continue
        selected.append(proposal)
        used_ends.add(end_id)
        used_starts.add(start_id)
        per_frame_count[t] = per_frame_count.get(t, 0) + 1

    if not selected:
        return nodes_by_id, edges

    next_node_id = _next_node_id(nodes_by_id)
    frame_cache: dict[int, np.ndarray] = {}
    new_edges: list[dict[str, object]] = []
    for _, end_id, start_id, t, _ in selected:
        source = nodes_by_id[end_id]
        target = nodes_by_id[start_id]
        previous_id = end_id
        inserted_ids: list[int] = []
        for k in (1, 2):
            frac = k / 3.0
            mid_t = int(source["t"]) + k
            midpoint = (
                float(source["z"]) + (float(target["z"]) - float(source["z"])) * frac,
                float(source["y"]) + (float(target["y"]) - float(source["y"])) * frac,
                float(source["x"]) + (float(target["x"]) - float(source["x"])) * frac,
            )
            refined_point = refine_synthetic_midpoint(dataset, mid_t, midpoint, frame_cache, stats)
            node_id = next_node_id
            next_node_id += 1
            nodes_by_id[node_id] = {
                "node_id": node_id,
                "t": mid_t,
                "z": refined_point[0],
                "y": refined_point[1],
                "x": refined_point[2],
            }
            inserted_ids.append(node_id)
            current = nodes_by_id[node_id]
            new_edges.append({
                "source_id": previous_id,
                "target_id": node_id,
                "edge_prob": None,
                "distance_um": edge_distance_um(nodes_by_id[previous_id], current),
                "gap2_recovered": 1,
            })
            previous_id = node_id
        new_edges.append({
            "source_id": previous_id,
            "target_id": start_id,
            "edge_prob": None,
            "distance_um": edge_distance_um(nodes_by_id[previous_id], target),
            "gap2_recovered": 1,
        })
        stats["gap2_pairs_selected"] += 1
        stats["gap2_added_nodes"] += len(inserted_ids)
        stats["gap2_added_edges"] += 3

    return nodes_by_id, [*edges, *new_edges]


def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
    frame_cache: dict[int, np.ndarray] | None = None,
    deepcenter_cache: dict[tuple[str, int], np.ndarray] | None = None,
) -> list[dict[str, object]]:
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges
    frame_cache = frame_cache if frame_cache is not None else {}
    deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
 
    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))
 
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
 
    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()
    used_sources: set[int] = set()  
 
    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue
 
        
        
        
        
        
        candidate_tree = None
        if SAFE_DIV_REQUIRE_MUTUAL_NN:
            candidate_positions = np.stack([_position_um(nodes_by_id[cid]) for cid in candidate_ids])
            candidate_tree = cKDTree(candidate_positions)
 
        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
 
            
            
            
            
            mutual_nn_id = None
            if candidate_tree is not None:
                _, nn_idx = candidate_tree.query(_position_um(existing_child))
                mutual_nn_id = candidate_ids[int(nn_idx)]
 
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue
 
                
                if SAFE_DIV_REQUIRE_MUTUAL_NN and candidate_id != mutual_nn_id:
                    stats["safe_division_mutual_nn_rejected"] += 1
                    continue
 
                
                
                
                
                if SAFE_DIV_REQUIRE_DIVERGENCE:
                    c1_succ = out_by_source.get(existing_child_id, [])
                    q_succ = out_by_source.get(candidate_id, [])
                    if len(c1_succ) != 1 or len(q_succ) != 1:
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    c1_grandchild = nodes_by_id.get(int(c1_succ[0]["target_id"]))
                    q_grandchild = nodes_by_id.get(int(q_succ[0]["target_id"]))
                    if (
                        c1_grandchild is None or q_grandchild is None
                        or int(c1_grandchild["t"]) != t + 2
                        or int(q_grandchild["t"]) != t + 2
                    ):
                        stats["safe_division_divergence_rejected"] += 1
                        continue
                    grandchild_dist = edge_distance_um(c1_grandchild, q_grandchild)
                    if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
                        stats["safe_division_divergence_rejected"] += 1
                        continue
 
                stats["safe_division_geometric_candidates"] += 1
                if DEEPCENTER_SAFE_DIV_VETO and not deepcenter_accept_repair_point(
                    dataset,
                    int(candidate["t"]),
                    node_point(candidate),
                    deepcenter_bundle,
                    frame_cache,
                    deepcenter_cache,
                    stats,
                    "safe_div",
                    DEEPCENTER_SAFE_DIV_THRESHOLD,
                ):
                    continue
                
                
                
                
                
                
                if SAFE_DIV_SISTER_SYMMETRY_TAU > 0.0:
                    _sym_denom = max((child_dist + parent_dist) / 2.0, 1e-6)
                    if abs(child_dist - parent_dist) / _sym_denom > SAFE_DIV_SISTER_SYMMETRY_TAU:
                        stats["safe_division_symmetry_rejected"] += 1
                        continue
                score = parent_dist + 0.15 * sister_dist
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist))
 
        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _ in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            if source_id in used_sources:
                continue
            candidate = nodes_by_id[candidate_id]
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
            })
            used_targets.add(candidate_id)
            used_sources.add(source_id)
            added_this_frame += 1
 
    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    component_edges: dict[int, list[dict[str, object]]] = {root: [] for root in components}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id in parent and target_id in parent:
            component_edges.setdefault(find(source_id), []).append(edge)

    keep: set[int] = set()
    for root, members in components.items():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_before_rescue = len(nodes_by_id) - len(keep)
    if removed_before_rescue <= 0:
        return nodes_by_id, edges

    if ADAPTIVE_SHORT_TRACK_RESCUE:
        removed_frac = removed_before_rescue / max(len(nodes_by_id), 1)
        if removed_frac >= SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC:
            budget = min(
                SHORT_TRACK_RESCUE_MAX_NODES_ABS,
                max(0, int(round(len(nodes_by_id) * SHORT_TRACK_RESCUE_MAX_NODES_FRAC))),
            )
            stats["short_track_rescue_triggered"] = 1
            stats["short_track_rescue_budget"] = budget
            proposals: list[tuple[float, int, float, int, list[int]]] = []
            for root, members in components.items():
                if set(members) & keep:
                    continue
                if len(members) < SHORT_TRACK_RESCUE_MIN_LEN or len(members) >= OUTPUT_MIN_TRACK_LEN:
                    continue
                c_edges = component_edges.get(root, [])
                if not c_edges:
                    continue
                probs: list[float] = []
                dists: list[float] = []
                for edge in c_edges:
                    try:
                        prob = float(edge.get("edge_prob", 0.0))
                    except (TypeError, ValueError):
                        prob = 0.0
                    if np.isfinite(prob):
                        probs.append(prob)
                    try:
                        dist = float(edge.get("distance_um", np.nan))
                    except (TypeError, ValueError):
                        dist = np.nan
                    if np.isfinite(dist):
                        dists.append(dist)
                mean_prob = float(np.mean(probs)) if probs else 0.0
                mean_dist = float(np.mean(dists)) if dists else float("inf")
                if mean_prob < SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB:
                    continue
                if mean_dist > SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM:
                    continue
                score = mean_prob - 0.02 * mean_dist + 0.004 * len(members)
                proposals.append((score, len(members), mean_prob, root, members))
            proposals.sort(reverse=True)
            rescued_nodes = 0
            rescued_components = 0
            for _, size, _, _, members in proposals:
                if budget <= 0 or rescued_nodes + size > budget:
                    continue
                keep.update(members)
                rescued_nodes += size
                rescued_components += 1
            stats["short_track_rescue_components"] = rescued_components
            stats["short_track_rescue_nodes"] = rescued_nodes

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


def filter_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    raw_edges: list[dict[str, object]],
    dataset: str | None = None,
    deepcenter_bundle: dict[str, object] | None = None,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    stats = {
        "raw_edges": len(raw_edges),
        "dropped_nonconsecutive_edges": 0,
        "dropped_long_edges": 0,
        "dropped_multi_parent_edges": 0,
        "dropped_multi_child_edges": 0,
        "dropped_division_edges": 0,
        "gap_candidates": 0,
        "gap_pairs_selected": 0,
        "gap_reused_existing": 0,
        "gap_inserted_synthetic": 0,
        "gap_added_nodes": 0,
        "gap_added_edges": 0,
        "gap_skipped_node_cap": 0,
        "gap_density_nodes_scored": 0,
        "gap_density_candidates_expanded": 0,
        "gap_density_candidates_restricted": 0,
        "gap_density_selected_outside_base": 0,
        "gap_density_step_delta_milli_sum": 0,
        "gap_refined_synthetic": 0,
        "gap_refine_failed": 0,
        "gap_refine_rejected_shift": 0,
        "pruned_isolated_nodes": 0,
        "motion_relink_edges": 0,
        "motion_relink_tight_edges": 0,
        "motion_relink_relaxed_edges": 0,
        "motion_relink_frames": 0,
        "motion_relink_replaced_raw_edges": 0,
        "motion_relink_fallback_raw": 0,
        "motion_relink_skipped_large_frame": 0,
        "gap2_candidates": 0,
        "gap2_pairs_selected": 0,
        "gap2_added_nodes": 0,
        "gap2_added_edges": 0,
        "gap2_skipped_cap": 0,
        "safe_division_candidates": 0,
        "safe_division_geometric_candidates": 0,  
        "safe_divisions_added": 0,
        "safe_division_skipped_cap": 0,
        "safe_division_mutual_nn_rejected": 0,
        "safe_division_divergence_rejected": 0,
        "safe_division_symmetry_rejected": 0,  
        "deepcenter_gap_checked": 0,
        "deepcenter_gap_bypassed_strong_motion": 0,
        "deepcenter_gap_bypassed_observed_node": 0,
        "deepcenter_gap_accepted": 0,
        "deepcenter_gap_rejected": 0,
        "deepcenter_gap_missing": 0,
        "deepcenter_safe_div_checked": 0,
        "deepcenter_safe_div_accepted": 0,
        "deepcenter_safe_div_rejected": 0,
        "deepcenter_safe_div_missing": 0,
        "short_track_components_removed": 0,
        "short_track_nodes_removed": 0,
        "short_track_edges_removed": 0,
        "short_track_filter_skipped_all": 0,
        "short_track_rescue_triggered": 0,
        "short_track_rescue_components": 0,
        "short_track_rescue_nodes": 0,
        "short_track_rescue_budget": 0,
        "linefit_smoothed_nodes": 0,
        "linefit_skipped_nodes": 0,
    }

    edges: list[dict[str, object]] = []
    for edge in raw_edges:
        source = nodes_by_id.get(int(edge["source_id"]))
        target = nodes_by_id.get(int(edge["target_id"]))
        if source is None or target is None:
            continue
        if OUTPUT_ENFORCE_NEXT_FRAME and int(target["t"]) != int(source["t"]) + 1:
            stats["dropped_nonconsecutive_edges"] += 1
            continue
        distance_um = edge_distance_um(source, target)
        edge["distance_um"] = distance_um
        if OUTPUT_EDGE_MAX_UM > 0 and distance_um > OUTPUT_EDGE_MAX_UM:
            stats["dropped_long_edges"] += 1
            continue
        edges.append(edge)

    if OUTPUT_MOTION_RELINK:
        learned_edge_probs: dict[tuple[int, int], float] = {}
        for edge in edges:
            prob = edge.get("edge_prob")
            if prob is None:
                continue
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                continue
            if np.isfinite(prob):
                key = (int(edge["source_id"]), int(edge["target_id"]))
                learned_edge_probs[key] = max(learned_edge_probs.get(key, float("-inf")), prob)
        motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
        if motion_edges:
            stats["motion_relink_replaced_raw_edges"] = len(edges)
            edges = motion_edges
        else:
            stats["motion_relink_fallback_raw"] = 1

    if OUTPUT_SINGLE_PARENT_REPAIR and edges:
        best_by_target: dict[int, dict[str, object]] = {}
        for edge in edges:
            target_id = int(edge["target_id"])
            prev = best_by_target.get(target_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_target[target_id] = edge
        kept_ids = {id(edge) for edge in best_by_target.values()}
        stats["dropped_multi_parent_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    if OUTPUT_SINGLE_CHILD_REPAIR and edges:
        best_by_source: dict[int, dict[str, object]] = {}
        for edge in edges:
            source_id = int(edge["source_id"])
            prev = best_by_source.get(source_id)
            if prev is None or edge_sort_key(edge) > edge_sort_key(prev):
                best_by_source[source_id] = edge
        kept_ids = {id(edge) for edge in best_by_source.values()}
        stats["dropped_multi_child_edges"] = sum(1 for edge in edges if id(edge) not in kept_ids)
        edges = [edge for edge in edges if id(edge) in kept_ids]

    print(f"  [{dataset}] after edge-filter+motion-relink: {len(nodes_by_id)} nodes, {len(edges)} edges")
    repair_frame_cache: dict[int, np.ndarray] = {}
    deepcenter_heatmap_cache: dict[tuple[str, int], np.ndarray] = {}
    nodes_by_id, edges = close_single_frame_gaps(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )
    nodes_by_id, edges = recover_strict_gap2(nodes_by_id, edges, stats, dataset=dataset)
    print(f"  [{dataset}] after gap-closing (single-frame + gap2): {len(nodes_by_id)} nodes, {len(edges)} edges")
    edges = add_safe_divisions_postlink(
        nodes_by_id,
        edges,
        stats,
        dataset=dataset,
        deepcenter_bundle=deepcenter_bundle,
        frame_cache=repair_frame_cache,
        deepcenter_cache=deepcenter_heatmap_cache,
    )

    _geo_cands = stats['safe_division_geometric_candidates']
    _post_veto_cands = stats['safe_division_candidates']
    _rejected_by_dc = _geo_cands - _post_veto_cands
    print(
        f"  [{dataset}] after safe-division repair: {len(nodes_by_id)} nodes, {len(edges)} edges"
        f" (geometric_candidates={_geo_cands}, deepcenter_rejected={_rejected_by_dc},"
        f" post_veto_candidates={_post_veto_cands}, added={stats['safe_divisions_added']},"
        f" cap_skipped={stats['safe_division_skipped_cap']},"
        f" mutual_nn_rejected={stats['safe_division_mutual_nn_rejected']},"
        f" divergence_rejected={stats['safe_division_divergence_rejected']})"
    )
    if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:
        by_source: dict[int, list[dict[str, object]]] = {}
        for edge in edges:
            by_source.setdefault(int(edge["source_id"]), []).append(edge)

        filtered: list[dict[str, object]] = []
        for source_id, source_edges in by_source.items():
            if len(source_edges) <= 1:
                filtered.extend(source_edges)
                continue

            ranked = sorted(source_edges, key=edge_sort_key, reverse=True)
            source = nodes_by_id[source_id]
            top1 = ranked[0]
            top2 = ranked[1]
            d1 = float(top1["distance_um"])
            d2 = float(top2["distance_um"])
            sister = edge_distance_um(nodes_by_id[int(top1["target_id"])], nodes_by_id[int(top2["target_id"])])
            valid_division = (
                max(d1, d2) <= DIV_PARENT_MAX_UM
                and sister <= DIV_SISTER_MAX_UM
                and int(nodes_by_id[int(top1["target_id"])] ["t"]) == int(source["t"]) + 1
                and int(nodes_by_id[int(top2["target_id"])] ["t"]) == int(source["t"]) + 1
            )
            if valid_division:
                filtered.extend([top1, top2])
                stats["dropped_division_edges"] += max(0, len(ranked) - 2)
            elif DIV_DROP_TO_SINGLE_IF_BAD:
                filtered.append(top1)
                stats["dropped_division_edges"] += len(ranked) - 1
            else:
                filtered.extend(ranked)
        edges = filtered

    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(edge["source_id"]) for edge in edges} | {int(edge["target_id"]) for edge in edges}
        if incident:
            kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in incident}
            stats["pruned_isolated_nodes"] = len(nodes_by_id) - len(kept_nodes)
            nodes_by_id = kept_nodes
            edges = [edge for edge in edges if int(edge["source_id"]) in nodes_by_id and int(edge["target_id"]) in nodes_by_id]

    print(f"  [{dataset}] after division-geometry-filter+prune-isolated: {len(nodes_by_id)} nodes, {len(edges)} edges")
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    print(f"  [{dataset}] after short-track filtering: {len(nodes_by_id)} nodes, {len(edges)} edges"
          f" (components_removed={stats['short_track_components_removed']})")
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    print(f"  [{dataset}] FINAL: {len(nodes_by_id)} nodes, {len(edges)} edges")

    return nodes_by_id, edges, stats


DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

def write_test_submission(tag: str = "base") -> None:
    
    
    geffs = sorted((REPO_DIR / "predictions").glob(f"*/{METHOD}/split_0/*.geff"))
    print(f"Found {len(geffs)} prediction graphs")
    if len(geffs) != len(test_stems):
        found = {path.stem for path in geffs}
        missing = sorted(set(test_stems) - found)
        raise RuntimeError(f"Expected {len(test_stems)} graphs, found {len(geffs)}. Missing: {missing[:10]}")

    stats_rows: list[dict[str, object]] = []
    seen_datasets: set[str] = set()
    row_id = 0
    total_nodes = 0
    total_edges = 0

    with SUBMISSION_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for geff_path in geffs:
            dataset = geff_path.stem
            seen_datasets.add(dataset)
            graph = graph_from_geff(geff_path)

            nodes_by_id: dict[int, dict[str, object]] = {}
            for row in graph.node_attrs().iter_rows(named=True):
                node_id = int(row["node_id"])
                nodes_by_id[node_id] = {
                    "node_id": node_id,
                    "t": int(row["t"]),
                    "z": float(row["z"]),
                    "y": float(row["y"]),
                    "x": float(row["x"]),
                }

            raw_edges: list[dict[str, object]] = []
            for row in graph.edge_attrs().iter_rows(named=True):
                edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
                raw_edges.append({
                    "source_id": int(row["source_id"]),
                    "target_id": int(row["target_id"]),
                    "edge_prob": None if edge_prob is None else float(edge_prob),
                })

            raw_node_count = len(nodes_by_id)
            nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)
            if not nodes_by_id:
                raise AssertionError(f"{dataset}: post-processing removed every node")

            for node_id in sorted(nodes_by_id):
                node = nodes_by_id[node_id]
                writer.writerow({
                    "id": row_id,
                    "dataset": dataset,
                    "row_type": "node",
                    "node_id": int(node["node_id"]),
                    "t": int(node["t"]),
                    "z": max(0, int(round(float(node["z"])))),
                    "y": max(0, int(round(float(node["y"])))),
                    "x": max(0, int(round(float(node["x"])))),
                    "source_id": -1,
                    "target_id": -1,
                })
                row_id += 1

            division_sources: dict[int, int] = {}
            for edge in edges:
                source_id = int(edge["source_id"])
                target_id = int(edge["target_id"])
                if source_id not in nodes_by_id or target_id not in nodes_by_id:
                    raise AssertionError(f"{dataset}: dangling edge after filtering")
                writer.writerow({
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
                })
                row_id += 1
                division_sources[source_id] = division_sources.get(source_id, 0) + 1

            node_count = len(nodes_by_id)
            edge_count = len(edges)
            total_nodes += node_count
            total_edges += edge_count
            stats_rows.append({
                "dataset": dataset,
                "raw_nodes": raw_node_count,
                "nodes": node_count,
                "raw_edges": filter_stats["raw_edges"],
                "edges": edge_count,
                "division_like_sources": sum(1 for count in division_sources.values() if count >= 2),
                "edge_to_node_ratio": edge_count / max(node_count, 1),
                "gap_added_nodes_frac": filter_stats.get("gap_added_nodes", 0) / max(raw_node_count, 1),
                **filter_stats,
            })

    expected_datasets = set(test_stems)
    missing_datasets = sorted(expected_datasets - seen_datasets)
    extra_datasets = sorted(seen_datasets - expected_datasets)
    if missing_datasets or extra_datasets:
        raise AssertionError({"missing": missing_datasets[:10], "extra": extra_datasets[:10]})
    assert row_id == total_nodes + total_edges, "Internal row counter mismatch"
    assert total_nodes > 0, "No node rows produced"

    header = SUBMISSION_PATH.open().readline().strip().split(",")
    assert header == CSV_COLUMNS, f"Bad CSV header: {header}"

    stats = pd.DataFrame(stats_rows).sort_values("dataset").reset_index(drop=True)
    stats["predict_minutes_total"] = predict_seconds / 60.0
    stats["experiment_tag"] = f"{EXPERIMENT_TAG}:{tag}"
    stats.to_csv(RUN_STATS_PATH, index=False)

    print(f"Wrote {SUBMISSION_PATH} with {row_id:,} rows")
    print(f"Node rows: {total_nodes:,} | edge rows: {total_edges:,}")
    print(f"Wrote {RUN_STATS_PATH}")
    display(pd.read_csv(SUBMISSION_PATH, nrows=8))


write_test_submission("base")


# ==== HOCT consensus veto (post-ILP stage): arming cell ==============================
# Controlled by BIOHUB_HOCT_VETO, set in the configuration cell:
#   "0"  disabled: nothing below runs; submission.csv stays exactly what the base notebook wrote.
#   "1"  arm lf-hoctveto: after the notebook's own post-processing (gap closing, safe-division
#        gates, short-track filtering, line-fit smoothing) has produced the FINAL track graph of a
#        test movie, HOCT general_v0 (royerlab, arXiv 2607.11754) is run over that same node set --
#        a 3-micrometre sphere is painted around every final node, intensity features come from the
#        raw frames -- and every NON-division edge that HOCT does not also propose is dropped.
#        Nodes are never changed. Both outgoing edges of a node with two children stay untouched.
#   "2"  arm lf-hoctveto-div: as "1", but division edges are vetoed too; a daughter whose edge HOCT
#        does not propose becomes a track start (its node is kept).
# Measured offline on the honest 20 videos with the converged linker (ec_ref50_hoct_img.json):
# keeping only edges proposed by BOTH linkers gains +0.0040 [+0.0006, +0.0058], positive on both
# prefixes, and cuts scorer-visible false divisions 55 -> 29; the edge UNION loses 0.004.
#
# How it hooks in (runtime-hardened 2026-09-11, after both arms exceeded Kaggle's 12-hour limit):
# this cell only ARMS the veto. `write_test_submission` (defined in the previous cell, which also
# wrote the base submission.csv) is re-bound to a wrapper that, only while it runs, swaps
# `filter_output_graph` for a version applying the veto to its output. Nothing is re-written here.
# The veto is applied inside the notebook's FINAL write of submission.csv, exactly once:
#   - when the post-process sweep cell selects a configuration it calls write_test_submission(...)
#     itself; that call now goes through the wrapper (no extra linker pass, no extra HOCT pass);
#   - otherwise the small finalize cell inserted after the sweep (hoct_veto_finalize_cell.py) sees
#     that no wrapped write has completed and re-writes the base submission.csv once with the veto.
# The 2026-09-11 arms ran HOCT twice per video (once here on the base write, once inside the
# sweep's re-write) because this cell re-wrote immediately AND hooked the sweep; that is removed.
# Validator scoring (score_validator_config) calls filter_output_graph directly and is untouched.
#
# Wall-clock budget (environment variables, read once when this cell runs):
#   BIOHUB_HOCT_DEADLINE_H   (default 10.0)  notebook hours after which no further video is vetoed;
#   BIOHUB_HOCT_MAX_VIDEO_S  (default 900)  a video whose predicted HOCT time exceeds this is skipped.
# Predicted HOCT time per video = 9 s per 1,000 final nodes + 10 s (fit on the 2026-09-11 public
# runs on a T4: 6,151 nodes 58 s, 20,727 nodes 119 s, 25,622 nodes 148 s, 70,251 nodes 640 s).
# A video is skipped -- its base graph is written unchanged -- when the prediction is above the
# per-video cap, when the deadline has passed, or when elapsed + prediction would pass it. The
# chunked retries after a solver failure re-check the deadline before each retry.
# Elapsed time = age of this Python process, read from /proc/self/stat (the IPython kernel process
# runs every cell, so its start is the session start); when PID 1 (the container's init) started at
# most one hour earlier its age is used instead, since it also covers the seconds before the kernel
# came up. Without /proc the time since this cell ran plus the detector's own timer is used and the
# summary says so (that fallback under-estimates).
# Fail-safe: a per-video HOCT failure keeps that video's base graph; when HOCT cannot be installed
# or loaded every video is passed through; when the wrapped write itself fails, submission.csv and
# run_stats.csv are restored from a backup taken before the write and the exception is swallowed, so
# the notebook completes with a valid file. The previous submission.csv is kept next to the final one
# as submission_before_hoct_veto.csv for inspection. One HOCT_VETO_SUMMARY line (videos vetoed /
# skipped for budget / failed, elapsed seconds, deadline) is printed by the finalize cell.
import glob as _hv_glob
import hashlib as _hv_hashlib
import math as _hv_math
import os as _hv_os
import shutil as _hv_shutil
import subprocess as _hv_subprocess
import sys as _hv_sys
import time as _hv_time
import traceback as _hv_traceback
from collections import Counter as _hv_Counter

import numpy as _hv_np

_HV_MODE = int(_hv_os.environ.get("BIOHUB_HOCT_VETO", "0"))
_HV_DEADLINE_H = float(_hv_os.environ.get("BIOHUB_HOCT_DEADLINE_H", "10.0"))
_HV_MAX_VIDEO_S = float(_hv_os.environ.get("BIOHUB_HOCT_MAX_VIDEO_S", "900"))
_HV_SEC_PER_1000_NODES = 9.0      # slope of the predicted HOCT time (see the header)
_HV_FIXED_S = 10.0                # intercept: sphere painting, volume read, graph conversion
_HV_SCALE_ZYX = (1.625, 0.40625, 0.40625)
_HV_RADIUS_UM = 3.0
_HV_TILE, _HV_OVERLAP = (5, 32, 128, 128), (1, 8, 16, 16)
_HV_MAX_DELTA_T = 1
_HV_BACKUP_SUFFIX = ".hoct_veto_backup"
_HV_KEEP_PREVIOUS_NAME = "submission_before_hoct_veto.csv"
_HV_STATE: dict = {
    "model": None, "cache": {}, "totals": None,
    "counts": {"vetoed": 0, "skipped_budget": 0, "failed": 0, "write_failures": 0},
    "applied_writes": 0,            # wrapped writes of submission.csv that completed
    "disabled_reason": None,        # set when HOCT cannot be installed/loaded: later videos fail fast
    "hoct_seconds": 0.0,            # wall time spent inside the HOCT stage (all videos)
    "t_import_monotonic": _hv_time.monotonic(),
    "clock": None,                  # elapsed-time source actually used (for the summary line)
}


def _hv_log(msg: str) -> None:
    print(f"[hoct_veto {_hv_time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- pure graph logic (unit-tested offline)

def _hv_apply_veto(edges: list, hoct_pairs: set, mode: int) -> tuple[list, dict]:
    """Drop pipeline edges that HOCT does not propose.

    edges      : the notebook's edge dicts (source_id, target_id, ...).
    hoct_pairs : set of (source_id, target_id) in the pipeline's node ids.
    mode       : 1 keeps both edges of every node with >= 2 children untouched; 2 vetoes all edges.
    Returns (kept edges, counters).
    """
    out_deg = _hv_Counter(int(e["source_id"]) for e in edges)
    kept = []
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        if (s, t) in hoct_pairs or (mode == 1 and out_deg[s] >= 2):
            kept.append(e)
    out_deg_after = _hv_Counter(int(e["source_id"]) for e in kept)
    counters = {
        "edges_before": len(edges), "edges_after": len(kept), "removed": len(edges) - len(kept),
        "divisions_before": sum(1 for v in out_deg.values() if v >= 2),
        "divisions_after": sum(1 for v in out_deg_after.values() if v >= 2),
    }
    return kept, counters


def _hv_snap_to_nodes(t: _hv_np.ndarray, zyx: _hv_np.ndarray, det: _hv_np.ndarray) -> _hv_np.ndarray:
    """Index of the nearest node (same frame, micrometre distance) for every HOCT node; one-to-one."""
    from scipy.spatial import cKDTree
    scale = _hv_np.asarray(_HV_SCALE_ZYX)
    idx_out = _hv_np.full(len(t), -1, dtype=int)
    for tt in _hv_np.unique(t):
        sel = _hv_np.where(t == tt)[0]
        det_sel = _hv_np.where(det[:, 0].astype(int) == int(tt))[0]
        if len(det_sel) == 0:
            raise RuntimeError(f"HOCT node at frame {tt} but no pipeline node there")
        dist, k = cKDTree(det[det_sel, 1:] * scale).query(zyx[sel] * scale, k=1)
        idx_out[sel] = det_sel[k]
    if len(_hv_np.unique(idx_out)) != len(idx_out):
        raise RuntimeError("HOCT nodes do not map one-to-one onto pipeline nodes")
    return idx_out


# ---------------------------------------------------------------- wall-clock budget (unit-tested offline)

def _hv_proc_age_s(pid) -> "float | None":
    """Seconds since process `pid` started: /proc/<pid>/stat field 22 (start time in clock ticks after
    boot) against /proc/uptime. None when unreadable (no /proc, no such process)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            stat = fh.read()
        start_ticks = int(stat[stat.rindex(")") + 2:].split()[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        return uptime - start_ticks / _hv_os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _hv_notebook_elapsed_s() -> float:
    """Seconds since the notebook session started (see the header for the reasoning)."""
    self_age = _hv_proc_age_s("self")
    if self_age is not None:
        init_age = _hv_proc_age_s(1)
        if init_age is not None and self_age <= init_age <= self_age + 3600.0:
            _HV_STATE["clock"] = "/proc/1/stat"
            return init_age
        _HV_STATE["clock"] = "/proc/self/stat"
        return self_age
    _HV_STATE["clock"] = "cell-relative+predict_seconds (no /proc; under-estimates)"
    detector = globals().get("predict_seconds", 0.0)
    return _hv_time.monotonic() - _HV_STATE["t_import_monotonic"] + float(detector or 0.0)


def _hv_estimate_s(n_nodes: int) -> float:
    """Predicted HOCT wall time for one video from its final node count."""
    return _HV_SEC_PER_1000_NODES * n_nodes / 1000.0 + _HV_FIXED_S


def _hv_budget_decision(n_nodes: int, elapsed_s: float, deadline_s: float, max_video_s: float) -> tuple[str, float]:
    """Pure budget rule. Returns (decision, predicted seconds); decision is "run", "skip_video_cap"
    (prediction above the per-video cap) or "skip_deadline" (deadline passed, or elapsed + prediction
    would pass it). Every video is judged on its own, so a small video can still run late."""
    est = _hv_estimate_s(n_nodes)
    if est > max_video_s:
        return "skip_video_cap", est
    if elapsed_s >= deadline_s or elapsed_s + est > deadline_s:
        return "skip_deadline", est
    return "run", est


# ---------------------------------------------------------------- HOCT plumbing (Kaggle only)

def _hv_find(pattern: str) -> str:
    hits = sorted(set(_hv_glob.glob(f"/kaggle/input/{pattern}") + _hv_glob.glob(f"/kaggle/input/*/{pattern}")
                      + _hv_glob.glob(f"/kaggle/input/*/*/{pattern}") + _hv_glob.glob(f"/kaggle/input/*/*/*/{pattern}")))
    if not hits:
        raise FileNotFoundError(f"HOCT veto: nothing matches /kaggle/input/**/{pattern}")
    return hits[0]


def _hv_install_and_load():
    if _HV_STATE["model"] is not None:
        return _HV_STATE["model"]
    wheel_dir = _hv_os.path.dirname(_hv_find("hoct-0.2.0-py3-none-any.whl"))
    weights = _hv_find("general_v0.pt")
    cmd = [_hv_sys.executable, "-m", "pip", "install", "--quiet", "--no-index", "--no-deps",
           "--find-links", wheel_dir, "hoct==0.2.0", "spatial-graph==0.1.1", "pooch==1.9.0"]
    _hv_log("installing offline wheels from " + wheel_dir)
    _hv_subprocess.run(cmd, check=True)
    import importlib
    importlib.invalidate_caches()
    import hoct  # noqa: E402
    import torch  # noqa: E402
    from hoct import load_model
    model = load_model(weights, device="cuda")
    _hv_log(f"hoct {hoct.__version__} loaded general_v0 from {weights} "
            f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params)")
    _HV_STATE["model"] = model
    return model


def _hv_release_gpu() -> None:
    """Return cached GPU memory after a failure so the notebook's own models are not squeezed."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _hv_rasterize_spheres(det: _hv_np.ndarray, shape: tuple) -> _hv_np.ndarray:
    """One 3-um anisotropic sphere per node; labels 1..K per frame; overlaps go to the nearest centre."""
    T, Z, Y, X = shape
    labels = _hv_np.zeros(shape, dtype=_hv_np.int16)
    sz, sy, sx = _HV_SCALE_ZYX
    rz, ry, rx = (int(_hv_math.ceil(_HV_RADIUS_UM / s)) for s in _HV_SCALE_ZYX)
    r2 = _HV_RADIUS_UM ** 2
    t_all = det[:, 0].astype(int)
    for t in range(T):
        rows = det[t_all == t]
        if len(rows) > 32000:
            raise RuntimeError(f"frame {t}: {len(rows)} nodes exceed the int16 label range")
        best = _hv_np.full((Z, Y, X), _hv_np.inf, dtype=_hv_np.float32)
        lab = labels[t]
        for k, (_, cz, cy, cx) in enumerate(rows, start=1):
            z0, z1 = max(0, int(_hv_math.floor(cz)) - rz), min(Z, int(_hv_math.ceil(cz)) + rz + 1)
            y0, y1 = max(0, int(_hv_math.floor(cy)) - ry), min(Y, int(_hv_math.ceil(cy)) + ry + 1)
            x0, x1 = max(0, int(_hv_math.floor(cx)) - rx), min(X, int(_hv_math.ceil(cx)) + rx + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            zz = (_hv_np.arange(z0, z1) - cz) * sz
            yy = (_hv_np.arange(y0, y1) - cy) * sy
            xx = (_hv_np.arange(x0, x1) - cx) * sx
            d2 = (zz[:, None, None] ** 2 + yy[None, :, None] ** 2 + xx[None, None, :] ** 2).astype(_hv_np.float32)
            sub_best = best[z0:z1, y0:y1, x0:x1]
            hit = (d2 <= r2) & (d2 < sub_best)
            sub_best[hit] = d2[hit]
            lab[z0:z1, y0:y1, x0:x1][hit] = k
    return labels


def _hv_read_volume(dataset: str) -> _hv_np.ndarray:
    """Raw (T, Z, Y, X) uint16 frames of a test movie via the notebook's own frame reader."""
    zarr_path = TEST_DIR / f"{dataset}.zarr"
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    T = int(meta["shape"][0])
    cache: dict = {}
    frames = [read_test_frame(dataset, t, cache) for t in range(T)]
    return _hv_np.stack(frames, axis=0)


def _hv_predict_pairs(model, labels, images, det, ids, n_chunks: int) -> set:
    """Run HOCT (whole movie, or n_chunks overlapping runs of frames stitched at the seams) and return
    its proposed edges as (source_id, target_id) pairs in the pipeline's node ids."""
    import torch
    from hoct import predict
    from tracksdata.functional import TilingScheme
    T = labels.shape[0]
    starts = [round(i * T / n_chunks) for i in range(n_chunks)] + [T]
    pairs: set = set()
    for i in range(n_chunks):
        s, e_own = starts[i], starts[i + 1]
        e_frames = min(T, e_own + 1)
        sub = _hv_np.where((det[:, 0] >= s) & (det[:, 0] < e_frames))[0]
        det_sub = det[sub].copy()
        det_sub[:, 0] -= s
        with torch.inference_mode():
            sol = predict(model, labels=labels[s:e_frames], images=None if images is None else images[s:e_frames],
                          scale=(1.0, *_HV_SCALE_ZYX), max_delta_t=_HV_MAX_DELTA_T,
                          tiling_scheme=TilingScheme(tile_shape=_HV_TILE, overlap_shape=_HV_OVERLAP))
        nodes = sol.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
        edges = sol.edge_attrs(attr_keys=[])
        t_loc = nodes["t"].to_numpy().astype(int)
        zyx = _hv_np.stack([nodes[c].to_numpy() for c in ("z", "y", "x")], 1).astype(float)
        snap = _hv_snap_to_nodes(t_loc, zyx, det_sub)          # index into det_sub
        hoct_to_pid = {int(n): int(ids[sub[k]]) for n, k in zip(nodes["node_id"].to_list(), snap)}
        src_t = {int(n): int(tt) + s for n, tt in zip(nodes["node_id"].to_list(), t_loc)}
        for a, b in zip(edges["source_id"].to_list(), edges["target_id"].to_list()):
            if s <= src_t[int(a)] < e_own:                    # this run owns edges whose source is in [s, e_own)
                pairs.add((hoct_to_pid[int(a)], hoct_to_pid[int(b)]))
        del sol
        torch.cuda.empty_cache()
    return pairs


def _hv_hoct_pairs(dataset: str, nodes_by_id: dict) -> set:
    ids = _hv_np.array(sorted(nodes_by_id), dtype=int)
    det = _hv_np.array([[float(nodes_by_id[i]["t"]), float(nodes_by_id[i]["z"]),
                         float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])] for i in ids])
    key = (dataset, _hv_hashlib.sha256(det.tobytes()).hexdigest())
    if key in _HV_STATE["cache"]:
        _hv_log(f"[{dataset}] HOCT edges reused from cache (identical final node set)")
        return _HV_STATE["cache"][key]
    model = _hv_install_and_load()
    t0 = _hv_time.time()
    volume = _hv_read_volume(dataset)
    labels = _hv_rasterize_spheres(det, volume.shape)
    _hv_log(f"[{dataset}] {len(ids)} final nodes -> spheres r={_HV_RADIUS_UM} um, volume {volume.shape} read+painted in {_hv_time.time() - t0:.0f}s")
    pairs = None
    deadline_s = _HV_DEADLINE_H * 3600.0
    for n_chunks in (1, 2, 4):
        if n_chunks > 1:
            elapsed = _hv_notebook_elapsed_s()
            if elapsed + _hv_estimate_s(len(ids)) > deadline_s:
                raise RuntimeError(f"HOCT retry with {n_chunks} chunks refused: elapsed {elapsed:.0f}s plus "
                                   f"the predicted {_hv_estimate_s(len(ids)):.0f}s would pass the {deadline_s:.0f}s deadline")
        try:
            t1 = _hv_time.time()
            pairs = _hv_predict_pairs(model, labels, volume, det, ids, n_chunks)
            _hv_log(f"[{dataset}] HOCT proposed {len(pairs)} edges in {_hv_time.time() - t1:.0f}s (time chunks={n_chunks})")
            break
        except RuntimeError as exc:   # tracksdata raises RuntimeError when SCIP cannot solve the whole-movie ILP
            _hv_log(f"[{dataset}] HOCT with {n_chunks} chunk(s) failed: {str(exc)[:160]}; retrying with more chunks")
            _hv_release_gpu()
    if pairs is None:
        raise RuntimeError(f"HOCT_VETO_FAILED {dataset}: HOCT produced no solution")
    _HV_STATE["cache"][key] = pairs
    return pairs


# ---------------------------------------------------------------- one video: budget, veto, fail-safe

def _hv_veto_video(dataset: str, nodes_by_id: dict, edges: list) -> tuple[list, dict]:
    """Apply the veto to one video's FINAL graph, or leave it untouched. Never raises.

    Returns (edges, counters); counters carries "status" -- vetoed / skip_deadline / skip_video_cap /
    failed -- and the edge and division counts (unchanged counts when the graph passed through).
    """
    n_nodes = len(nodes_by_id)
    counts = _HV_STATE["counts"]
    out_deg = _hv_Counter(int(e["source_id"]) for e in edges)
    untouched = {"edges_before": len(edges), "edges_after": len(edges), "removed": 0,
                 "divisions_before": sum(1 for v in out_deg.values() if v >= 2)}
    untouched["divisions_after"] = untouched["divisions_before"]

    if n_nodes == 0 or not edges:            # nothing HOCT could veto; do not spend a run on it
        counts["vetoed"] += 1
        print(f"HOCT_VETO_DATASET mode={_HV_MODE} dataset={dataset} edges_before={len(edges)} edges_after={len(edges)} "
              f"removed=0 divisions_before={untouched['divisions_before']} divisions_after={untouched['divisions_after']}", flush=True)
        return edges, {**untouched, "status": "vetoed"}

    if _HV_STATE["disabled_reason"] is not None:
        counts["failed"] += 1
        print(f"HOCT_VETO_FAILED mode={_HV_MODE} dataset={dataset} nodes={n_nodes} reason=hoct_unavailable "
              f"({_HV_STATE['disabled_reason']}); base graph kept", flush=True)
        return edges, {**untouched, "status": "failed"}

    elapsed = _hv_notebook_elapsed_s()
    deadline_s = _HV_DEADLINE_H * 3600.0
    decision, est = _hv_budget_decision(n_nodes, elapsed, deadline_s, _HV_MAX_VIDEO_S)
    if decision != "run":
        counts["skipped_budget"] += 1
        print(f"HOCT_VETO_SKIPPED mode={_HV_MODE} dataset={dataset} reason={decision} nodes={n_nodes} predicted_s={est:.0f} "
              f"elapsed_s={elapsed:.0f} deadline_s={deadline_s:.0f} max_video_s={_HV_MAX_VIDEO_S:.0f}; base graph kept", flush=True)
        return edges, {**untouched, "status": decision}

    t0 = _hv_time.time()
    try:
        _hv_install_and_load()
    except Exception as exc:
        _HV_STATE["disabled_reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        counts["failed"] += 1
        _hv_log("HOCT could not be installed or loaded; every video passes through unchanged\n" + _hv_traceback.format_exc())
        print(f"HOCT_VETO_FAILED mode={_HV_MODE} dataset={dataset} nodes={n_nodes} reason=install_or_load "
              f"({_HV_STATE['disabled_reason']}); base graph kept", flush=True)
        _hv_release_gpu()
        return edges, {**untouched, "status": "failed"}

    try:
        hoct_pairs = _hv_hoct_pairs(dataset, nodes_by_id)
        kept, c = _hv_apply_veto(edges, hoct_pairs, _HV_MODE)
    except Exception as exc:
        spent = _hv_time.time() - t0
        _HV_STATE["hoct_seconds"] += spent
        counts["failed"] += 1
        _hv_log(f"[{dataset}] HOCT veto FAILED after {spent:.0f}s; base graph kept\n" + _hv_traceback.format_exc())
        print(f"HOCT_VETO_FAILED mode={_HV_MODE} dataset={dataset} nodes={n_nodes} reason={type(exc).__name__} "
              f"seconds={spent:.0f}; base graph kept", flush=True)
        _hv_release_gpu()
        return edges, {**untouched, "status": "failed"}

    _HV_STATE["hoct_seconds"] += _hv_time.time() - t0
    counts["vetoed"] += 1
    print(f"HOCT_VETO_DATASET mode={_HV_MODE} dataset={dataset} edges_before={c['edges_before']} "
          f"edges_after={c['edges_after']} removed={c['removed']} divisions_before={c['divisions_before']} "
          f"divisions_after={c['divisions_after']}", flush=True)
    return kept, {**c, "status": "vetoed"}


# ---------------------------------------------------------------- backup of the previous outputs

def _hv_output_paths() -> list:
    """submission.csv and run_stats.csv as written by the notebook (globals from the paths cell)."""
    out = []
    for name in ("SUBMISSION_PATH", "RUN_STATS_PATH"):
        p = globals().get(name)
        if p is not None:
            out.append(str(p))
    return out


def _hv_backup_outputs() -> dict:
    """Copy the current outputs to <file>.hoct_veto_backup so a failed write can be undone.
    Returns {original path: backup path} for the files that existed."""
    backup: dict = {}
    for p in _hv_output_paths():
        try:
            if _hv_os.path.isfile(p):
                b = p + _HV_BACKUP_SUFFIX
                _hv_shutil.copy2(p, b)
                backup[p] = b
        except Exception as exc:
            _hv_log(f"could not back up {p}: {type(exc).__name__}: {exc}")
    return backup


def _hv_restore_outputs(backup: dict) -> bool:
    """Put the backed-up files back (os.replace is atomic on one filesystem). True when submission.csv
    was restored."""
    restored_submission = False
    submission = str(globals().get("SUBMISSION_PATH", ""))
    for p, b in backup.items():
        try:
            _hv_os.replace(b, p)
            if p == submission:
                restored_submission = True
        except Exception as exc:
            _hv_log(f"could not restore {p} from {b}: {type(exc).__name__}: {exc}")
    return restored_submission


def _hv_keep_previous_submission(backup: dict) -> None:
    """After a successful wrapped write: keep the previous submission.csv next to the final one for
    inspection (submission_before_hoct_veto.csv) and drop the other backups."""
    submission = str(globals().get("SUBMISSION_PATH", ""))
    for p, b in backup.items():
        try:
            if p == submission:
                _hv_os.replace(b, _hv_os.path.join(_hv_os.path.dirname(p), _HV_KEEP_PREVIOUS_NAME))
            else:
                _hv_os.remove(b)
        except Exception as exc:
            _hv_log(f"could not tidy backup {b}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- hook into write_test_submission

def _hv_install_hook() -> None:
    orig_write = write_test_submission
    orig_filter = filter_output_graph

    def veto_filter_output_graph(nodes_by_id, raw_edges, *args, **kwargs):
        nodes_by_id, edges, stats = orig_filter(nodes_by_id, raw_edges, *args, **kwargs)
        try:
            dataset = kwargs.get("dataset") if "dataset" in kwargs else (args[0] if args else None)
            edges, c = _hv_veto_video(str(dataset), nodes_by_id, edges)       # never raises
            stats["hoct_veto_status"] = c["status"]
            stats["hoct_veto_removed_edges"] = c["removed"]
            stats["hoct_veto_divisions_before"] = c["divisions_before"]
            stats["hoct_veto_divisions_after"] = c["divisions_after"]
            tot = _HV_STATE["totals"]
            for k in ("edges_before", "edges_after", "removed", "divisions_before", "divisions_after"):
                tot[k] = tot.get(k, 0) + c[k]
        except Exception:
            _hv_log("bookkeeping error in the veto hook; the graph is passed through as post-processed\n" + _hv_traceback.format_exc())
        return nodes_by_id, edges, stats

    def write_test_submission_with_veto(tag: str = "base"):
        _HV_STATE["totals"] = {}
        backup = _hv_backup_outputs()
        globals()["filter_output_graph"] = veto_filter_output_graph
        try:
            result = orig_write(tag)
        except Exception as exc:
            globals()["filter_output_graph"] = orig_filter
            _HV_STATE["counts"]["write_failures"] += 1
            restored = _hv_restore_outputs(backup)
            _hv_log(f"write_test_submission({tag!r}) FAILED with the veto hooked; previous outputs restored={restored}\n"
                    + _hv_traceback.format_exc())
            if not restored:
                _hv_log("no previous submission.csv to fall back on; re-running the notebook's own write without the veto")
                return orig_write(tag)
            print(f"HOCT_VETO_WRITE_FAILED mode={_HV_MODE} tag={tag} error={type(exc).__name__} "
                  f"restored_previous_submission=True", flush=True)
            return None
        finally:
            globals()["filter_output_graph"] = orig_filter
        _hv_keep_previous_submission(backup)
        _HV_STATE["applied_writes"] += 1
        c = _HV_STATE["totals"]
        cnt = _HV_STATE["counts"]
        print(f"HOCT_VETO_ACTIVE mode={_HV_MODE} edges_before={c.get('edges_before', 0)} edges_after={c.get('edges_after', 0)} "
              f"removed={c.get('removed', 0)} divisions_before={c.get('divisions_before', 0)} "
              f"divisions_after={c.get('divisions_after', 0)}", flush=True)
        if c.get("removed", 0) == 0:
            print(f"HOCT_VETO_NO_OP mode={_HV_MODE}: no edge removed (vetoed={cnt['vetoed']} skipped_budget={cnt['skipped_budget']} "
                  f"failed={cnt['failed']}); submission.csv equals the notebook's own write", flush=True)
        return result

    globals()["write_test_submission"] = write_test_submission_with_veto


# ---------------------------------------------------------------- finalize (called from the finalize cell)

def _hv_print_summary() -> None:
    c = _HV_STATE["counts"]
    elapsed = _hv_notebook_elapsed_s()
    print(f"HOCT_VETO_SUMMARY mode={_HV_MODE} vetoed={c['vetoed']} skipped_budget={c['skipped_budget']} failed={c['failed']} "
          f"write_failures={c['write_failures']} applied_writes={_HV_STATE['applied_writes']} "
          f"hoct_seconds={_HV_STATE['hoct_seconds']:.0f} elapsed_s={elapsed:.0f} deadline_h={_HV_DEADLINE_H} "
          f"max_video_s={_HV_MAX_VIDEO_S:.0f} clock={_HV_STATE['clock']}", flush=True)


def _hv_finalize() -> None:
    """Run after the post-process sweep. If the sweep re-wrote submission.csv the veto is already in
    it; if it kept the base file, re-write that file once with the veto. Never raises."""
    if _HV_MODE not in (1, 2):
        return
    try:
        if _HV_STATE["applied_writes"] > 0:
            _hv_log("the veto was applied inside the sweep's re-write of submission.csv; nothing to re-write")
        elif _HV_STATE["counts"]["write_failures"] > 0:
            _hv_log("a wrapped write failed earlier and the previous submission.csv was restored; not retrying")
        else:
            _hv_log("the sweep kept the base submission.csv: re-writing it once with the HOCT consensus veto")
            write_test_submission("base")
    except Exception:
        _HV_STATE["counts"]["write_failures"] += 1
        _hv_log("finalize failed; submission.csv is whatever the notebook wrote last\n" + _hv_traceback.format_exc())
    _hv_print_summary()


# ---------------------------------------------------------------- cell entry point
if _HV_MODE == 0:
    print("HOCT_VETO_DISABLED mode=0 (base behaviour; submission.csv unchanged)", flush=True)
elif _HV_MODE in (1, 2):
    _hv_install_hook()
    _hv_elapsed_at_arming = _hv_notebook_elapsed_s()
    print(f"HOCT_VETO_ARMED mode={_HV_MODE} ({'division edges protected' if _HV_MODE == 1 else 'division edges vetoed too'}) "
          f"elapsed_s={_hv_elapsed_at_arming:.0f} deadline_h={_HV_DEADLINE_H} max_video_s={_HV_MAX_VIDEO_S:.0f} "
          f"clock={_HV_STATE['clock']}; applied inside the notebook's final write of submission.csv", flush=True)
else:
    raise RuntimeError(f"BIOHUB_HOCT_VETO must be 0, 1 or 2, got {_HV_MODE!r}")


from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

_guard_submission = Path("/kaggle/working/submission.csv")
_guard_columns = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
if not _guard_submission.is_file():
    raise FileNotFoundError(_guard_submission)
_guard_frame = pd.read_csv(_guard_submission)
if _guard_frame.empty or _guard_frame.columns.tolist() != _guard_columns:
    raise RuntimeError("Retention-guard submission schema changed")
if _guard_frame["id"].tolist() != list(range(len(_guard_frame))):
    raise RuntimeError("Retention-guard row IDs are not contiguous")
if set(_guard_frame["row_type"].unique()) != {"node", "edge"}:
    raise RuntimeError("Retention-guard row types changed")

_guard_datasets = sorted(_guard_frame["dataset"].astype(str).unique())
_guard_expected = sorted(
    path.name.removesuffix(".zarr")
    for path in TEST_DIR.iterdir()
    if path.name.endswith(".zarr")
)
if _guard_datasets != _guard_expected:
    raise RuntimeError({"expected": _guard_expected, "actual": _guard_datasets})

_guard_records = []
for _guard_path in sorted(Path("/kaggle/working").glob("retention_guard_*.jsonl")):
    for _guard_line in _guard_path.read_text().splitlines():
        if _guard_line.strip():
            _guard_records.append(json.loads(_guard_line))
if not _guard_records:
    raise RuntimeError("No frame-retention diagnostics were produced")

_guard_keys = [
    (str(row["dataset"]), int(row["frame"])) for row in _guard_records
]
if len(_guard_keys) != len(set(_guard_keys)):
    raise RuntimeError("Duplicate frame-retention diagnostics")
if sorted(set(movie for movie, _ in _guard_keys)) != _guard_expected:
    raise RuntimeError("Frame-retention diagnostics do not cover every movie")
for _guard_record in _guard_records:
    if (
        float(_guard_record["minimum_retention"])
        != 0.9
        or int(_guard_record["primary_candidates"]) < 0
        or int(_guard_record["blended_candidates"]) < 0
    ):
        raise RuntimeError("Frame-retention diagnostic contract changed")
    _guard_expected_use_primary = bool(
        int(_guard_record["primary_candidates"]) > 0
        and float(_guard_record["retention"])
        < 0.9
    )
    if bool(_guard_record["use_primary"]) != _guard_expected_use_primary:
        raise RuntimeError("Frame-retention decision is inconsistent")

_guard_topology = {}
for _guard_movie, _guard_group in _guard_frame.groupby("dataset", sort=True):
    _guard_nodes = _guard_group[_guard_group["row_type"].eq("node")]
    _guard_edges = _guard_group[_guard_group["row_type"].eq("edge")]
    if _guard_nodes.empty or _guard_nodes["t"].lt(0).any():
        raise RuntimeError(f"{_guard_movie}: invalid biological node time")
    if _guard_nodes[["z", "y", "x"]].lt(0).any().any():
        raise RuntimeError(f"{_guard_movie}: negative biological coordinate")
    _guard_node_time = dict(zip(
        _guard_nodes["node_id"].astype(int),
        _guard_nodes["t"].astype(int),
    ))
    _guard_incoming = Counter()
    _guard_outgoing = Counter()
    for _guard_edge in _guard_edges.itertuples():
        _guard_source = int(_guard_edge.source_id)
        _guard_target = int(_guard_edge.target_id)
        if (
            _guard_source not in _guard_node_time
            or _guard_target not in _guard_node_time
            or _guard_node_time[_guard_target]
            != _guard_node_time[_guard_source] + 1
        ):
            raise RuntimeError(f"{_guard_movie}: invalid lineage edge")
        _guard_incoming[_guard_target] += 1
        _guard_outgoing[_guard_source] += 1
    _guard_max_in = max(_guard_incoming.values(), default=0)
    _guard_max_out = max(_guard_outgoing.values(), default=0)
    if _guard_max_in > 1 or _guard_max_out > 2:
        raise RuntimeError(f"{_guard_movie}: invalid lineage degree")
    _guard_topology[_guard_movie] = {
        "nodes": int(len(_guard_nodes)),
        "edges": int(len(_guard_edges)),
        "max_indegree": int(_guard_max_in),
        "max_outdegree": int(_guard_max_out),
        "division_parents": int(sum(
            value == 2 for value in _guard_outgoing.values()
        )),
    }

_guard_by_movie = {}
for _guard_movie in _guard_expected:
    _guard_movie_records = [
        row for row in _guard_records if row["dataset"] == _guard_movie
    ]
    _guard_by_movie[_guard_movie] = {
        "frames": int(len(_guard_movie_records)),
        "fallback_frames": int(sum(
            bool(row["use_primary"]) for row in _guard_movie_records
        )),
        "minimum_retention": float(min(
            row["retention"] for row in _guard_movie_records
        )),
        "median_retention": float(pd.Series(
            [row["retention"] for row in _guard_movie_records]
        ).median()),
    }

_guard_digest = hashlib.sha256(_guard_submission.read_bytes()).hexdigest()
_guard_report = {
    "experiment": "harmonic_bidirectional_association_v1",
    "status": "clean_graph_audit_pass_candidate_unverified_quality",
    "parent_experiment": "paired_bidirectional_primary_weight020_vs_forward_v1",
    "method_attribution": "fixed-90 dual-seed baseline with harmonic mutual-support association fusion (rule from public CC0 notebook yusuketogashi/no-hack-biohub-cell-another-approch-3rd v18)",
    "source_kernel": "raykkretzschmar/biohub-bidirectional-primary-union13-diagnostic-v1",
    "source_notebook_sha256": "3e65ca691941949196bf417030ea84fccafe16baaec174b2a63540451bb937e8",
    "public_output_used": False,
    "metric_hack_used": False,
    "organizer_labels_used_for_configuration": False,
    "leaderboard_feedback_used_for_configuration": True,
    "configuration": {
        "minimum_candidate_retention": 0.9,
        "fallback_scope": "individual_frame",
        "detector_threshold": 0.96875,
        "secondary_detection_weight": 0.475,
        "secondary_edge_weight": 0.15,
        "bidirectional_primary_weight": 0.30,
        "secondary_link_mode": "low_margin_consensus",
        "secondary_low_margin_max": 0.35,
        "edge_candidate_threshold": 0.48,
        "ilp_appearance_weight": 0.0,
        "ilp_disappearance_weight": 1.5,
        "gap_close_um": 5.8,
        "deepcenter_gap_threshold": 0.25,
        "deepcenter_gap_confirm_min_span_um": 8.5,
    },
    "hardware": {
        "visible_gpu_count": int(torch.cuda.device_count()),
    },
    "diagnostics": {
        "rows": int(len(_guard_records)),
        "fallback_frames": int(sum(
            bool(row["use_primary"]) for row in _guard_records
        )),
        "by_movie": _guard_by_movie,
    },
    "submission": {
        "sha256": _guard_digest,
        "rows": int(len(_guard_frame)),
        "datasets": _guard_datasets,
    },
    "topology": _guard_topology,
    "quality_promotion": {
        "status": "candidate_unverified",
        "required_receipt": "bidirectional_blend_union13_receipt.json",
        "required_condition": "promote=true",
        "execute_push_submit": "FORBIDDEN_UNTIL_REQUIRED_CONDITION",
        "validated_receipt_sha256": None,
    },
}
Path("/kaggle/working/dual_seed_frame_retention_guard_report.json").write_text(
    json.dumps(_guard_report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(_guard_report, indent=2, sort_keys=True))









TRAIN_DIR = COMP_DIR / "train"

VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1") != "0"
VALIDATOR_N_PER_TYPE = int(os.environ.get("BIOHUB_VALIDATOR_N_PER_TYPE", "2"))
VALIDATOR_MATCH_RADIUS_UM = float(os.environ.get("BIOHUB_VALIDATOR_MATCH_RADIUS_UM", "7.0"))
VALIDATOR_NODE_COUNT_PENALTY_A = float(os.environ.get("BIOHUB_VALIDATOR_NODE_COUNT_PENALTY_A", "0.1"))
VALIDATOR_DIVISION_WEIGHT = float(os.environ.get("BIOHUB_VALIDATOR_DIVISION_WEIGHT", "0.1"))
VALIDATOR_STATS_PATH = WORKING_DIR / "validator_results.csv"

val_stems: list[str] = []
if VALIDATOR_ENABLE and TRAIN_DIR.exists():
    train_stems_all = sorted(p.name[:-5] for p in TRAIN_DIR.iterdir() if p.name.endswith(".zarr"))
    test_stem_set = set(test_stems)  
    overlap = [s for s in train_stems_all if s in test_stem_set]
    if overlap:
        print(f"VALIDATOR: excluding {len(overlap)} TRAIN stem(s) that also appear in TEST_DIR: {overlap}")
    candidates = [s for s in train_stems_all if s not in test_stem_set]

    
    
    
    
    
    
    
    def _stem_has_gt_division(stem: str) -> bool:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        try:
            graph = graph_from_geff(gt_path)
        except Exception:
            return False
        out_degree: dict[int, int] = {}
        for row in graph.edge_attrs().iter_rows(named=True):
            s = int(row["source_id"])
            out_degree[s] = out_degree.get(s, 0) + 1
        return any(d >= 2 for d in out_degree.values())

    by_prefix: dict[str, list[str]] = {}
    for s in candidates:
        by_prefix.setdefault(s.split("_")[0], []).append(s)

    division_flags: dict[str, bool] = {}
    for prefix, stems in by_prefix.items():
        for s in stems:
            division_flags[s] = _stem_has_gt_division(s)

    for prefix, stems in sorted(by_prefix.items()):
        ranked = sorted(stems, key=lambda s: (not division_flags[s], s))
        val_stems.extend(ranked[:VALIDATOR_N_PER_TYPE])
    n_division_selected = sum(1 for s in val_stems if division_flags[s])
    print(f"VALIDATOR: selected {len(val_stems)} held-out TRAIN samples "
          f"({VALIDATOR_N_PER_TYPE} per embryo-type prefix, {len(by_prefix)} prefixes found, "
          f"{n_division_selected} contain a GT division)")
    print(val_stems)
elif VALIDATOR_ENABLE:
    print(f"VALIDATOR: TRAIN_DIR not found at {TRAIN_DIR} -- skipping.")
else:
    print("VALIDATOR: disabled (BIOHUB_VALIDATOR_ENABLE=0).")


def _merge_validator_shards(worker_count: int, stems: list[str], method_prefix: str) -> Path:
    """Same logic as _merge_prediction_shards (Cell 6), parameterized for an
    arbitrary stem list and method prefix instead of the global test_stems/
    METHOD -- that function is hardcoded to the real test run and isn't
    safe to call directly for a different sample set."""
    import shutil as _shutil
    shard_dirs: list[Path] = []
    seen: set[str] = set()
    expected_all = set(stems)

    for shard_index in range(worker_count):
        shard_method = f"{method_prefix}_gpu{shard_index}"
        shard_dir = _prediction_dir_for_method(shard_method)
        expected = set(stems[shard_index::worker_count])
        found = {p.stem for p in sorted(shard_dir.glob("*.geff"))}
        if found != expected:
            raise RuntimeError(
                f"VALIDATOR shard {shard_index} output mismatch: "
                f"missing={sorted(expected - found)}, extra={sorted(found - expected)}"
            )
        overlap_ds = seen & found
        if overlap_ds:
            raise RuntimeError(f"VALIDATOR: duplicate datasets across shards: {sorted(overlap_ds)}")
        seen.update(found)
        shard_dirs.append(shard_dir)

    if seen != expected_all:
        raise RuntimeError(
            f"VALIDATOR: merged shards do not cover the held-out set: "
            f"missing={sorted(expected_all - seen)}, extra={sorted(seen - expected_all)}"
        )

    username_roots = {shard_dir.parents[1] for shard_dir in shard_dirs}
    if len(username_roots) != 1:
        raise RuntimeError(f"VALIDATOR: shards used inconsistent prediction roots: {username_roots}")

    final_root = next(iter(username_roots)) / method_prefix
    final_dir = final_root / "split_0"
    staging_dir = final_root / "split_0_val_staging"
    if staging_dir.exists():
        _shutil.rmtree(staging_dir) if staging_dir.is_dir() else staging_dir.unlink()
    staging_dir.mkdir(parents=True, exist_ok=False)

    for shard_dir in shard_dirs:
        for source in sorted(shard_dir.glob("*.geff")):
            destination = staging_dir / source.name
            if destination.exists():
                raise RuntimeError(f"VALIDATOR: refusing to overwrite duplicate output: {destination}")
            _shutil.move(str(source), str(destination))

    merged = {p.stem for p in staging_dir.glob("*.geff")}
    if merged != expected_all:
        raise RuntimeError(
            f"VALIDATOR: staged directory failed verification: "
            f"missing={sorted(expected_all - merged)}, extra={sorted(merged - expected_all)}"
        )

    if final_dir.exists():
        _shutil.rmtree(final_dir) if final_dir.is_dir() else final_dir.unlink()
    staging_dir.rename(final_dir)
    for shard_dir in shard_dirs:
        _shutil.rmtree(shard_dir.parent)
    print(f"VALIDATOR: merged {len(merged)} prediction graphs into {final_dir}")
    return final_dir


predict_val_seconds = None
if VALIDATOR_ENABLE and val_stems:
    val_splits_path = REPO_DIR / "kaggle_val_splits.json"
    val_splits_path.write_text(json.dumps([{"split": 0, "train": [], "test": val_stems}], indent=2))
    val_method_prefix = f"{METHOD}_val"

    predict_val_cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", str(TRAIN_DIR),
        "--splits", str(val_splits_path.name),
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        predict_val_cmd.append("--use-ilp")

    _val_start = time.time()
    val_worker_count = min(2, _torch.cuda.device_count(), len(val_stems))
    if val_worker_count >= 2:
        cuda_tokens = _visible_cuda_tokens(val_worker_count)
        val_processes: dict[int, subprocess.Popen] = {}
        val_commands: dict[int, list[str]] = {}
        print(f"VALIDATOR: launching {val_worker_count} shards on CUDA devices {cuda_tokens}")
        for shard_index in range(val_worker_count):
            shard_cmd = [*predict_val_cmd, "--method", f"{val_method_prefix}_gpu{shard_index}",
                         "--slice", f"{shard_index}::{val_worker_count}"]
            shard_env = {**os.environ, "PYTHONPATH": "src"}
            shard_env["CUDA_VISIBLE_DEVICES"] = cuda_tokens[shard_index]
            val_commands[shard_index] = shard_cmd
            val_processes[shard_index] = subprocess.Popen(shard_cmd, cwd=REPO_DIR, env=shard_env)
        _wait_for_prediction_shards(val_processes, val_commands)
        _merge_validator_shards(val_worker_count, val_stems, val_method_prefix)
    else:
        print("VALIDATOR: using single-process prediction (fewer than 2 GPUs or samples).")
        subprocess.run([*predict_val_cmd, "--method", val_method_prefix],
                        cwd=REPO_DIR, env={**os.environ, "PYTHONPATH": "src"}, check=True)
    predict_val_seconds = time.time() - _val_start
    print(f"VALIDATOR: prediction completed in {predict_val_seconds / 60:.2f} minutes")








from scipy.optimize import linear_sum_assignment


def match_nodes_bipartite(pred_nodes: dict, gt_nodes: dict, max_dist: float = 7.0):
    pred_by_t: dict[int, list[int]] = {}
    for pid, (t, *_r) in pred_nodes.items():
        pred_by_t.setdefault(int(t), []).append(pid)
    gt_by_t: dict[int, list[int]] = {}
    for gid, (t, *_r) in gt_nodes.items():
        gt_by_t.setdefault(int(t), []).append(gid)

    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}
    for t, p_ids in pred_by_t.items():
        g_ids = gt_by_t.get(t, [])
        if not g_ids:
            continue
        voxel_scale = np.array(VOXEL_SCALE_UM, dtype=float)
        p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=float) * voxel_scale
        g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=float) * voxel_scale
        diff = p_pos[:, None, :] - g_pos[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=-1))
        BIG = 1e6
        cost_gated = np.where(cost <= max_dist, cost, BIG)
        row_ind, col_ind = linear_sum_assignment(cost_gated)
        for r, c in zip(row_ind, col_ind):
            if cost_gated[r, c] >= BIG:
                continue
            pred_to_gt[p_ids[r]] = g_ids[c]
            gt_to_pred[g_ids[c]] = p_ids[r]
    return pred_to_gt, gt_to_pred


def compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    gt_edge_set = set(gt_edges)
    gt_outgoing: dict[int, set[int]] = {}
    gt_incoming_source: dict[int, int] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming_source[t] = s

    tp = 0
    fp = 0
    matched_gt_edges = set()
    for s, t in pred_edges:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, ())
        if is_tp:
            tp += 1
            matched_gt_edges.add((ms, mt))
            continue
        is_fp = (mt is not None and mt in gt_incoming_source) or (
            ms is not None and bool(gt_outgoing.get(ms))
        )
        if is_fp:
            fp += 1
    fn = len(gt_edge_set - matched_gt_edges)
    return tp, fp, fn


def edge_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom else 0.0


def adjusted_jaccard(jaccard: float, t_pred: int, t_true, a: float = 0.1) -> float:
    if not t_true or t_true <= 0:
        return jaccard
    return max(0.0, jaccard * (1.0 - a * (t_pred - t_true) / t_true))


def weakly_connected_components(node_ids, edges):
    parent = {n: n for n in node_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s, t in edges:
        if s in parent and t in parent:
            union(s, t)
    return {n: find(n) for n in node_ids}


def compute_division_confusion(pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred):
    gt_out: dict[int, set[int]] = {}
    gt_in: dict[int, int] = {}
    for s, t in gt_edges:
        gt_out.setdefault(s, set()).add(t)
        gt_in[t] = s

    pred_out: dict[int, set[int]] = {}
    for s, t in pred_edges:
        pred_out.setdefault(s, set()).add(t)

    pred_node_ids = list(pred_nodes.keys())
    pred_edge_list = list(pred_edges)
    components = weakly_connected_components(pred_node_ids, pred_edge_list)
    fork_components = {
        components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components
    }
    gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

    def lineage_descendants(root_child: int) -> set[int]:
        seen = {root_child}
        stack = [root_child]
        while stack:
            cur = stack.pop()
            for nxt in gt_out.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    tp = 0
    fn = 0
    tp_gt_sources: set[int] = set()

    for gsrc in gt_division_sources:
        children = sorted(gt_out[gsrc])
        if len(children) < 2:
            continue
        anchor_candidates = [gsrc]
        if gsrc in gt_in:
            anchor_candidates.append(gt_in[gsrc])
        anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]

        lineage_hit_components: list[set[int]] = []
        ok = True
        for child in children[:2]:
            lineage = lineage_descendants(child)
            hit_comp_ids = {
                components[p_id]
                for gt_id in lineage
                if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components
            }
            if not hit_comp_ids:
                ok = False
                break
            lineage_hit_components.append(hit_comp_ids)

        if not ok or not anchor_pred_nodes:
            fn += 1
            continue

        anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
        if not anchor_comp_ids:
            fn += 1
            continue

        found = any(
            comp_id in lineage_hit_components[0]
            and comp_id in lineage_hit_components[1]
            and comp_id in fork_components
            for comp_id in anchor_comp_ids
        )
        if found:
            tp += 1
            tp_gt_sources.add(gsrc)
        else:
            fn += 1

    fp = 0
    for n, outs in pred_out.items():
        if len(outs) < 2:
            continue
        g = pred_to_gt.get(n)
        if g is None or g not in gt_out or g in tp_gt_sources:
            continue
        fp += 1

    return tp, fp, fn


def decompose_errors(pred_nodes, gt_nodes, pred_edges, gt_edges, pred_to_gt, gt_to_pred):
    """Splits error mass into detection vs. fragmentation vs. wrong-association,
    using the exact same pred_to_gt/gt_to_pred matching compute_edge_confusion
    uses. Division errors are already isolated by compute_division_confusion;
    this covers everything else -- the diagnostic breakdown for deciding
    whether further gains are in detection, linking, or fragmentation."""
    gt_edge_set = set(gt_edges)
    pred_edge_set = set(pred_edges)
    gt_outgoing: dict[int, set[int]] = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)

    missed_gt_nodes = sum(1 for g in gt_nodes if g not in gt_to_pred)
    spurious_pred_nodes = sum(1 for p in pred_nodes if p not in pred_to_gt)

    recovered = fragmented = lost_to_detection = 0
    for gs, gtid in gt_edge_set:
        ps, pt = gt_to_pred.get(gs), gt_to_pred.get(gtid)
        if ps is None or pt is None:
            lost_to_detection += 1
        elif (ps, pt) in pred_edge_set:
            recovered += 1
        else:
            fragmented += 1

    wrong_association = 0
    for ps, pt in pred_edge_set:
        ms, mt = pred_to_gt.get(ps), pred_to_gt.get(pt)
        if ms is not None and mt is not None and mt not in gt_outgoing.get(ms, ()):
            wrong_association += 1

    return {
        "missed_gt_nodes": missed_gt_nodes,
        "spurious_pred_nodes": spurious_pred_nodes,
        "edges_recovered": recovered,
        "edges_fragmented": fragmented,
        "edges_lost_to_detection": lost_to_detection,
        "wrong_association_edges": wrong_association,
    }


def _find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key_recursive(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def read_estimated_true_node_count(geff_path: Path):
    for candidate in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        found = _find_key_recursive(payload, "estimated_number_of_nodes")
        if found is not None:
            try:
                return float(found)
            except (TypeError, ValueError):
                continue
    return None


def graph_to_plain(graph):
    nodes: dict[int, tuple] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes[node_id] = (int(row["t"]), float(row["z"]), float(row["y"]), float(row["x"]))
    edges: list[tuple[int, int]] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edges.append((int(row["source_id"]), int(row["target_id"])))
    return nodes, edges


def nodes_by_id_to_plain(nodes_by_id):
    return {nid: (int(n["t"]), float(n["z"]), float(n["y"]), float(n["x"])) for nid, n in nodes_by_id.items()}


def score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true):
    p2g, g2p = match_nodes_bipartite(pred_nodes_plain, gt_nodes_plain, max_dist=VALIDATOR_MATCH_RADIUS_UM)
    tp, fp, fn = compute_edge_confusion(pred_edges_plain, gt_edges_plain, p2g, g2p)
    jac = edge_jaccard(tp, fp, fn)
    t_pred = len(pred_nodes_plain)
    adj = adjusted_jaccard(jac, t_pred, t_true, a=VALIDATOR_NODE_COUNT_PENALTY_A)
    div_tp, div_fp, div_fn = compute_division_confusion(
        pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, p2g, g2p
    )
    errors = decompose_errors(pred_nodes_plain, gt_nodes_plain, pred_edges_plain, gt_edges_plain, p2g, g2p)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    row = {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn, "edge_jaccard": jac,
        "t_pred": t_pred, "t_true": t_true, "adjusted_edge_jaccard": adj,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn, "div_jaccard": div_jac,
        "weight": tp + fp + fn,
    }
    row.update(errors)
    return row


def aggregate_official(sample_rows):
    total_w = sum(r["weight"] for r in sample_rows) or 1
    weighted_adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in sample_rows) / total_w
    div_tp = sum(r["div_tp"] for r in sample_rows)
    div_fp = sum(r["div_fp"] for r in sample_rows)
    div_fn = sum(r["div_fn"] for r in sample_rows)
    div_jac = edge_jaccard(div_tp, div_fp, div_fn)
    return {
        "adjusted_edge_jaccard": weighted_adj,
        "division_jaccard": div_jac,
        "proxy_score": weighted_adj + VALIDATOR_DIVISION_WEIGHT * div_jac,
        "div_tp": div_tp, "div_fp": div_fp, "div_fn": div_fn,
        "missed_gt_nodes": sum(r["missed_gt_nodes"] for r in sample_rows),
        "spurious_pred_nodes": sum(r["spurious_pred_nodes"] for r in sample_rows),
        "edges_recovered": sum(r["edges_recovered"] for r in sample_rows),
        "edges_fragmented": sum(r["edges_fragmented"] for r in sample_rows),
        "edges_lost_to_detection": sum(r["edges_lost_to_detection"] for r in sample_rows),
        "wrong_association_edges": sum(r["wrong_association_edges"] for r in sample_rows),
    }

import copy as _copy





PP_SWEEP_KEYS = [
    "SAFE_DIV_MAX_UM", "SAFE_DIV_SISTER_MAX_UM", "SAFE_DIV_DIVERGE_UM",
    "SAFE_DIV_SISTER_SYMMETRY_TAU", "SAFE_DIV_EXISTING_CHILD_MAX_UM",
    "SAFE_DIV_FRAME_FRAC_CAP", "SAFE_DIV_GLOBAL_FRAC_CAP",
    "DEEPCENTER_SAFE_DIV_THRESHOLD", "DEEPCENTER_GAP_THRESHOLD",
    "GAP_CLOSE_UM", "OUTPUT_MIN_TRACK_LEN",
    "SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB", "MOTION_RELINK_TIGHT_UM",
    "MOTION_RELINK_RELAXED_UM", "GAP2_MAX_STEP_UM", "GAP2_MAX_TOTAL_UM",
    "MOTION_RELINK_LEARNED_BONUS", "MOTION_RELINK_VELOCITY_WEIGHT",
    "GAP_CLOSE_REUSE_UM", "OUTPUT_EDGE_MAX_UM",
]
PP_BASE_CONFIG = {key: globals()[key] for key in PP_SWEEP_KEYS}
print("Post-process base configuration (public 0.939):")
for key in PP_SWEEP_KEYS:
    print(f"  {key:<40} {PP_BASE_CONFIG[key]}")


def pp_apply(config: dict) -> dict:
    saved = {key: globals()[key] for key in config}
    for key, value in config.items():
        if key not in PP_SWEEP_KEYS:
            raise KeyError(f"{key} is not a sweepable post-process constant")
        globals()[key] = type(PP_BASE_CONFIG[key])(value)
    return saved


def pp_restore(saved: dict) -> None:
    for key, value in saved.items():
        globals()[key] = value


VAL_RAW_GRAPHS: dict[str, tuple[dict, list]] = {}
VAL_GT: dict[str, tuple[list, list, object]] = {}

if VALIDATOR_ENABLE and val_stems:
    val_pred_paths = {
        stem: found
        for stem in val_stems
        if (found := next((REPO_DIR / "predictions").rglob(f"{stem}.geff"), None)) is not None
    }
    missing = [s for s in val_stems if s not in val_pred_paths]
    if missing:
        raise RuntimeError(f"VALIDATOR: no prediction .geff for {missing}")
    for stem in val_stems:
        gt_path = TRAIN_DIR / f"{stem}.geff"
        if not gt_path.exists():
            raise RuntimeError(f"VALIDATOR: missing GT {gt_path}")
        gt_graph = graph_from_geff(gt_path)
        gt_nodes_plain, gt_edges_plain = graph_to_plain(gt_graph)
        t_true = read_estimated_true_node_count(gt_path)
        if t_true is None:
            raise RuntimeError(f"VALIDATOR: estimated_number_of_nodes missing for {stem}")
        VAL_GT[stem] = (gt_nodes_plain, gt_edges_plain, t_true)

        pred_graph = graph_from_geff(val_pred_paths[stem])
        raw_nodes_by_id: dict[int, dict[str, object]] = {}
        for row in pred_graph.node_attrs().iter_rows(named=True):
            node_id = int(row["node_id"])
            raw_nodes_by_id[node_id] = {
                "node_id": node_id, "t": int(row["t"]),
                "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
            }
        raw_edges = []
        for row in pred_graph.edge_attrs().iter_rows(named=True):
            edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
            raw_edges.append({
                "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                "edge_prob": None if edge_prob is None else float(edge_prob),
            })
        VAL_RAW_GRAPHS[stem] = (raw_nodes_by_id, raw_edges)
    print(f"VALIDATOR: cached {len(VAL_RAW_GRAPHS)} raw prediction graphs + GT")


def score_validator_config(config: dict, label: str, verbose: bool = False) -> tuple[dict, list]:
    """Run the full post-process on every cached validator graph with the
    given overrides and score it with the official metric formula."""
    saved = pp_apply(config)
    _real_test_dir = TEST_DIR
    globals()["TEST_DIR"] = TRAIN_DIR
    rows = []
    t0 = time.time()
    try:
        for stem in val_stems:
            raw_nodes_by_id, raw_edges = VAL_RAW_GRAPHS[stem]
            nodes_copy = _copy.deepcopy(raw_nodes_by_id)
            edges_copy = _copy.deepcopy(raw_edges)
            processed_nodes, processed_edges, _stage_stats = filter_output_graph(
                nodes_copy, edges_copy, dataset=stem,
                deepcenter_bundle=globals().get("DEEPCENTER_VETO_DETECTOR"),
            )
            gt_nodes_plain, gt_edges_plain, t_true = VAL_GT[stem]
            pred_nodes_plain = nodes_by_id_to_plain(processed_nodes)
            pred_edges_plain = [(int(e["source_id"]), int(e["target_id"])) for e in processed_edges]
            row = score_sample(pred_nodes_plain, pred_edges_plain, gt_nodes_plain, gt_edges_plain, t_true)
            row["stem"] = stem
            row["config"] = label
            row["safe_divisions_added"] = _stage_stats.get("safe_divisions_added", 0)
            rows.append(row)
    finally:
        globals()["TEST_DIR"] = _real_test_dir
        pp_restore(saved)
    summary = aggregate_official(rows)
    summary["n_samples"] = len(rows)
    summary["config"] = label
    summary["seconds"] = time.time() - t0
    if verbose:
        for row in rows:
            print(f"  {row['stem']:<28} edge_jaccard={row['edge_jaccard']:.4f} "
                  f"adj={row['adjusted_edge_jaccard']:.4f} T_pred={row['t_pred']} T_true={row['t_true']} "
                  f"div(tp/fp/fn)=({row['div_tp']}/{row['div_fp']}/{row['div_fn']}) "
                  f"safe_div_added={row['safe_divisions_added']}")
    print(f"[{label}] n={summary['n_samples']} adjusted_edge_jaccard={summary['adjusted_edge_jaccard']:.4f} "
          f"division_jaccard={summary['division_jaccard']:.4f} (tp/fp/fn={summary['div_tp']}/{summary['div_fp']}/{summary['div_fn']}) "
          f"PROXY_SCORE={summary['proxy_score']:.4f}  [{summary['seconds']/60:.1f} min]")
    return summary, rows


validator_sample_rows: list[dict[str, object]] = []
validator_summary_rows: list[dict[str, object]] = []
PP_RESULTS: dict[str, dict] = {}

if VALIDATOR_ENABLE and val_stems:
    print()
    print("=" * 78)
    print("LOCAL VALIDATOR -- base configuration (public 0.939), official metric formula")
    print("(https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md)")
    print("=" * 78)
    base_summary, base_rows = score_validator_config({}, "base", verbose=True)
    PP_RESULTS["base"] = base_summary
    validator_sample_rows.extend(base_rows)
    validator_summary_rows.append(base_summary)
    with VALIDATOR_STATS_PATH.open("w", newline="") as f:
        fieldnames = sorted({k for row in base_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in base_rows:
            writer.writerow(row)
    print(f"Per-sample validator rows written to {VALIDATOR_STATS_PATH}")
else:
    print("VALIDATOR: disabled or no held-out samples available -- skipping scoring.")




PP_CANDIDATES: dict[str, dict] = {
    "gap45": {"GAP_CLOSE_UM": 4.5},
    "tight55": {"MOTION_RELINK_TIGHT_UM": 5.5},
    "relaxed9": {"MOTION_RELINK_RELAXED_UM": 9.0},
    "bonus125": {"MOTION_RELINK_LEARNED_BONUS": 1.25},
    "gap2step40": {"GAP2_MAX_STEP_UM": 4.0},
    "reuse28": {"GAP_CLOSE_REUSE_UM": 2.8},
    "dcgap035": {"DEEPCENTER_GAP_THRESHOLD": 0.35},
}
PP_SELECT_MARGIN = float(os.environ.get("BIOHUB_PPSWEEP_SELECT_MARGIN", "0.002"))
PP_MAX_ADJ_LOSS = float(os.environ.get("BIOHUB_PPSWEEP_MAX_ADJ_LOSS", "0.0005"))
PP_SWEEP_RESULTS_PATH = WORKING_DIR / "ppsweep_results.csv"
PP_SELECTED_PATH = WORKING_DIR / "ppsweep_selected.json"

selected_label = "base"
selected_config: dict = {}

if VALIDATOR_ENABLE and val_stems and "base" in PP_RESULTS:
    base_summary = PP_RESULTS["base"]
    print("=" * 78)
    print(f"POST-PROCESS SWEEP -- {len(PP_CANDIDATES)} candidates x {len(val_stems)} held-out videos")
    print("=" * 78)
    for label, config in PP_CANDIDATES.items():
        summary, rows = score_validator_config(config, label)
        PP_RESULTS[label] = summary
        validator_sample_rows.extend(rows)

    
    
    positive = [
        label for label, summary in PP_RESULTS.items()
        if label != "base"
        and summary["proxy_score"] >= base_summary["proxy_score"] + 0.0005
        and summary["adjusted_edge_jaccard"] >= base_summary["adjusted_edge_jaccard"] - PP_MAX_ADJ_LOSS
    ]
    positive.sort(key=lambda l: PP_RESULTS[l]["proxy_score"], reverse=True)
    combo_config: dict = {}
    for label in positive:
        for key, value in PP_CANDIDATES[label].items():
            combo_config.setdefault(key, value)
    if len(positive) >= 2:
        combo_label = "combo(" + "+".join(positive) + ")"
        summary, rows = score_validator_config(combo_config, combo_label)
        PP_RESULTS[combo_label] = summary
        PP_CANDIDATES[combo_label] = combo_config
        validator_sample_rows.extend(rows)

    print()
    print("=" * 78)
    print("SWEEP TABLE (sorted by PROXY_SCORE)")
    print("=" * 78)
    ranked = sorted(PP_RESULTS.items(), key=lambda kv: kv[1]["proxy_score"], reverse=True)
    for label, summary in ranked:
        delta = summary["proxy_score"] - base_summary["proxy_score"]
        print(f"  {label:<40} proxy={summary['proxy_score']:.4f} ({delta:+.4f})  "
              f"adj={summary['adjusted_edge_jaccard']:.4f}  divJ={summary['division_jaccard']:.4f} "
              f"(tp/fp/fn={summary['div_tp']}/{summary['div_fp']}/{summary['div_fn']})")
    pd.DataFrame([
        {"config": label, **{k: v for k, v in summary.items() if k != "config"},
         "overrides": json.dumps(PP_CANDIDATES.get(label, {}), sort_keys=True)}
        for label, summary in ranked
    ]).to_csv(PP_SWEEP_RESULTS_PATH, index=False)

    best_label, best_summary = ranked[0]
    if (
        best_label != "base"
        and best_summary["proxy_score"] >= base_summary["proxy_score"] + PP_SELECT_MARGIN
        and best_summary["adjusted_edge_jaccard"] >= base_summary["adjusted_edge_jaccard"] - PP_MAX_ADJ_LOSS
    ):
        selected_label = best_label
        selected_config = dict(PP_CANDIDATES[best_label])
    print()
    print(f"SELECTED: {selected_label}  overrides={selected_config}  "
          f"(margin rule: >= +{PP_SELECT_MARGIN} proxy and adj loss <= {PP_MAX_ADJ_LOSS})")
    print(f"  base     proxy={base_summary['proxy_score']:.4f} adj={base_summary['adjusted_edge_jaccard']:.4f} divJ={base_summary['division_jaccard']:.4f}")
    sel = PP_RESULTS[selected_label]
    print(f"  selected proxy={sel['proxy_score']:.4f} adj={sel['adjusted_edge_jaccard']:.4f} divJ={sel['division_jaccard']:.4f}")

    with VALIDATOR_STATS_PATH.open("w", newline="") as f:
        fieldnames = sorted({k for row in validator_sample_rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in validator_sample_rows:
            writer.writerow(row)
else:
    print("SWEEP: validator unavailable -- keeping the base configuration.")

PP_SELECTED_PATH.write_text(json.dumps({
    "selected": selected_label,
    "overrides": selected_config,
    "base_proxy": PP_RESULTS.get("base", {}).get("proxy_score"),
    "selected_proxy": PP_RESULTS.get(selected_label, {}).get("proxy_score"),
    "held_out_stems": list(val_stems),
}, indent=2, sort_keys=True) + "\n")

if selected_config:
    print()
    print(f"Re-writing submission.csv with the selected post-process configuration: {selected_label}")
    _saved = pp_apply(selected_config)
    try:
        write_test_submission(selected_label)
    finally:
        pp_restore(_saved)
    for key, value in selected_config.items():
        os.environ["BIOHUB_" + key] = str(value)
else:
    print("Keeping the base submission.csv (public 0.939 configuration).")


_final = pd.read_csv(SUBMISSION_PATH)
assert _final.columns.tolist() == CSV_COLUMNS, _final.columns.tolist()
assert _final["id"].tolist() == list(range(len(_final))), "row ids not contiguous"
_expected_sets = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))
assert sorted(_final["dataset"].astype(str).unique()) == _expected_sets, "dataset mismatch"
for _ds, _grp in _final.groupby("dataset"):
    _n = _grp[_grp.row_type.eq("node")]
    _e = _grp[_grp.row_type.eq("edge")]
    _t = dict(zip(_n.node_id.astype(int), _n.t.astype(int)))
    assert all(_t[int(s)] + 1 == _t[int(d)] for s, d in zip(_e.source_id, _e.target_id)), f"{_ds}: bad edge time"
    assert _e.target_id.value_counts().max() <= 1, f"{_ds}: multi-parent"
    assert _e.source_id.value_counts().max() <= 2, f"{_ds}: out-degree > 2"
    print(f"  {_ds}: nodes={len(_n)} edges={len(_e)} divisions={(_e.source_id.value_counts() == 2).sum()}")
print(f"Final submission.csv rows={len(_final)}  config={selected_label}")

# ==== HOCT consensus veto: finalize cell (inserted after the post-process sweep) =======
# If the sweep re-wrote submission.csv, the veto was applied inside that write (the arming cell
# re-bound write_test_submission) and nothing is re-written here. If the sweep kept the base file,
# it is re-written once with the veto. Either way exactly one HOCT_VETO_SUMMARY line is printed:
# videos vetoed / skipped for the wall-clock budget / failed, elapsed seconds, deadline used.
# With BIOHUB_HOCT_VETO=0 (or without the arming cell) this cell does nothing.
if globals().get("_HV_MODE", 0) in (1, 2):
    _hv_finalize()











print("=" * 78)
print("PIPELINE MANIFEST -- resolved state, not just config")
print("=" * 78)

_secondary_weights_env = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "")
_secondary_ready = bool(_secondary_weights_env and Path(_secondary_weights_env).exists())
print(f"Dual-seed ensemble:      requested=True  weights_found={_secondary_ready}"
      f"{'  <-- FALLING BACK TO SINGLE-SEED, check BIOHUB_SECONDARY_WEIGHTS' if not _secondary_ready else ''}")

_bidir_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))
print(f"Bidirectional fusion:    weight={_bidir_weight}  "
      f"active={_bidir_weight > 0.0}  mode={os.environ.get('BIOHUB_BIDIRECTIONAL_FUSION_MODE', '(unset)')}")

_dc_requested = os.environ.get("BIOHUB_USE_DEEPCENTER_VETO", "0") != "0"
_dc_bundle = globals().get("DEEPCENTER_VETO_DETECTOR")
_dc_loaded = "DEEPCENTER_VETO_DETECTOR" in globals() and _dc_bundle is not None
_dc_path = _dc_bundle.get("path") if _dc_loaded else None
print(f"DeepCenter veto:         requested={_dc_requested}  loaded={_dc_loaded}"
      f"{'  <-- REQUESTED BUT NOT LOADED, gap/division vetoes are no-ops' if _dc_requested and not _dc_loaded else ''}")
if _dc_loaded:
    print(f"  - checkpoint file:     {_dc_path}")
    print(f"  - expected epoch:      {os.environ.get('BIOHUB_DEEPCENTER_EXPECTED_EPOCH')}")
print(f"  - gap veto:            {os.environ.get('BIOHUB_DEEPCENTER_GAP_VETO', '0') != '0'}")
print(f"  - safe-div veto:       {os.environ.get('BIOHUB_DEEPCENTER_SAFE_DIV_VETO', '0') != '0'}")

print(f"Safe-div thresholds:     parent<={os.environ.get('BIOHUB_SAFE_DIV_MAX_UM')}um  "
      f"sister<={os.environ.get('BIOHUB_SAFE_DIV_SISTER_MAX_UM')}um  "
      f"global_cap={os.environ.get('BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP')}  "
      f"frame_cap={os.environ.get('BIOHUB_SAFE_DIV_FRAME_FRAC_CAP')}")

print(f"Validator:               enabled={VALIDATOR_ENABLE}  "
      f"held_out_samples={len(val_stems)}  match_radius={VALIDATOR_MATCH_RADIUS_UM}um")

print("=" * 78)

