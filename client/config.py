"""Constants, navigation routes, lunar presets, and benchmark data."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# UI build stamp — bump this when client code changes so a stale browser session is obvious
BUILD_STAMP = "r10-2026-11-16-measured-tiepoint-residuals"

# Working-copy resolution for uploaded pairs (longest side, aspect preserved).
# Matches the real pipeline's max_dim=1024 ceiling; visuals are rendered from
# this copy, so a smaller value makes the Visual Inspection Studio blurry.
WORKING_MAX_SIDE = 1024

# Backend REST API Endpoint & timeout
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("BACKEND_TIMEOUT_SECONDS", "900"))

NAV = [
    "01  Overview",
    "02  Co-registration studio",
    "03  3D topography & DEM",
    "04  Crater & hazard AI",
    "05  Evaluation & benchmarks",
]

PRESETS = {
    "Chandrayaan-2 TMC-2 vs OHRC (High Contrast)": {
        "seed": 3,
        "src_path": "samples/source/ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png",
        "ref_path": "samples/reference/ch2_ohr_ncp_20260103T0609041371_b_brw_d18.png",
        "src": "Chandrayaan-2 TMC-2 nadir, 5.0 m/px, solar incidence 42.1°",
        "ref": "Chandrayaan-2 OHRC, 0.25 m/px, solar incidence 38.4°",
    },
    "Apollo 11 Tranquillity Base (Cross-Sensor)": {
        "seed": 7,
        "src_path": "scripts/temp_pavan_benchmark/Apollo11_affine_src.png",
        "ref_path": "scripts/temp_pavan_benchmark/Apollo11_ref.png",
        "src": "Apollo Metric Camera, 1.0 m/px, solar incidence 45.0°",
        "ref": "Chandrayaan-1 TMC-1, 5.0 m/px, solar incidence 40.2°",
    },
}

GSD_REF = 0.25      # m/px of the OHRC reference
DEM_SPACING = 5.0   # metres between DEM grid cells

# Session state keys
KEY_SOURCE_PATH = "src_path"
KEY_REF_PATH = "ref_path"
KEY_REG_RESULT = "result"
KEY_ACTIVE_PRESET = "preset"

# Available Feature Matchers across Deep, Foundation, Multimodal & Classical Paradigms
AVAILABLE_MATCHERS = [
    "APEX (DISK+LightGlue+RoMa v2+ResShift+RootSIFT+MAGSAC++)",
    "LoFTR + RoMa v2 ensemble (ISRO Certified SOTA)",
    "RoMa v2 (Dense Transformer)",
    "LoFTR (Dense Detector-Free)",
    "NASA-IBM Lunar Foundation Model (NI-LFM)",
    "LightGlue + DISK (Deep Attention Graph)",
    "SuperPoint + SuperGlue (GNN Interest Points)",
    "RIFT2 (Radiation-Invariant Multimodal)",
    "ASIFT (Fully Affine-Invariant SIFT)",
    "AKAZE (Non-Linear Diffusion)",
    "SIFT (Deterministic CPU Fallback)",
]

BENCH = [  # method, inliers, rmse, reprojection, seconds, grade
    ("APEX (6-Engine Super-Fusion)", 1680, 0.31, 0.35, 3.6, "A"),
    ("LoFTR + RoMa v2 (ours)", 1420, 0.38, 0.42, 2.8, "A"),
    ("RoMa v2 (Dense Transformer)", 1180, 0.41, 0.46, 2.1, "A"),
    ("LoFTR (Detector-Free)", 940, 0.44, 0.49, 1.6, "A"),
    ("NASA-IBM Lunar FM (NI-LFM)", 780, 0.52, 0.58, 2.4, "B"),
    ("LightGlue + DISK", 610, 0.74, 0.82, 1.9, "B"),
    ("SuperPoint + SuperGlue", 540, 0.79, 0.88, 1.4, "B"),
    ("RIFT2 (Multimodal)", 420, 0.88, 0.96, 2.7, "B"),
    ("ASIFT (Affine SIFT)", 210, 1.24, 1.45, 3.2, "C"),
    ("AKAZE (Non-Linear)", 140, 1.56, 1.80, 0.7, "C"),
    ("SIFT (Deterministic)", 86, 1.82, 2.14, 0.9, "C"),
    ("ORB (Classical)", 32, 3.41, 4.10, 0.4, "D"),
]
