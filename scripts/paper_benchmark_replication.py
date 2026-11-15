"""
ISRO Research Paper Benchmark Replication Suite
===============================================
Replicates the empirical findings and Table 3 benchmark metrics from:
"Comparative Analysis of Image Registration Algorithms for Chandrayaan-2 Lunar Datasets" (2025)

Sensors Evaluated:
- OHRC ↔ NAC (High-Resolution Optical)
- IIRS ↔ WAC (Hyperspectral-to-Panchromatic)
- DFSAR ↔ SELENE (Synthetic Aperture Radar-to-Optical)
Regions: Equatorial & Polar
Algorithms: SIFT, ASIFT, AKAZE, RIFT2, SuperGlue
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("clarity.benchmark.paper_replication")

# ─────────────────────────────────────────────────────────────────────────────
# Ground-Truth Paper Table 3 Reference Data
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TABLE_3_RECORDS = [
    # OHRC-NAC Equatorial
    {"sensor_pair": "OHRC-NAC", "region": "Equatorial", "algorithm": "SIFT", "rmse_x": 4.3168, "rmse_y": 3.5132, "total_rmse": 5.5658, "time_s": 0.81, "matches": 184, "status": "Success"},
    {"sensor_pair": "OHRC-NAC", "region": "Equatorial", "algorithm": "ASIFT", "rmse_x": 1.9961, "rmse_y": 1.6384, "total_rmse": 2.5826, "time_s": 809.82, "matches": 412, "status": "Success"},
    {"sensor_pair": "OHRC-NAC", "region": "Equatorial", "algorithm": "AKAZE", "rmse_x": 3.2505, "rmse_y": 2.2743, "total_rmse": 3.9671, "time_s": 0.80, "matches": 156, "status": "Success"},
    {"sensor_pair": "OHRC-NAC", "region": "Equatorial", "algorithm": "RIFT2", "rmse_x": 1.5034, "rmse_y": 1.1963, "total_rmse": 1.9213, "time_s": 36.88, "matches": 340, "status": "Success"},
    {"sensor_pair": "OHRC-NAC", "region": "Equatorial", "algorithm": "SuperGlue", "rmse_x": 0.6249, "rmse_y": 0.5718, "total_rmse": 0.8471, "time_s": 3.81, "matches": 582, "status": "Superior"},

    # OHRC-NAC Polar
    {"sensor_pair": "OHRC-NAC", "region": "Polar", "algorithm": "SIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "OHRC-NAC", "region": "Polar", "algorithm": "ASIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "OHRC-NAC", "region": "Polar", "algorithm": "AKAZE", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "OHRC-NAC", "region": "Polar", "algorithm": "RIFT2", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "OHRC-NAC", "region": "Polar", "algorithm": "SuperGlue", "rmse_x": 0.7812, "rmse_y": 0.6945, "total_rmse": 1.0452, "time_s": 4.12, "matches": 428, "status": "Superior"},

    # IIRS-WAC Equatorial
    {"sensor_pair": "IIRS-WAC", "region": "Equatorial", "algorithm": "SIFT", "rmse_x": 5.1204, "rmse_y": 4.2811, "total_rmse": 6.6738, "time_s": 0.95, "matches": 92, "status": "Success"},
    {"sensor_pair": "IIRS-WAC", "region": "Equatorial", "algorithm": "ASIFT", "rmse_x": 2.4510, "rmse_y": 2.1105, "total_rmse": 3.2343, "time_s": 740.15, "matches": 280, "status": "Success"},
    {"sensor_pair": "IIRS-WAC", "region": "Equatorial", "algorithm": "AKAZE", "rmse_x": 4.1022, "rmse_y": 3.6540, "total_rmse": 5.4939, "time_s": 0.89, "matches": 114, "status": "Success"},
    {"sensor_pair": "IIRS-WAC", "region": "Equatorial", "algorithm": "RIFT2", "rmse_x": 1.8210, "rmse_y": 1.4520, "total_rmse": 2.3292, "time_s": 42.10, "matches": 295, "status": "Success"},
    {"sensor_pair": "IIRS-WAC", "region": "Equatorial", "algorithm": "SuperGlue", "rmse_x": 0.7105, "rmse_y": 0.6432, "total_rmse": 0.9584, "time_s": 4.05, "matches": 490, "status": "Superior"},

    # IIRS-WAC Polar
    {"sensor_pair": "IIRS-WAC", "region": "Polar", "algorithm": "SIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "IIRS-WAC", "region": "Polar", "algorithm": "ASIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "IIRS-WAC", "region": "Polar", "algorithm": "AKAZE", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "IIRS-WAC", "region": "Polar", "algorithm": "RIFT2", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "IIRS-WAC", "region": "Polar", "algorithm": "SuperGlue", "rmse_x": 0.8421, "rmse_y": 0.7930, "total_rmse": 1.1568, "time_s": 4.30, "matches": 380, "status": "Superior"},

    # DFSAR-SELENE Equatorial
    {"sensor_pair": "DFSAR-SELENE", "region": "Equatorial", "algorithm": "SIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Equatorial", "algorithm": "ASIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Equatorial", "algorithm": "AKAZE", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Equatorial", "algorithm": "RIFT2", "rmse_x": 2.1045, "rmse_y": 1.8920, "total_rmse": 2.8301, "time_s": 55.40, "matches": 185, "status": "Success"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Equatorial", "algorithm": "SuperGlue", "rmse_x": 0.9120, "rmse_y": 0.8250, "total_rmse": 1.2298, "time_s": 4.60, "matches": 320, "status": "Superior"},

    # DFSAR-SELENE Polar
    {"sensor_pair": "DFSAR-SELENE", "region": "Polar", "algorithm": "SIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Polar", "algorithm": "ASIFT", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Polar", "algorithm": "AKAZE", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Polar", "algorithm": "RIFT2", "rmse_x": None, "rmse_y": None, "total_rmse": None, "time_s": None, "matches": 0, "status": "Failed"},
    {"sensor_pair": "DFSAR-SELENE", "region": "Polar", "algorithm": "SuperGlue", "rmse_x": 0.9850, "rmse_y": 0.9100, "total_rmse": 1.3410, "time_s": 4.85, "matches": 275, "status": "Superior"},
]


def get_paper_reference_dataframe() -> pd.DataFrame:
    """Returns the ground truth Table 3 benchmarks as a pandas DataFrame."""
    return pd.DataFrame(PAPER_TABLE_3_RECORDS)


def evaluate_registration_accuracy(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    gt_homography: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes exact RMSE_X, RMSE_Y, and Total RMSE against ground-truth transformation or RANSAC model.
    Enforces AGENTS.md Gate 1 (4-point trap) & Gate 2 (MVR >= 15 inliers).
    """
    import cv2

    n_pts = len(pts_src)
    if n_pts < 15:
        # GATE 2: MINIMUM VIABLE REGISTRATION
        return {
            "status": "FAILED QUALITY GATE",
            "reason": f"Gate 2 Breach: Inlier count {n_pts} < 15 minimum viable threshold.",
            "inliers": n_pts,
            "rmse_x": None,
            "rmse_y": None,
            "total_rmse": None,
        }

    if gt_homography is not None:
        H = gt_homography
        inliers_mask = np.ones(n_pts, dtype=bool)
    else:
        # Robust estimation using MAGSAC++ / RANSAC
        H, mask = cv2.findHomography(pts_src, pts_ref, cv2.USAC_MAGSAC, 3.0)
        if H is None or mask is None:
            return {
                "status": "FAILED QUALITY GATE",
                "reason": "Homography estimation failed to converge.",
                "inliers": 0,
                "rmse_x": None,
                "rmse_y": None,
                "total_rmse": None,
            }
        inliers_mask = mask.ravel().astype(bool)

    inlier_count = int(inliers_mask.sum())
    if inlier_count < 15:
        return {
            "status": "FAILED QUALITY GATE",
            "reason": f"Gate 2 Breach: Surviving inliers {inlier_count} < 15.",
            "inliers": inlier_count,
            "rmse_x": None,
            "rmse_y": None,
            "total_rmse": None,
        }

    # Transform source points with H
    pts_src_in = pts_src[inliers_mask]
    pts_ref_in = pts_ref[inliers_mask]

    pts_h = np.hstack([pts_src_in, np.ones((len(pts_src_in), 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / (proj[:, 2:] + 1e-12)

    residuals = proj - pts_ref_in
    res_x = residuals[:, 0]
    res_y = residuals[:, 1]

    rmse_x = float(np.sqrt(np.mean(res_x ** 2)))
    rmse_y = float(np.sqrt(np.mean(res_y ** 2)))
    total_rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

    # GATE 1: OVERFIT PREVENTION & THE 4-POINT TRAP
    if (inlier_count == 4 and total_rmse < 1e-4) or (inlier_count < 15 and total_rmse < 1e-4):
        return {
            "status": "FAILED QUALITY GATE",
            "reason": "Gate 1 Breach: Mathematical Illusion (Overfit with zero residual).",
            "inliers": inlier_count,
            "rmse_x": None,
            "rmse_y": None,
            "total_rmse": None,
        }

    return {
        "status": "Superior" if total_rmse < 1.0 else "Success",
        "inliers": inlier_count,
        "rmse_x": round(rmse_x, 4),
        "rmse_y": round(rmse_y, 4),
        "total_rmse": round(total_rmse, 4),
    }


def run_benchmark_on_images(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    algorithms: Optional[List[str]] = None,
    gt_homography: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Executes selected algorithms against the given image pair, measures execution time,
    RMSE, inliers, and formats results alongside Table 3 paper standards.
    """
    from src.features.deep_matcher import LunarMatcher

    algos = algorithms or ["superglue", "rift2", "akaze", "sift"]
    records = []

    for algo in algos:
        t0 = time.time()
        try:
            matcher = LunarMatcher(matcher_type=algo)
            pts_s, pts_r, conf = matcher.match(img_src, img_ref)
            elapsed = time.time() - t0
            n_matches = len(pts_s)

            if n_matches < 4:
                records.append({
                    "algorithm": algo.upper(),
                    "matches": n_matches,
                    "rmse_x": None,
                    "rmse_y": None,
                    "total_rmse": None,
                    "time_s": round(elapsed, 2),
                    "status": "Failed (0 inliers)",
                })
            else:
                eval_res = evaluate_registration_accuracy(pts_s, pts_r, gt_homography)
                records.append({
                    "algorithm": algo.upper(),
                    "matches": eval_res["inliers"],
                    "rmse_x": eval_res["rmse_x"],
                    "rmse_y": eval_res["rmse_y"],
                    "total_rmse": eval_res["total_rmse"],
                    "time_s": round(elapsed, 2),
                    "status": eval_res["status"],
                })
        except Exception as e:
            elapsed = time.time() - t0
            records.append({
                "algorithm": algo.upper(),
                "matches": 0,
                "rmse_x": None,
                "rmse_y": None,
                "total_rmse": None,
                "time_s": round(elapsed, 2),
                "status": f"Error ({str(e)[:30]})",
            })

    return pd.DataFrame(records)
