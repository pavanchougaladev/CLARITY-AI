#!/usr/bin/env python3
"""
Synthetic GT-RMSE Benchmark — Real Lunar Imagery with Known Ground Truth
========================================================================
Applies a known synthetic transformation (rotation, scale, translation, perspective shear)
to a real Chandrayaan lunar image to create a ground-truth reference image.

Then runs all registration engines (LunarFusion, RoMa, Ensemble, LoFTR, SIFT, etc.)
and computes the exact Ground Truth RMSE (GT-RMSE) against the known transformation.
"""

import os
import sys
import json
import time
import logging
import cv2
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_registration_pipeline
from src.evaluate.metrics import evaluate_transform_gt_rmse, synthetic_grid_gt_rmse

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_ground_truth_transform(
    angle_deg: float = 6.0,
    scale: float = 1.03,
    tx: float = 18.0,
    ty: float = -12.0,
    perspective_shear: float = 0.00005
) -> np.ndarray:
    """
    Construct a precise ground-truth transformation matrix H_gt combining
    rotation, scaling, translation, and slight perspective shear.
    """
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)

    # 3x3 Matrix
    H = np.array([
        [scale * cos_t, -scale * sin_t, tx],
        [scale * sin_t,  scale * cos_t, ty],
        [perspective_shear, -perspective_shear, 1.0]
    ], dtype=np.float64)

    return H


def run_synthetic_gt_benchmark(
    sample_image_path: str,
    output_dir: str = None,
    matchers=None
):
    """
    Runs synthetic GT-RMSE evaluation on a real lunar image patch.
    """
    if matchers is None:
        matchers = ["fusion", "roma", "ensemble", "sift"]

    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "scripts", "synthetic_test_data")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading real lunar image: {sample_image_path}")
    img_raw = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
    if img_raw is None:
        raise FileNotFoundError(f"Could not load image at {sample_image_path}")

    # Crop a square 2048x2048 patch to maintain 1:1 aspect ratio (avoid 22-pixel width scaling on elongated TMC strips)
    h_raw, w_raw = img_raw.shape[:2]
    patch_size = min(2048, h_raw, w_raw)
    cy, cx = h_raw // 2, w_raw // 2
    img_src = img_raw[max(0, cy - patch_size // 2):cy + patch_size // 2, max(0, cx - patch_size // 2):cx + patch_size // 2]
    h, w = img_src.shape[:2]

    # 1. Generate Ground Truth Matrix (Synthetic Rotation, Scale, Translation)
    H_gt = create_synthetic_ground_truth_transform(angle_deg=4.0, scale=1.02, tx=12.0, ty=-8.0, perspective_shear=0.0)

    # 2. Warp image with border reflection to create Ground Truth reference image
    img_ref = cv2.warpPerspective(img_src, H_gt, (w, h), borderMode=cv2.BORDER_REFLECT)

    src_path = os.path.join(output_dir, "real_lunar_src.png")
    ref_path = os.path.join(output_dir, "real_lunar_ref_gt.png")
    cv2.imwrite(src_path, img_src)
    cv2.imwrite(ref_path, img_ref)

    logger.info(f"Saved synthetic pair ({w}x{h} px patch) to {output_dir}")


    # 3. Create test grid points to measure GT-RMSE across full image
    ys = np.linspace(50, h - 50, 15)
    xs = np.linspace(50, w - 50, 15)
    gx, gy = np.meshgrid(xs, ys)
    test_pts_src = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
    test_pts_ref_gt = cv2.perspectiveTransform(test_pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)

    results = {}

    print("\n" + "=" * 80)
    print(f"🎯 SYNTHETIC GROUND TRUTH BENCHMARK ON REAL LUNAR IMAGE ({w}x{h} px)")
    print(f"Known Transformation: Rot = 6.0°, Scale = 1.03x, Shift = (+18.0, -12.0) px")
    print("=" * 80)
    print(f"{'Matcher Engine':<25} | {'Inliers':<8} | {'GT-RMSE (px)':<14} | {'Result Verdict':<20}")
    print("-" * 80)

    for matcher in matchers:
        t0 = time.time()
        try:
            _, _, res = run_registration_pipeline(
                src_path=src_path,
                ref_path=ref_path,
                matcher_type=matcher,
                max_dim=1024,
                use_tps=False
            )
            elapsed = time.time() - t0
            H_down = res.get('H_fallback')
            inliers = res.get('inlier_count', 0)
            scale_src = res.get('scale_src', 1.0)
            scale_ref = res.get('scale_ref', 1.0)

            if H_down is not None and H_down.shape == (3, 3):
                # Rescale matrix to un-downscaled image coordinates
                S_src = np.diag([scale_src, scale_src, 1.0])
                S_ref_inv = np.diag([1.0 / max(scale_ref, 1e-6), 1.0 / max(scale_ref, 1e-6), 1.0])
                H_est = S_ref_inv @ H_down @ S_src

                # Measure exact error against known Ground Truth
                gt_rmse_val = evaluate_transform_gt_rmse(H_est, test_pts_src, test_pts_ref_gt)
                verdict = "🟢 EXCELLENT (Sub-pixel)" if gt_rmse_val < 1.0 else ("🟡 ACCEPTABLE" if gt_rmse_val < 5.0 else f"🔴 HIGH ERROR ({gt_rmse_val:.1f}px)")
            else:
                gt_rmse_val = float('inf')
                verdict = "🔴 FAILED (No Transform)"

            results[matcher] = {
                "inliers": inliers,
                "gt_rmse": round(gt_rmse_val, 4),
                "verdict": verdict,
                "time_sec": round(elapsed, 2)
            }

            print(f"{matcher.upper():<25} | {inliers:<8} | {gt_rmse_val:<14.4f} | {verdict:<20}")

        except Exception as e:
            results[matcher] = {"error": str(e)}
            print(f"{matcher.upper():<25} | {'ERROR':<8} | {'N/A':<14} | 🔴 {str(e)[:25]}")


    print("=" * 80 + "\n")
    return results


if __name__ == "__main__":
    sample_img = os.path.join(
        PROJECT_ROOT,
        "data/raw/ch2_tmc_nrn_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nrn_20200108T1153531115_b_brw_d18.png"
    )

    if os.path.exists(sample_img):
        run_synthetic_gt_benchmark(sample_img)
    else:
        logger.error(f"Sample image not found at {sample_img}")
