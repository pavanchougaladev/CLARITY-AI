#!/usr/bin/env python3
"""
Real-Image Benchmark Runner — Chandrayaan-2 Lunar Image Registration
=====================================================================
Replicates and evaluates the exact benchmark protocol specified in pavan.pdf:
  - 7 unique real lunar image strips (Apollo11, Chandrayaan-1/2 TMC, Tycho crater, OHRC)
  - 3 ground-truth transform perturbations per strip:
      1. Affine (rotation +/-3 deg, scale +/-1.5%, translation +/-12 px)
      2. Perspective (affine + projective tilt)
      3. Photometric (affine + photometric gamma shift simulating sun angle)
  - 4 core matching algorithms (RoMa v2 APEX, DISK, SIFT STANDARD, ORB) + Extended Software Engines (LunarFusion, RoMa-Omega, NI-LFM, LoFTR)
  - Evaluates GT-RMSE on 32x32 interior grid, Inlier RMSE, Inliers count, Spatial Occupancy (8x8 grid), Uniformity (chi-squared score), ECC score, and Execution Time.
"""

import os
import sys
import json
import time
import logging
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_registration_pipeline, robust_read
from src.evaluate.metrics import evaluate_transform_gt_rmse, synthetic_grid_gt_rmse, match_density_grid, reprojection_error_per_point

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# Output paths
RESULTS_JSON = os.path.join(PROJECT_ROOT, "scripts", "pavan_benchmark_results.json")


def compute_ecc_score(img_src: np.ndarray, img_ref: np.ndarray, H_est: np.ndarray) -> float:
    """Compute Enhanced Correlation Coefficient (ECC) between warped source and reference."""
    try:
        h, w = img_ref.shape[:2]
        warped = cv2.warpPerspective(img_src, H_est, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Mask out borders
        mask_src = np.ones_like(img_src, dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask_src, H_est, (w, h), borderMode=cv2.BORDER_CONSTANT)
        valid = (warped_mask > 200)
        
        if np.sum(valid) < 100:
            return 0.0
            
        src_vals = warped[valid].astype(np.float32)
        ref_vals = img_ref[valid].astype(np.float32)
        
        # Zero-mean normalized cross-correlation
        src_mean = src_vals - np.mean(src_vals)
        ref_mean = ref_vals - np.mean(ref_vals)
        
        denom = np.sqrt(np.sum(src_mean ** 2) * np.sum(ref_mean ** 2))
        if denom < 1e-6:
            return 0.0
            
        ecc = np.sum(src_mean * ref_mean) / denom
        return float(np.clip(ecc, -1.0, 1.0))
    except Exception as e:
        logger.debug(f"ECC computation error: {e}")
        return 0.0


def compute_uniformity(points: np.ndarray, img_shape: tuple, grid: tuple = (8, 8)) -> float:
    """Compute Chi-squared based spatial uniformity score (1 = perfectly uniform, 0 = clustered)."""
    if len(points) == 0:
        return 0.0
    density, coverage = match_density_grid(img_shape, points, grid)
    total = np.sum(density)
    if total == 0:
        return 0.0
    expected = total / density.size
    chi2 = np.sum((density - expected) ** 2 / expected)
    # Convert chi2 to a normalized 0..1 uniformity index
    max_chi2 = total * (density.size - 1)
    if max_chi2 < 1e-6:
        return 1.0
    score = 1.0 - (chi2 / max_chi2)
    return float(np.clip(score, 0.0, 1.0))


def build_transform(transform_type: str, angle_deg: float = 2.5, scale: float = 1.012, tx: float = 8.5, ty: float = -6.2) -> np.ndarray:
    """Construct ground-truth 3x3 Homography matrix based on transform mode."""
    rad = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(rad), np.sin(rad)
    
    if transform_type == "affine":
        H = np.array([
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t,  scale * cos_t, ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    elif transform_type == "perspective":
        H = np.array([
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t,  scale * cos_t, ty],
            [0.00004, -0.00003, 1.0]
        ], dtype=np.float64)
    elif transform_type == "photometric":
        H = np.array([
            [scale * cos_t, -scale * sin_t, tx],
            [scale * sin_t,  scale * cos_t, ty],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    else:
        H = np.eye(3, dtype=np.float64)
    return H


def apply_photometric_distortion(img: np.ndarray, gamma: float = 1.35) -> np.ndarray:
    """Apply non-linear gamma transform to simulate solar illumination angle differences."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def load_lunar_strips() -> list:
    """Locate 7 unique real Chandrayaan lunar image strips across workspace data directories."""
    raw_files = [
        # Apollo 11 region
        ("Apollo11", "data/raw/ch1_tmc_nca_20090529T1724252721_d_img_apl/browse/calibrated/20090529/ch1_tmc_nca_20090529T1724252721_b_brw_apl.png"),
        # Chandrayaan-1 / 2 TMC Strips
        ("chandrayaana1.1", "data/raw/ch1_tmc_ncf_20090529T0853239926_d_img_d18/browse/calibrated/20090529/ch1_tmc_ncf_20090529T0853239926_b_brw_d18.png"),
        ("chandrayaana1.2", "data/raw/ch2_tmc_nrn_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nrn_20200108T1153531115_b_brw_d18.png"),
        ("chandrayaana1.3", "data/raw/ch2_tmc_nra_20200108T1153531115_d_img_d18/browse/raw/20200108/ch2_tmc_nra_20200108T1153531115_b_brw_d18.png"),
        ("moon1", "data/raw/ch2_tmc_nrn_20191208T0221109733_d_img_g26/browse/raw/20191208/ch2_tmc_nrn_20191208T0221109733_b_brw_g26.png"),
        ("moon2", "data/raw/ch2_ohr_nrp_20230820T0559124374_d_img_n18/browse/raw/20230820/ch2_ohr_nrp_20230820T0559124374_b_brw_n18.png"),
        ("tycho crater2", "data/raw/ch2_tmc_nrn_20191210T1903498259_d_img_d18/browse/raw/20191210/ch2_tmc_nrn_20191210T1903498259_b_brw_d18.png"),
    ]
    
    validated_strips = []
    for name, rel_path in raw_files:
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if os.path.exists(full_path):
            validated_strips.append((name, full_path))
        else:
            logger.warning(f"Strip file not found: {full_path}")
            
    return validated_strips


def main():
    logger.info("Starting Pavan PDF Real-Image Benchmark Protocol...")
    
    strips = load_lunar_strips()
    if not strips:
        logger.error("No lunar strip images found!")
        return

    logger.info(f"Loaded {len(strips)} unique real lunar strip datasets.")
    
    transforms = ["affine", "perspective", "photometric"]
    matchers = ["roma", "disk", "sift", "orb"]
    extended_matchers = ["fusion", "loftr", "ni_lfm"]
    
    # Store benchmark detailed outputs
    per_test_detail = []
    algorithm_summary = {}
    
    temp_dir = os.path.join(PROJECT_ROOT, "scripts", "temp_pavan_benchmark")
    os.makedirs(temp_dir, exist_ok=True)
    
    for matcher in matchers + extended_matchers:
        algorithm_summary[matcher] = {
            "gt_rmse_list": [],
            "inlier_rmse_list": [],
            "inliers_list": [],
            "coverage_list": [],
            "uniformity_list": [],
            "ecc_list": [],
            "time_list": [],
            "pairs_below_0_1px": 0,
            "total_runs": 0
        }

    total_pair_tests = len(strips) * len(transforms)
    logger.info(f"Running {total_pair_tests} test pairs across {len(matchers)} core algorithms ({total_pair_tests * len(matchers)} total runs)...")
    
    test_counter = 0
    for strip_name, strip_path in strips:
        # Load and crop a square 1024x1024 region with sufficient texture
        img_raw = cv2.imread(strip_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            logger.error(f"Failed to load strip {strip_name} at {strip_path}")
            continue
            
        h_raw, w_raw = img_raw.shape[:2]
        crop_sz = min(1024, h_raw, w_raw)
        cy, cx = h_raw // 2, w_raw // 2
        img_ref = img_raw[cy - crop_sz // 2 : cy + crop_sz // 2, cx - crop_sz // 2 : cx + crop_sz // 2]
        h_crop, w_crop = img_ref.shape[:2]
        
        ref_file = os.path.join(temp_dir, f"{strip_name}_ref.png")
        cv2.imwrite(ref_file, img_ref)
        
        for t_type in transforms:
            test_counter += 1
            logger.info(f"\n--- [{test_counter}/{total_pair_tests}] Strip: {strip_name} | Transform: {t_type} ---")
            
            # Create synthetic warp ground-truth matrix H_gt
            if t_type == "affine":
                H_gt = build_transform("affine", angle_deg=2.5, scale=1.012, tx=8.5, ty=-6.2)
                img_src = cv2.warpPerspective(img_ref, np.linalg.inv(H_gt), (w_crop, h_crop), borderMode=cv2.BORDER_REFLECT)
            elif t_type == "perspective":
                H_gt = build_transform("perspective", angle_deg=-2.0, scale=0.988, tx=-10.0, ty=7.4)
                img_src = cv2.warpPerspective(img_ref, np.linalg.inv(H_gt), (w_crop, h_crop), borderMode=cv2.BORDER_REFLECT)
            elif t_type == "photometric":
                H_gt = build_transform("photometric", angle_deg=1.8, scale=1.008, tx=5.5, ty=-4.0)
                img_src_warped = cv2.warpPerspective(img_ref, np.linalg.inv(H_gt), (w_crop, h_crop), borderMode=cv2.BORDER_REFLECT)
                img_src = apply_photometric_distortion(img_src_warped, gamma=1.35)
                
            src_file = os.path.join(temp_dir, f"{strip_name}_{t_type}_src.png")
            cv2.imwrite(src_file, img_src)
            
            # Evaluate all matchers on this pair
            apex_record = None
            
            for matcher in matchers + extended_matchers:
                t0 = time.time()
                try:
                    _, _, res = run_registration_pipeline(
                        src_path=src_file,
                        ref_path=ref_file,
                        matcher_type=matcher,
                        max_dim=1024,
                        use_tps=False
                    )
                    elapsed = time.time() - t0
                    
                    H_est = res.get('H_fallback')
                    inlier_count = int(res.get('inlier_count', 0))
                    inlier_pts_src = res.get('pts_src', np.empty((0, 2)))
                    inlier_pts_ref = res.get('pts_ref', np.empty((0, 2)))
                    
                    if H_est is not None and H_est.shape == (3, 3):
                        # Compute GT RMSE across 32x32 interior grid against ground truth
                        gt_rmse_val = float(synthetic_grid_gt_rmse(H_est, H_gt, img_shape=(h_crop, w_crop), grid_size=32))
                        
                        # Inlier reprojection RMSE (transform source points by H_est)
                        if len(inlier_pts_src) > 0 and len(inlier_pts_ref) > 0:
                            pts_s_reshaped = np.asarray(inlier_pts_src, dtype=np.float32).reshape(-1, 1, 2)
                            pts_s_proj = cv2.perspectiveTransform(pts_s_reshaped, H_est.astype(np.float64)).reshape(-1, 2)
                            errs = np.linalg.norm(pts_s_proj - inlier_pts_ref, axis=1)
                            inlier_rmse_val = float(np.sqrt(np.mean(errs ** 2))) if len(errs) > 0 else 0.5
                        else:
                            inlier_rmse_val = float('inf')
                            
                        # Spatial coverage & uniformity on 8x8 grid
                        _, coverage = match_density_grid((h_crop, w_crop), inlier_pts_ref, (8, 8))
                        uniformity = compute_uniformity(inlier_pts_ref, (h_crop, w_crop), (8, 8))
                        
                        # ECC photometric correlation score
                        ecc_val = float(compute_ecc_score(img_src, img_ref, H_est))
                    else:
                        gt_rmse_val = float('nan')
                        inlier_rmse_val = float('nan')
                        coverage = 0.0
                        uniformity = 0.0
                        ecc_val = 0.0
                    
                    # Accumulate summary stats
                    stats = algorithm_summary[matcher]
                    stats["gt_rmse_list"].append(gt_rmse_val)
                    stats["inlier_rmse_list"].append(inlier_rmse_val)
                    stats["inliers_list"].append(inlier_count)
                    stats["coverage_list"].append(coverage)
                    stats["uniformity_list"].append(uniformity)
                    stats["ecc_list"].append(ecc_val)
                    stats["time_list"].append(elapsed)
                    stats["total_runs"] += 1
                    if gt_rmse_val < 0.1:
                        stats["pairs_below_0_1px"] += 1
                        
                    # Save per-test details for RoMa (APEX)
                    if matcher == "roma":
                        apex_record = {
                            "strip": strip_name,
                            "transform": t_type,
                            "gt_rmse": round(gt_rmse_val, 4),
                            "inlier_rmse": round(inlier_rmse_val, 4),
                            "inliers": inlier_count,
                            "coverage": f"{int(round(coverage * 100))}%",
                            "ecc": round(ecc_val, 4),
                            "time_sec": round(elapsed, 1)
                        }
                        
                    logger.info(f"  > [{matcher.upper()}] GT-RMSE: {gt_rmse_val:.4f} px | Inliers: {inlier_count} | Coverage: {coverage*100:.1f}% | Time: {elapsed:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error executing {matcher} on {strip_name}_{t_type}: {e}")
                    
            if apex_record:
                per_test_detail.append(apex_record)

    # Compile final aggregated table data
    summary_table = []
    for matcher in matchers + extended_matchers:
        s = algorithm_summary[matcher]
        if s["total_runs"] == 0:
            continue
        gt_arr = np.array(s["gt_rmse_list"])
        summary_table.append({
            "matcher": matcher,
            "mean_gt_rmse": round(float(np.mean(gt_arr)), 4),
            "max_gt_rmse": round(float(np.max(gt_arr)), 4),
            "pct_below_0_1px": f"{int(round(s['pairs_below_0_1px'] / s['total_runs'] * 100))}%",
            "mean_inlier_rmse": round(float(np.mean(s["inlier_rmse_list"])), 4),
            "mean_inliers": int(round(np.mean(s["inliers_list"]))),
            "mean_coverage": f"{int(round(np.mean(s['coverage_list']) * 100))}%",
            "mean_uniformity": round(float(np.mean(s["uniformity_list"])), 3),
            "mean_ecc": round(float(np.mean(s["ecc_list"])), 4),
            "mean_time_sec": round(float(np.mean(s["time_list"])), 2)
        })

    benchmark_data = {
        "summary": summary_table,
        "per_test_detail_apex": per_test_detail,
        "metadata": {
            "total_strips": len(strips),
            "transforms_per_strip": len(transforms),
            "total_test_pairs": total_pair_tests,
            "total_runs": test_counter * len(matchers + extended_matchers)
        }
    }
    
    with open(RESULTS_JSON, "w") as f:
        json.dump(benchmark_data, f, indent=2)
        
    logger.info(f"\n🎉 Benchmark complete! Results saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
