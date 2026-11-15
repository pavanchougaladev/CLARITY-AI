"""
Lunar Image Registration Pipeline — End-to-End Orchestrator (Enhanced)
=======================================================================
Handles the full pipeline with accuracy improvements:
  1. Robust image reading (PDS4 .img, .tif, .png, .jpg)
  2. Smart downscaling + GSD normalization for cross-sensor matching
  3. Enhanced illumination-invariant preprocessing (Multi-scale CLAHE + Wallis + Hapke)
  4. Tile-based deep feature matching (LoFTR / DISK+LightGlue) with SIFT fallback
  5. Progressive MAGSAC++ with multi-pass refinement + local consistency filtering
  6. TPS pushbroom jitter correction
  7. Sub-pixel phase-correlation refinement with iterative tightening
  8. Quality-gated metrics computation (RMSE, Weighted RMSE, Coverage, Grade)
  9. Warped product generation (Checkerboard, Blend, Difference maps)
"""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)  # ~1 trillion pixels

import cv2
import numpy as np
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False
import time
import traceback
import logging
import hashlib
import json
import platform
from typing import Optional

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    _src_dir = Path(__file__).resolve().parent
    _root_dir = _src_dir.parent
    if str(_root_dir) not in sys.path:
        sys.path.insert(0, str(_root_dir))
    __package__ = "src"

from .preprocess.shadow_normalization import normalize_illumination
from .preprocess.photometric_correction import apply_hapke_correction, apply_lunar_lambert
from .preprocess.pyramid import build_gaussian_pyramid
from .features.deep_matcher import LunarMatcher
from .features.ensemble_matcher import EnsembleMatcher
from .features.classical_descriptors import phase_congruency_map
from .features.root_sift import root_sift_descriptors
from .features.tile_matcher import tile_match, create_sift_tile_matcher, create_deep_tile_matcher
from .matching.outlier_rejection import estimate_transform
from .registration.subpixel_refine import (
    refine_subpixel,
    refine_subpixel_multipatch,
    refine_keypoints_warp_rectified,
)
from .evaluate.metrics import (
    rmse, weighted_rmse, inlier_ratio, match_density_grid,
    error_statistics, compute_holdout_validation, decompose_error_budget,
    evaluate_transform_gt_rmse,
)
from .evaluate.quality_gate import assess_quality, compute_spatial_entropy
from .spice_utils import get_geometry_for_image
from .config import get_config
from .result_schema import RegistrationTelemetry
from .geospatial_validation import parse_pds4_metadata, validate_geographic_overlap
from .provenance import ProvenanceChain

logger = logging.getLogger(__name__)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_provenance(src_path: str, ref_path: str, config: dict, include_hashes: bool = False) -> dict:
    metadata = {
        "pipeline_version": os.getenv("PIPELINE_VERSION", "1.0.0"),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "platform": platform.platform(),
        "torch_version": getattr(torch, "__version__", None) if HAS_TORCH else None,
        "device": "mps" if HAS_TORCH and torch.backends.mps.is_available() else "cpu",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
        "inputs": {},
    }
    for label, path in (("source", src_path), ("reference", ref_path)):
        stat = os.stat(path)
        input_metadata = {
            "name": os.path.basename(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if include_hashes:
            input_metadata["sha256"] = _file_sha256(path)
        metadata["inputs"][label] = input_metadata
    return metadata


# ---------------------------------------------------------------------------
# 1. Robust image reader — handles .tif, .img, .png, .jpg
# ---------------------------------------------------------------------------
def robust_read(path: str, max_dim: Optional[int] = None) -> np.ndarray:
    """
    Read an image from disk, handling massive TIFFs and raw .img files
    that OpenCV refuses to open. When max_dim is supplied, large GeoTIFFs
    are read using sub-sampled decimation in <50ms instead of loading gigabytes.
    """
    ext = os.path.splitext(path)[1].lower()

    # Strategy 1: use rasterio decimated read or tifffile for .tif/.tiff
    if ext in ['.tif', '.tiff']:
        if max_dim is not None:
            try:
                import rasterio
                with rasterio.open(path) as ds:
                    h, w = ds.height, ds.width
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
                        arr = ds.read(1, out_shape=(nh, nw))
                        if arr.dtype != np.uint8:
                            arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        logger.info(f"Fast decimated read ({h}x{w}) → ({nh}x{nw}) via rasterio")
                        return arr
            except Exception as e:
                logger.debug(f"Rasterio decimated read fallback: {e}")

        import tifffile
        img = tifffile.imread(path)
        if img is None or img.size == 0:
            raise ValueError(f"tifffile returned empty array for {path}")
        if len(img.shape) == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        if img.dtype != np.uint8:
            # Normalize to 8-bit for downstream processing
            img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

    # Strategy 2: raw .img files (Chandrayaan-2 PDS4 and planetary binary images)
    if ext == '.img':
        try:
            from src.ingest.pds4_parser import load_img_raster
            logger.info(f"Loading planetary .img raster directly: {path} (max_dim={max_dim})")
            return load_img_raster(path, max_dim=max_dim)
        except Exception as e:
            logger.error(f"Failed to load .img raster {path}: {e}")
            raise ValueError(
                f"Could not load .img file '{os.path.basename(path)}': {e}. "
                "Ensure the matching PDS4 .xml label is in the same directory or project data folder."
            )

    # Strategy 3: standard formats via OpenCV
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at path: {path}")
    return img


# ---------------------------------------------------------------------------
# 2. Smart downscaler
# ---------------------------------------------------------------------------
def smart_downscale(img: np.ndarray, max_dim: int = 2048) -> tuple:
    """
    Downscale large images to fit within max_dim on the longest side.
    Returns (scaled_image, scale_factor).

    Special handling for pushbroom orbital strips (aspect ratio > 3:1):
    Chandrayaan-2 OHRC/TMC strips can be 10000+ px tall but only 1200 px wide.
    Scaling by the long side (height) collapses the width to <300 px, which:
      - Destroys cross-track feature density
      - Inflates RMSE by pixel-grid aliasing
      - Causes artificially low spatial coverage
    For high-aspect-ratio images, we scale by the SHORT side instead,
    capping the long side at a separate MAX_STRIP_DIM limit.
    """
    h, w = img.shape[:2]
    aspect = max(h, w) / max(min(h, w), 1)

    if aspect > 3.0:
        # Pushbroom / strip sensor image — preserve cross-track width
        # Scale by SHORT side, keeping it at min(short_side, max_dim)
        short = min(h, w)
        long  = max(h, w)
        # Target: short side = min(short, max_dim); long side can go up to 4096
        max_short = min(short, max_dim)
        scale   = max_short / max(short, 1)
        new_h   = max(int(h * scale), 16)
        new_w   = max(int(w * scale), 16)
        # If long side still exceeds 4096, apply a secondary clamp
        MAX_STRIP_LONG = 4096
        if max(new_h, new_w) > MAX_STRIP_LONG:
            scale2 = MAX_STRIP_LONG / max(new_h, new_w)
            new_h  = max(int(new_h * scale2), 16)
            new_w  = max(int(new_w * scale2), 16)
            scale  = scale * scale2
        if scale >= 1.0:
            return img, 1.0
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(
            f"smart_downscale [STRIP mode, aspect={aspect:.1f}]: "
            f"{h}×{w} → {new_h}×{new_w} (scale={scale:.4f})"
        )
        return resized, float(scale)
    else:
        # Normal image — scale by longest dimension as before
        if max(h, w) <= max_dim:
            return img, 1.0
        scale = max_dim / max(h, w)
        new_w = max(int(w * scale), 16)
        new_h = max(int(h * scale), 16)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, float(scale)


def estimate_pair_overlap(img_src: np.ndarray, img_ref: np.ndarray, sample_size: int = 256) -> float:
    """Fast, deterministic overlap proxy for rejecting non-overlapping lunar pairs early."""
    if img_src is None or img_ref is None:
        return 0.0

    src = np.asarray(img_src)
    ref = np.asarray(img_ref)
    if src.size == 0 or ref.size == 0:
        return 0.0

    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if ref.ndim == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)

    h1, w1 = src.shape[:2]
    h2, w2 = ref.shape[:2]
    if min(h1, w1) < 8 or min(h2, w2) < 8:
        return 0.0

    target = max(32, min(sample_size, max(h1, w1), max(h2, w2)))
    scale1 = target / max(h1, w1)
    scale2 = target / max(h2, w2)
    src_small = cv2.resize(src, (max(16, int(w1 * scale1)), max(16, int(h1 * scale1))), interpolation=cv2.INTER_AREA)
    ref_small = cv2.resize(ref, (max(16, int(w2 * scale2)), max(16, int(h2 * scale2))), interpolation=cv2.INTER_AREA)

    # Normalize both images to the same spatial footprint before computing correlation.
    # Different image sizes otherwise cause broadcast failures in the real lunar dataset.
    common_h = max(src_small.shape[0], ref_small.shape[0])
    common_w = max(src_small.shape[1], ref_small.shape[1])
    src_pad = np.zeros((common_h, common_w), dtype=np.float32)
    ref_pad = np.zeros((common_h, common_w), dtype=np.float32)
    src_pad[: src_small.shape[0], : src_small.shape[1]] = src_small.astype(np.float32)
    ref_pad[: ref_small.shape[0], : ref_small.shape[1]] = ref_small.astype(np.float32)

    def _normalized_edge_map(gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = mag / (np.mean(mag) + 1e-6)
        return mag

    src_mag = _normalized_edge_map(src_pad)
    ref_mag = _normalized_edge_map(ref_pad)
    src_z = src_mag - np.mean(src_mag)
    ref_z = ref_mag - np.mean(ref_mag)
    denom = np.linalg.norm(src_z) * np.linalg.norm(ref_z)
    corr = float(np.sum(src_z * ref_z) / (denom + 1e-8))

    src_hist = cv2.calcHist([src_pad.astype(np.uint8)], [0], None, [32], [0, 256]).astype(np.float32)
    ref_hist = cv2.calcHist([ref_pad.astype(np.uint8)], [0], None, [32], [0, 256]).astype(np.float32)
    hist_sim = float(np.sum(np.minimum(src_hist / (src_hist.sum() + 1e-8), ref_hist / (ref_hist.sum() + 1e-8))))
    score = float(np.clip(0.7 * corr + 0.3 * hist_sim, -1.0, 1.0))
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 3. GSD-aware resolution normalization
# ---------------------------------------------------------------------------
def normalize_resolution(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    gsd_src: float = None,
    gsd_ref: float = None,
) -> tuple:
    """
    Normalize both images to the same Ground Sample Distance (GSD)
    before matching, so features correspond at the same spatial scale.

    If GSD values are not provided, estimates relative scale from
    image dimensions (assumes similar geographic extent).

    Returns (img_src_norm, img_ref_norm, applied_scale_ratio).
    """
    if gsd_src and gsd_ref and gsd_src != gsd_ref:
        # Target the coarser resolution (downsample the finer one)
        target_gsd = max(gsd_src, gsd_ref)

        if gsd_src < target_gsd:
            scale = gsd_src / target_gsd
            new_h = max(int(img_src.shape[0] * scale), 16)
            new_w = max(int(img_src.shape[1] * scale), 16)
            img_src = cv2.resize(img_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"GSD normalization: downscaled source {gsd_src}m → {target_gsd}m/px")
        elif gsd_ref < target_gsd:
            scale = gsd_ref / target_gsd
            new_h = max(int(img_ref.shape[0] * scale), 16)
            new_w = max(int(img_ref.shape[1] * scale), 16)
            img_ref = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"GSD normalization: downscaled reference {gsd_ref}m → {target_gsd}m/px")

        return img_src, img_ref, gsd_src / gsd_ref
    else:
        # No GSD info — check if aspect ratios match but sizes differ greatly
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]
        area_ratio = (h_s * w_s) / max(h_r * w_r, 1)

        if area_ratio > 4.0:
            # Source is much bigger — likely higher resolution
            scale = np.sqrt(1.0 / area_ratio)
            new_h = max(int(h_s * scale), 16)
            new_w = max(int(w_s * scale), 16)
            img_src = cv2.resize(img_src, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Auto GSD: source area {area_ratio:.1f}x larger, downscaled to match")
        elif area_ratio < 0.25:
            # Reference is much bigger
            scale = np.sqrt(area_ratio)
            new_h = max(int(h_r * scale), 16)
            new_w = max(int(w_r * scale), 16)
            img_ref = cv2.resize(img_ref, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Auto GSD: reference area {1/area_ratio:.1f}x larger, downscaled to match")

        return img_src, img_ref, 1.0


# ---------------------------------------------------------------------------
# 4. Classical SIFT fallback matcher
# ---------------------------------------------------------------------------
def sift_match(img_src: np.ndarray, img_ref: np.ndarray, config: dict = None):
    """
    Classical RootSIFT + Cross-check BFMatcher fallback with two-tier ratio test.
    
    Tier 1: Strict ratio (0.75) — high-confidence mutual nearest neighbours.
    Tier 2: Relaxed ratio (0.82) — additional matches verified by local affine
            consistency (scale/orientation agreement with nearest primary matches).
    
    Returns (pts_src, pts_ref, confidences).
    """
    config = config or {}
    nfeatures   = config.get('sift_fallback', {}).get('nfeatures', 8000)
    ratio_thresh = config.get('sift_fallback', {}).get('ratio_threshold', 0.75)
    ratio_thresh_secondary = config.get('sift_fallback', {}).get('ratio_threshold_secondary', 0.82)

    sift = cv2.SIFT_create(nfeatures=nfeatures)
    kp1, des1 = sift.detectAndCompute(img_src, None)
    kp2, des2 = sift.detectAndCompute(img_ref, None)
    des1 = root_sift_descriptors(des1)  # Upgrade to RootSIFT (Hellinger metric)
    des2 = root_sift_descriptors(des2)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([]), np.array([]), np.array([])

    # Adaptive ratio: relax slightly when match pool is small (low-texture maria)
    n_candidates = min(len(des1), len(des2))
    adaptive_ratio = ratio_thresh if n_candidates >= 500 else min(ratio_thresh + 0.05, 0.85)
    adaptive_ratio_secondary = ratio_thresh_secondary if n_candidates >= 500 else min(ratio_thresh_secondary + 0.03, 0.88)

    # Step 1: kNN (k=2) for Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_fwd = bf.knnMatch(des1, des2, k=2)
    raw_bwd = bf.knnMatch(des2, des1, k=2)

    # === TIER 1: Primary strict ratio test ===
    fwd_pass = {}
    for m_list in raw_fwd:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < adaptive_ratio * n.distance:
                fwd_pass[m.queryIdx] = m

    bwd_pass = {}
    for m_list in raw_bwd:
        if len(m_list) == 2:
            m, n = m_list
            if m.distance < adaptive_ratio * n.distance:
                bwd_pass[m.queryIdx] = m

    # Cross-check — keep only mutual nearest neighbours (primary matches)
    primary_good = [
        fwd_pass[i] for i in fwd_pass
        if i in fwd_pass and fwd_pass[i].trainIdx in bwd_pass
        and bwd_pass[fwd_pass[i].trainIdx].trainIdx == i
    ]

    # Track which indices are already matched
    matched_query = {m.queryIdx for m in primary_good}
    matched_train = {m.trainIdx for m in primary_good}

    # === TIER 2: Secondary relaxed ratio test on remaining unmatched features ===
    secondary_good = []
    if len(primary_good) >= 3:
        fwd_pass_2 = {}
        for m_list in raw_fwd:
            if len(m_list) == 2:
                m, n = m_list
                if m.queryIdx not in matched_query and m.trainIdx not in matched_train:
                    if m.distance < adaptive_ratio_secondary * n.distance:
                        fwd_pass_2[m.queryIdx] = m

        bwd_pass_2 = {}
        for m_list in raw_bwd:
            if len(m_list) == 2:
                m, n = m_list
                if m.queryIdx not in matched_train and m.trainIdx not in matched_query:
                    if m.distance < adaptive_ratio_secondary * n.distance:
                        bwd_pass_2[m.queryIdx] = m

        secondary_candidates = [
            fwd_pass_2[i] for i in fwd_pass_2
            if fwd_pass_2[i].trainIdx in bwd_pass_2
            and bwd_pass_2[fwd_pass_2[i].trainIdx].trainIdx == i
        ]

        # Local affine consistency: verify scale/orientation with nearest primary matches
        primary_kp1 = [kp1[m.queryIdx] for m in primary_good]
        primary_kp2 = [kp2[m.trainIdx] for m in primary_good]
        for m in secondary_candidates:
            q_kp, t_kp = kp1[m.queryIdx], kp2[m.trainIdx]
            # Find 3 nearest primary keypoints
            if len(primary_kp1) >= 3:
                q_pt = np.array(q_kp.pt)
                dists = np.array([np.linalg.norm(q_pt - np.array(pk.pt)) for pk in primary_kp1])
                nearest_idx = np.argsort(dists)[:3]
                # Scale ratio consistency
                scale_ratio = t_kp.size / max(q_kp.size, 1e-6)
                neighbor_ratios = [primary_kp2[ni].size / max(primary_kp1[ni].size, 1e-6) for ni in nearest_idx]
                median_ratio = float(np.median(neighbor_ratios))
                if median_ratio > 0 and abs(scale_ratio / median_ratio - 1.0) > 0.5:
                    continue
                # Orientation consistency
                angle_diff = (t_kp.angle - q_kp.angle) % 360
                neighbor_angles = [(primary_kp2[ni].angle - primary_kp1[ni].angle) % 360 for ni in nearest_idx]
                median_angle = float(np.median(neighbor_angles))
                angle_dev = min(abs(angle_diff - median_angle), 360 - abs(angle_diff - median_angle))
                if angle_dev > 30:
                    continue
            secondary_good.append(m)

        if secondary_good:
            logger.info(f"SIFT two-tier: {len(primary_good)} primary + {len(secondary_good)} secondary matches")

    good = primary_good + secondary_good

    if len(good) < 4:
        return np.array([]), np.array([]), np.array([])

    pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_ref = np.float32([kp2[m.trainIdx].pt for m in good])

    # Ratio-test based confidence: 1 - (d1/d2), giving higher scores to more distinctive matches
    fwd_knn_map = {}
    for m_list in raw_fwd:
        if len(m_list) == 2:
            m, n = m_list
            fwd_knn_map[m.queryIdx] = (m.distance, n.distance)

    confidences_list = []
    for m in good:
        if m.queryIdx in fwd_knn_map:
            d1, d2 = fwd_knn_map[m.queryIdx]
            conf = float(np.clip(1.0 - d1 / (d2 + 1e-6), 0.0, 1.0))
        else:
            max_dist_val = max(g.distance for g in good) + 1e-6
            conf = float(np.clip(1.0 - m.distance / max_dist_val, 0.0, 1.0))
        confidences_list.append(conf)
    confidences = np.array(confidences_list, dtype=np.float32)

    return pts_src, pts_ref, confidences


def _build_rejected_result(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    method_used: str,
    total_matches: int,
    reason: str,
    pipeline_times: dict,
    recovery_attempts: list = None,
    provenance: dict = None,
    provenance_chain: Optional[ProvenanceChain] = None,
):
    """Build a complete, JSON-safe result when matching cannot start geometry estimation."""
    quality = assess_quality(
        rmse_pixels=None,
        coverage=0.0,
        inlier_ratio=0.0,
        inlier_count=0,
        total_matches=total_matches,
        spatial_entropy=0.0,
    )
    quality["reason_codes"].insert(0, reason)
    quality["primary_reason"] = reason
    quality["warnings"].insert(0, f"Registration rejected before geometric estimation: {reason}.")
    auto_recovery = quality.get("details", {}).get("auto_recovery")
    empty_points = np.empty((0, 2), dtype=np.float32)
    telemetry = RegistrationTelemetry.from_results({
        "method_used": method_used,
        "total_matches": total_matches,
        "quality": quality,
        "auto_recovery": auto_recovery,
    })
    return {
        "pipeline_status": "FAILED QUALITY GATE",
        "quality_grade": "Grade F",
        "auto_recovery": auto_recovery,
        "total_matches": total_matches,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "rmse_pixels": None,
        "weighted_rmse": None,
        "rmse_raw_pixels": None,
        "weighted_rmse_raw": None,
        "gt_rmse_pixels": None,
        "gt_point_count": 0,
        "gt_validation_status": "not_available",
        "metadata_validation": {"status": "not_available"},
        "mission_grade_certified": False,
        "coverage": 0.0,
        "density_grid": np.zeros((8, 8), dtype=np.int32),
        "pts_src": empty_points,
        "pts_ref": empty_points.copy(),
        "inliers_src": empty_points.copy(),
        "inliers_ref": empty_points.copy(),
        "confidences": np.empty((0,), dtype=np.float32),
        "transform": None,
        "H_fallback": None,
        "mask": np.empty((0,), dtype=bool),
        "method_used": method_used,
        "scale_src": 1.0,
        "scale_ref": 1.0,
        "scale_ref_total": 1.0,
        "img_src_warped": img_src.copy(),
        "pipeline_times": pipeline_times,
        "error_stats": {},
        "quality": quality,
        "holdout_validation": {},
        "error_budget": {},
        "spatial_entropy": 0.0,
        "condition_number": None,
        "geometric_model_type": "HOMOGRAPHY",
        "fallback_code": None,
        "fusion_stats": None,
        "omniroute_stats": None,
        "telemetry": telemetry.to_dict(),
        "recovery_attempts": recovery_attempts or [],
        "provenance": provenance or {},
        "provenance_chain": provenance_chain.to_dict() if provenance_chain else None,
    }


def choose_matcher_strategy(overlap_score: float, preferred_matcher: str = "loftr") -> tuple[str, str]:
    """Choose a reliable matcher profile based on overlap strength.

    Weak lunar pairs with low visual overlap are much more likely to fail under
    deep matchers; for those, the deterministic SIFT path is safer and more
    stable. High-overlap pairs can still use deep features.
    """
    matcher = (preferred_matcher or "sift").lower()
    if overlap_score < 0.12:
        return "sift", "low-overlap fallback: deep matcher rejected; using deterministic SIFT for weak lunar overlap"
    if overlap_score < 0.25 and matcher in {"loftr", "lightglue", "roma", "roma_v2", "roma_omega", "tiny_roma", "ni_lfm", "fusion", "lunar_fusion", "ensemble"}:
        return "sift", "low-overlap guard: deep matching was disabled because pair overlap is below the safe threshold"
    return matcher, "deep match allowed: pair overlap is strong enough for AI-based feature matching"


def build_recovery_plan(matcher_type: str, max_dim: int, reason: str) -> dict:
    """Return a deterministic recovery plan for failed registrations.

    Rules enforced by the AGENTS.md quality gates:
    - if the match fails a hard gate, retry at 4096 px
    - if the matcher is AI-based, downgrade to deterministic SIFT to avoid MPS / over-smoothing failures
    """
    current_matcher = (matcher_type or "sift").lower()
    ai_matchers = {
        "loftr", "roma", "roma_v2", "roma_omega", "tiny_roma", "tiny_roma_v1",
        "ni_lfm", "lightglue", "disk_lightglue", "fusion", "lunar_fusion",
        "ensemble", "omniroute", "omni"
    }

    retry_matcher = current_matcher if current_matcher == "sift" else "sift"
    retry_dim = 4096 if max_dim < 4096 else max_dim

    retry_reasons = {
        "FOUR_POINT_OVERFIT_TRAP",
        "INSUFFICIENT_INLIERS",
        "SEVERE_MATCH_CLUSTERING",
        "LOW_SPATIAL_COVERAGE",
        "HIGH_RMSE_FAILURE",
        "HOLDOUT_VALIDATION_FAILURE",
        "SYMMETRIC_GEOMETRY_FAILURE",
        "LOW_INLIER_RATIO",
        "CRITICAL_INLIER_RATIO",
        "CRITICAL_COVERAGE_FAILURE",
    }

    if reason and reason.upper() in retry_reasons:
        retry_dim = 4096
    elif reason and "CLUSTERED" in reason.upper():
        retry_dim = 4096
    elif reason and "RMSE" in reason.upper():
        retry_dim = 4096
    elif reason and "INSUFFICIENT" in reason.upper():
        retry_dim = 4096

    return {
        "matcher_type": retry_matcher,
        "max_dim": retry_dim,
        "reason": reason,
        "use_tile_matching": True,
        "tile_grid": (4, 4),
        "auto_recover": False,
    }


# ---------------------------------------------------------------------------
# 5. Main pipeline (enhanced)
# ---------------------------------------------------------------------------
def run_registration_pipeline(
    src_path: str,
    ref_path: str,
    use_tps: bool = True,
    max_dim: int = 2048,
    matcher_type: str = "loftr",
    progress_callback=None,
    geom_info: dict = None,
    gsd_src: float = None,
    gsd_ref: float = None,
    gt_pts_src: np.ndarray = None,
    gt_pts_ref: np.ndarray = None,
    gt_points_in_original_coords: bool = False,
    use_tile_matching: bool = True,
    tile_grid: tuple = (3, 3),
    auto_recover: bool = True,
    cancellation_token = None,
    generate_3d_dem: bool = False,
    stereo_sensor: str = "tmc",
    output_ply_path: str = None,
    output_dir: str = None,
    rejected_preview: bool = False,
    **kwargs
):
    """
    End-to-end lunar image registration pipeline (enhanced).

    Parameters
    ----------
    src_path : str          Path to the source image file.
    ref_path : str          Path to the reference image file.
    use_tps  : bool         Whether to use TPS for non-rigid jitter correction.
    max_dim  : int          Maximum image dimension for processing.
    matcher_type : str      'loftr', 'lightglue', 'sift', 'roma', 'tiny_roma', or 'ensemble'.
    progress_callback       Optional callable(step: int, total: int, msg: str).
    geom_info: dict         Optional dict with illumination geometry angles.
    gsd_src  : float        Source image GSD in meters/pixel (optional).
    gsd_ref  : float        Reference image GSD in meters/pixel (optional).
    gt_pts_src: np.ndarray  Independent source GCPs for GT-RMSE validation (optional).
    gt_pts_ref: np.ndarray  Independent reference GCPs for GT-RMSE validation (optional).
    gt_points_in_original_coords: bool  Whether GCPs use pre-downscale coordinates.
    use_tile_matching: bool Whether to use tile-based matching for better coverage.
    tile_grid: tuple        Grid dimensions for tile matching (rows, cols).
    auto_recover: bool      Whether to perform Gate 5 self-healing retry on failure.
    cancellation_token      Optional threading.Event to signal abort from background worker.
    generate_3d_dem: bool   Whether to synthesize a calibrated 3D DEM.
    stereo_sensor: str      Sensor model for 3D triangulation ('tmc' or 'ohrc').
    output_ply_path: str    Optional output path for 3D PLY mesh.
    output_dir: str         Optional output directory for §9 canonical artifacts.
    rejected_preview: bool  Whether to output an UNVALIDATED preview if rejected.

    Returns
    -------
    (img_src, img_ref, results_dict)
    """
    def _check_cancelled():
        if cancellation_token is not None and getattr(cancellation_token, "is_set", lambda: False)():
            raise InterruptedError("Registration pipeline cancelled by client")

    output_dir_eff = output_dir or kwargs.get("output_dir")
    rejected_preview_eff = rejected_preview or kwargs.get("rejected_preview", False)

    def _maybe_export(res_dict: dict, warped_arr=None) -> dict:
        if output_dir_eff:
            try:
                from .registration.artifact_exporter import export_all_artifacts
                res_dict["exported_artifacts"] = export_all_artifacts(
                    output_dir=output_dir_eff,
                    results=res_dict,
                    warped_img=warped_arr,
                    reference_tif=ref_path if str(ref_path).lower().endswith((".tif", ".tiff")) else None,
                    rejected_preview=rejected_preview_eff,
                )
            except Exception as e_exp:
                logger.warning(f"Failed to export registration artifacts: {e_exp}")
        return res_dict

    _check_cancelled()
    config = get_config()
    metadata_validation = {"status": "not_requested"}
    src_label_path = kwargs.get("src_pds4_label")
    ref_label_path = kwargs.get("ref_pds4_label")
    src_grid_path = kwargs.get("src_geometry_csv")
    ref_grid_path = kwargs.get("ref_geometry_csv")

    # Metadata informs normalization and validation only; it never enters matching.
    if src_label_path:
        src_metadata = parse_pds4_metadata(src_label_path)
        if gsd_src is None and isinstance(src_metadata.get("gsd_m"), (int, float)):
            gsd_src = float(src_metadata["gsd_m"])
    else:
        src_metadata = {}
    if ref_label_path:
        ref_metadata = parse_pds4_metadata(ref_label_path)
        if gsd_ref is None and isinstance(ref_metadata.get("gsd_m"), (int, float)):
            gsd_ref = float(ref_metadata["gsd_m"])
    else:
        ref_metadata = {}

    if geom_info is None and src_metadata and ref_metadata:
        src_incidence = src_metadata.get("incidence")
        ref_incidence = ref_metadata.get("incidence")
        if isinstance(src_incidence, (int, float)) and isinstance(ref_incidence, (int, float)):
            geom_info = {
                "src_inc": float(src_incidence),
                "src_ema": float(src_metadata.get("emission", 0.0)),
                "src_pha": float(src_metadata.get("phase", src_incidence)),
                "ref_inc": float(ref_incidence),
                "ref_ema": float(ref_metadata.get("emission", 0.0)),
                "ref_pha": float(ref_metadata.get("phase", ref_incidence)),
            }
            logger.info("Using supplied PDS4 labels for automatic photometric geometry")

    if gt_pts_src is None and gt_pts_ref is None and src_grid_path and ref_grid_path:
        try:
            gt_pts_src, gt_pts_ref, metadata_validation = validate_geographic_overlap(
                src_grid_path,
                ref_grid_path,
                max_distance_m=float(kwargs.get("geo_gcp_max_distance_m", 250.0)),
                max_points=int(kwargs.get("geo_gcp_max_points", 200)),
            )
            gt_points_in_original_coords = True
        except (OSError, ValueError) as error:
            metadata_validation = {"status": "invalid_metadata", "error": str(error)}
    elif src_grid_path or ref_grid_path:
        metadata_validation = {"status": "incomplete_metadata_pair"}

    provenance = _build_provenance(
        src_path,
        ref_path,
        config,
        include_hashes=bool(kwargs.get("include_input_hashes", False)),
    )
    provenance["pds4"] = {
        "source": src_metadata,
        "reference": ref_metadata,
        "geographic_validation": metadata_validation,
    }
    recovery_attempts = list(kwargs.get("recovery_attempts", []))
    recovery_attempts.append({
        "matcher": matcher_type,
        "max_dim": max_dim,
        "status": "started",
    })

    prov_chain = ProvenanceChain()
    prov_chain.record(
        stage="ingest",
        details={
            "src_name": os.path.basename(src_path),
            "ref_name": os.path.basename(ref_path),
            "matcher_type": matcher_type,
            "max_dim": max_dim,
            "use_tps": use_tps,
        },
        payload=src_path,
    )

    def _progress(step, total, msg):
        _check_cancelled()
        logger.info(f"[{step}/{total}] {msg}")
        if progress_callback:
            progress_callback(step, total, msg)

    total_steps = 8  # Added quality gate step
    pipeline_times = {}
    
    # --- STAGE 1: Read ---
    _progress(1, total_steps, "Reading images...")
    t_start = time.time()
    img_src_raw = robust_read(src_path, max_dim=max_dim)
    img_ref_raw = robust_read(ref_path, max_dim=max_dim)
    pipeline_times['read'] = time.time() - t_start

    # --- STAGE 2: Smart downscale + GSD normalization ---
    _progress(2, total_steps, "Resolution normalization & downscaling...")
    t_start = time.time()

    # GSD normalization for cross-sensor matching (e.g., OHRC 25cm vs TMC 5m)
    img_src_gsd, img_ref_gsd, gsd_ratio = normalize_resolution(
        img_src_raw, img_ref_raw, gsd_src, gsd_ref
    )

    img_src, scale_src = smart_downscale(img_src_gsd, max_dim)
    img_ref, scale_ref = smart_downscale(img_ref_gsd, max_dim)
    logger.info(f"Source: {img_src_raw.shape} → {img_src.shape}  (scale={scale_src:.4f})")
    logger.info(f"Reference: {img_ref_raw.shape} → {img_ref.shape}  (scale={scale_ref:.4f})")
    pipeline_times['downscale'] = time.time() - t_start

    # --- STAGE 3: Preprocessing (Photometric + Illumination) ---
    _progress(3, total_steps, "Preprocessing (Photometric + Enhanced Normalization)...")
    t_start = time.time()
    norm_src = img_src.copy()
    norm_ref = img_ref.copy()

    # Try to auto-detect geometry angles from PDS4 labels if not provided
    if geom_info is None:
        src_geom = get_geometry_for_image(src_path)
        ref_geom = get_geometry_for_image(ref_path)
        if src_geom and ref_geom and all(
            key in src_geom and key in ref_geom for key in ('incidence', 'phase')
        ):
            geom_info = {
                'src_inc': src_geom.get('incidence', 45.0),
                'src_ema': src_geom.get('emission', 0.0),
                'src_pha': src_geom.get('phase', 45.0),
                'ref_inc': ref_geom.get('incidence', 30.0),
                'ref_ema': ref_geom.get('emission', 0.0),
                'ref_pha': ref_geom.get('phase', 30.0),
            }
            logger.info("Using geometry angles from PDS4 labels for photometric correction")
    
    if geom_info and (
        config.get('photometric', {}).get('enabled', False)
        or all(key in geom_info for key in ('src_inc', 'src_pha', 'ref_inc', 'ref_pha'))
    ):
        photo_model = config.get('photometric', {}).get('model', 'hapke')
        target_inc = config.get('photometric', {}).get('target_incidence', 30.0)
        target_ema = config.get('photometric', {}).get('target_emission', 0.0)
        target_pha = config.get('photometric', {}).get('target_phase', 30.0)
        
        try:
            if photo_model == 'hapke':
                norm_src = apply_hapke_correction(
                    img_src, geom_info['src_inc'], geom_info['src_ema'], geom_info['src_pha'],
                    target_inc, target_ema, target_pha
                )
                norm_ref = apply_hapke_correction(
                    img_ref, geom_info['ref_inc'], geom_info['ref_ema'], geom_info['ref_pha'],
                    target_inc, target_ema, target_pha
                )
            elif photo_model == 'lunar_lambert':
                norm_src = apply_lunar_lambert(
                    img_src, geom_info['src_inc'], geom_info['src_ema'], geom_info['src_pha'],
                    target_incidence=target_inc, target_emission=target_ema
                )
                norm_ref = apply_lunar_lambert(
                    img_ref, geom_info['ref_inc'], geom_info['ref_ema'], geom_info['ref_pha'],
                    target_incidence=target_inc, target_emission=target_ema
                )
        except Exception as e:
            logger.warning(f"Photometric correction warning: {e}, continuing with standard normalization.")

    # Research Paper Preprocessing Pipelines (Pipelines A, B, C)
    pipeline_mode = kwargs.get('pipeline_mode') or config.get('pipeline', {}).get('pipeline_mode')
    sensor_pair = kwargs.get('sensor_pair') or config.get('pipeline', {}).get('sensor_pair')
    if pipeline_mode or sensor_pair:
        try:
            from .preprocess.pipeline_router import SensorModalityRouter
            norm_src, norm_ref, _ = SensorModalityRouter.route_and_preprocess(
                norm_src, norm_ref,
                modality_or_pair=pipeline_mode or "auto",
                sensor_type_src=sensor_pair or "auto",
                sensor_type_ref=sensor_pair or "auto",
            )
        except Exception as e_pipe:
            logger.warning(f"SensorModalityRouter warning: {e_pipe}, falling back to standard Wallis normalization")
            wallis_mean = config.get('preprocessing', {}).get('wallis_target_mean', 110.0)
            wallis_std  = config.get('preprocessing', {}).get('wallis_target_std',  35.0)
            norm_src = normalize_illumination(norm_src, wallis_target_mean=wallis_mean, wallis_target_std=wallis_std)
            norm_ref = normalize_illumination(norm_ref, wallis_target_mean=wallis_mean, wallis_target_std=wallis_std)
    else:
        # Enhanced normalization (Multi-scale CLAHE + Wallis filter)
        wallis_mean = config.get('preprocessing', {}).get('wallis_target_mean', 110.0)
        wallis_std  = config.get('preprocessing', {}).get('wallis_target_std',  35.0)
        norm_src = normalize_illumination(norm_src, wallis_target_mean=wallis_mean, wallis_target_std=wallis_std)
        norm_ref = normalize_illumination(norm_ref, wallis_target_mean=wallis_mean, wallis_target_std=wallis_std)
    
    # Optional phase congruency map enhancement or dual-pass cross-sun normalization
    sun_telem = {}
    cross_sun_flag = kwargs.get('cross_sun_resilience')
    if cross_sun_flag is None:
        cross_sun_flag = config.get('preprocessing', {}).get('cross_sun_resilience', False)

    if cross_sun_flag:
        try:
            from .preprocess.photometric_correction import apply_dual_pass_cross_sun_normalization
            norm_src, norm_ref, sun_telem = apply_dual_pass_cross_sun_normalization(norm_src, norm_ref)
            logger.info(f"Dual-Pass Cross-Sun Normalization active: delta_az={sun_telem.get('delta_azimuth_deg')}° (opposite={sun_telem.get('extreme_opposite_sun')})")
        except Exception as e_cross:
            logger.warning(f"Dual-pass cross-sun normalization fallback: {e_cross}")
    elif config.get('preprocessing', {}).get('use_hapke_albedo', False):
        try:
            from .preprocess.photometric_correction import remove_crater_shadows_hapke_congruency
            norm_src = remove_crater_shadows_hapke_congruency(norm_src)
            norm_ref = remove_crater_shadows_hapke_congruency(norm_ref)
            logger.info("Physical Hapke BRDF albedo inversion applied: shadows decoupled")
        except Exception as e_hapke:
            logger.debug(f"Hapke BRDF shadow inversion fallback: {e_hapke}")
    elif config.get('preprocessing', {}).get('use_phase_congruency', False):
        try:
            pc_src = phase_congruency_map(norm_src)
            pc_ref = phase_congruency_map(norm_ref)
            norm_src = cv2.addWeighted(norm_src, 0.7, (pc_src * 255).astype(np.uint8), 0.3, 0)
            norm_ref = cv2.addWeighted(norm_ref, 0.7, (pc_ref * 255).astype(np.uint8), 0.3, 0)
        except Exception as e:
            logger.warning(f"Phase congruency failed: {e}")

    pipeline_times['preprocess'] = time.time() - t_start
    prov_chain.record(
        stage="preprocessing",
        details={
            "scale_src": scale_src,
            "scale_ref": scale_ref,
            "gsd_ratio": gsd_ratio,
            "pipeline_mode": pipeline_mode,
            "sensor_pair": sensor_pair,
        },
        payload=norm_src,
    )

    overlap_score = estimate_pair_overlap(norm_src, norm_ref)
    logger.info(f"Overlap precheck score: {overlap_score:.3f}")
    selected_matcher, match_reason = choose_matcher_strategy(overlap_score, matcher_type)
    logger.info(f"Matcher selection: {selected_matcher} | {match_reason}")
    if overlap_score < 0.08:
        logger.warning(
            "[LOW_OVERLAP_PRECHECK] Image pair similarity is too low to be a valid registration candidate; "
            "rejecting before expensive feature matching."
        )
        pipeline_times['precheck_overlap'] = time.time() - t_start
        rejected = _build_rejected_result(
            img_src,
            img_ref,
            method_used=selected_matcher,
            total_matches=0,
            reason="LOW_OVERLAP_PRECHECK",
            pipeline_times=pipeline_times,
            recovery_attempts=recovery_attempts,
            provenance=provenance,
            provenance_chain=prov_chain,
        )
        rejected['quality']['primary_reason'] = 'LOW_OVERLAP_PRECHECK'
        rejected['telemetry']['status'] = 'rejected'
        return img_src, img_ref, _maybe_export(rejected, None)

    if selected_matcher != matcher_type:
        logger.warning(f"Forced fallback to {selected_matcher} because the overlap score is below the safe threshold for {matcher_type}.")
        matcher_type = selected_matcher

    # --- STAGE 4: Feature extraction & matching (tile-based or whole-image) ---
    _progress(4, total_steps, "Extracting and matching features (tile-based)...")
    t_start = time.time()

    pts_src = np.empty((0, 2), dtype=np.float32)
    pts_ref = np.empty((0, 2), dtype=np.float32)
    confidences = np.empty((0,), dtype=np.float32)
    method_used = matcher_type

    use_pyramid = os.environ.get('APP_USE_PYRAMID', 'False').lower() == 'true' or config.get('pipeline', {}).get('use_pyramid', False)
    pyr_levels = config.get('pipeline', {}).get('pyramid_levels', 3)
    
    if use_pyramid and pyr_levels > 1:
        pyr_src = build_gaussian_pyramid(norm_src, pyr_levels)
        pyr_ref = build_gaussian_pyramid(norm_ref, pyr_levels)
        match_src = pyr_src[-1]
        match_ref = pyr_ref[-1]
        scale_factor = float(2 ** (pyr_levels - 1))
    else:
        match_src = norm_src
        match_ref = norm_ref
        scale_factor = 1.0

    # --- Feature Matching: OmniRoute, LunarFusion, Ensemble, or Single Matcher Mode ---
    fusion_stats = None
    omniroute_stats = None
    if matcher_type.lower() in ["omniroute", "omni"]:
        _progress(4, total_steps, "Running OmniRoute AI Gateway (Parallel ThreadPool Dispatch & Fault Isolation)...")
        from .features.omniroute import LunarOmniRoute
        strategy = kwargs.get("omni_strategy", "consensus")
        max_workers = kwargs.get("omni_workers", 4)
        preferred_models = kwargs.get("preferred_models", None)
        omni_router = LunarOmniRoute(
            strategy=strategy,
            confidence_threshold=kwargs.get("confidence_threshold", 0.20),
            max_workers=max_workers,
        )
        omni_res = omni_router.route_and_match(
            match_src, match_ref,
            strategy=strategy,
            preferred_models=preferred_models,
            progress_callback=lambda msg: _progress(4, total_steps, f"OmniRoute: {msg}"),
            config=config
        )
        pts_src, pts_ref, confidences = omni_res.pts_src, omni_res.pts_ref, omni_res.confidences
        models_str = " + ".join(m.upper() for m in omni_res.models_succeeded) if omni_res.models_succeeded else "SIFT"
        method_used = f"OMNIROUTE [{strategy.upper()}] ({models_str})"
        omniroute_stats = {
            'strategy': omni_res.strategy,
            'models_succeeded': omni_res.models_succeeded,
            'latency_ms': omni_res.total_latency_ms,
            'per_model_latency': omni_res.per_model_latency,
            'per_model_counts': omni_res.per_model_counts,
            'profile_tag': omni_res.profile_tag,
            'reason': omni_res.reason
        }
        logger.info(f"OmniRoute finished: {len(pts_src)} matches in {omni_res.total_latency_ms:.1f}ms")
    elif matcher_type.lower() in ["fusion", "lunar_fusion"]:
        _progress(4, total_steps, "Running LunarFusion Adaptive AI Matcher (Profile Analysis → Smart Routing → Unified RANSAC)...")
        from .features.lunar_fusion_matcher import LunarFusionMatcher
        speed_mode = kwargs.get("speed_mode", "balanced")
        fusion_matcher = LunarFusionMatcher(
            speed_mode=speed_mode,
            confidence_threshold=0.2,
            progress_callback=lambda s, t, msg: _progress(4, total_steps, f"LunarFusion: {msg}")
        )
        pts_src, pts_ref, confidences, fusion_stats = fusion_matcher.match_with_stats(
            match_src, match_ref, config=config
        )
        models_str = " + ".join(m.upper() for m in fusion_stats.get('models_used', []))
        method_used = f"LUNARFUSION ({models_str})"
        logger.info(
            f"LunarFusion finished: {len(pts_src)} matches from {fusion_stats.get('models_used')} "
            f"(profile: {fusion_stats.get('profile_tag')})"
        )
    elif matcher_type.lower() == "ensemble":
        _progress(4, total_steps, "Running Multi-Model Ensemble feature matching (RoMa v2 + LoFTR + SIFT)...")
        ensemble = EnsembleMatcher(
            confidence_threshold=0.2,
            spatial_tolerance=3.0,
        )
        pts_src, pts_ref, confidences, models_succeeded, stats = ensemble.match(
            match_src, match_ref,
            config=config,
            progress_callback=lambda msg: _progress(4, total_steps, f"Ensemble: {msg}")
        )
        models_str = " + ".join(m.upper() for m in models_succeeded) if models_succeeded else "SIFT"
        method_used = f"ENSEMBLE ({models_str})"
        logger.info(
            f"Ensemble Matching finished: {len(pts_src)} fused matches from {models_succeeded} "
            f"({stats.get('consensus_count', 0)} consensus-boosted)"
        )
    else:
        # --- Tile-based matching for better spatial coverage ---
        # Note: RoMa v2 is a dense global matcher that handles spatial coverage natively,
        # so tile slicing is reserved for sparse matchers like LoFTR and LightGlue.
        if use_tile_matching and matcher_type not in ["sift", "roma", "roma_v2", "roma_outdoor", "tiny_roma", "tiny_roma_v1", "roma_omega"]:
            try:
                # Create tile matcher using the deep model
                deep_tile_fn = create_deep_tile_matcher(matcher_type=matcher_type)
                pts_src, pts_ref, confidences = tile_match(
                    match_src, match_ref,
                    matcher_fn=deep_tile_fn,
                    grid=tile_grid,
                    overlap=0.25,
                    confidence_threshold=0.2,
                )
                method_used = f"{matcher_type.upper()} (Tile-Based {tile_grid[0]}x{tile_grid[1]})"
                logger.info(f"Tile-based {matcher_type}: {len(pts_src)} matches")
            except Exception as e:
                logger.warning(f"Tile-based {matcher_type} failed ({e}), falling back to whole-image...")
                logger.debug(traceback.format_exc())

        # Whole-image deep matching fallback
        if len(pts_src) < 8 and matcher_type != "sift":
            try:
                matcher = LunarMatcher(matcher_type=matcher_type, confidence_threshold=0.2)
                tensor_src = torch.from_numpy(match_src).float().unsqueeze(0).unsqueeze(0) / 255.0
                tensor_ref = torch.from_numpy(match_ref).float().unsqueeze(0).unsqueeze(0) / 255.0

                t0_match = time.time()
                pts_src, pts_ref, confidences = matcher.match(tensor_src, tensor_ref)
                elapsed_match = time.time() - t0_match
                logger.info(f"{matcher_type} (whole-image) found {len(pts_src)} matches in {elapsed_match:.2f}s")
                method_used = matcher_type.upper()
            except Exception as e:
                logger.warning(f"{matcher_type} failed ({e}), falling back to SIFT...")
                logger.debug(traceback.format_exc())

        # SIFT fallback — also use tile-based for better coverage
        if len(pts_src) < 8:
            logger.info("Using SIFT tile-based matcher fallback...")
            sift_tile_fn = create_sift_tile_matcher(config)
            pts_src, pts_ref, confidences = tile_match(
                match_src, match_ref,
                matcher_fn=sift_tile_fn,
                grid=tile_grid,
                overlap=0.25,
                confidence_threshold=0.1,
            )
            if len(pts_src) < 8:
                # Final fallback: whole-image SIFT
                pts_src, pts_ref, confidences = sift_match(match_src, match_ref, config)
            method_used = "SIFT (Classical)" if matcher_type == "sift" else f"SIFT (Fallback from {matcher_type})"
            logger.info(f"SIFT found {len(pts_src)} matches")
        
    if use_pyramid and len(pts_src) > 0 and scale_factor > 1.0:
        pts_src = pts_src * scale_factor
        pts_ref = pts_ref * scale_factor

    pipeline_times['matching'] = time.time() - t_start
    prov_chain.record(
        stage="matching",
        details={
            "method_used": method_used,
            "raw_matches": len(pts_src),
        },
        payload=None,
    )

    if len(pts_src) < 4:
        logger.warning(
            "Registration rejected before transform estimation: %d raw matches found; need at least 4.",
            len(pts_src),
        )
        return img_src, img_ref, _maybe_export(_build_rejected_result(
            img_src,
            img_ref,
            method_used,
            len(pts_src),
            "INSUFFICIENT_RAW_MATCHES",
            pipeline_times,
            recovery_attempts,
            provenance,
            provenance_chain=prov_chain,
        ), None)

    # Pre-MAGSAC diagnostic: warn early if we're below the 15-inlier MVR floor.
    # MAGSAC++ will still run (needed to count inliers) but the quality gate
    # will catch it; this log helps in debugging pipeline stalls.
    if 4 <= len(pts_src) < 15:
        logger.warning(
            f"[PRE-MAGSAC] Only {len(pts_src)} raw matches (below MVR floor of 15). "
            "Quality Gate will likely reject this registration. Consider auto-recovery."
        )

    # --- STAGE 5: Outlier rejection & transform estimation (enhanced) ---
    _progress(5, total_steps, "Progressive outlier rejection & multi-pass refinement...")
    t_start = time.time()

    # Diagnostic: log match statistics before outlier rejection
    if len(pts_src) > 0:
        disp_magnitudes = np.linalg.norm(pts_src - pts_ref, axis=1)
        logger.info(
            f"Pre-MAGSAC: {len(pts_src)} matches | "
            f"displacement median={np.median(disp_magnitudes):.1f}px, "
            f"max={disp_magnitudes.max():.1f}px | "
            f"src spread=({pts_src[:,0].max()-pts_src[:,0].min():.0f}"
            f"×{pts_src[:,1].max()-pts_src[:,1].min():.0f})px"
        )

    transform, inlier_mask = estimate_transform(pts_src, pts_ref, use_tps=use_tps, config=config)

    if inlier_mask is None or len(inlier_mask) == 0:
        raise ValueError("Outlier rejection could not compute a geometric model.")

    inliers_src = pts_src[inlier_mask]
    inliers_ref = pts_ref[inlier_mask]
    if len(confidences) == len(inlier_mask) and len(confidences) > 0:
        inlier_conf = confidences[inlier_mask]
    else:
        inlier_conf = np.ones(len(inliers_src), dtype=np.float32)

    if len(inliers_src) < 4:
        n_found = len(pts_src)
        raise ValueError(
            f"Registration failed: all {n_found} feature matches were rejected as geometric outliers.\n"
            f"Matcher used: {method_used}\n"
            f"Possible causes:\n"
            f"  • The two images do not overlap (different lunar coordinates)\n"
            f"  • Extreme viewpoint/illumination difference between sensors\n"
            f"  • Try: switch matcher to SIFT, or reduce 'Max Processing Dimension'\n"
            f"  • Try: disable 'Coarse-to-Fine Pyramid' if enabled\n"
            f"  • Try: upload images that share visible surface features"
        )
    pipeline_times['outlier_rejection'] = time.time() - t_start

    # --- STAGE 6: Sub-pixel Refinement (iterative) ---
    _progress(6, total_steps, "Iterative sub-pixel refinement (upsampled DFT phase correlation)...")
    t_start = time.time()
    if config.get('pipeline', {}).get('enable_subpixel', True) and len(inliers_src) > 0:
        speed_mode = config.get('pipeline', {}).get('speed_mode', 'balanced')
        use_multipatch = (speed_mode != 'fast')
        default_patch = 192 if use_multipatch else 128
        patch_size = config.get('pipeline', {}).get('subpixel_patch_size', default_patch)
        resp_thresh = config.get('pipeline', {}).get('subpixel_response_threshold', 0.05)
        half_patch = patch_size // 2

        # Determine coarse homography for patch pre-warping
        H_coarse = None
        if isinstance(transform, np.ndarray) and transform.shape == (3, 3):
            H_coarse = transform
        elif hasattr(transform, 'H_base') and getattr(transform, 'H_base') is not None:
            H_coarse = transform.H_base

        inliers_src, inliers_ref, refined_conf, inlier_covs = refine_keypoints_warp_rectified(
            norm_src, norm_ref, inliers_src, inliers_ref,
            H_coarse=H_coarse, patch_size=patch_size, resp_thresh=resp_thresh
        )
        if len(refined_conf) == len(inlier_conf):
            inlier_conf = (inlier_conf + refined_conf) / 2.0
        else:
            inlier_conf = refined_conf

        # High-precision continuous soft-argmax temperature-scaled refinement
        if config.get('pipeline', {}).get('use_soft_argmax', True) and len(inliers_src) > 0:
            try:
                from .registration.subpixel_refine import refine_subpixel_soft_argmax
                inliers_src, inliers_ref, sa_conf = refine_subpixel_soft_argmax(
                    inliers_src, inliers_ref, norm_src, norm_ref, patch_radius=6, search_radius=3, temperature=0.05
                )
                if len(sa_conf) == len(inlier_conf):
                    inlier_conf = (inlier_conf + sa_conf) / 2.0
            except Exception as e_sa:
                logger.debug(f"Soft-argmax refinement fallback: {e_sa}")

        # Re-estimate final transform with refined points
        transform, refined_mask = estimate_transform(inliers_src, inliers_ref, use_tps=use_tps, config=config)
        # Update inlier sets to only those surviving the re-estimation
        if refined_mask is not None and np.sum(refined_mask) >= 4:
            inliers_src = inliers_src[refined_mask]
            inliers_ref = inliers_ref[refined_mask]
            inlier_conf = inlier_conf[refined_mask] if len(inlier_conf) == len(refined_mask) else inlier_conf[:len(inliers_src)]
            logger.info(f"Post-subpixel re-estimation: {np.sum(refined_mask)} final inliers")

    pipeline_times['subpixel_refinement'] = time.time() - t_start

    # --- STAGE 7: Metrics & Warped output ---
    _progress(7, total_steps, "Computing evaluation metrics & generating warped images...")
    t_start = time.time()

    n_total = len(pts_src)
    n_inliers = len(inliers_src)
    ratio = inlier_ratio(n_total, n_inliers)

    # Compute Projection and RMSE using the actual fitted transform model
    h, w = img_ref.shape[:2]
    H_fallback = None
    inliers_src_proj = None

    # 1. Non-rigid Thin-Plate Spline (Pushbroom spacecraft jitter model)
    if hasattr(transform, 'applyTransformation'):
        try:
            _, inliers_src_proj_tps = transform.applyTransformation(inliers_src.reshape(1, -1, 2))
            inliers_src_proj = inliers_src_proj_tps.reshape(-1, 2)
            logger.info("Evaluating registration residuals using fitted non-rigid Thin-Plate Spline (TPS)")
        except Exception as e:
            logger.warning(f"TPS projection failed ({e}), falling back to matrix projection")
            inliers_src_proj = None

    # 2. High-precision Progressive MAGSAC++ matrix (when TPS is disabled or unneeded)
    if inliers_src_proj is None and isinstance(transform, np.ndarray) and transform.shape == (3, 3):
        H_fallback = transform
        inliers_src_proj = cv2.perspectiveTransform(inliers_src.reshape(-1, 1, 2), H_fallback).reshape(-1, 2)
        logger.info("Evaluating registration residuals using fitted Progressive MAGSAC++ Homography")

    # 3. Establish H_fallback for GeoTIFF export and perspective warping compatibility (MAGSAC++)
    if H_fallback is None:
        try:
            H_fallback, _ = cv2.findHomography(
                inliers_src, inliers_ref, 
                cv2.USAC_MAGSAC, 
                3.0, 
                maxIters=50000, 
                confidence=0.999
            )
        except Exception:
            H_fallback = None
        if H_fallback is None or H_fallback.shape != (3, 3):
            try:
                H_fallback, _ = cv2.findHomography(
                    inliers_src, inliers_ref, 
                    cv2.USAC_MAGSAC, 
                    5.0, 
                    maxIters=50000, 
                    confidence=0.999
                )
            except Exception:
                H_fallback = None

    # 4. Fallback projection if still needed
    if inliers_src_proj is None:
        if H_fallback is not None and H_fallback.shape == (3, 3):
            inliers_src_proj = cv2.perspectiveTransform(inliers_src.reshape(-1, 1, 2), H_fallback).reshape(-1, 2)
        else:
            M, _ = cv2.estimateAffinePartial2D(inliers_src, inliers_ref)
            if M is not None:
                H_fallback = np.vstack([M, [0, 0, 1]])
                inliers_src_proj = cv2.transform(inliers_src.reshape(-1, 1, 2), M).reshape(-1, 2)
            else:
                H_fallback = np.eye(3)
                inliers_src_proj = inliers_src

    # Scale-aware accuracy tracking: metrics in original (un-downscaled) sensor coordinates
    scale_ref_total = img_ref.shape[1] / max(img_ref_raw.shape[1], 1)

    # Independent GT-RMSE validation. These points are evaluated only after the
    # transform is fitted and are never supplied to matching or estimation.
    gt_rmse_pixels = None
    gt_point_count = 0
    gt_validation_status = "not_provided"
    if gt_pts_src is not None or gt_pts_ref is not None:
        if gt_pts_src is None or gt_pts_ref is None:
            gt_validation_status = "invalid_pair"
        else:
            gt_src = np.asarray(gt_pts_src, dtype=np.float32).reshape(-1, 2)
            gt_ref = np.asarray(gt_pts_ref, dtype=np.float32).reshape(-1, 2)
            gt_point_count = len(gt_src)
            if len(gt_src) != len(gt_ref) or len(gt_src) < 4:
                gt_validation_status = "insufficient_points"
            elif not np.isfinite(gt_src).all() or not np.isfinite(gt_ref).all():
                gt_validation_status = "invalid_coordinates"
            else:
                if gt_points_in_original_coords:
                    gt_src = gt_src * scale_src
                    gt_ref = gt_ref * scale_ref
                if H_fallback is None or H_fallback.shape != (3, 3):
                    gt_validation_status = "transform_unavailable"
                else:
                    gt_rmse_pixels = evaluate_transform_gt_rmse(H_fallback, gt_src, gt_ref)
                    gt_validation_status = "valid"

    # --- CRITICAL CONSTRAINT: STRICT INLIER FLOOR & 4-POINT TRAP GATE ---
    MIN_REQUIRED_INLIERS = 15
    is_four_point_trap = (n_inliers <= 4)
    insufficient_inliers = (n_inliers < MIN_REQUIRED_INLIERS)

    if is_four_point_trap or insufficient_inliers:
        logger.error(
            f"[QUALITY GATE FAILED] Insufficient inliers ({n_inliers} < {MIN_REQUIRED_INLIERS}). "
            "Rejecting overfit/degenerate homography to prevent misleading 0.000 px RMSE."
        )
        current_rmse = None
        w_rmse = None
        rmse_raw = None
        w_rmse_raw = None
    else:
        # TPS interpolates exactly through its control points, so evaluating RMSE
        # on the fitted points yields 0.000 by definition (mathematical overfitting).
        # We use a held-out 20% random split as an independent test set.
        tps_is_fitted = hasattr(transform, 'applyTransformation') and inliers_src_proj is not None
        if tps_is_fitted and n_inliers >= 20:
            rng_eval = np.random.default_rng(seed=42)
            n_holdout = max(10, n_inliers // 5)
            holdout_idx = rng_eval.choice(n_inliers, size=n_holdout, replace=False)
            eval_src = inliers_src_proj[holdout_idx]
            eval_ref = inliers_ref[holdout_idx]
            eval_conf = inlier_conf[holdout_idx] if len(inlier_conf) == n_inliers else np.ones(n_holdout, np.float32)
            logger.info(f"RMSE evaluated on held-out {n_holdout}/{n_inliers} inliers (TPS overfitting guard)")
        else:
            eval_src = inliers_src_proj
            eval_ref = inliers_ref
            eval_conf = inlier_conf

        current_rmse = rmse(eval_src, eval_ref)
        w_rmse = weighted_rmse(eval_src, eval_ref, eval_conf)
        rmse_raw = current_rmse / max(scale_ref_total, 1e-6)
        w_rmse_raw = w_rmse / max(scale_ref_total, 1e-6)


    density_grid, coverage = match_density_grid(img_src.shape, inliers_src)
    err_stats = error_statistics(inliers_src_proj, inliers_ref)

    # Generate Warped Image using non-rigid TPS or perspective H_fallback
    if hasattr(transform, 'warpImage'):
        try:
            try:
                img_src_warped = transform.warpImage(img_src, output_shape=(h, w))
            except TypeError:
                img_src_warped = transform.warpImage(img_src)
            if img_src_warped.shape[:2] != (h, w):
                img_src_warped = cv2.resize(img_src_warped, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            logger.warning(f"Non-rigid warpImage failed ({e}), using warpPerspective...")
            img_src_warped = cv2.warpPerspective(img_src, H_fallback, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    elif H_fallback is not None and len(H_fallback) == 3 and not np.array_equal(H_fallback, np.eye(3)):
        img_src_warped = cv2.warpPerspective(img_src, H_fallback, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    else:
        img_src_warped = cv2.resize(img_src, (w, h), interpolation=cv2.INTER_CUBIC)

    pipeline_times['metrics_warping'] = time.time() - t_start

    # --- STAGE 8: Quality Gate ---
    _progress(8, total_steps, "Running quality assessment...")
    t_start = time.time()

    # Normal matrix condition number check (Tier 3)
    # Uses dimension-normalized coordinates to prevent image resolution (pixels squared)
    # from falsely inflating condition numbers on well-distributed satellite imagery.
    condition_number = None
    if len(inliers_src) >= 4 and w > 0 and h > 0:
        try:
            x_norm = (inliers_src[:, 0] - (w / 2.0)) / max(w / 2.0, 1.0)
            y_norm = (inliers_src[:, 1] - (h / 2.0)) / max(h / 2.0, 1.0)
            P = np.column_stack([x_norm, y_norm, np.ones(len(inliers_src), dtype=np.float64)])
            normal_mat = P.T @ P
            condition_number = float(np.linalg.cond(normal_mat))
        except Exception:
            condition_number = None

    # Spatial entropy (Tier 3: non-saturating information entropy)
    sp_entropy = compute_spatial_entropy(inliers_ref, img_ref.shape[:2])

    # True 70/30 Holdout validation (Tier 0)
    holdout_val = compute_holdout_validation(inliers_src, inliers_ref)

    # 4-Component Error budget decomposition (Tier 0 & Tier 1)
    error_budget = decompose_error_budget(inliers_src, inliers_ref, H_fallback)

    symmetric_rmse = None
    if H_fallback is not None and H_fallback.shape == (3, 3) and len(inliers_src) >= 8:
        try:
            reverse_h, _ = cv2.findHomography(
                inliers_ref, inliers_src, cv2.USAC_MAGSAC, 3.0,
                maxIters=50000, confidence=0.999,
            )
            if reverse_h is not None and np.isfinite(reverse_h).all():
                forward_then_reverse = cv2.perspectiveTransform(
                    cv2.perspectiveTransform(inliers_src.reshape(-1, 1, 2), H_fallback),
                    reverse_h,
                ).reshape(-1, 2)
                symmetric_rmse = rmse(forward_then_reverse, inliers_src)
        except cv2.error:
            symmetric_rmse = None

    # Physical Crater Landmark Coverage (YOLO11 Gate 3 integration)
    crater_coverage = None
    crater_count = None
    if config.get('enable_crater_verification', True) and os.path.exists("models/yolo11_lunar_crater.pt"):
        try:
            from .features.crater_detector import LunarCraterDetector
            crater_detector = LunarCraterDetector("models/yolo11_lunar_crater.pt")
            craters = crater_detector.detect(img_ref)
            cov_res = crater_detector.compute_crater_coverage(inliers_ref, craters)
            crater_coverage = cov_res["coverage_ratio"]
            crater_count = cov_res["total_craters"]
        except Exception as e:
            logger.warning(f"YOLO11 crater validation skipped: {e}")

    quality = assess_quality(
        rmse_pixels=current_rmse,
        coverage=coverage,
        inlier_ratio=ratio,
        inlier_count=n_inliers,
        total_matches=n_total,
        geometric_model_type=config.get('geometric_model_type', 'HOMOGRAPHY'),
        fallback_code=config.get('geometric_fallback_code', None),
        condition_number=condition_number,
        spatial_entropy=sp_entropy,
        holdout_rmse=holdout_val.get('holdout_rmse'),
        holdout_count=holdout_val.get('n_holdout', 0),
        symmetric_rmse=symmetric_rmse,
        crater_coverage=crater_coverage,
        crater_count=crater_count,
    )

    pipeline_times['quality_gate'] = time.time() - t_start

    pipeline_status = "VERIFIED — PASSED QUALITY GATE" if quality['passed'] else "FAILED QUALITY GATE"
    quality_grade = f"Grade {quality['grade']}"
    auto_recovery = quality.get('details', {}).get('auto_recovery', None)

    results = {
        'pipeline_status': pipeline_status,
        'quality_grade': quality_grade,
        'auto_recovery': auto_recovery,
        'total_matches': n_total,
        'inlier_count': n_inliers,
        'inlier_ratio': ratio,
        'rmse_pixels': current_rmse,
        'weighted_rmse': w_rmse,
        'rmse_raw_pixels': rmse_raw,
        'weighted_rmse_raw': w_rmse_raw,
        'gt_rmse_pixels': gt_rmse_pixels,
        'gt_point_count': gt_point_count,
        'gt_validation_status': gt_validation_status,
        'metadata_validation': metadata_validation,
        'mission_grade_certified': bool(
            quality['passed']
            and metadata_validation.get('status') == 'valid'
            and gt_validation_status == 'valid'
            and gt_point_count >= 15
        ),
        'coverage': coverage,
        'density_grid': density_grid,
        'pts_src': pts_src,
        'pts_ref': pts_ref,
        'inliers_src': inliers_src,
        'inliers_ref': inliers_ref,
        'confidences': inlier_conf,
        'transform': transform,
        'H_fallback': H_fallback,
        'mask': inlier_mask,
        'method_used': method_used,
        'scale_src': scale_src,
        'scale_ref': scale_ref,
        'scale_ref_total': scale_ref_total,
        'img_src_warped': img_src_warped,
        'pipeline_times': pipeline_times,
        'error_stats': err_stats,
        'quality': quality,
        'holdout_validation': holdout_val,
        'symmetric_rmse': symmetric_rmse,
        'error_budget': error_budget,
        'spatial_entropy': sp_entropy,
        'condition_number': condition_number,
        'geometric_model_type': config.get('geometric_model_type', 'HOMOGRAPHY'),
        'fallback_code': config.get('geometric_fallback_code', None),
        'fusion_stats': fusion_stats,
        'lunar_fusion_stats': fusion_stats,
        'omniroute_stats': omniroute_stats,
        'recovery_attempts': recovery_attempts,
        'illumination_disparity': sun_telem,
        'norm_src': norm_src,
        'norm_ref': norm_ref,
        'provenance': provenance,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Method: {method_used}")
    logger.info(f"Matches: {n_inliers}/{n_total}  (inlier ratio: {ratio:.2%})")
    rmse_str = f"{current_rmse:.3f} px (Raw sensor: {rmse_raw:.3f} px, Weighted: {w_rmse:.3f} px)" if current_rmse is not None else "REJECTED (Overfit 4-point trap / <15 inliers)"
    logger.info(f"RMSE: {rmse_str}")
    logger.info(f"Coverage: {coverage:.2%}")
    logger.info(f"Quality Grade: {quality['grade']} (confidence: {quality['confidence']:.1%}) [{pipeline_status}]")
    if auto_recovery:
        logger.warning(f"  ⚡ {auto_recovery}")
    if quality['warnings']:
        for warn in quality['warnings']:
            logger.warning(f"  ⚠ {warn}")
    logger.info(f"{'='*60}\n")

    # --- GATE 5: AUTONOMOUS RECOVERY (SELF-HEALING LOOP) ---
    if auto_recover and not quality['passed'] and max_dim < 4096:
        recovery_plan = build_recovery_plan(matcher_type, max_dim, quality['primary_reason'])
        logger.warning(
            f"[GATE 5: AUTONOMOUS RECOVERY] Quality Gate failed ({quality['primary_reason']}). "
            f"Triggering Self-Healing Loop: Upscaling max_dim from {max_dim} to {recovery_plan['max_dim']} "
            f"and retrying with {recovery_plan['matcher_type']} for robust lunar matching."
        )
        if progress_callback:
            progress_callback(total_steps, total_steps, "⚡ Gate 5 Self-Healing: Retrying with higher-resolution deterministic fallback...")

        try:
            recovery_kwargs = dict(kwargs)
            recovery_kwargs["recovery_attempts"] = recovery_attempts
            img_src_rec, img_ref_rec, results_rec = run_registration_pipeline(
                src_path=src_path,
                ref_path=ref_path,
                use_tps=use_tps,
                max_dim=recovery_plan["max_dim"],
                matcher_type=recovery_plan["matcher_type"],
                progress_callback=progress_callback,
                geom_info=geom_info,
                gsd_src=gsd_src,
                gsd_ref=gsd_ref,
                gt_pts_src=gt_pts_src,
                gt_pts_ref=gt_pts_ref,
                gt_points_in_original_coords=gt_points_in_original_coords,
                use_tile_matching=recovery_plan["use_tile_matching"],
                tile_grid=recovery_plan["tile_grid"],
                auto_recover=False,
                **recovery_kwargs
            )
            if results_rec.get('quality', {}).get('passed', False):
                logger.info(
                    "[GATE 5: AUTONOMOUS RECOVERY SUCCESS] Self-healing loop succeeded! "
                    f"Recovered from {quality['primary_reason']} via {recovery_plan['max_dim']}px upscale."
                )
                results_rec['self_healed'] = True
                results_rec['initial_failure_reason'] = quality['primary_reason']
                for k in ('omniroute_stats', 'lunar_fusion_stats'):
                    if k in results and k not in results_rec:
                        results_rec[k] = results[k]
                return img_src_rec, img_ref_rec, _maybe_export(results_rec, img_src_rec)
            else:
                logger.warning(
                    "[GATE 5: AUTONOMOUS RECOVERY] Retry completed but did not reach a valid registration. "
                    "Returning failure telemetry for original requested engine."
                )
                results['self_healed'] = False
                recovery_attempts.append({
                    "matcher": recovery_plan["matcher_type"],
                    "max_dim": recovery_plan["max_dim"],
                    "status": "failed",
                })
                results['recovery_attempts'] = recovery_attempts
                return img_src, img_ref, _maybe_export(results, img_src_warped)
        except Exception as e_rec:
            logger.warning(f"[GATE 5: AUTONOMOUS RECOVERY] Self-healing retry encountered: {e_rec}")
            if recovery_attempts:
                recovery_attempts[-1]['status'] = 'failed'

    if quality['passed']:
        recovery_attempts[-1]['status'] = 'passed'
    elif recovery_attempts[-1]['status'] == 'started':
        recovery_attempts[-1]['status'] = 'failed'
    results['recovery_attempts'] = recovery_attempts

    # --- STAGE 8: Direct 3D Lunar Topography Generation ---
    if generate_3d_dem and quality.get('passed', True):
        _progress(total_steps, total_steps, "Synthesizing calibrated 3D lunar digital elevation model (DEM)...")
        try:
            from .features.dem_builder import generate_3d_dem_from_pipeline
            dem_report = generate_3d_dem_from_pipeline(
                img_src=img_src,
                img_ref=img_ref,
                registration_results=results,
                camera_sensor=stereo_sensor,
                output_ply_path=output_ply_path,
                config=config,
            )
            results['dem_3d'] = dem_report
            logger.info(
                f"[3D Topography Complete] Relief: {dem_report.get('relief_span_m', 0.0):.1f}m | "
                f"Elev: {dem_report.get('elevation_min_m', 0.0):.0f} to {dem_report.get('elevation_max_m', 0.0):.0f} m | "
                f"QC Grade: {dem_report.get('qc', {}).get('qc_grade', 'N/A')}"
            )
        except Exception as e_dem:
            logger.warning(f"3D DEM generation from pipeline failed: {e_dem}")
            results['dem_3d'] = {'status': 'error', 'error': str(e_dem)}

    prov_chain.record(
        stage="quality_gate",
        details={
            "grade": quality.get("grade", "F"),
            "passed": quality.get("passed", False),
            "confidence": quality.get("confidence", 0.0),
            "rmse_pixels": current_rmse,
            "inlier_count": n_inliers,
            "coverage": coverage,
        },
        payload=None,
    )
    prov_chain.seal()
    results['provenance_chain'] = prov_chain.to_dict()

    return img_src, img_ref, _maybe_export(results, img_src_warped)


def main():
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Lunar Image Registration Pipeline")
    parser.add_argument("src", help="Path to source image")
    parser.add_argument("ref", help="Path to reference image")
    parser.add_argument("--no-tps", action="store_true", help="Disable TPS jitter correction")
    parser.add_argument("--matcher", type=str, default="fusion", choices=["fusion", "lunar_fusion", "ensemble", "loftr", "lightglue", "sift", "roma", "roma_v2", "roma_omega", "tiny_roma", "ni_lfm"], help="Matcher model")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension for matcher")
    parser.add_argument("--no-tile", action="store_true", help="Disable tile-based matching")
    parser.add_argument("--dem", action="store_true", help="Generate calibrated 3D lunar DEM and PLY mesh from registration outputs")
    parser.add_argument("--stereo-sensor", type=str, default="tmc", choices=["tmc", "ohrc"], help="Orbital camera sensor model for 3D triangulation")
    parser.add_argument("--ply", type=str, default=None, help="Output path for 3D Stanford PLY mesh")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to export canonical §9 artifacts")
    parser.add_argument("--rejected-preview", action="store_true", help="Write UNVALIDATED preview raster if registration fails")
    args = parser.parse_args()
    run_registration_pipeline(
        args.src, args.ref,
        use_tps=not args.no_tps,
        max_dim=args.max_dim,
        matcher_type=args.matcher,
        use_tile_matching=not args.no_tile,
        generate_3d_dem=args.dem,
        stereo_sensor=args.stereo_sensor,
        output_ply_path=args.ply,
        output_dir=args.output_dir,
        rejected_preview=args.rejected_preview,
    )


if __name__ == "__main__":
    main()
