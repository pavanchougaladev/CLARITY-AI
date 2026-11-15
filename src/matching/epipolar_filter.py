"""
Epipolar / Fundamental Matrix Pre-Filter
=========================================
ISRO LunarFusion Co-Registration Suite — Geometric Constraint Filter

Enforces the epipolar constraint x'^T F x = 0 across multi-temporal or stereo
lunar orbital frames (e.g., TMC-2 fore/aft along-track stereo or cross-orbit OHRC).

Outliers whose Sampson epipolar error exceeds the pixel threshold are pruned
prior to non-rigid TPS or projective homography estimation, eliminating repetitive
crater aliasing and shadow edge mismatches.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def compute_sampson_distance(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Compute first-order Sampson epipolar distance for point pairs.

    d_Sampson^2 = (x2^T F x1)^2 / ((F x1)_1^2 + (F x1)_2^2 + (F^T x2)_1^2 + (F^T x2)_2^2)

    Parameters
    ----------
    F : np.ndarray
        3x3 Fundamental Matrix.
    pts1 : np.ndarray
        (N, 2) points in image 1.
    pts2 : np.ndarray
        (N, 2) points in image 2.

    Returns
    -------
    np.ndarray
        (N,) array of Sampson distances in pixels.
    """
    N = len(pts1)
    if N == 0:
        return np.empty((0,), dtype=np.float32)

    p1_h = np.hstack([pts1, np.ones((N, 1), dtype=np.float64)])  # (N, 3)
    p2_h = np.hstack([pts2, np.ones((N, 1), dtype=np.float64)])  # (N, 3)

    # Epipolar lines in image 2: l2 = F @ p1
    Fx1 = (F @ p1_h.T).T  # (N, 3)
    # Epipolar lines in image 1: l1 = F^T @ p2
    Ftx2 = (F.T @ p2_h.T).T  # (N, 3)

    # Algebraic error: p2^T @ F @ p1
    algebraic_error = np.sum(p2_h * Fx1, axis=1)  # (N,)

    # Denominator: (Fx1)_0^2 + (Fx1)_1^2 + (Ftx2)_0^2 + (Ftx2)_1^2
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    denom = np.maximum(denom, 1e-12)

    sampson_dist = np.sqrt((algebraic_error ** 2) / denom)
    return sampson_dist.astype(np.float32)


def epipolar_filter_matches(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    threshold_px: float = 2.0,
    min_inliers: int = 8,
    ransac_confidence: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Filter candidate correspondences against the multi-view epipolar constraint.

    Parameters
    ----------
    pts_src : np.ndarray
        (N, 2) coordinates in source scene.
    pts_ref : np.ndarray
        (N, 2) coordinates in reference scene.
    confidences : np.ndarray, optional
        (N,) match confidence values.
    threshold_px : float
        Maximum allowable Sampson epipolar error in pixels (default 2.0 px).
    min_inliers : int
        Minimum inliers required to accept the estimated Fundamental matrix.
    ransac_confidence : float
        RANSAC confidence level for Fundamental matrix estimation.

    Returns
    -------
    (filtered_src, filtered_ref, filtered_confidences, inlier_mask, stats_dict)
    """
    N = len(pts_src)
    full_mask = np.ones(N, dtype=bool)
    stats: Dict[str, Any] = {
        "applied": False,
        "input_matches": N,
        "output_matches": N,
        "pruned_outliers": 0,
        "fundamental_matrix": None,
        "mean_sampson_px": None,
        "status": "SKIPPED",
    }

    if N < 8:
        stats["status"] = "INSUFFICIENT_POINTS"
        return pts_src, pts_ref, confidences, full_mask, stats

    pts_s = np.asarray(pts_src, dtype=np.float32).reshape(-1, 2)
    pts_r = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 2)

    try:
        # Estimate Fundamental Matrix with RANSAC
        F, inlier_mask = cv2.findFundamentalMat(
            pts_s,
            pts_r,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=threshold_px,
            confidence=ransac_confidence,
            maxIters=5000,
        )

        if F is None or F.shape != (3, 3):
            stats["status"] = "F_ESTIMATION_FAILED"
            return pts_src, pts_ref, confidences, full_mask, stats

        # If multiple F matrices returned, pick the first 3x3
        if F.shape[0] > 3:
            F = F[:3, :3]

        # Compute exact Sampson distance for all candidate correspondences
        sampson_errors = compute_sampson_distance(F, pts_s, pts_r)
        epipolar_inliers = sampson_errors <= threshold_px
        inlier_count = int(np.sum(epipolar_inliers))

        if inlier_count < min_inliers:
            logger.warning(
                f"[EpipolarFilter] Only {inlier_count} matches passed epipolar constraint "
                f"(min {min_inliers}). Skipping filter to avoid starvation."
            )
            stats["status"] = "BELOW_MIN_INLIERS"
            return pts_src, pts_ref, confidences, full_mask, stats

        filtered_src = pts_s[epipolar_inliers]
        filtered_ref = pts_r[epipolar_inliers]
        filtered_conf = confidences[epipolar_inliers] if confidences is not None else None

        stats.update({
            "applied": True,
            "status": "APPLIED",
            "output_matches": inlier_count,
            "pruned_outliers": N - inlier_count,
            "mean_sampson_px": float(np.mean(sampson_errors[epipolar_inliers])),
            "fundamental_matrix": F.tolist(),
        })

        logger.info(
            f"[EpipolarFilter] Retained {inlier_count}/{N} matches "
            f"({stats['pruned_outliers']} epipolar outliers pruned, mean Sampson: {stats['mean_sampson_px']:.2f}px)"
        )
        return filtered_src, filtered_ref, filtered_conf, epipolar_inliers, stats

    except Exception as exc:
        logger.warning(f"[EpipolarFilter] Exception during epipolar verification: {exc}")
        stats["status"] = f"ERROR: {exc}"
        return pts_src, pts_ref, confidences, full_mask, stats
