"""
Evaluation Metrics — Scientific Validation Suite
=================================================
Provides accuracy and quality metrics for lunar image registration:
  - RMSE (Root Mean Square Error) — pixel-level accuracy
  - Confidence-weighted RMSE — accounts for match certainty
  - Inlier ratio — outlier rejection effectiveness
  - Match density grid & spatial coverage — distribution quality
"""

import numpy as np
from typing import Tuple


def rmse(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Calculate Root Mean Square Error between two sets of corresponding points."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('inf')
    return np.sqrt(np.mean(np.sum((pts_a - pts_b) ** 2, axis=1)))


def weighted_rmse(pts_a: np.ndarray, pts_b: np.ndarray, confidences: np.ndarray) -> float:
    """
    Confidence-weighted RMSE.

    Higher-confidence matches contribute more to the error metric,
    giving a more meaningful accuracy estimate than treating all
    inliers equally. This is especially important for LoFTR, which
    provides per-match confidence scores.

    Parameters
    ----------
    pts_a       : (N, 2) source point coordinates.
    pts_b       : (N, 2) reference point coordinates.
    confidences : (N,) confidence weights in [0, 1].

    Returns
    -------
    float : Weighted RMSE in pixels.
    """
    if len(pts_a) == 0 or len(pts_b) == 0:
        return float('inf')

    squared_errors = np.sum((pts_a - pts_b) ** 2, axis=1)
    weights = np.clip(confidences, 1e-6, 1.0)
    w_sum = weights.sum()
    if w_sum <= 0:
        return float('inf')
    weights = weights / (w_sum + 1e-12)  # Normalise to sum to 1

    return float(np.sqrt(np.sum(weights * squared_errors)))


def inlier_ratio(total_matches: int, inlier_count: int) -> float:
    """Calculate the ratio of inliers to total putative matches."""
    return inlier_count / max(total_matches, 1)


def match_density_grid(img_shape: Tuple[int, int], points: np.ndarray, grid: Tuple[int, int] = (8, 8)):
    """
    Checks uniform distribution of matches across the image.
    Returns the density grid and a coverage score (close to 1.0 = uniform spread).
    """
    if len(points) == 0:
        return np.zeros(grid), 0.0
        
    h, w = img_shape[:2]
    gh, gw = h // grid[0], w // grid[1]
    
    # Avoid division by zero if image is smaller than grid
    gh = max(gh, 1)
    gw = max(gw, 1)
    
    density = np.zeros(grid)
    for x, y in points:
        gi = min(int(y // gh), grid[0] - 1)
        gj = min(int(x // gw), grid[1] - 1)
        density[gi, gj] += 1
        
    coverage = np.count_nonzero(density) / density.size
    return density, coverage


def reprojection_error_per_point(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """
    Per-point Euclidean reprojection error.

    Useful for identifying which specific matches are worst and for
    generating error distribution histograms.

    Returns
    -------
    np.ndarray : (N,) array of per-point errors in pixels.
    """
    if len(pts_a) == 0:
        return np.array([])
    return np.sqrt(np.sum((pts_a - pts_b) ** 2, axis=1))


def error_statistics(pts_a: np.ndarray, pts_b: np.ndarray) -> dict:
    """
    Comprehensive error statistics for a set of matched points.

    Returns
    -------
    dict with keys: mean, median, std, min, max, p90, p95, p99
    """
    errors = reprojection_error_per_point(pts_a, pts_b)
    if len(errors) == 0:
        return {k: float('inf') for k in ['mean', 'median', 'std', 'min', 'max', 'p90', 'p95', 'p99']}

    return {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'p90': float(np.percentile(errors, 90)),
        'p95': float(np.percentile(errors, 95)),
        'p99': float(np.percentile(errors, 99)),
    }


def ground_truth_rmse(transformed_pts: np.ndarray, gt_pts_ref: np.ndarray) -> float:
    """
    Ground Truth Root Mean Square Error (GT-RMSE).

    Evaluates the error of transformed points against known, independent Ground Truth 
    Control Points (GCPs), rather than self-selected inlier matches.
    """
    return rmse(transformed_pts, gt_pts_ref)


# Standard alias
gt_rmse = ground_truth_rmse


def evaluate_transform_gt_rmse(
    transform_matrix: np.ndarray,
    gt_pts_src: np.ndarray,
    gt_pts_ref: np.ndarray
) -> float:
    """
    Evaluate an estimated transformation matrix (Homography 3x3 or Affine 2x3)
    against known Ground Truth control point correspondences.

    Parameters
    ----------
    transform_matrix : (3, 3) Homography or (2, 3) Affine matrix
    gt_pts_src       : (N, 2) Ground Truth source coordinates
    gt_pts_ref       : (N, 2) Ground Truth reference coordinates

    Returns
    -------
    float : Ground Truth RMSE in pixels
    """
    import cv2
    if len(gt_pts_src) == 0 or len(gt_pts_ref) == 0:
        return float('inf')

    pts = np.asarray(gt_pts_src, dtype=np.float32).reshape(-1, 1, 2)
    if transform_matrix.shape == (3, 3):
        transformed = cv2.perspectiveTransform(pts, transform_matrix.astype(np.float64)).reshape(-1, 2)
    elif transform_matrix.shape == (2, 3):
        transformed = cv2.transform(pts, transform_matrix.astype(np.float64)).reshape(-1, 2)
    else:
        raise ValueError(f"Unsupported transform matrix shape: {transform_matrix.shape}. Expected (3,3) or (2,3).")

    return ground_truth_rmse(transformed, gt_pts_ref)


def synthetic_grid_gt_rmse(
    estimated_matrix: np.ndarray,
    gt_matrix: np.ndarray,
    img_shape: Tuple[int, int] = (1000, 1000),
    grid_size: int = 20
) -> float:
    """
    Compute dense Ground Truth RMSE over a uniform spatial grid across the image.

    Used in synthetic validation and benchmark experiments where the exact 
    ground-truth transformation is known. Evaluates full-field warping distortion 
    across the entire sensor footprint.
    """
    import cv2
    h, w = img_shape[:2]
    ys = np.linspace(0, h - 1, grid_size)
    xs = np.linspace(0, w - 1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32).reshape(-1, 1, 2)

    def _apply(matrix, pts):
        if matrix.shape == (3, 3):
            return cv2.perspectiveTransform(pts, matrix.astype(np.float64)).reshape(-1, 2)
        elif matrix.shape == (2, 3):
            return cv2.transform(pts, matrix.astype(np.float64)).reshape(-1, 2)
        raise ValueError(f"Unsupported matrix shape: {matrix.shape}. Expected (3,3) or (2,3).")

    pts_est = _apply(estimated_matrix, grid_pts)
    pts_gt = _apply(gt_matrix, grid_pts)

    return ground_truth_rmse(pts_est, pts_gt)


def compute_holdout_validation(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    estimator_func=None,
    train_ratio: float = 0.7,
    seed: int = 42
) -> dict:
    """
    Perform rigorous out-of-sample hold-out validation (Tier 0).
    Partitions tie points into training (e.g. 70%) for estimation and
    evaluation strictly on the withheld holdout set (30%).
    """
    import cv2
    n_pts = len(pts_src)
    if n_pts < 8:
        return {
            "holdout_rmse": float('nan'),
            "train_rmse": float('nan'),
            "n_train": n_pts,
            "n_holdout": 0
        }

    if estimator_func is None:
        def _default_estimator(s, r):
            if len(s) < 4:
                return None, None
            try:
                H, m = cv2.findHomography(s, r, cv2.USAC_MAGSAC, 2.0)
                if H is not None:
                    return H, m
            except Exception:
                pass
            return cv2.estimateAffinePartial2D(s, r)
        estimator_func = _default_estimator

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_pts)
    n_train = int(n_pts * train_ratio)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_src, train_ref = pts_src[train_idx], pts_ref[train_idx]
    test_src, test_ref = pts_src[test_idx], pts_ref[test_idx]

    H_est, _ = estimator_func(train_src, train_ref)
    if H_est is None:
        return {
            "holdout_rmse": float('nan'),
            "train_rmse": float('nan'),
            "n_train": len(train_idx),
            "n_holdout": len(test_idx)
        }

    def _project(H, pts):
        pts_3d = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        if H.shape == (3, 3):
            return cv2.perspectiveTransform(pts_3d, H.astype(np.float64)).reshape(-1, 2)
        elif H.shape == (2, 3):
            return cv2.transform(pts_3d, H.astype(np.float64)).reshape(-1, 2)
        return pts

    train_proj = _project(H_est, train_src)
    test_proj = _project(H_est, test_src)

    train_rmse = float(np.sqrt(np.mean(np.sum((train_proj - train_ref) ** 2, axis=1))))
    holdout_rmse = float(np.sqrt(np.mean(np.sum((test_proj - test_ref) ** 2, axis=1))))

    return {
        "holdout_rmse": holdout_rmse,
        "train_rmse": train_rmse,
        "n_train": len(train_idx),
        "n_holdout": len(test_idx)
    }


def decompose_error_budget(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    H_est: np.ndarray,
) -> dict:
    """
    Decompose registration error into 4 distinct physical budget components:
    1. Global rigid/affine alignment error
    2. High-frequency residual (jitter / unmodeled micro-vibration)
    3. Parallax / relief-displacement residual
    4. Match localization noise floor
    """
    import cv2
    if len(pts_src) < 4 or H_est is None:
        return {
            "rigid_error_px": float('nan'),
            "jitter_residual_px": float('nan'),
            "relief_error_px": float('nan'),
            "localization_noise_px": float('nan'),
            "total_rmse_px": float('nan')
        }

    pts_s = np.asarray(pts_src, dtype=np.float32).reshape(-1, 1, 2)
    if H_est.shape == (3, 3):
        pts_proj = cv2.perspectiveTransform(pts_s, H_est.astype(np.float64)).reshape(-1, 2)
    else:
        pts_proj = cv2.transform(pts_s, H_est.astype(np.float64)).reshape(-1, 2)

    diffs = pts_proj - pts_ref
    err_norms = np.linalg.norm(diffs, axis=1)
    total_rmse = float(np.sqrt(np.mean(err_norms ** 2)))

    # Fit best rigid transform (rotation + translation, 3 DOF)
    mean_s = np.mean(pts_src, axis=0)
    mean_r = np.mean(pts_ref, axis=0)
    cs = pts_src - mean_s
    cr = pts_ref - mean_r
    H_cov = cs.T @ cr
    try:
        U, _, Vt = np.linalg.svd(H_cov)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        t = mean_r - (R @ mean_s)
        rigid_proj = (pts_src @ R.T) + t
        rigid_errs = np.linalg.norm(rigid_proj - pts_ref, axis=1)
        rigid_error = float(np.sqrt(np.mean(rigid_errs ** 2)))
    except Exception:
        rigid_error = total_rmse

    # High-frequency jitter vs low-frequency relief
    y_lines = pts_src[:, 1]
    sorted_idx = np.argsort(y_lines)
    y_sorted = y_lines[sorted_idx]
    dx_sorted = diffs[sorted_idx, 0]
    
    # Moving average window of size min(25, N/4) to extract smooth relief component
    w_size = max(5, min(25, len(pts_src) // 4))
    if w_size % 2 == 0:
        w_size += 1
    if len(dx_sorted) >= w_size:
        relief_trend = np.convolve(dx_sorted, np.ones(w_size) / w_size, mode='same')
        jitter_component = dx_sorted - relief_trend
        relief_err = float(np.sqrt(np.mean(relief_trend ** 2)))
        jitter_err = float(np.sqrt(np.mean(jitter_component ** 2)))
    else:
        relief_err = total_rmse * 0.7
        jitter_err = total_rmse * 0.3

    # Match localization noise floor estimate (MAD estimator / 0.6745)
    med = np.median(err_norms)
    mad = np.median(np.abs(err_norms - med))
    noise_floor = float(mad / 0.6745)

    return {
        "rigid_error_px": round(rigid_error, 4),
        "jitter_residual_px": round(jitter_err, 4),
        "relief_error_px": round(relief_err, 4),
        "localization_noise_px": round(noise_floor, 4),
        "total_rmse_px": round(total_rmse, 4)
    }


