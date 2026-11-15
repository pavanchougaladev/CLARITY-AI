"""
Triplet Cycle-Consistency Validation
=====================================
ISRO LunarFusion Co-Registration Suite — Geometric Topology Verification

Implements multi-view cycle closure constraint:
For 3 overlapping lunar orbital scenes A, B, and C with pairwise homographies:
  H_AB: A -> B
  H_BC: B -> C
  H_CA: C -> A

The composite loop transformation:
  H_cycle = H_CA @ H_BC @ H_AB
must satisfy H_cycle(x) ~ x for all valid spatial points x in scene A.

Surfaces closure residuals (mean, max, RMSE) in pixels across a uniform spatial grid
to detect drift, cumulative bias, or registration inconsistency across overlapping strips.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List


def check_triplet_closure(
    H_AB: np.ndarray,
    H_BC: np.ndarray,
    H_CA: Optional[np.ndarray] = None,
    H_AC: Optional[np.ndarray] = None,
    image_shape: Tuple[int, int] = (2048, 2048),
    tolerance_px: float = 5.0,
    grid_size: int = 16,
) -> Dict[str, Any]:
    """
    Validate geometric cycle-consistency across 3 overlapping lunar scenes.

    Parameters
    ----------
    H_AB : np.ndarray
        3x3 Homography mapping scene A -> scene B.
    H_BC : np.ndarray
        3x3 Homography mapping scene B -> scene C.
    H_CA : np.ndarray, optional
        3x3 Homography mapping scene C -> scene A. If None, H_AC must be provided.
    H_AC : np.ndarray, optional
        3x3 Homography mapping scene A -> scene C. If provided, H_CA is computed as inv(H_AC).
    image_shape : tuple of (int, int)
        (height, width) of source scene A to sample evaluation grid.
    tolerance_px : float
        Maximum allowable mean loop residual in pixels for certifying consistency.
    grid_size : int
        Number of grid divisions per axis for evaluation points (grid_size x grid_size).

    Returns
    -------
    dict
        Structured closure telemetry including mean, max, RMSE residuals, and pass/fail status.
    """
    H_AB = np.asarray(H_AB, dtype=np.float64)
    H_BC = np.asarray(H_BC, dtype=np.float64)

    if H_CA is None:
        if H_AC is None:
            raise ValueError("Either H_CA (C->A) or H_AC (A->C) must be provided.")
        H_AC = np.asarray(H_AC, dtype=np.float64)
        if np.abs(np.linalg.det(H_AC)) < 1e-12:
            raise ValueError("H_AC is singular and cannot be inverted for triplet closure.")
        H_CA = np.linalg.inv(H_AC)
    else:
        H_CA = np.asarray(H_CA, dtype=np.float64)

    # Normalize scale of homographies so det is not wildly diverging
    for H in (H_AB, H_BC, H_CA):
        if H.shape != (3, 3):
            raise ValueError(f"Expected 3x3 homography matrix, received shape {H.shape}")
        if np.abs(H[2, 2]) > 1e-9:
            H /= H[2, 2]

    # Composite loop transformation: A -> B -> C -> A
    H_cycle = H_CA @ H_BC @ H_AB

    h, w = image_shape
    # Generate uniform 2D grid over scene A (leaving 5% margin from edges)
    xs = np.linspace(w * 0.05, w * 0.95, grid_size)
    ys = np.linspace(h * 0.05, h * 0.95, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts_A = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (N, 2)

    # Project pts_A through H_cycle using homogeneous coordinates
    N = len(pts_A)
    ones = np.ones((N, 1), dtype=np.float64)
    pts_homo = np.hstack([pts_A, ones])  # (N, 3)

    pts_projected_homo = (H_cycle @ pts_homo.T).T  # (N, 3)

    # Check for points projecting behind the camera plane / negative scale
    w_coords = pts_projected_homo[:, 2]
    valid_mask = np.abs(w_coords) > 1e-7

    if not np.any(valid_mask):
        return {
            "status": "FAIL",
            "is_consistent": False,
            "mean_residual_px": float("inf"),
            "max_residual_px": float("inf"),
            "rmse_residual_px": float("inf"),
            "tolerance_px": tolerance_px,
            "closure_ratio": 0.0,
            "grid_points_tested": N,
            "valid_projections": 0,
            "H_cycle": H_cycle.tolist(),
            "reason": "Degenerate projection: all projective scale factors w are near zero.",
        }

    pts_projected = pts_projected_homo[valid_mask, :2] / w_coords[valid_mask, np.newaxis]
    pts_original = pts_A[valid_mask]

    # Euclidean residuals between starting coordinates and loop-closed coordinates
    residuals = np.linalg.norm(pts_original - pts_projected, axis=1)

    mean_res = float(np.mean(residuals))
    max_res = float(np.max(residuals))
    rmse_res = float(np.sqrt(np.mean(residuals ** 2)))
    closure_ratio = float(np.mean(residuals <= tolerance_px))
    is_consistent = bool(mean_res <= tolerance_px and closure_ratio >= 0.80)

    return {
        "status": "PASS" if is_consistent else "FAIL",
        "is_consistent": is_consistent,
        "mean_residual_px": round(mean_res, 4),
        "max_residual_px": round(max_res, 4),
        "rmse_residual_px": round(rmse_res, 4),
        "tolerance_px": float(tolerance_px),
        "closure_ratio": round(closure_ratio, 4),
        "grid_points_tested": N,
        "valid_projections": int(np.sum(valid_mask)),
        "H_cycle": [[round(float(val), 6) for val in row] for row in H_cycle],
    }
