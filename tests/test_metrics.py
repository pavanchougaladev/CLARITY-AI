import numpy as np
import pytest
import cv2
from src.evaluate.metrics import (
    rmse,
    weighted_rmse,
    inlier_ratio,
    match_density_grid,
    error_statistics,
    ground_truth_rmse,
    gt_rmse,
    evaluate_transform_gt_rmse,
    synthetic_grid_gt_rmse,
)

def test_rmse(synthetic_points):
    src, ref, _ = synthetic_points
    # We shifted by (5, 5), so error is sqrt(5^2 + 5^2) = sqrt(50) = 7.071
    val = rmse(src, ref)
    assert np.isclose(val, np.sqrt(50))

def test_weighted_rmse(synthetic_points):
    src, ref, conf = synthetic_points
    # Since all errors are exactly sqrt(50), weighted rmse should also be sqrt(50)
    val = weighted_rmse(src, ref, conf)
    assert np.isclose(val, np.sqrt(50))

def test_inlier_ratio():
    assert inlier_ratio(100, 50) == 0.5
    assert inlier_ratio(0, 0) == 0.0

def test_match_density_grid():
    points = np.array([[10, 10], [90, 90]])
    grid, coverage = match_density_grid((100, 100), points, grid=(2, 2))
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1
    assert grid[1, 1] == 1
    assert coverage == 0.5 # 2 out of 4 cells filled

def test_error_statistics(synthetic_points):
    src, ref, _ = synthetic_points
    stats = error_statistics(src, ref)
    assert np.isclose(stats['mean'], np.sqrt(50))
    assert np.isclose(stats['min'], np.sqrt(50))
    assert np.isclose(stats['max'], np.sqrt(50))


def test_ground_truth_rmse():
    """Verify GT-RMSE on independent ground truth control points."""
    gt_src = np.array([[10.0, 20.0], [50.0, 60.0], [100.0, 150.0]], dtype=np.float32)
    # Perfectly aligned points
    assert np.isclose(ground_truth_rmse(gt_src, gt_src), 0.0)
    assert np.isclose(gt_rmse(gt_src, gt_src), 0.0)

    # Known offset: dx = 3, dy = 4 -> Euclidean distance = 5.0
    gt_ref = gt_src + np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(ground_truth_rmse(gt_src, gt_ref), 5.0)


def test_evaluate_transform_gt_rmse():
    """Verify GT-RMSE evaluation on an estimated homography / affine matrix."""
    gt_src = np.array([[10.0, 20.0], [50.0, 60.0], [100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
    
    # 2D Affine shift by (+10, -5)
    M_gt = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]], dtype=np.float32)
    gt_ref = cv2.transform(gt_src.reshape(-1, 1, 2), M_gt).reshape(-1, 2)

    # Exact transform should give GT-RMSE == 0
    err_exact = evaluate_transform_gt_rmse(M_gt, gt_src, gt_ref)
    assert np.isclose(err_exact, 0.0, atol=1e-5)

    # In Homography form (3x3)
    H_gt = np.vstack([M_gt, [0.0, 0.0, 1.0]])
    err_homography = evaluate_transform_gt_rmse(H_gt, gt_src, gt_ref)
    assert np.isclose(err_homography, 0.0, atol=1e-5)

    # Dense synthetic grid evaluation
    grid_err = synthetic_grid_gt_rmse(H_gt, H_gt, img_shape=(500, 500))
    assert np.isclose(grid_err, 0.0, atol=1e-5)


def test_ransac_vs_least_squares_gt_rmse():
    """
    Demonstrate how RANSAC drastically improves GT-RMSE compared to Non-RANSAC
    when outlier matches are present (such as false crater pairings).
    """
    np.random.seed(42)
    # Generate 100 true ground-truth points across a lunar tile
    pts_src = np.random.uniform(50, 450, size=(100, 2)).astype(np.float32)
    
    # True transformation: 15 degree rotation + translation
    theta = np.deg2rad(15)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    H_gt = np.array([
        [cos_t, -sin_t, 25.0],
        [sin_t,  cos_t, -15.0],
        [0.0,    0.0,    1.0]
    ], dtype=np.float64)

    # Ground truth reference points
    pts_ref_gt = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)

    # Putative matches: 70% genuine inliers (with slight 0.2px sub-pixel noise)
    matches_src = pts_src.copy()
    matches_ref = pts_ref_gt + np.random.normal(0, 0.2, size=pts_ref_gt.shape).astype(np.float32)

    # 30% outliers: completely corrupted match coordinates
    outlier_idx = np.random.choice(100, size=30, replace=False)
    matches_ref[outlier_idx] += np.random.uniform(100, 300, size=(30, 2))

    # 1. Non-RANSAC estimation (Ordinary Least Squares / method=0)
    H_no_ransac, _ = cv2.findHomography(matches_src, matches_ref, method=0)

    # 2. RANSAC estimation (cv2.RANSAC)
    H_ransac, mask = cv2.findHomography(matches_src, matches_ref, cv2.RANSAC, 3.0)

    # Independent GT validation: evaluate on 50 fresh uncorrupted test points
    test_src = np.random.uniform(50, 450, size=(50, 2)).astype(np.float32)
    test_ref_gt = cv2.perspectiveTransform(test_src.reshape(-1, 1, 2), H_gt).reshape(-1, 2)

    gt_rmse_no_ransac = evaluate_transform_gt_rmse(H_no_ransac, test_src, test_ref_gt)
    gt_rmse_ransac = evaluate_transform_gt_rmse(H_ransac, test_src, test_ref_gt)

    # RANSAC should achieve sub-pixel or near sub-pixel GT-RMSE, while Non-RANSAC is heavily corrupted (>50 px)
    assert gt_rmse_ransac < 1.0, f"RANSAC GT-RMSE should be < 1.0 px, got {gt_rmse_ransac:.3f} px"
    assert gt_rmse_no_ransac > 50.0, f"Non-RANSAC should suffer from outliers, got {gt_rmse_no_ransac:.3f} px"
    assert gt_rmse_ransac < gt_rmse_no_ransac


def test_tps_vs_homography_rmse():
    """Verify that TPS models non-linear pushbroom jitter with sub-pixel RMSE."""
    from src.matching.tps_transformer import create_thin_plate_spline_transformer

    x, y = np.meshgrid(np.linspace(50, 450, 6), np.linspace(50, 450, 6))
    pts_src = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32)
    jitter_x = 1.5 * np.sin(pts_src[:, 1] / 50.0)
    pts_ref = pts_src.copy()
    pts_ref[:, 0] += jitter_x

    # Planar homography cannot bend to sinusoidal line jitter
    H, _ = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    proj_H = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), H).reshape(-1, 2)
    rmse_H = rmse(proj_H, pts_ref)

    # TPS captures non-linear spacecraft motion
    tps = create_thin_plate_spline_transformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts_src))]
    tps.estimateTransformation(pts_src.reshape(1, -1, 2), pts_ref.reshape(1, -1, 2), matches)
    _, proj_tps = tps.applyTransformation(pts_src.reshape(1, -1, 2))
    rmse_tps = rmse(proj_tps.reshape(-1, 2), pts_ref)

    assert rmse_tps < 0.1
    assert rmse_H > 0.5
    assert rmse_tps < rmse_H

