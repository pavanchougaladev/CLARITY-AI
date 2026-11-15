"""
Unit Tests for Epipolar / Fundamental Matrix Pre-Filter (Feature 6)
"""

import pytest
import numpy as np
import cv2
from src.matching.epipolar_filter import compute_sampson_distance, epipolar_filter_matches
from src.matching.outlier_rejection import estimate_transform


def test_sampson_distance_zero_on_pure_inliers():
    # Setup camera matrices and points satisfying epipolar geometry
    # Pure horizontal stereo baseline: F = [e2]_x @ H
    # Epipolar line in image 2 is horizontal: l2 = [0, 1, -y]
    # For horizontal baseline, F is [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
    F = np.array([
        [0.0,  0.0, 0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0, 0.0]
    ], dtype=np.float64)

    # Point 1: (100, 200), Point 2: (150, 200) -> same y coordinate on horizontal epipolar line
    pts1 = np.array([[100.0, 200.0]], dtype=np.float32)
    pts2 = np.array([[150.0, 200.0]], dtype=np.float32)

    dist = compute_sampson_distance(F, pts1, pts2)
    assert dist[0] == pytest.approx(0.0, abs=1e-5)

    # Outlier with vertical parallax: Point 2: (150, 250) -> y error = 50px
    pts2_outlier = np.array([[150.0, 250.0]], dtype=np.float32)
    dist_outlier = compute_sampson_distance(F, pts1, pts2_outlier)
    assert dist_outlier[0] > 10.0


def test_epipolar_filter_prunes_gross_outliers():
    # Generate 50 points on a synthetic planar scene undergoing affine motion
    np.random.seed(42)
    pts_s = np.random.uniform(50, 800, (40, 2)).astype(np.float32)
    # Reference points with translation + slight rotation
    M = np.array([[1.0, 0.02, 30.0], [-0.02, 1.0, 15.0]], dtype=np.float32)
    pts_r = cv2.transform(pts_s.reshape(-1, 1, 2), M).reshape(-1, 2)

    # Add 10 gross outliers with random vertical shifts
    outliers_s = np.random.uniform(100, 700, (10, 2)).astype(np.float32)
    outliers_r = np.random.uniform(100, 700, (10, 2)).astype(np.float32)

    all_s = np.vstack([pts_s, outliers_s])
    all_r = np.vstack([pts_r, outliers_r])

    f_s, f_r, _, mask, stats = epipolar_filter_matches(all_s, all_r, threshold_px=3.0)

    assert stats["applied"] is True
    assert stats["pruned_outliers"] >= 8  # Caught the vast majority of gross outliers
    assert len(f_s) < len(all_s)


def test_estimate_transform_with_epipolar_config():
    # Verify estimate_transform works seamlessly with epipolar_filter: True
    np.random.seed(123)
    pts_s = np.random.uniform(100, 900, (30, 2)).astype(np.float32)
    H_gt = np.array([[1.01, 0.005, 20.0], [-0.005, 1.01, 15.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    pts_r = cv2.perspectiveTransform(pts_s.reshape(-1, 1, 2), H_gt).reshape(-1, 2)

    cfg = {"pipeline": {"epipolar_filter": True, "epipolar_threshold_px": 2.5}}
    M_res, inliers = estimate_transform(pts_s, pts_r, use_tps=False, config=cfg)

    assert M_res is not None
    assert int(np.sum(inliers)) >= 15
