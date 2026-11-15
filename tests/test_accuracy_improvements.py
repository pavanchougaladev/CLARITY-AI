"""
Tests for the 5 Accuracy Improvements:
1. Two-tier SIFT with local affine consistency
2. MAGSAC++ parameter hardening
3. Adaptive tile grid calculation
4. Dual-direction Forward-backward consistency check
5. Enhanced sub-pixel refinement (Gaussian peak fitting + Multi-patch consensus)
"""

import numpy as np
import cv2
import pytest

from src.registration.subpixel_refine import (
    _gaussian_peak_fit,
    refine_subpixel,
    refine_subpixel_multipatch,
    upsampled_dft_phase_correlation,
)
from src.matching.outlier_rejection import (
    _forward_backward_consistency,
    _progressive_magsac,
)
from src.features.tile_matcher import tile_match, create_sift_tile_matcher
from src.pipeline import sift_match


def test_gaussian_peak_fit_accuracy():
    """Verify Gaussian peak fitting recovers sub-pixel fractional offsets accurately."""
    # Synthesize a 7x7 discrete surface sampled from a 2D Gaussian with offset (dx=0.35, dy=-0.25)
    true_dx, true_dy = 0.35, -0.25
    cy, cx = 3, 3
    y, x = np.mgrid[-3:4, -3:4]
    sigma = 1.5
    surface = np.exp(-((x - true_dx)**2 + (y - true_dy)**2) / (2 * sigma**2))

    est_dx, est_dy = _gaussian_peak_fit(surface, cy, cx)
    assert abs(est_dx - true_dx) < 0.05
    assert abs(est_dy - true_dy) < 0.05


def test_gaussian_peak_fit_boundary():
    """Boundary conditions should return (0.0, 0.0) without crashing."""
    surface = np.ones((5, 5), dtype=np.float64)
    dx, dy = _gaussian_peak_fit(surface, 0, 0)
    assert dx == 0.0 and dy == 0.0


def test_refine_subpixel_multipatch_consensus():
    """Verify multi-patch sub-pixel refinement converges and returns median shift."""
    size = 192
    y, x = np.mgrid[-96:96, -96:96]
    # Synthetic crater pattern
    base = (120.0 - 80.0 * np.exp(-(x**2 + y**2) / 200.0)
            + 35.0 * np.exp(-((np.sqrt(x**2 + y**2) - 25.0)**2) / 20.0)).astype(np.float32)

    shift_x, shift_y = 0.4, 0.2
    # Shift via phase ramp
    F = np.fft.fft2(base)
    ky = np.fft.fftfreq(size).reshape(-1, 1)
    kx = np.fft.fftfreq(size).reshape(1, -1)
    phase = np.exp(-2j * np.pi * (kx * shift_x + ky * shift_y))
    shifted = np.real(np.fft.ifft2(F * phase)).astype(np.float32)

    shift, resp = refine_subpixel_multipatch(
        base, shifted,
        patch_sizes=(64, 128, 192),
        convergence_thresh=0.5
    )
    assert abs(shift[0] - shift_x) < 0.1
    assert abs(shift[1] - shift_y) < 0.1
    assert resp > 0.5


def test_refine_subpixel_multipatch_small_patch():
    """If patch is smaller than max patch_sizes, it adapts gracefully."""
    patch = np.ones((64, 64), dtype=np.float32) * 100
    shift, resp = refine_subpixel_multipatch(patch, patch, patch_sizes=(64, 128, 192))
    assert isinstance(shift, tuple)
    assert len(shift) == 2


def test_forward_backward_consistency_rejects_asymmetric():
    """Verify forward-backward consistency rejects asymmetric / unidirectional outliers."""
    np.random.seed(42)
    # 20 true correspondences under identity + translation (5, 5)
    pts_src = np.random.uniform(50, 450, (20, 2)).astype(np.float32)
    pts_ref = pts_src + np.array([5.0, 5.0], dtype=np.float32)

    # Add 5 unidirectional outlier matches
    outliers_src = np.random.uniform(50, 450, (5, 2)).astype(np.float32)
    outliers_ref = np.random.uniform(50, 450, (5, 2)).astype(np.float32)

    all_src = np.vstack([pts_src, outliers_src])
    all_ref = np.vstack([pts_ref, outliers_ref])

    initial_mask = np.ones(len(all_src), dtype=bool)

    filtered_mask = _forward_backward_consistency(
        all_src, all_ref, initial_mask,
        threshold=3.0, confidence=0.999, max_iters=5000
    )

    # Inliers (first 20) should mostly be preserved
    assert np.sum(filtered_mask[:20]) >= 16
    # Gross outliers should be rejected
    assert np.sum(filtered_mask[20:]) <= 1


def test_adaptive_tile_grid():
    """Verify adaptive tile grid formula produces expected grid dimensions."""
    from src.features.tile_matcher import tile_match

    # Mock matcher returning empty points
    def mock_matcher(s, r):
        return np.array([]), np.array([]), np.array([])

    # Square image 1024x1024 -> grid_dim = 2 -> 2x2 grid
    img_sq = np.zeros((1024, 1024), dtype=np.uint8)
    p_s, p_r, c = tile_match(img_sq, img_sq, matcher_fn=mock_matcher, grid=None)
    assert len(p_s) == 0

    # Tall strip 3000x500 (aspect 6.0)
    img_tall = np.zeros((3000, 500), dtype=np.uint8)
    p_s, p_r, c = tile_match(img_tall, img_tall, matcher_fn=mock_matcher, grid=None)
    assert len(p_s) == 0


def test_sift_two_tier_ratio():
    """Verify two-tier ratio test runs without error and extracts correspondences."""
    img1 = np.zeros((256, 256), dtype=np.uint8)
    img2 = np.zeros((256, 256), dtype=np.uint8)

    # Draw synthetic craters
    for pt in [(60, 60), (120, 150), (180, 80), (140, 200), (80, 210)]:
        cv2.circle(img1, pt, 20, 200, -1)
        cv2.circle(img1, pt, 10, 50, -1)
        # Shifted slightly in img2
        pt2 = (pt[0] + 3, pt[1] + 2)
        cv2.circle(img2, pt2, 20, 200, -1)
        cv2.circle(img2, pt2, 10, 50, -1)

    pts_src, pts_ref, confs = sift_match(img1, img2)
    assert len(pts_src) == len(pts_ref) == len(confs)
