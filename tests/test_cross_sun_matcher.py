"""
Tests for the cross-sun gradient-magnitude matching fallback.
"""

import cv2
import numpy as np

from conftest import generate_synthetic_lunar_terrain
from src.features.cross_sun_matcher import (
    gradient_magnitude_image,
    match_cross_sun_gradient,
    merge_correspondences,
)
from src.pipeline import sift_match


def test_gradient_magnitude_ignores_shadow_polarity():
    """Inverting intensity must not change the relief signature used for matching."""
    img = generate_synthetic_lunar_terrain(192, 192, seed=808)
    inverted = (255 - img).astype(np.uint8)

    grad_a = gradient_magnitude_image(img)
    grad_b = gradient_magnitude_image(inverted)

    assert grad_a.shape == img.shape
    assert grad_a.dtype == np.uint8
    assert np.array_equal(grad_a, grad_b)


def test_cross_sun_gradient_matching_recovers_known_shift():
    """Gradient RootSIFT must recover a known translation across shadow inversion."""
    img = generate_synthetic_lunar_terrain(256, 256, seed=809)
    inverted = (255 - img).astype(np.uint8)
    shift = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, -4.0]], dtype=np.float32)
    ref = cv2.warpAffine(
        inverted, shift, (256, 256),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
    )

    pts_src, pts_ref, conf, stats = match_cross_sun_gradient(
        img, ref, matcher_fn=sift_match, config={}
    )

    assert stats["engine"] == "cross_sun_gradient_rootsift"
    assert stats["raw_matches"] == len(pts_src)
    assert len(pts_src) >= 30, f"Expected a dense recovery, got {len(pts_src)} matches"
    assert len(pts_src) == len(pts_ref) == len(conf)

    H, mask = cv2.findHomography(pts_src, pts_ref, cv2.RANSAC, 3.0)
    assert H is not None
    assert int(mask.sum()) >= 25

    # Recovered mapping must be close to the applied translation.
    assert abs(float(H[0, 2]) - 6.0) < 1.0
    assert abs(float(H[1, 2]) + 4.0) < 1.0
    assert abs(float(H[0, 0]) - 1.0) < 0.02
    assert abs(float(H[1, 1]) - 1.0) < 0.02


def test_merge_correspondences_suppresses_duplicate_pairs():
    """Fusion must not double-count the same tie-point from several channels."""
    src_a = np.array([[10.0, 10.0], [50.0, 60.0]], dtype=np.float32)
    ref_a = np.array([[12.0, 11.0], [52.0, 61.0]], dtype=np.float32)
    src_b = np.array([[10.2, 10.1], [90.0, 90.0]], dtype=np.float32)
    ref_b = np.array([[12.1, 11.1], [92.0, 91.0]], dtype=np.float32)

    merged_src, merged_ref, merged_conf = merge_correspondences(
        [
            (src_a, ref_a, np.array([0.9, 0.8], dtype=np.float32)),
            (src_b, ref_b, np.array([0.7, 0.6], dtype=np.float32)),
        ],
        tolerance_px=2.0,
    )

    assert len(merged_src) == 3
    assert merged_src[0].tolist() == [10.0, 10.0]  # highest-confidence duplicate wins
    assert merged_conf[0] == 0.9
