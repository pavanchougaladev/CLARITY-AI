"""
test_omniroute.py
==================
Unit and integration tests for OmniRoute AI Gateway & Multi-Engine Router.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.features.omniroute import LunarOmniRoute, OmniRouteResult


def test_omniroute_initialization():
    """Verify OmniRoute initializes with all supported routing strategies."""
    for strat in ["consensus", "auto_fallback", "adaptive_profile", "fastest_wins"]:
        router = LunarOmniRoute(strategy=strat)
        assert router.strategy == strat
        assert router.confidence_threshold == 0.20


def test_omniroute_synthetic_sift_matching():
    """Verify OmniRoute runs classical SIFT matching on textured synthetic images."""
    router = LunarOmniRoute(strategy="consensus")

    # Generate synthetic textured test images
    np.random.seed(42)
    img_src = np.random.randint(40, 220, (128, 128), dtype=np.uint8)
    # Apply small translation
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    import cv2
    img_ref = cv2.warpAffine(img_src, M, (128, 128))

    res = router.route_and_match(img_src, img_ref, preferred_models=["sift"])
    assert isinstance(res, OmniRouteResult)
    assert res.strategy == "consensus"
    assert "sift" in res.models_attempted
    assert res.total_latency_ms > 0


def test_omniroute_auto_fallback_mechanism():
    """Verify that if primary models fail or return zero matches, auto_fallback falls back."""
    router = LunarOmniRoute(strategy="auto_fallback")
    img_dummy = np.zeros((64, 64), dtype=np.uint8)

    # Mock the execute single model to fail for first model and succeed for fallback
    with patch.object(router, '_execute_single_model') as mock_exec:
        def side_effect(m_name, *args, **kwargs):
            if m_name == "roma":
                return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), 12.0
            else:
                pts = np.array([[10., 10.], [20., 20.], [30., 30.], [40., 40.]] * 10, dtype=np.float32)
                return pts, pts, np.ones(len(pts), dtype=np.float32), 8.0

        mock_exec.side_effect = side_effect
        res = router.route_and_match(img_dummy, img_dummy, strategy="auto_fallback", preferred_models=["roma", "sift"])
        assert "sift" in res.models_succeeded
        assert len(res.pts_src) > 0


def test_omniroute_fastest_wins_race():
    """Verify fastest_wins race returns as soon as the first valid candidate arrives."""
    router = LunarOmniRoute(strategy="fastest_wins")
    img_dummy = np.zeros((64, 64), dtype=np.uint8)

    with patch.object(router, '_execute_single_model') as mock_exec:
        pts = np.array([[10., 10.], [20., 20.]] * 10, dtype=np.float32)
        mock_exec.return_value = (pts, pts, np.ones(len(pts), dtype=np.float32), 5.0)

        res = router.route_and_match(img_dummy, img_dummy, strategy="fastest_wins", preferred_models=["sift"])
        assert res.strategy == "fastest_wins"
        assert len(res.pts_src) > 0


def test_omniroute_pipeline_integration(tmp_path):
    """Verify run_registration_pipeline executes with matcher_type='omniroute'."""
    import cv2
    from src.pipeline import run_registration_pipeline

    # Create synthetic crater images with sufficient tie points for MVR
    np.random.seed(123)
    img_base = np.full((200, 200), 120, dtype=np.uint8)
    for cx, cy, r in [(50, 50, 20), (110, 100, 25), (70, 120, 15), (150, 40, 18), (140, 150, 22), (40, 160, 16)]:
        cv2.circle(img_base, (cx, cy), r, 60, -1)
        cv2.circle(img_base, (cx, cy), r, 200, 2)
    noise = np.random.randint(-10, 10, (200, 200), dtype=np.int16)
    img_src = np.clip(img_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Warp slightly for reference
    M = cv2.getRotationMatrix2D((100, 100), 2.0, 1.0)
    M[:, 2] += [4.0, 3.0]
    img_ref = cv2.warpAffine(img_src, M, (200, 200))

    src_path = str(tmp_path / "src.png")
    ref_path = str(tmp_path / "ref.png")
    cv2.imwrite(src_path, img_src)
    cv2.imwrite(ref_path, img_ref)

    src_out, ref_out, results = run_registration_pipeline(
        src_path, ref_path,
        use_tps=False,
        max_dim=256,
        matcher_type="omniroute",
        omni_strategy="consensus",
        preferred_models=["sift"]
    )

    assert results is not None
    assert "omniroute_stats" in results
    assert results["omniroute_stats"]["strategy"] == "consensus"
    assert "inlier_count" in results
    assert results["inlier_count"] > 0
    if results.get("rmse_pixels") is not None:
        assert results["rmse_pixels"] < 10.0

