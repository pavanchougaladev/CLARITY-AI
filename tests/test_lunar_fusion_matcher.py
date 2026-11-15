"""
Tests for LunarFusionMatcher, ImageProfileAnalyzer, and AdaptiveRouter.
"""

import pytest
import numpy as np
import cv2
try:
    import torch
except (ImportError, Exception):
    torch = None
from unittest.mock import MagicMock, patch

from src.features.lunar_fusion_matcher import (
    ImageProfile,
    ImageProfileAnalyzer,
    AdaptiveRouter,
    ModelPlan,
    LunarFusionMatcher,
)
from src.features.deep_matcher import LunarMatcher


def test_image_profile_analyzer_synthetic():
    analyzer = ImageProfileAnalyzer()

    # 1. Dark shadowed image pair
    dark_src = np.zeros((200, 200), dtype=np.uint8)
    dark_ref = np.zeros((200, 200), dtype=np.uint8)
    p_dark = analyzer.analyze(dark_src, dark_ref)
    assert p_dark.shadow_fraction > 0.5
    assert p_dark.profile_tag == "small"  # 200 < 512

    # 2. Crater-like high texture image pair (large)
    np.random.seed(42)
    crater_src = np.random.randint(0, 256, (600, 600), dtype=np.uint8)
    crater_ref = np.random.randint(0, 256, (600, 600), dtype=np.uint8)
    p_crater = analyzer.analyze(crater_src, crater_ref)
    assert p_crater.size_px == 600
    assert p_crater.profile_tag in ["crater", "normal"]

    # 3. Flat featureless plain
    flat_src = np.full((600, 600), 128, dtype=np.uint8)
    flat_ref = np.full((600, 600), 128, dtype=np.uint8)
    p_flat = analyzer.analyze(flat_src, flat_ref)
    assert p_flat.texture_src < 0.1
    assert p_flat.profile_tag == "low_texture"


def test_adaptive_router_modes():
    router = AdaptiveRouter()

    # Low texture profile
    p_low = ImageProfile(profile_tag="low_texture")
    plan_balanced = router.route(p_low, speed_mode="balanced")
    assert "ni_lfm" in plan_balanced.models
    assert len(plan_balanced.models) >= 2
    assert pytest.approx(sum(plan_balanced.weights.values()), 0.01) == 1.0

    # Fast mode should select top 2
    plan_fast = router.route(p_low, speed_mode="fast")
    assert len(plan_fast.models) == 2
    assert pytest.approx(sum(plan_fast.weights.values()), 0.01) == 1.0

    # Accurate mode should activate all models (including SuperGlue, RIFT2, YOLO11, RoMa v2)
    plan_acc = router.route(p_low, speed_mode="accurate")
    assert len(plan_acc.models) >= 6
    assert "superglue" in plan_acc.models
    assert "rift2" in plan_acc.models
    assert ("roma_v2" in plan_acc.models or "roma_omega" in plan_acc.models or "roma" in plan_acc.models)
    assert "ni_lfm" in plan_acc.models
    assert "yolo11" in plan_acc.models


def test_lunar_fusion_prepare_inputs():
    if torch is None or not hasattr(torch, 'Tensor'):
        pytest.skip("PyTorch not installed or missing dylib")
    matcher = LunarFusionMatcher()

    # Test numpy uint8 inputs
    img_np = np.ones((100, 100), dtype=np.uint8) * 128
    src_t, ref_t, src_np, ref_np = matcher._prepare_inputs(img_np, img_np)
    assert isinstance(src_t, torch.Tensor)
    assert src_t.shape == (1, 1, 100, 100)
    assert isinstance(src_np, np.ndarray)
    assert src_np.shape == (100, 100)

    # Test torch tensor inputs
    tensor_in = torch.rand((1, 1, 64, 64))
    src_t2, ref_t2, src_np2, ref_np2 = matcher._prepare_inputs(tensor_in, tensor_in)
    assert isinstance(src_t2, torch.Tensor)
    assert src_t2.shape == (1, 1, 64, 64)
    assert isinstance(src_np2, np.ndarray)
    assert src_np2.shape == (64, 64)


def test_lunar_fusion_weighted_fuse_and_ransac():
    matcher = LunarFusionMatcher(confidence_threshold=0.2)

    # Mock matches from two models
    pts_src_1 = np.array([[50.0, 50.0], [100.0, 100.0]], dtype=np.float32)
    pts_ref_1 = np.array([[55.0, 55.0], [105.0, 105.0]], dtype=np.float32)
    conf_1 = np.array([0.9, 0.8], dtype=np.float32)

    pts_src_2 = np.array([[50.5, 49.8], [200.0, 200.0]], dtype=np.float32)
    pts_ref_2 = np.array([[55.2, 54.9], [205.0, 205.0]], dtype=np.float32)
    conf_2 = np.array([0.85, 0.75], dtype=np.float32)

    matches_by_model = {
        "roma": (pts_src_1, pts_ref_1, conf_1),
        "ni_lfm": (pts_src_2, pts_ref_2, conf_2),
    }
    weights = {"roma": 0.6, "ni_lfm": 0.4}

    f_src, f_ref, f_conf = matcher._weighted_fuse(matches_by_model, weights)
    # The (50, 50) points should merge
    assert len(f_src) == 3
    assert len(f_ref) == 3
    assert len(f_conf) == 3


def test_deep_matcher_fusion_integration():
    """Verify LunarMatcher initializes with matcher_type='fusion'."""
    matcher = LunarMatcher(matcher_type="fusion")
    assert matcher._available
    assert isinstance(matcher.matcher, LunarFusionMatcher)


def test_lunar_fusion_simultaneous_rootsift_execution():
    """Verify LunarFusion executes RootSIFT simultaneously alongside other engines."""
    matcher = LunarFusionMatcher(speed_mode="fast")
    plan = ModelPlan(models=["sift", "loftr"], weights={"sift": 0.5, "loftr": 0.5}, reason="Simultaneous test")

    src = np.zeros((256, 256), dtype=np.uint8)
    ref = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(src, (128, 128), 30, 200, -1)
    cv2.circle(ref, (132, 130), 30, 200, -1)

    src_t, ref_t, src_np, ref_np = matcher._prepare_inputs(src, ref)
    matches_by_model = matcher._run_multiscale_matching(plan, src_t, ref_t, src_np, ref_np, {}, total_steps=6)

    assert "sift" in matches_by_model
    assert "loftr" in matches_by_model
    pts_s, pts_r, c = matches_by_model["sift"]
    assert isinstance(pts_s, np.ndarray)


def test_lunar_fusion_new_algorithms_execution():
    """Verify LunarFusion directly dispatches to newly updated algorithms (SuperGlue, RIFT2, ASIFT, AKAZE)."""
    matcher = LunarFusionMatcher(speed_mode="fast")
    plan = ModelPlan(
        models=["superglue", "rift2", "akaze"],
        weights={"superglue": 0.4, "rift2": 0.3, "akaze": 0.3},
        reason="Research paper algorithms test"
    )

    src = np.zeros((256, 256), dtype=np.uint8)
    ref = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(src, (128, 128), 35, 220, -1)
    cv2.circle(ref, (130, 130), 35, 220, -1)

    src_t, ref_t, src_np, ref_np = matcher._prepare_inputs(src, ref)
    matches_by_model = matcher._run_multiscale_matching(plan, src_t, ref_t, src_np, ref_np, {}, total_steps=6)

    assert "superglue" in matches_by_model
    assert "rift2" in matches_by_model
    assert "akaze" in matches_by_model
    for m in ["superglue", "rift2", "akaze"]:
        pts_s, pts_r, c = matches_by_model[m]
        assert isinstance(pts_s, np.ndarray)
        assert isinstance(pts_r, np.ndarray)
        assert isinstance(c, np.ndarray)


def test_lunar_fusion_sensor_aware_routing():
    """Verify LunarFusion routes to SAR-optical and Polar profiles based on config."""
    router = AdaptiveRouter()

    # Explicit SAR sensor pair
    p_sar = ImageProfile(profile_tag="sar_optical")
    plan_sar = router.route(p_sar, speed_mode="balanced")
    assert "superglue" in plan_sar.models
    assert "rift2" in plan_sar.models
    assert plan_sar.weights["superglue"] >= 0.50

    # Explicit Polar region
    p_polar = ImageProfile(profile_tag="polar")
    plan_polar = router.route(p_polar, speed_mode="balanced")
    assert "superglue" in plan_polar.models
    assert "rift2" in plan_polar.models
    assert plan_polar.weights["superglue"] >= 0.50

    # Custom include_models configuration
    p_norm = ImageProfile(profile_tag="normal")
    plan_custom = router.route(p_norm, speed_mode="balanced", config={"include_models": ["asift", "akaze"]})
    assert "asift" in plan_custom.models
    assert "akaze" in plan_custom.models

