"""
Unit and integration tests for YOLO11 Crater Landmark Matcher
and its integration into LunarFusion Adaptive AI.
"""

import numpy as np
import pytest
import cv2
import os

from src.features.crater_detector import _ULTRALYTICS_AVAILABLE
from src.features.crater_matcher import (
    YOLO11CraterMatcher,
    match_craters_yolo11,
)
from src.features.lunar_fusion_matcher import (
    LunarFusionMatcher,
    AdaptiveRouter,
    ImageProfile,
    ModelPlan,
)

pytestmark = pytest.mark.skipif(
    not _ULTRALYTICS_AVAILABLE,
    reason="ultralytics not installed; install 'ultralytics>=8.3.0' to run crater matcher tests",
)


def create_synthetic_crater_patch(size=640, shift_x=0, shift_y=0):
    """Generate synthetic lunar surface with distinct craters."""
    img = np.full((size, size), 128, dtype=np.uint8)
    noise = np.random.normal(0, 6, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    craters = [
        (160, 160, 45),
        (460, 160, 35),
        (320, 320, 55),
        (160, 480, 40),
        (480, 480, 50),
    ]
    for cx, cy, r in craters:
        x = cx + shift_x
        y = cy + shift_y
        if 0 <= x < size and 0 <= y < size:
            # Inner dark bowl
            cv2.circle(img, (x, y), r, 50, -1)
            # Bright sunlit rim
            cv2.circle(img, (x - int(r * 0.2), y - int(r * 0.2)), r, 220, 3)
    return img


def test_crater_matcher_initialization():
    """Verify YOLO11CraterMatcher initializes cleanly."""
    matcher = YOLO11CraterMatcher(conf=0.15)
    assert matcher is not None
    assert matcher.detector is not None


def test_crater_matcher_empty_images():
    """Verify empty/uniform images return empty match tuples without error."""
    black1 = np.zeros((300, 300), dtype=np.uint8)
    black2 = np.zeros((300, 300), dtype=np.uint8)

    matcher = YOLO11CraterMatcher(conf=0.20)
    pts_s, pts_r, conf = matcher.match(black1, black2)

    assert len(pts_s) == 0
    assert len(pts_r) == 0
    assert len(conf) == 0


def test_crater_matcher_synthetic_matching():
    """Verify matching on synthetic lunar crater scenes."""
    np.random.seed(42)
    img_src = create_synthetic_crater_patch(640, shift_x=0, shift_y=0)
    img_ref = create_synthetic_crater_patch(640, shift_x=12, shift_y=-8)

    matcher = YOLO11CraterMatcher(conf=0.10, min_similarity=0.25, include_rim_points=True)
    pts_s, pts_r, conf = matcher.match(img_src, img_ref)

    # If YOLO11 detected craters on this synthetic patch
    if len(pts_s) > 0:
        assert len(pts_s) == len(pts_r) == len(conf)
        assert pts_s.shape[1] == 2
        assert pts_r.shape[1] == 2
        # Check shift direction
        diffs = pts_r - pts_s
        # Mean shift should be in direction of (+12, -8)
        assert np.mean(diffs[:, 0]) > 0  # positive X shift
        assert np.mean(diffs[:, 1]) < 0  # negative Y shift


def test_match_craters_yolo11_helper():
    """Verify top-level function match_craters_yolo11 works with config dict."""
    img = np.ones((200, 200), dtype=np.uint8) * 100
    pts_s, pts_r, conf = match_craters_yolo11(img, img, config={"yolo11_conf": 0.25})
    assert isinstance(pts_s, np.ndarray)
    assert isinstance(pts_r, np.ndarray)
    assert isinstance(conf, np.ndarray)


def test_lunar_fusion_router_includes_yolo11():
    """Verify AdaptiveRouter includes yolo11 in crater and normal profiles."""
    router = AdaptiveRouter()

    # Crater profile must include yolo11
    p_crater = ImageProfile(profile_tag="crater")
    plan_crater = router.route(p_crater, speed_mode="balanced")
    assert "yolo11" in plan_crater.models
    assert plan_crater.weights["yolo11"] > 0.0

    # Normal profile must include yolo11
    p_normal = ImageProfile(profile_tag="normal")
    plan_normal = router.route(p_normal, speed_mode="balanced")
    assert "yolo11" in plan_normal.models
    assert plan_normal.weights["yolo11"] > 0.0


def test_lunar_fusion_gate_4_zero_match_yolo11():
    """Verify Gate 4 strictly zeroes YOLO11 weight if it returns 0 raw matches."""
    matcher = LunarFusionMatcher()
    profile = ImageProfile(profile_tag="crater")
    plan = ModelPlan(
        models=["roma_v2", "yolo11", "loftr", "sift"],
        weights={"roma_v2": 0.40, "yolo11": 0.30, "loftr": 0.20, "sift": 0.10},
        reason="Crater terrain test"
    )

    # Simulated matches where YOLO11 had 0 matches
    pts_s = np.random.uniform(50, 450, (50, 2)).astype(np.float32)
    pts_r = pts_s + np.array([2.0, 2.0], dtype=np.float32)
    conf = np.full(50, 0.8, dtype=np.float32)

    matches_by_model = {
        "roma_v2": (pts_s, pts_r, conf),
        "loftr": (pts_s, pts_r, conf),
        "sift": (pts_s, pts_r, conf),
        "yolo11": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
    }

    weights, raw_counts, inliers, ai_decision, dominant = matcher._redistribute_weights_and_evaluate(
        plan, matches_by_model, profile
    )

    # Gate 4: Zero matches MUST result in strictly 0.0 weight
    assert weights["yolo11"] == 0.0
    assert dominant != "yolo11"
    assert "YOLO11 Crater" in ai_decision
