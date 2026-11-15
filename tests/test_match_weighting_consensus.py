"""
Tests for Match Weighting, Zero-Match Exclusion, and Dominance Consensus Logic.

Validates the three critical pipeline constraints:
1. Zero-Match Exclusion: Any engine returning 0 matches gets strictly 0% weight.
2. Weight Redistribution: 100% weight is distributed only among active engines (>0 matches),
   proportional to inlier count and confidence.
3. Dominance Logic: Never declare an engine with 0 matches as dominant; only highlight the
   top-weighted engine that actually contributed valid matches.
"""

import pytest
import numpy as np

from src.features.lunar_fusion_matcher import (
    ImageProfile,
    ModelPlan,
    LunarFusionMatcher,
)


def test_zero_match_exclusion_and_weight_redistribution():
    """Verify that zero-match engines receive 0% weight and active engines get 100% distributed."""
    matcher = LunarFusionMatcher()
    profile = ImageProfile(profile_tag="crater")
    plan = ModelPlan(
        models=["roma_v2", "loftr", "sift", "lightglue", "ni_lfm"],
        weights={"roma_v2": 0.35, "loftr": 0.25, "sift": 0.15, "lightglue": 0.15, "ni_lfm": 0.10},
        reason="Crater terrain"
    )

    # Simulated matches with consistent geometric transform (shift + tiny noise):
    # roma_v2:   0 matches (crashed or failed)
    # lightglue: 0 matches
    # loftr:     400 matches, high confidence
    # sift:      1200 matches, solid confidence
    # ni_lfm:    200 matches, moderate confidence
    np.random.seed(42)
    sift_s = np.random.uniform(10, 490, (1200, 2)).astype(np.float32)
    sift_r = sift_s + np.array([5.0, -3.0], dtype=np.float32) + np.random.normal(0, 0.2, (1200, 2)).astype(np.float32)

    loftr_s = np.random.uniform(50, 450, (400, 2)).astype(np.float32)
    loftr_r = loftr_s + np.array([5.0, -3.0], dtype=np.float32) + np.random.normal(0, 0.2, (400, 2)).astype(np.float32)

    ni_s = np.random.uniform(100, 400, (200, 2)).astype(np.float32)
    ni_r = ni_s + np.array([5.0, -3.0], dtype=np.float32) + np.random.normal(0, 0.2, (200, 2)).astype(np.float32)

    matches_by_model = {
        "roma_v2":   (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        "lightglue": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        "loftr": (loftr_s, loftr_r, np.full(400, 0.85, dtype=np.float32)),
        "sift": (sift_s, sift_r, np.full(1200, 0.75, dtype=np.float32)),
        "ni_lfm": (ni_s, ni_r, np.full(200, 0.70, dtype=np.float32)),
    }

    weights, raw_counts, inliers, ai_decision, dominant = matcher._redistribute_weights_and_evaluate(
        plan, matches_by_model, profile
    )

    # Constraint 1: Zero-Match Exclusion
    assert weights["roma_v2"] == 0.0, f"Expected 0.0 for roma_v2, got {weights['roma_v2']}"
    assert weights["lightglue"] == 0.0, f"Expected 0.0 for lightglue, got {weights['lightglue']}"

    # Constraint 2: Weight Redistribution (sum of all weights = 1.0)
    total_w = sum(weights.values())
    assert pytest.approx(total_w, 1e-5) == 1.0, f"Total weight must sum to 1.0, got {total_w}"
    assert weights["sift"] > 0.0
    assert weights["loftr"] > 0.0
    assert weights["ni_lfm"] > 0.0

    # SIFT has largest score (1200 * 0.75 = 900 vs LoFTR 400 * 0.85 = 340)
    assert weights["sift"] > weights["loftr"] > weights["ni_lfm"]

    # Constraint 3: Dominance Logic
    assert dominant != "roma_v2", "RoMa v2 returned 0 matches and must NEVER be declared dominant!"
    assert dominant != "lightglue", "LightGlue returned 0 matches and must NEVER be declared dominant!"
    assert dominant == "sift", f"Expected SIFT as dominant engine, got {dominant}"

    # AI Decision text validation
    assert "SIFT dominant" in ai_decision, f"AI Decision must highlight SIFT: {ai_decision}"
    assert "RoMa v2" in ai_decision and "strictly 0% weight" in ai_decision, "AI Decision must document RoMa v2 0% exclusion"
    assert "DISK+LightGlue" in ai_decision and "strictly 0% weight" in ai_decision, "AI Decision must document DISK+LightGlue 0% exclusion"
    assert "RoMa v2 dominant" not in ai_decision, "AI Decision must never declare RoMa v2 dominant when 0 matches"


def test_single_active_engine_gets_100_percent():
    """Verify that if only SIFT succeeds, it receives exactly 100% weight."""
    matcher = LunarFusionMatcher()
    profile = ImageProfile(profile_tag="shadow")
    plan = ModelPlan(
        models=["loftr", "ni_lfm", "sift"],
        weights={"loftr": 0.40, "ni_lfm": 0.40, "sift": 0.20},
        reason="Shadow terrain"
    )

    matches_by_model = {
        "loftr":  (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        "ni_lfm": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        "sift": (
            np.ones((150, 2), dtype=np.float32) * 50.0,
            np.ones((150, 2), dtype=np.float32) * 52.0,
            np.full(150, 0.80, dtype=np.float32)
        ),
    }

    weights, raw_counts, inliers, ai_decision, dominant = matcher._redistribute_weights_and_evaluate(
        plan, matches_by_model, profile
    )

    assert weights["loftr"] == 0.0
    assert weights["ni_lfm"] == 0.0
    assert pytest.approx(weights["sift"], 1e-5) == 1.0
    assert dominant == "sift"
    assert "SIFT dominant (100% consensus weight" in ai_decision


def test_all_zero_matches():
    """Verify graceful handling when no matcher returns matches."""
    matcher = LunarFusionMatcher()
    profile = ImageProfile(profile_tag="normal")
    plan = ModelPlan(
        models=["roma_v2", "sift"],
        weights={"roma_v2": 0.7, "sift": 0.3},
        reason="Normal terrain"
    )

    matches_by_model = {
        "roma_v2": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        "sift":    (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
    }

    weights, raw_counts, inliers, ai_decision, dominant = matcher._redistribute_weights_and_evaluate(
        plan, matches_by_model, profile
    )

    assert weights["roma_v2"] == 0.0
    assert weights["sift"] == 0.0
    assert dominant is None
    assert "No feature matchers returned valid raw matches" in ai_decision


def test_lunar_fusion_match_with_stats_end_to_end(monkeypatch):
    """End-to-end verification of match_with_stats ensuring zero-match exclusion & dominance."""
    matcher = LunarFusionMatcher(speed_mode="fast")
    # Synthetic small images
    src_img = np.zeros((100, 100), dtype=np.uint8)
    ref_img = np.zeros((100, 100), dtype=np.uint8)

    # Mock _run_single_matcher so that sift succeeds with 50 matches, loftr fails / returns 0
    def mock_run(m_type, src_t, ref_t, src_np, ref_np, cfg):
        if m_type == "sift":
            pts = np.linspace(10, 80, 50).reshape(-1, 1)
            pts2d = np.hstack([pts, pts]).astype(np.float32)
            conf = np.full(50, 0.9, dtype=np.float32)
            return pts2d, pts2d + 1.0, conf
        else:
            # e.g. loftr returns 0
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    monkeypatch.setattr(matcher, "_run_single_matcher", mock_run)

    f_src, f_ref, f_conf, stats = matcher.match_with_stats(src_img, ref_img)

    assert len(f_src) > 0
    assert stats["weights"]["sift"] == 1.0
    # Any other model in plan has 0% weight
    for m, w in stats["weights"].items():
        if m != "sift":
            assert w == 0.0, f"Expected 0% for {m}, got {w}"

    assert stats["dominant_engine"] == "sift"
    assert "SIFT dominant" in stats["reason"]
    assert stats["zero_match_excluded"] != [] or len(stats["models_used"]) == 1
