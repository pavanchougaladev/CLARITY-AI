"""
tests/test_antigravity_gates.py
===============================
Antigravity Five-Tier Mathematical Certification Quality Gate Auditing Suite
=============================================================================
Mathematically verifies all five quality gates defined in AGENTS.md,
IEEE_SYSTEM_SPECIFICATION.md, and src/evaluate/quality_gate.py:

  - Gate 1: Overfit Prevention & The 4-Point Trap Elimination
  - Gate 2: Minimum Viable Registration (MVR Floor >= 15 Inliers)
  - Gate 3: Spatial Distribution & Shannon Spatial Entropy (H_s >= 0.35, Cov >= 15%)
  - Gate 4: Zero-Hallucination Weight Nullification (0 Matches -> W_m = 0.0%)
  - Gate 5: Autonomous Self-Healing Recovery Loop (4096px Upscale & CPU Fallback)
"""

import pytest
import numpy as np
import cv2

from src.evaluate.quality_gate import (
    assess_quality,
    compute_spatial_entropy,
    MIN_REQUIRED_INLIERS,
)
from src.features.lunar_fusion_matcher import (
    LunarFusionMatcher,
    ModelPlan,
    ImageProfile,
)
from src.pipeline import build_recovery_plan


# ─────────────────────────────────────────────────────────────────────────────
# GATE 1: OVERFIT PREVENTION & THE 4-POINT TRAP ELIMINATION
# ─────────────────────────────────────────────────────────────────────────────

class TestGate1FourPointTrap:
    """
    Gate 1 mathematically audits and rejects degenerate overfit matrices.
    If a matcher produces <= 4 inliers with zero residual (0.000 px), standard
    RANSAC terminates with an illusory perfect fit. Gate 1 flags this as an
    overfit illusion and assigns Grade F.
    """

    def test_four_inliers_zero_residual_hard_rejection(self, four_point_trap_cluster):
        """Assert that 4 inliers with 0.000 px RMSE trigger Grade F and GATE_1 breach."""
        res = assess_quality(
            rmse_pixels=four_point_trap_cluster["rmse_pixels"],
            coverage=four_point_trap_cluster["coverage"],
            inlier_ratio=1.0,
            inlier_count=four_point_trap_cluster["inlier_count"],
            total_matches=four_point_trap_cluster["total_matches"],
        )
        assert res["grade"] == "F", f"Expected Grade F, got {res['grade']}"
        assert res["passed"] is False
        assert "FOUR_POINT_OVERFIT_TRAP" in res["reason_codes"]
        assert "GATE_1_FOUR_POINT_TRAP" in res["reason_codes"] or "GATE_1_FOUR_POINT_TRAP" in res["gate_telemetry"]
        assert "BREACHED" in res["gate_telemetry"]["GATE_1_4_POINT_TRAP"]
        assert any("4-POINT TRAP" in w for w in res["warnings"])
        # Gate 5 auto-recovery instruction must be triggered
        assert res["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "ACTION_REQUIRED"
        assert "auto_recovery" in res["details"]

    def test_sub_fifteen_inliers_near_zero_residual_rejection(self):
        """Assert that N < 15 with RMSE < 0.001 px (e.g. 8 inliers, 0.0002 px) is rejected as overfit."""
        res = assess_quality(
            rmse_pixels=0.0002,
            coverage=0.40,
            inlier_ratio=1.0,
            inlier_count=8,
            total_matches=8,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "FOUR_POINT_OVERFIT_TRAP" in res["reason_codes"]
        assert "BREACHED" in res["gate_telemetry"]["GATE_1_4_POINT_TRAP"]

    def test_four_point_trap_with_artificially_high_coverage(self):
        """Even if four points artificially span corners (high bounding box coverage), Gate 1 must reject."""
        res = assess_quality(
            rmse_pixels=0.000,
            coverage=0.85, # Artificially high coverage
            inlier_ratio=1.0,
            inlier_count=4,
            total_matches=4,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "FOUR_POINT_OVERFIT_TRAP" in res["reason_codes"]

    def test_three_inliers_hard_rejection(self):
        """Fewer than 4 inliers (e.g. 3 inliers) with zero residual must be rejected."""
        res = assess_quality(
            rmse_pixels=0.000,
            coverage=0.30,
            inlier_ratio=1.0,
            inlier_count=3,
            total_matches=3,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "FOUR_POINT_OVERFIT_TRAP" in res["reason_codes"]


# ─────────────────────────────────────────────────────────────────────────────
# GATE 2: MINIMUM VIABLE REGISTRATION (MVR FLOOR)
# ─────────────────────────────────────────────────────────────────────────────

class TestGate2MinimumViableRegistration:
    """
    Gate 2 enforces a strict statistical floor of N >= 15 verified inliers.
    Any candidate with < 15 inliers lacks geometric degrees-of-freedom redundancy
    and is rejected regardless of RANSAC confidence or reported sub-pixel RMSE.
    """

    @pytest.mark.parametrize("inlier_count", [1, 5, 8, 12, 14])
    def test_insufficient_inliers_hard_rejection(self, inlier_count):
        """Assert that inlier counts 1, 5, 8, 12, 14 all fail Gate 2 with Grade F."""
        res = assess_quality(
            rmse_pixels=0.35, # Excellent RMSE, but lacks inliers
            coverage=0.60,
            inlier_ratio=0.85,
            inlier_count=inlier_count,
            total_matches=inlier_count + 2,
        )
        assert res["grade"] == "F", f"Expected Grade F for {inlier_count} inliers, got {res['grade']}"
        assert res["passed"] is False
        assert "INSUFFICIENT_INLIERS" in res["reason_codes"]
        assert "BREACHED" in res["gate_telemetry"]["GATE_2_MIN_VIABLE_REGISTRATION"]
        assert f"< {MIN_REQUIRED_INLIERS} Inliers" in res["gate_telemetry"]["GATE_2_MIN_VIABLE_REGISTRATION"]

    def test_high_ransac_confidence_does_not_bypass_mvr(self):
        """Assert that RANSAC 0.999 confidence and 100% inlier ratio cannot bypass the 15-inlier floor."""
        res = assess_quality(
            rmse_pixels=0.25,
            coverage=0.75,
            inlier_ratio=1.0, # 100% inlier ratio
            inlier_count=13, # Still < 15
            total_matches=13,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "INSUFFICIENT_INLIERS" in res["reason_codes"]

    def test_fifteen_inliers_satisfies_mvr_floor(self):
        """Assert that exactly 15 verified inliers (with good coverage and RMSE) passes MVR Gate 2."""
        res = assess_quality(
            rmse_pixels=0.45,
            coverage=0.60,
            inlier_ratio=0.75,
            inlier_count=15, # Exactly 15 inliers
            total_matches=20,
        )
        assert res["passed"] is True
        assert res["gate_telemetry"]["GATE_2_MIN_VIABLE_REGISTRATION"] == "PASSED"
        assert res["grade"] in ["A", "B"]


# ─────────────────────────────────────────────────────────────────────────────
# GATE 3: SPATIAL DISTRIBUTION & SHANNON SPATIAL ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

class TestGate3SpatialDistributionAndEntropy:
    """
    Gate 3 audits the spatial spread of inlier correspondences across the crater grid.
    Divides canvas into an 8x8 uniform grid (K = 64 cells).
    Rejects registrations with:
      1. Spatial coverage < 15%
      2. Normalized Shannon spatial information entropy H_s < 0.35
    """

    def test_shannon_entropy_mathematical_limits(self):
        """Verify Shannon spatial entropy calculation on 8x8 grid: empty=0, clustered=0, uniform=1."""
        img_shape = (512, 512)
        grid = (8, 8)
        
        # 1. Empty point set
        h_empty = compute_spatial_entropy(np.empty((0, 2)), img_shape, grid)
        assert h_empty == 0.0

        # 2. Extreme clustering: all 100 points in exactly 1 cell (cell 0, 0)
        pts_clustered = np.ones((100, 2), dtype=np.float32) * 15.0
        h_clustered = compute_spatial_entropy(pts_clustered, img_shape, grid)
        assert h_clustered == 0.0, f"Expected 0.0 entropy for single cell cluster, got {h_clustered}"

        # 3. Uniform distribution: exactly 1 point per cell across all 64 cells
        pts_uniform = []
        for gi in range(8):
            for gj in range(8):
                pts_uniform.append([gj * 64 + 32, gi * 64 + 32])
        pts_uniform = np.array(pts_uniform, dtype=np.float32)
        h_uniform = compute_spatial_entropy(pts_uniform, img_shape, grid)
        assert np.isclose(h_uniform, 1.0, atol=1e-4), f"Expected 1.0 for uniform distribution, got {h_uniform}"

    def test_low_spatial_entropy_rejection(self, clustered_points_fixture):
        """Assert that tie points clustered into a tiny region yield H_s < 0.35 and trigger Gate 3 breach."""
        pts = clustered_points_fixture["pts_src"]
        img_shape = clustered_points_fixture["img_shape"]
        
        h_s = compute_spatial_entropy(pts, img_shape, grid=(8, 8))
        assert h_s < 0.35, f"Expected H_s < 0.35 for clustered fixture, got {h_s:.3f}"
        
        res = assess_quality(
            rmse_pixels=0.35,
            coverage=clustered_points_fixture["coverage"],
            inlier_ratio=0.80,
            inlier_count=clustered_points_fixture["inlier_count"],
            total_matches=clustered_points_fixture["total_matches"],
            spatial_entropy=h_s,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "SEVERE_MATCH_CLUSTERING" in res["reason_codes"]
        assert "BREACHED" in res["gate_telemetry"]["GATE_3_SPATIAL_DISTRIBUTION"]

    def test_low_spatial_coverage_rejection_at_eight_percent(self):
        """Coverage < 15% (e.g. 8%) must be rejected as Severe Match Clustering."""
        res = assess_quality(
            rmse_pixels=0.30,
            coverage=0.08, # 8% coverage
            inlier_ratio=0.85,
            inlier_count=80,
            total_matches=100,
            spatial_entropy=0.25,
        )
        assert res["grade"] == "F"
        assert res["passed"] is False
        assert "SEVERE_MATCH_CLUSTERING" in res["reason_codes"]
        assert "BREACHED" in res["gate_telemetry"]["GATE_3_SPATIAL_DISTRIBUTION"]

    def test_high_density_inlier_exception(self):
        """For pushbroom strips with >= 200 inliers and sub-3px RMSE, density implies valid spatial sampling."""
        res = assess_quality(
            rmse_pixels=0.55,
            coverage=0.14, # Borderline coverage (14%)
            inlier_ratio=0.65,
            inlier_count=240, # >= 200 inliers
            total_matches=370,
            spatial_entropy=0.45,
        )
        # With >= 200 verified inliers, the dense exception allows passing
        assert res["passed"] is True
        assert res["grade"] in ["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────────────────
# GATE 4: ZERO-HALLUCINATION WEIGHT NULLIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestGate4ZeroHallucinationEnforcement:
    """
    Gate 4 checks the raw match integer for every candidate engine in LunarFusion.
    If an engine returns N_raw = 0 matches:
      1. Its assigned consensus weight MUST be strictly W_m = 0.0%.
      2. It MUST NEVER be declared as the "dominant" engine.
      3. Active engines (> 0 matches) receive 100% of the redistributed weight.
    """

    def test_zero_raw_match_strictly_nullifies_weight(self):
        """Engines with 0 matches must receive exactly 0.0% weight."""
        matcher = LunarFusionMatcher(speed_mode="accurate")
        plan = ModelPlan(
            models=["roma_v2", "loftr", "sift"],
            weights={"roma_v2": 0.50, "loftr": 0.30, "sift": 0.20},
            reason="Multi-model verification"
        )
        profile = ImageProfile(profile_tag="normal")

        # Simulate: roma_v2 fails completely (0 matches), loftr gets 50, sift gets 30
        matches_by_model = {
            "roma_v2": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
            "loftr": (np.ones((50, 2), dtype=np.float32) * 50, np.ones((50, 2), dtype=np.float32) * 52, np.ones(50, dtype=np.float32) * 0.9),
            "sift": (np.ones((30, 2), dtype=np.float32) * 100, np.ones((30, 2), dtype=np.float32) * 102, np.ones(30, dtype=np.float32) * 0.8),
        }

        weights, raw_counts, inliers, decision, dominant = matcher._redistribute_weights_and_evaluate(
            plan, matches_by_model, profile
        )

        # 1. Zero-match nullification
        assert weights["roma_v2"] == 0.0, f"Expected 0.0% weight for roma_v2, got {weights['roma_v2']}"
        assert raw_counts["roma_v2"] == 0

        # 2. Dominance constraint: 0-match engine cannot be dominant
        assert dominant != "roma_v2", f"Zero-match engine was declared dominant: {dominant}"
        assert dominant in ["loftr", "sift"]

        # 3. Sum of weights among active engines must equal 1.0 (100%)
        active_weight_sum = sum(weights.values())
        assert np.isclose(active_weight_sum, 1.0, atol=1e-5), f"Weights do not sum to 1.0: {active_weight_sum}"

    def test_all_engines_zero_matches_handled_gracefully(self):
        """If all engines yield 0 raw matches, all weights must be 0% and dominant is None."""
        matcher = LunarFusionMatcher(speed_mode="balanced")
        plan = ModelPlan(
            models=["loftr", "sift"],
            weights={"loftr": 0.70, "sift": 0.30},
            reason="Dual engine test"
        )
        profile = ImageProfile(profile_tag="crater")

        matches_by_model = {
            "loftr": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
            "sift": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        }

        weights, raw_counts, inliers, decision, dominant = matcher._redistribute_weights_and_evaluate(
            plan, matches_by_model, profile
        )

        assert weights["loftr"] == 0.0
        assert weights["sift"] == 0.0
        assert dominant is None
        assert "All weights 0%" in decision

    def test_cross_modal_engine_zero_match_nullification(self):
        """In SAR-Optical Pipeline C, if classical SIFT yields 0 matches and RIFT2 yields 180, SIFT weight is 0%."""
        matcher = LunarFusionMatcher(speed_mode="accurate")
        plan = ModelPlan(
            models=["rift2", "sift"],
            weights={"rift2": 0.60, "sift": 0.40},
            reason="SAR-Optical modality"
        )
        profile = ImageProfile(profile_tag="sar_optical")

        matches_by_model = {
            "rift2": (np.random.rand(180, 2).astype(np.float32) * 200, np.random.rand(180, 2).astype(np.float32) * 200, np.ones(180, dtype=np.float32) * 0.85),
            "sift": (np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)),
        }

        weights, raw_counts, inliers, decision, dominant = matcher._redistribute_weights_and_evaluate(
            plan, matches_by_model, profile
        )

        assert weights["sift"] == 0.0
        assert weights["rift2"] == 1.0
        assert dominant == "rift2"


# ─────────────────────────────────────────────────────────────────────────────
# GATE 5: AUTONOMOUS SELF-HEALING RECOVERY LOOP
# ─────────────────────────────────────────────────────────────────────────────

class TestGate5AutonomousSelfHealingRecovery:
    """
    Gate 5 executes the self-healing loop when an image pair fails Gate 1, 2, or 3:
      Recovery Action 1: Upscale 'Max processing dimension (px)' to 4096.
      Recovery Action 2: Fall back from AI matchers (MPS divergence) to deterministic CPU math (SIFT).
      Outcome: Re-evaluate and certify the recovered solution to Grade C or higher.
    """

    def test_build_recovery_plan_upscales_to_4096(self):
        """Assert that failure reasons trigger 4096 px resolution upscale."""
        reasons = [
            "FOUR_POINT_OVERFIT_TRAP",
            "INSUFFICIENT_INLIERS",
            "SEVERE_MATCH_CLUSTERING",
            "LOW_SPATIAL_COVERAGE",
            "HIGH_RMSE_FAILURE",
        ]
        for r in reasons:
            plan = build_recovery_plan(matcher_type="loftr", max_dim=2048, reason=r)
            assert plan["max_dim"] == 4096, f"Expected 4096 px for {r}, got {plan['max_dim']}"

    def test_build_recovery_plan_falls_back_to_deterministic_sift(self):
        """Assert that AI matchers (LoFTR, RoMa, LunarFusion) fall back to deterministic SIFT."""
        ai_matchers = ["loftr", "roma", "roma_v2", "fusion", "lunar_fusion", "omniroute"]
        for m in ai_matchers:
            plan = build_recovery_plan(matcher_type=m, max_dim=2048, reason="INSUFFICIENT_INLIERS")
            assert plan["matcher_type"] == "sift", f"Expected fallback to sift for {m}, got {plan['matcher_type']}"
            assert plan["use_tile_matching"] is True

    def test_gate_telemetry_requests_autonomous_recovery(self):
        """Assert that any gate breach sets GATE_5_AUTONOMOUS_RECOVERY to 'ACTION_REQUIRED'."""
        # 1. Breached Gate 1
        res1 = assess_quality(rmse_pixels=0.000, coverage=0.5, inlier_ratio=1.0, inlier_count=4, total_matches=4)
        assert res1["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "ACTION_REQUIRED"

        # 2. Breached Gate 2
        res2 = assess_quality(rmse_pixels=0.4, coverage=0.5, inlier_ratio=0.8, inlier_count=10, total_matches=12)
        assert res2["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "ACTION_REQUIRED"

        # 3. Breached Gate 3
        res3 = assess_quality(rmse_pixels=0.4, coverage=0.05, inlier_ratio=0.8, inlier_count=50, total_matches=60)
        assert res3["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "ACTION_REQUIRED"

        # 4. Nominal passed registration -> NOMINAL
        res_pass = assess_quality(rmse_pixels=0.3, coverage=0.7, inlier_ratio=0.9, inlier_count=300, total_matches=330)
        assert res_pass["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "NOMINAL"

    def test_simulated_self_healing_recovery_certifies_grade_c_or_higher(self):
        """Simulate initial failure followed by recovery upscale: solution recovers to Grade C or higher."""
        # Initial attempt: failed due to low inliers at coarse resolution
        initial_res = assess_quality(
            rmse_pixels=0.42,
            coverage=0.20,
            inlier_ratio=0.50,
            inlier_count=11, # < 15 inliers
            total_matches=22,
        )
        assert initial_res["grade"] == "F"
        assert initial_res["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "ACTION_REQUIRED"

        # Trigger recovery plan
        rec_plan = build_recovery_plan("loftr", 2048, initial_res["primary_reason"])
        assert rec_plan["max_dim"] == 4096
        assert rec_plan["matcher_type"] == "sift"

        # Simulated recovery execution at 4096px with SIFT tile matching:
        # resolves high-frequency craters, yielding 185 inliers and 0.38 px RMSE
        recovered_res = assess_quality(
            rmse_pixels=0.38,
            coverage=0.62,
            inlier_ratio=0.78,
            inlier_count=185,
            total_matches=237,
            spatial_entropy=0.68,
        )
        assert recovered_res["passed"] is True
        assert recovered_res["grade"] in ["A", "B", "C"]
        assert recovered_res["confidence"] >= 0.65
        assert recovered_res["gate_telemetry"]["GATE_5_AUTONOMOUS_RECOVERY"] == "NOMINAL"
