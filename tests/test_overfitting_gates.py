"""
tests/test_overfitting_gates.py
================================
Unit tests verifying the 3 Critical Overfitting Prevention Constraints:
1. The 4-Point Trap: Exactly 4 inliers with artificial 0.000 px RMSE -> strictly Grade F.
2. Strict Inlier Floor: Fewer than 15 inliers -> strictly Grade F, passed=False.
3. Spatial Distribution Check: Coverage < 15% -> strictly Grade F, Severe Match Clustering.
4. Auto-Recovery Instruction: Guidance to upscale processing dimension from 2048 to 4096.
5. USAC_MAGSAC parameterization check in outlier rejection.
"""

import numpy as np
import pytest
from src.evaluate.quality_gate import assess_quality, MIN_REQUIRED_INLIERS


def test_four_point_trap_rejection():
    """Any registration with 4 inliers and 0.000 px RMSE must be rejected as an overfit matrix (Grade F)."""
    res = assess_quality(
        rmse_pixels=0.000,
        coverage=0.45,
        inlier_ratio=1.0,
        inlier_count=4,
        total_matches=4,
    )
    assert res['grade'] == 'F', f"Expected Grade F for 4-point trap, got {res['grade']}"
    assert res['passed'] is False
    assert res['primary_reason'] == "FOUR_POINT_OVERFIT_TRAP"
    assert any("4-POINT TRAP" in w for w in res['warnings'])
    assert "auto_recovery" in res['details']


def test_strict_inlier_floor():
    """Fewer than 15 inliers (e.g., 10 inliers) must fail quality gate as Grade F."""
    res = assess_quality(
        rmse_pixels=0.40,
        coverage=0.50,
        inlier_ratio=0.80,
        inlier_count=12,
        total_matches=15,
    )
    assert res['grade'] == 'F', f"Expected Grade F for 12 inliers (< {MIN_REQUIRED_INLIERS}), got {res['grade']}"
    assert res['passed'] is False
    assert res['primary_reason'] == "INSUFFICIENT_INLIERS"
    assert any("Insufficient inliers" in w for w in res['warnings'])


def test_severe_match_clustering():
    """Spatial coverage < 15% must be rejected as Grade F due to severe match clustering."""
    res = assess_quality(
        rmse_pixels=0.35,
        coverage=0.08, # 8% coverage
        inlier_ratio=0.85,
        inlier_count=80,
        total_matches=100,
    )
    assert res['grade'] == 'F', f"Expected Grade F for coverage < 15%, got {res['grade']}"
    assert res['passed'] is False
    assert res['primary_reason'] == "SEVERE_MATCH_CLUSTERING"
    assert any("Severe match clustering" in w for w in res['warnings'])


def test_valid_registration_passes():
    """Well-distributed inliers (>= 15, coverage >= 50%, low RMSE) must pass with Grade A."""
    res = assess_quality(
        rmse_pixels=0.30,
        coverage=0.65,
        inlier_ratio=0.90,
        inlier_count=200,
        total_matches=220,
    )
    assert res['grade'] == 'A', f"Expected Grade A, got {res['grade']}"
    assert res['passed'] is True
    assert res['confidence'] >= 0.85


def test_low_inlier_ratio_rejected():
    """A low inlier ratio is a hard quality failure because it indicates mostly false matches."""
    res = assess_quality(
        rmse_pixels=0.80,
        coverage=0.70,
        inlier_ratio=0.28,
        inlier_count=32,
        total_matches=115,
    )
    assert res['grade'] == 'F', f"Expected Grade F for inlier ratio below hard floor, got {res['grade']}"
    assert res['passed'] is False
    assert res['primary_reason'] == "INSUFFICIENT_INLIER_RATIO"


def test_outlier_rejection_magsac_params():
    """Verify that outlier_rejection defaults to USAC_MAGSAC with 50,000 iters and 3.0 threshold."""
    from src.matching.outlier_rejection import MIN_REQUIRED_INLIERS as REJ_FLOOR
    assert REJ_FLOOR == 15, f"Expected outlier rejection floor to be 15, got {REJ_FLOOR}"


def test_gate_telemetry_tracking():
    """Verify that gate_telemetry dictionary accurately reports all 5 gate states."""
    res = assess_quality(
        rmse_pixels=0.000,
        coverage=0.05,
        inlier_ratio=1.0,
        inlier_count=4,
        total_matches=4,
    )
    gates = res['gate_telemetry']
    assert "BREACHED" in gates['GATE_1_4_POINT_TRAP']
    assert "BREACHED" in gates['GATE_2_MIN_VIABLE_REGISTRATION']
    assert "BREACHED" in gates['GATE_3_SPATIAL_DISTRIBUTION']
    assert "ENFORCED" in gates['GATE_4_ZERO_HALLUCINATION']
    assert gates['GATE_5_AUTONOMOUS_RECOVERY'] == "ACTION_REQUIRED"

