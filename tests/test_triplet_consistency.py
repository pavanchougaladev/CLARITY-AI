"""
Unit Tests for Triplet Cycle-Consistency Validation (Feature 1)
"""

import pytest
import numpy as np
from src.evaluate.triplet_consistency import check_triplet_closure


def test_perfect_identity_closure():
    # 3 identity matrices: A->B, B->C, C->A
    I = np.eye(3, dtype=np.float64)
    res = check_triplet_closure(I, I, I, image_shape=(1000, 1000), tolerance_px=1.0)

    assert res["status"] == "PASS"
    assert res["is_consistent"] is True
    assert res["mean_residual_px"] < 1e-4
    assert res["closure_ratio"] == 1.0


def test_translation_loop_closure():
    # A->B: shift (+50, +20)
    # B->C: shift (-10, +30)
    # C->A: shift (-40, -50)
    # Sum of shifts = 0 -> perfect loop
    H_AB = np.array([[1, 0, 50], [0, 1, 20], [0, 0, 1]], dtype=np.float64)
    H_BC = np.array([[1, 0, -10], [0, 1, 30], [0, 0, 1]], dtype=np.float64)
    H_CA = np.array([[1, 0, -40], [0, 1, -50], [0, 0, 1]], dtype=np.float64)

    res = check_triplet_closure(H_AB, H_BC, H_CA, image_shape=(1000, 1000), tolerance_px=2.0)
    assert res["status"] == "PASS"
    assert res["is_consistent"] is True
    assert res["mean_residual_px"] < 1e-3


def test_loop_closure_via_H_AC():
    # If H_AC is provided instead of H_CA: H_AC = (H_CA)^-1
    H_AB = np.array([[1, 0, 25], [0, 1, 10], [0, 0, 1]], dtype=np.float64)
    H_BC = np.array([[1, 0, 15], [0, 1, 15], [0, 0, 1]], dtype=np.float64)
    # H_AC = H_BC @ H_AB = shift (40, 25)
    H_AC = H_BC @ H_AB

    res = check_triplet_closure(H_AB, H_BC, H_AC=H_AC, image_shape=(800, 800), tolerance_px=1.0)
    assert res["status"] == "PASS"
    assert res["is_consistent"] is True
    assert res["mean_residual_px"] < 1e-3


def test_inconsistent_triplet_drift():
    # Severe drift in loop: A->B (+10), B->C (+10), C->A (+10) -> net loop error = 30px
    H_AB = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    H_BC = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    H_CA = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

    res = check_triplet_closure(H_AB, H_BC, H_CA, image_shape=(1000, 1000), tolerance_px=5.0)
    assert res["status"] == "FAIL"
    assert res["is_consistent"] is False
    assert res["mean_residual_px"] >= 25.0
