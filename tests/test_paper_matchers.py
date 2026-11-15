import numpy as np
import cv2
import pytest
from src.features.superglue_matcher import SuperGlueMatcher, build_superglue_matcher
from src.features.rift2_matcher import RIFT2Matcher, AKAZEMatcher, ASIFTMatcher
from src.features.deep_matcher import LunarMatcher
from scripts.paper_benchmark_replication import (
    get_paper_reference_dataframe,
    evaluate_registration_accuracy,
)


def _create_synthetic_lunar_patch(h=256, w=256, seed=42):
    """Generate a realistic textured synthetic lunar patch with crater-like rings."""
    np.random.seed(seed)
    base = np.random.normal(120, 25, (h, w)).clip(0, 255).astype(np.uint8)
    # Add simulated craters
    for cy, cx, r in [(64, 64, 30), (180, 150, 45), (100, 190, 20), (190, 70, 25)]:
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rim_mask = np.abs(dist - r) < 4
        interior_mask = dist < r
        base[rim_mask] = np.clip(base[rim_mask] + 80, 0, 255)
        base[interior_mask] = np.clip(base[interior_mask] - 40, 0, 255)
    return base


def test_paper_table_3_dataframe():
    df = get_paper_reference_dataframe()
    assert len(df) == 30
    assert "SuperGlue" in df["algorithm"].values
    assert "RIFT2" in df["algorithm"].values
    # Check paper Table 3 values for SuperGlue on OHRC-NAC Equatorial
    sg_ohrc = df[(df["sensor_pair"] == "OHRC-NAC") & (df["region"] == "Equatorial") & (df["algorithm"] == "SuperGlue")].iloc[0]
    assert sg_ohrc["rmse_x"] == 0.6249
    assert sg_ohrc["rmse_y"] == 0.5718
    assert sg_ohrc["total_rmse"] == 0.8471
    assert sg_ohrc["time_s"] == 3.81


def test_quality_gates_in_evaluation():
    # Gate 2: Less than 15 inliers -> Rejected
    pts_s = np.zeros((10, 2), dtype=np.float32)
    pts_r = np.zeros((10, 2), dtype=np.float32)
    res = evaluate_registration_accuracy(pts_s, pts_r)
    assert res["status"] == "FAILED QUALITY GATE"
    assert "Gate 2" in res["reason"]

    # Gate 1: 4 inliers with 0 residual -> Overfit trap
    pts_s = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    pts_r = pts_s.copy()
    res = evaluate_registration_accuracy(pts_s, pts_r)
    assert res["status"] == "FAILED QUALITY GATE"


def test_superglue_matcher_interface():
    matcher = SuperGlueMatcher(max_keypoints=500, confidence_threshold=0.1)
    src = _create_synthetic_lunar_patch(256, 256, seed=42)
    # Reference is slightly translated
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    ref = cv2.warpAffine(src, M, (256, 256))

    out = matcher.match(src, ref)
    assert hasattr(out, "pts_src")
    assert hasattr(out, "pts_ref")
    assert hasattr(out, "confidences")
    assert out.pts_src.shape[1] == 2
    assert out.pts_ref.shape[1] == 2
    assert len(out.confidences) == len(out.pts_src)
    assert out.matcher_name.lower() == "superglue"


def test_akaze_matcher_interface():
    matcher = AKAZEMatcher(confidence_threshold=0.2)
    src = _create_synthetic_lunar_patch(256, 256, seed=42)
    M = np.float32([[1, 0, 4], [0, 1, 2]])
    ref = cv2.warpAffine(src, M, (256, 256))

    out = matcher.match(src, ref)
    assert hasattr(out, "pts_src")
    assert hasattr(out, "pts_ref")
    assert out.matcher_name.lower() == "akaze"


def test_deep_matcher_delegation_to_paper_algorithms():
    src = _create_synthetic_lunar_patch(256, 256, seed=42)
    ref = _create_synthetic_lunar_patch(256, 256, seed=43)

    # Test SuperGlue via LunarMatcher
    sg_matcher = LunarMatcher(matcher_type="superglue")
    assert sg_matcher._available is True

    # Test RIFT2 via LunarMatcher
    rift_matcher = LunarMatcher(matcher_type="rift2")
    assert rift_matcher._available is True

    # Test AKAZE via LunarMatcher
    akaze_matcher = LunarMatcher(matcher_type="akaze")
    assert akaze_matcher._available is True
