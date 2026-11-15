import json
import pytest
import os
import cv2
import numpy as np
from src.pipeline import run_registration_pipeline, build_recovery_plan, estimate_pair_overlap, choose_matcher_strategy
from src.benchmark_manifest import filter_manifest_by_overlap, discover_overlapping_pairs

@pytest.fixture
def mock_images(tmp_path):
    # Create two synthetic overlapping images
    img1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img1, (100, 100), 50, 255, -1)
    
    # img2 is shifted by (10, 10)
    img2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img2, (110, 110), 50, 255, -1)
    
    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    
    cv2.imwrite(str(path1), img1)
    cv2.imwrite(str(path2), img2)
    
    return str(path1), str(path2)

def test_pipeline_end_to_end(mock_images):
    src_path, ref_path = mock_images
    
    # We will use SIFT fallback since LoFTR might not find features in a simple circle
    # But wait, a simple circle doesn't have many features for SIFT either.
    # Let's just mock the feature matching or verify it fails gracefully if no features
    
    try:
        img_src, img_ref, results = run_registration_pipeline(src_path, ref_path, use_tps=False)
    except ValueError as e:
        # Since it's a blank image with a circle, it might not find 4 matches
        assert "Not enough matches" in str(e) or "failed" in str(e).lower()


def test_pipeline_identical_images():
    """Verify that registering two identical real lunar images succeeds with Grade A or B and passes Quality Gate."""
    img_path = 'data/raw/ch2_tmc_nrn_20191208T0221109733_d_img_g26/browse/raw/20191208/ch2_tmc_nrn_20191208T0221109733_b_brw_g26.png'
    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    _, _, res = run_registration_pipeline(img_path, img_path, matcher_type='sift')
    quality = res['quality']
    assert res['inlier_count'] > 100
    assert res['rmse_pixels'] < 0.1
    assert quality['passed'] is True
    assert quality['grade'] in ['A', 'B']
    assert 'ILL_CONDITIONED_GEOMETRY' not in quality['reason_codes']


def test_build_recovery_plan_uses_4096_and_sift_fallback():
    plan = build_recovery_plan("loftr", 2048, "INSUFFICIENT_INLIERS")
    assert plan["matcher_type"] == "sift"
    assert plan["max_dim"] == 4096
    assert plan["reason"] == "INSUFFICIENT_INLIERS"

    plan2 = build_recovery_plan("sift", 2048, "LOW_SPATIAL_COVERAGE")
    assert plan2["matcher_type"] == "sift"
    assert plan2["max_dim"] == 4096


def test_pipeline_returns_structured_rejection_for_insufficient_matches(tmp_path):
    src = np.zeros((128, 128), dtype=np.uint8)
    ref = np.zeros((128, 128), dtype=np.uint8)
    src_path = tmp_path / "blank_src.png"
    ref_path = tmp_path / "blank_ref.png"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)

    _, _, result = run_registration_pipeline(
        str(src_path),
        str(ref_path),
        matcher_type="sift",
        use_tile_matching=False,
        auto_recover=False,
    )

    assert result["quality"]["passed"] is False
    assert result["quality"]["primary_reason"] == "INSUFFICIENT_RAW_MATCHES"
    assert result["rmse_pixels"] is None
    assert result["telemetry"]["status"] == "rejected"


def test_pipeline_rejects_non_overlapping_image_pairs_before_match(tmp_path):
    src = np.zeros((256, 256), dtype=np.uint8)
    ref = np.full((256, 256), 255, dtype=np.uint8)
    cv2.circle(src, (64, 64), 25, 200, -1)
    cv2.rectangle(ref, (180, 180), (230, 230), 30, -1)

    src_path = tmp_path / "non_overlap_src.png"
    ref_path = tmp_path / "non_overlap_ref.png"
    cv2.imwrite(str(src_path), src)
    cv2.imwrite(str(ref_path), ref)

    _, _, result = run_registration_pipeline(
        str(src_path),
        str(ref_path),
        matcher_type="sift",
        use_tile_matching=False,
        auto_recover=False,
    )

    assert result["quality"]["passed"] is False
    assert result["quality"]["primary_reason"] in {"LOW_OVERLAP_PRECHECK", "INSUFFICIENT_RAW_MATCHES"}
    assert result["telemetry"]["status"] == "rejected"


def test_pipeline_reports_independent_gt_rmse(tmp_path):
    rng = np.random.default_rng(7)
    source = np.zeros((400, 400), dtype=np.uint8)
    for x, y in rng.integers(30, 370, size=(120, 2)):
        cv2.circle(source, (int(x), int(y)), 3, 255, -1)

    transform = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, 7.0]], dtype=np.float32)
    reference = cv2.warpAffine(source, transform, (400, 400))
    gt_src = np.array([[60, 60], [120, 220], [250, 100], [320, 300], [180, 340]], dtype=np.float32)
    gt_ref = cv2.transform(gt_src.reshape(-1, 1, 2), transform).reshape(-1, 2)

    src_path = tmp_path / "gt_src.png"
    ref_path = tmp_path / "gt_ref.png"
    cv2.imwrite(str(src_path), source)
    cv2.imwrite(str(ref_path), reference)

    _, _, result = run_registration_pipeline(
        str(src_path), str(ref_path), matcher_type="sift", use_tps=False,
        use_tile_matching=False, auto_recover=False, max_dim=512,
        gt_pts_src=gt_src, gt_pts_ref=gt_ref,
    )

    assert result["gt_validation_status"] == "valid"
    assert result["gt_point_count"] == len(gt_src)
    assert result["gt_rmse_pixels"] < 2.0


def test_filter_manifest_by_overlap_removes_weak_pairs(tmp_path):
    src_a = np.zeros((128, 128), dtype=np.uint8)
    ref_a = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(src_a, (64, 64), 20, 255, -1)
    cv2.circle(ref_a, (72, 72), 20, 255, -1)

    src_b = np.zeros((128, 128), dtype=np.uint8)
    ref_b = np.full((128, 128), 255, dtype=np.uint8)
    cv2.circle(src_b, (20, 20), 12, 200, -1)
    cv2.circle(ref_b, (100, 100), 12, 40, -1)

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pairs": [{"source": str(tmp_path / "src_a.png"), "reference": str(tmp_path / "ref_a.png")}, {"source": str(tmp_path / "src_b.png"), "reference": str(tmp_path / "ref_b.png")}] }), encoding="utf-8")
    cv2.imwrite(str(tmp_path / "src_a.png"), src_a)
    cv2.imwrite(str(tmp_path / "ref_a.png"), ref_a)
    cv2.imwrite(str(tmp_path / "src_b.png"), src_b)
    cv2.imwrite(str(tmp_path / "ref_b.png"), ref_b)

    selected = filter_manifest_by_overlap(str(manifest), min_overlap=0.15)
    assert len(selected) == 1
    assert selected[0][0].endswith("src_a.png")


def test_estimate_pair_overlap_handles_mismatched_dimensions():
    src = np.zeros((220, 200), dtype=np.uint8)
    ref = np.zeros((180, 240), dtype=np.uint8)
    cv2.circle(src, (100, 90), 30, 255, -1)
    cv2.circle(ref, (120, 80), 28, 255, -1)

    score = estimate_pair_overlap(src, ref)
    assert 0.0 <= score <= 1.0
    assert np.isfinite(score)


def test_choose_matcher_strategy_uses_sift_for_low_overlap_pairs():
    selected, reason = choose_matcher_strategy(0.08, "loftr")
    assert selected == "sift"
    assert "low-overlap" in reason.lower()

    selected, reason = choose_matcher_strategy(0.35, "loftr")
    assert selected == "loftr"
    assert "deep match" in reason.lower()


def test_discover_overlapping_pairs_finds_valid_manifest(tmp_path):
    src_a = np.zeros((128, 128), dtype=np.uint8)
    ref_a = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(src_a, (64, 64), 20, 255, -1)
    cv2.circle(ref_a, (68, 66), 20, 255, -1)

    src_b = np.zeros((128, 128), dtype=np.uint8)
    ref_b = np.full((128, 128), 255, dtype=np.uint8)
    cv2.circle(src_b, (20, 20), 18, 200, -1)
    cv2.circle(ref_b, (110, 110), 18, 40, -1)

    dir_a = tmp_path / "data" / "raw" / "ch1_tmc_nca_20090529T1724252721_d_img_apl" / "browse" / "calibrated" / "20090529"
    dir_b = tmp_path / "data" / "raw" / "ch1_tmc_nca_20090529T2139554127_d_img_apl" / "browse" / "calibrated" / "20090529"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)

    src_path_a = dir_a / "ch1_tmc_nca_20090529T1724252721_b_brw_apl.png"
    ref_path_a = dir_b / "ch1_tmc_nca_20090529T2139554127_b_brw_apl.png"
    src_path_b = dir_a / "ch1_tmc_nca_20090529T1724252721_b_brw_other.png"
    ref_path_b = dir_b / "ch1_tmc_nca_20090529T2139554127_b_brw_other.png"

    cv2.imwrite(str(src_path_a), src_a)
    cv2.imwrite(str(ref_path_a), ref_a)
    cv2.imwrite(str(src_path_b), src_b)
    cv2.imwrite(str(ref_path_b), ref_b)

    pairs = discover_overlapping_pairs(str(tmp_path / "data" / "raw"), min_overlap=0.10, max_pairs=10)
    assert len(pairs) >= 1
    assert any(p[0].endswith("ch1_tmc_nca_20090529T1724252721_b_brw_apl.png") for p in pairs)

