from src.evaluate.quality_gate import assess_quality

def test_quality_gate_grades():
    # Grade A: sub-half-pixel with high coverage
    res_a = assess_quality(rmse_pixels=0.3, coverage=0.8, inlier_ratio=0.9, inlier_count=500, total_matches=550)
    assert res_a['grade'] == 'A'
    assert res_a['passed'] is True
    assert res_a['confidence'] >= 0.85

    # Grade F: catastrophic error (non-overlapping)
    res_f = assess_quality(rmse_pixels=300.0, coverage=0.05, inlier_ratio=0.1, inlier_count=5, total_matches=50)
    assert res_f['grade'] == 'F'
    assert res_f['passed'] is False
    assert len(res_f['warnings']) > 0


def test_quality_gate_edge_cases():
    # Empty match case
    res_empty = assess_quality(rmse_pixels=float('inf'), coverage=0.0, inlier_ratio=0.0, inlier_count=0, total_matches=0)
    assert res_empty['grade'] == 'F'
    assert res_empty['passed'] is False


def test_rmse_failure_threshold_is_hard_rejection():
    result = assess_quality(
        rmse_pixels=17.39,
        coverage=0.83,
        inlier_ratio=0.50,
        inlier_count=2000,
        total_matches=4000,
        holdout_rmse=2.0,
        holdout_count=600,
    )
    assert result['grade'] == 'F'
    assert result['passed'] is False
    assert 'HIGH_RMSE_FAILURE' in result['reason_codes']


def test_holdout_failure_is_hard_rejection():
    result = assess_quality(
        rmse_pixels=0.5,
        coverage=0.8,
        inlier_ratio=0.8,
        inlier_count=100,
        total_matches=120,
        holdout_rmse=8.0,
        holdout_count=30,
    )
    assert result['grade'] == 'F'
    assert result['passed'] is False
    assert result['primary_reason'] == 'HOLDOUT_VALIDATION_FAILURE'


def test_symmetric_geometry_failure_is_hard_rejection():
    result = assess_quality(
        rmse_pixels=0.5,
        coverage=0.8,
        inlier_ratio=0.8,
        inlier_count=100,
        total_matches=120,
        symmetric_rmse=7.0,
    )
    assert result['grade'] == 'F'
    assert result['passed'] is False
    assert result['primary_reason'] == 'SYMMETRIC_GEOMETRY_FAILURE'
