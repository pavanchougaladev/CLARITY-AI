import pytest
from src.features.dem_builder import _SENSOR_PRESETS, ChandrayaanCameraModel

def test_sensor_geometry_consistency():
    """
    Verify physical optics relationship for all Chandrayaan sensors:
    GSD = (pixel_pitch_m * altitude_m) / focal_length_m
    """
    altitude_m = 100_000.0  # 100 km nominal circular orbit

    # 1. TMC-2 Validation
    tmc2 = _SENSOR_PRESETS["tmc"]
    f_m_tmc = tmc2.focal_length_mm * 1e-3
    pitch_m_tmc = tmc2.pixel_size_um * 1e-6
    expected_gsd_tmc = (pitch_m_tmc * altitude_m) / f_m_tmc
    assert abs(expected_gsd_tmc - tmc2.gsd_m) < 0.01, (
        f"TMC-2 GSD mismatch: computed {expected_gsd_tmc:.3f} m/px vs nominal {tmc2.gsd_m:.3f} m/px"
    )

    # 2. OHRC Validation
    ohrc = _SENSOR_PRESETS["ohrc"]
    f_m_ohrc = ohrc.focal_length_mm * 1e-3
    pitch_m_ohrc = ohrc.pixel_size_um * 1e-6
    expected_gsd_ohrc = (pitch_m_ohrc * altitude_m) / f_m_ohrc
    assert abs(expected_gsd_ohrc - ohrc.gsd_m) < 0.005, (
        f"OHRC GSD mismatch: computed {expected_gsd_ohrc:.3f} m/px vs nominal {ohrc.gsd_m:.3f} m/px"
    )

    # 3. Stereo Baseline Consistency (B/H ratio for ±25° convergence)
    # B/H = 2 * tan(25°) = 0.9326
    bh_ratio_tmc = tmc2.baseline_m / altitude_m
    assert abs(bh_ratio_tmc - 0.933) < 0.01, (
        f"TMC-2 B/H mismatch: computed {bh_ratio_tmc:.4f} vs expected 0.933"
    )
