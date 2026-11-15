import os
import numpy as np
import pytest

from src.geospatial_validation import build_geographic_gcp_pairs, load_geometry_grid, parse_pds4_metadata

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chandrayan-2 dataset")
_GRID_FILE = os.path.join(_DATA_DIR, "ch2_ohr_ncp_20220128T0018149733_g_grd_d18.csv")
_LABEL_FILE = os.path.join(_DATA_DIR, "ch2_ohr_ncp_20220128T0018149733_d_img_d18.xml")


def test_build_geographic_gcp_pairs():
    source = np.array([
        [10.0, -89.0, 100.0, 200.0],
        [10.1, -89.1, 300.0, 400.0],
        [10.2, -89.2, 500.0, 600.0],
        [20.0, -80.0, 700.0, 800.0],
    ])
    reference = source.copy()
    reference[:, 2:] += 5.0
    source_points, reference_points, diagnostics = build_geographic_gcp_pairs(
        source, reference, max_distance_m=50.0, max_points=20
    )
    assert diagnostics["status"] == "valid"
    assert len(source_points) == 4
    assert np.allclose(reference_points - source_points, 5.0)


@pytest.mark.skipif(
    not os.path.exists(_GRID_FILE),
    reason="Real Chandrayaan-2 geometry grid file not present; skipping data-dependent integration test",
)
def test_load_real_geometry_grid():
    grid = load_geometry_grid(_GRID_FILE)
    assert grid.shape[1] == 4
    assert len(grid) > 1000


@pytest.mark.skipif(
    not os.path.exists(_LABEL_FILE),
    reason="Real Chandrayaan-2 PDS4 label file not present; skipping data-dependent integration test",
)
def test_parse_real_pds4_metadata():
    metadata = parse_pds4_metadata(_LABEL_FILE)
    assert metadata["gsd_m"] == 0.26
    assert metadata["focal_length_mm"] == 2080.0
    assert metadata["start_time"].startswith("2022-01-28T00:18:14")
    assert len(metadata["corners"]) >= 4
    assert metadata["incidence"] == 88.425473
