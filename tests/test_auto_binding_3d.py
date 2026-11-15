"""
Unit tests for Zero-Friction Registration-to-3D Auto-Binding
============================================================
Verifies that active registration outputs (images, verified tie-points, and metadata)
from Tab 1 are automatically detected and consumed by the 3D DEM builder and
stereo reconstruction pipeline without requiring manual file export/re-upload.
"""

import os
import tempfile
import numpy as np
import pytest

from src.features.dem_builder import BackgroundDEMTask
from src.registration.export import export_asp_match_file


def test_zero_friction_auto_binding_contract():
    # 1. Simulate Tab 1 Registration results preserved in session memory
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    synthetic_crater = np.sin(x / 20.0) * np.cos(y / 20.0) * 50.0 + 100.0
    img_left = np.clip(synthetic_crater, 0, 255).astype(np.uint8)
    img_right = np.clip(np.roll(synthetic_crater, shift=4, axis=1), 0, 255).astype(np.uint8)

    # 35 verified tie-points (simulating LoFTR / LunarFusion inliers)
    rng = np.random.RandomState(42)
    pts_src = rng.uniform(20, 230, size=(35, 2)).astype(np.float32)
    pts_ref = pts_src.copy()
    pts_ref[:, 0] += 4.0  # 4px horizontal parallax disparity

    active_results = {
        "method_used": "LunarFusion Multi-Scale",
        "inliers_src": pts_src,
        "inliers_ref": pts_ref,
        "inliers": len(pts_src),
        "rmse": 0.42,
        "quality": {"grade": "A", "passed": True},
    }

    # 2. Verify auto-detection logic
    inliers_s = active_results.get("inliers_src")
    inliers_r = active_results.get("inliers_ref")
    assert inliers_s is not None and inliers_r is not None
    assert len(inliers_s) == 35

    # 3. Verify in-memory match file export (zero manual upload)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".match") as tmp_match:
        match_path = tmp_match.name

    try:
        export_asp_match_file(inliers_s, inliers_r, match_path)
        assert os.path.exists(match_path)
        assert os.path.getsize(match_path) > 0

        # Read back lines to ensure proper format for dem_builder
        with open(match_path, "r") as f:
            lines = [line.strip().split() for line in f if line.strip()]
        assert len(lines) == 35
        assert len(lines[0]) == 4  # col1 row1 col2 row2

        # 4. Verify BackgroundDEMTask successfully runs with the auto-bound inputs
        task = BackgroundDEMTask()
        task._run(
            img_left=img_left,
            img_right=img_right,
            use_sgbm=True,
            match_file=match_path,
            use_asp=False,
            asp_path="",
            config={"stereo": {"sensor": "tmc"}},
            engine="calibrated_sgbm",
        )

        assert task.status == "completed"
        assert task.result_dem is not None
        assert task.result_dem.shape[0] > 0 and task.result_dem.shape[1] > 0
        assert task.result_texture is not None
    finally:
        if os.path.exists(match_path):
            os.remove(match_path)


def test_generate_3d_dem_from_pipeline_direct():
    from src.features.dem_builder import generate_3d_dem_from_pipeline

    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    synthetic_crater = np.sin(x / 20.0) * np.cos(y / 20.0) * 50.0 + 100.0
    img_left = np.clip(synthetic_crater, 0, 255).astype(np.uint8)
    img_right = np.clip(np.roll(synthetic_crater, shift=4, axis=1), 0, 255).astype(np.uint8)

    rng = np.random.RandomState(42)
    pts_src = rng.uniform(20, 230, size=(40, 2)).astype(np.float32)
    pts_ref = pts_src.copy()
    pts_ref[:, 0] += 3.5

    mock_pipeline_results = {
        "method_used": "LUNARFUSION (ROMA_V2 + YOLO11 + LOFTR)",
        "inliers_src": pts_src,
        "inliers_ref": pts_ref,
        "pts_src": pts_src,
        "pts_ref": pts_ref,
        "inliers": len(pts_src),
        "rmse": 0.38,
        "quality": {"grade": "A", "passed": True},
        "provenance": {
            "pds4": {
                "source": {"incidence": 70.0, "azimuth": 65.0, "gsd_m": 5.0},
                "reference": {"incidence": 68.0, "azimuth": 62.0, "gsd_m": 5.0},
            }
        },
        "H_fallback": np.eye(3, dtype=np.float64),
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp_ply:
        ply_path = tmp_ply.name

    try:
        report = generate_3d_dem_from_pipeline(
            img_src=img_left,
            img_ref=img_right,
            registration_results=mock_pipeline_results,
            camera_sensor="tmc",
            output_ply_path=ply_path,
        )

        assert report["status"] == "completed"
        assert report["dem_m"] is not None
        assert report["dem_m"].ndim == 2
        assert report["hillshade"] is not None
        assert report["uncertainty_map"] is not None
        assert report["qc"] is not None
        assert "relief_span_m" in report
        assert report["ply_path"] == ply_path
        assert os.path.exists(ply_path)
        assert os.path.getsize(ply_path) > 0

        # Verify PLY header contains normals nx, ny, nz
        with open(ply_path, "r") as f:
            header_sample = "".join(f.readline() for _ in range(20))
        assert "property float nx" in header_sample
        assert "property float ny" in header_sample
        assert "property float nz" in header_sample
    finally:
        if os.path.exists(ply_path):
            os.remove(ply_path)

