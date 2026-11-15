import os
import pytest
import numpy as np
import cv2

from src.features.dem_builder import (
    ChandrayaanCameraModel,
    get_camera_model,
    _triangulate_disparity_to_dem,
    compute_dem_uncertainty_map,
    compute_hillshade,
    export_point_cloud_ply,
    evaluate_dem_qc,
    compare_against_reference_dem,
    build_asp_bundle_adjust_manifest,
)
from src.features.stereo_lunar_adapter import StereoLunarAdapter


def test_camera_models_defaults_and_override():
    cam_tmc = get_camera_model({"stereo": {"sensor": "tmc"}})
    assert cam_tmc.sensor == "tmc"
    assert cam_tmc.focal_length_mm == 140.0
    assert cam_tmc.gsd_m == 5.0
    assert np.isclose(cam_tmc.focal_px, 140.0 / (7.0 * 1e-3))

    cam_ohrc = get_camera_model({"stereo": {"sensor": "ohrc"}})
    assert cam_ohrc.sensor == "ohrc"
    assert cam_ohrc.focal_length_mm == 2200.0
    assert cam_ohrc.gsd_m == 0.25


def test_triangulate_disparity_to_dem():
    # Synthetic disparity map with a slope
    disp = np.full((50, 50), 10.0, dtype=np.float32)
    disp[20:30, 20:30] = 15.0  # Crater rim (closer/higher relief)
    focal_px = 20000.0
    baseline_m = 93300.0
    gsd_m = 5.0

    dem = _triangulate_disparity_to_dem(disp, focal_px, baseline_m, gsd_m, min_disparity=0.5)
    assert dem.shape == (50, 50)
    # Higher disparity corresponds to closer terrain, i.e., positive relative elevation
    assert dem[25, 25] > dem[5, 5]


def test_compute_dem_uncertainty_map():
    disp = np.full((40, 40), 12.0, dtype=np.float32)
    disp[0:5, 0:5] = 0.0  # Invalid pixels
    focal_px = 20000.0
    baseline_m = 93300.0
    gsd_m = 5.0

    sigma_z = compute_dem_uncertainty_map(disp, focal_px, baseline_m, gsd_m, sigma_disp_px=0.5)
    assert sigma_z.shape == (40, 40)
    assert np.all(sigma_z[0:5, 0:5] == 0.0)
    assert np.all(sigma_z[10:30, 10:30] > 0.0)


def test_compute_hillshade():
    # Synthetic crater bowl DEM
    y, x = np.ogrid[-25:25, -25:25]
    dem = np.sqrt(x**2 + y**2).astype(np.float32) * 5.0
    hillshade = compute_hillshade(dem, gsd_m=5.0)

    assert hillshade.shape == (50, 50)
    assert hillshade.dtype == np.uint8
    assert np.min(hillshade) >= 0 and np.max(hillshade) <= 255
    # Non-trivial terrain relief must produce contrast
    assert np.std(hillshade) > 5.0


def test_export_point_cloud_ply(tmp_path):
    dem = np.full((30, 30), 100.0, dtype=np.float32)
    dem[10:20, 10:20] = 150.0
    ply_file = tmp_path / "test_terrain.ply"

    exported = export_point_cloud_ply(str(ply_file), dem, gsd_m=5.0)
    assert os.path.exists(exported)
    with open(exported, "r") as f:
        content = f.read()
    assert "ply" in content
    assert "element vertex 900" in content
    assert "property float z" in content


def test_evaluate_dem_qc():
    dem = np.full((50, 50), 200.0, dtype=np.float32)
    dem[10:40, 10:40] += 50.0  # Relief span 50m
    qc = evaluate_dem_qc(dem, gsd_m=5.0)

    assert qc["qc_grade"] == "ACCEPTED"
    assert qc["relief_span_m"] == 50.0
    assert qc["void_percentage"] == 0.0


def test_compare_against_reference_dem():
    # Ground truth reference DEM
    ref_dem = np.full((50, 50), 1000.0, dtype=np.float32)
    # Generated DEM with a known 2.5m bias and 0.5m noise
    gen_dem = ref_dem + 2.5

    res = compare_against_reference_dem(gen_dem, ref_dem)
    assert res["status"] == "VALIDATED"
    assert np.isclose(res["mean_bias_m"], 2.5, atol=0.01)
    assert np.isclose(res["median_bias_m"], 2.5, atol=0.01)
    assert np.isclose(res["rmse_m"], 2.5, atol=0.01)


def test_stereo_lunar_adapter_reconstruction(tmp_path):
    adapter = StereoLunarAdapter(camera_sensor="tmc")
    # Synthetic lunar crater stereo pair with slight horizontal disparity
    img_l = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(img_l, (60, 60), 30, 200, -1)
    cv2.circle(img_l, (60, 60), 10, 50, -1)

    # Right image shifted horizontally
    img_r = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(img_r, (55, 60), 30, 200, -1)
    cv2.circle(img_r, (55, 60), 10, 50, -1)

    ply_path = str(tmp_path / "stereo_crater.ply")
    result = adapter.reconstruct(img_l, img_r, ply_export_path=ply_path, max_processing_dim=120)

    assert "dem_m" in result
    assert "uncertainty_m" in result
    assert "qc" in result
    assert os.path.exists(ply_path)
    assert result["engine"] in ["calibrated_sgbm_triangulation", "stereolunar_deep_mast3r"]


def test_build_asp_bundle_adjust_manifest(tmp_path):
    images = ["/data/img1.tif", "/data/img2.tif", "/data/img3.tif"]
    out_dir = str(tmp_path / "asp_bundle")
    manifest = build_asp_bundle_adjust_manifest(images, out_dir)

    assert manifest["num_images"] == 3
    assert os.path.exists(manifest["image_list"])
