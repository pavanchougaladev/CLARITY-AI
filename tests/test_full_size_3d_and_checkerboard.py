"""
Tests for 100% Full-Size 3D Map Rendering and Sub-Pixel Checkerboard Inspection Suite.
"""

import numpy as np
import cv2
import pytest
import os
import tempfile

from ui.app import create_checkerboard, compute_seam_alignment_continuity, generate_dem_preview
from src.features.dem_builder import (
    get_camera_model,
    export_point_cloud_ply,
    export_dem_geotiff,
    export_mesh_obj,
    despike_and_clamp_lunar_slopes,
    compute_photoclinometry_sfs,
)
from src.matching.tps_transformer import (
    ScipyThinPlateSplineTransformer,
    PushbroomJitterTransformer,
)


def test_checkerboard_and_seam_continuity():
    """Verify sub-pixel checkerboard generation and seam continuity scoring."""
    # Synthetic lunar crater image
    h, w = 256, 256
    img1 = np.full((h, w), 120, dtype=np.uint8)
    cv2.circle(img1, (128, 128), 40, 220, -1)
    cv2.circle(img1, (128, 128), 20, 60, -1)

    # Identical image
    img2 = img1.copy()

    # Checkerboard with seams
    checker = create_checkerboard(img1, img2, tiles=16, show_seams=True, seam_color=(0, 240, 255))
    assert checker.shape == (h, w, 3)
    assert checker.dtype == np.uint8

    # Continuity metric on perfectly aligned images
    metrics = compute_seam_alignment_continuity(img1, img2, tiles=16)
    assert "coherence_pct" in metrics
    assert "mean_seam_disparity_px" in metrics
    assert metrics["coherence_pct"] >= 95.0
    assert metrics["is_subpixel_locked"] is True

    # Shifted image (misaligned)
    M = np.float32([[1, 0, 15], [0, 1, 15]])
    img_shifted = cv2.warpAffine(img1, M, (w, h))
    metrics_shifted = compute_seam_alignment_continuity(img_shifted, img2, tiles=16)
    assert metrics_shifted["mean_seam_disparity_px"] > metrics["mean_seam_disparity_px"]


def test_100_percent_full_size_dem_preview():
    """Verify 100% full-size 3D surface map render from co-registered stereo pair."""
    h, w = 320, 320
    # Left & right stereo pair with parallax
    img_ref = np.full((h, w), 100, dtype=np.uint8)
    cv2.circle(img_ref, (160, 160), 50, 200, -1)

    img_warped = np.full((h, w), 100, dtype=np.uint8)
    # 3-pixel parallax displacement
    cv2.circle(img_warped, (163, 160), 50, 200, -1)

    fig, dem_m, tex, qc = generate_dem_preview(
        img_warped,
        img_ref,
        resolution_mode="Full-Size 100% Native",
        vertical_exaggeration=1.5,
        colorscale="Lunar Texture (Photorealistic)",
        sensor_name="tmc",
    )

    # Must preserve full resolution (320x320)
    assert dem_m.shape == (h, w)
    assert tex.shape == (h, w)
    assert qc["is_full_size"] is True
    assert qc["dim_w"] == w
    assert qc["dim_h"] == h
    assert qc["gsd_m"] > 0
    assert qc["relief_span_m"] >= 0.0

    # Verify Plotly figure structure
    assert len(fig.data) == 1
    assert fig.data[0].type == "surface"


def test_3d_product_exporters():
    """Verify PLY, GeoTIFF, and Wavefront OBJ exports on 3D terrain."""
    h, w = 64, 64
    dem_m = np.random.uniform(50.0, 350.0, (h, w)).astype(np.float32)
    tex = np.full((h, w), 150, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. PLY
        ply_path = os.path.join(tmpdir, "test.ply")
        export_point_cloud_ply(ply_path, dem_m, gsd_m=5.0, texture=tex, subsample_step=2)
        assert os.path.exists(ply_path)
        assert os.path.getsize(ply_path) > 100

        # 2. GeoTIFF
        tif_path = os.path.join(tmpdir, "test.tif")
        export_dem_geotiff(dem_m, tif_path, gsd_m=5.0)
        assert os.path.exists(tif_path)
        assert os.path.getsize(tif_path) > 100

        # 3. Wavefront OBJ
        obj_path = os.path.join(tmpdir, "test.obj")
        export_mesh_obj(obj_path, dem_m, gsd_m=5.0, subsample_step=2)
        assert os.path.exists(obj_path)
        assert os.path.getsize(obj_path) > 100


def test_non_rigid_warping_dimension_preservation():
    """Verify PushbroomJitter and TPS transformers warp with explicit output_shape."""
    src_img = np.full((200, 200), 100, dtype=np.uint8)
    cv2.circle(src_img, (100, 100), 30, 255, -1)

    pts_src = np.array([[50, 50], [150, 50], [150, 150], [50, 150], [100, 100]], dtype=np.float32)
    pts_ref = pts_src + np.array([2.0, 1.0], dtype=np.float32)

    # 1. ScipyThinPlateSplineTransformer
    tps = ScipyThinPlateSplineTransformer()
    tps.estimateTransformation(pts_src, pts_ref)
    warped_tps = tps.warpImage(src_img, output_shape=(240, 240))
    assert warped_tps.shape[:2] == (240, 240)

    # 2. PushbroomJitterTransformer
    pb = PushbroomJitterTransformer()
    pb.estimateTransformation(pts_src, pts_ref)
    warped_pb = pb.warpImage(src_img, output_shape=(240, 240))
    assert warped_pb.shape[:2] == (240, 240)
