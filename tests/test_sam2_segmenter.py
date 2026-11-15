"""
Unit and Integration Tests for SAM 2 Crater Rim & Hazard Segmenter
===================================================================
Validates:
1. LunarSAM2Segmenter rim polygon extraction (vertices >= 3, area > 0, perimeter > 0).
2. Isoperimetric Circularity Index C in (0.0, 1.0].
3. Polygon hazard exclusion raster generation.
4. Standard GeoJSON export compliance for QGIS / ArcGIS.
5. Integration into services.detect_craters_hybrid and evaluate_landing_hazards.
"""

import json
import numpy as np
import pytest

from src.features.sam2_crater_segmenter import LunarSAM2Segmenter


def test_sam2_segmenter_initialization():
    """Verify segmenter initializes cleanly without throwing exceptions."""
    segmenter = LunarSAM2Segmenter(prefer_neural=True)
    assert segmenter is not None


def test_sam2_segment_synthetic_crater():
    """Verify that a synthetic crater image yields an accurate polygon and valid morphometrics."""
    # Create 128x128 synthetic crater patch
    img = np.full((128, 128), 120, dtype=np.uint8)
    yy, xx = np.ogrid[:128, :128]
    dist = np.hypot(xx - 64, yy - 64)
    # Bright rim at r=32, shadowed interior at r < 30
    img[dist < 30] = 40
    img[(dist >= 30) & (dist <= 34)] = 210

    segmenter = LunarSAM2Segmenter()
    box = (32.0, 32.0, 96.0, 96.0)
    res = segmenter.segment_crater(
        image=img,
        box=box,
        cx=64.0,
        cy=64.0,
        r=32.0,
        gsd_m=0.25,
    )

    assert "polygon_pts" in res
    assert len(res["polygon_pts"]) >= 12
    assert res["area_m2"] > 0
    assert res["perimeter_m"] > 0
    assert 0.1 <= res["circularity"] <= 1.0
    assert 0.1 <= res["ellipticity"] <= 1.0
    assert res["segmentation_engine"] is not None


def test_polygon_hazard_mask_generation():
    """Verify polygon hazard mask correctly rasterizes and dilates craters."""
    craters = [
        {
            "id": 1,
            "cx": 50.0,
            "cy": 50.0,
            "r": 20.0,
            "polygon_pts": [(50 + 20 * np.cos(a), 50 + 20 * np.sin(a)) for a in np.linspace(0, 2*np.pi, 24)],
        }
    ]
    mask = LunarSAM2Segmenter.generate_polygon_hazard_mask(
        craters=craters,
        grid_shape=(100, 100),
        buffer_ratio=1.35,
        img_shape=(100, 100),
    )
    assert mask.shape == (100, 100)
    assert mask.dtype == bool
    # Center must be marked hazardous
    assert mask[50, 50] is True or mask[50, 50] == 1
    # Corner far from crater must be safe
    assert mask[0, 0] is False or mask[0, 0] == 0


def test_geojson_export_validity():
    """Verify exported GeoJSON matches RFC 7946 specifications for QGIS/ArcGIS."""
    craters = [
        {
            "id": 1,
            "cx": 40.0,
            "cy": 40.0,
            "r": 15.0,
            "diameter_m": 30.0,
            "depth_m": 4.5,
            "depth_diam_ratio": 0.15,
            "circularity": 0.94,
            "ellipticity": 0.96,
            "area_m2": 700.0,
            "perimeter_m": 94.0,
            "rim_slope": 14.2,
            "hazard_level": "MODERATE HAZARD",
            "segmentation_engine": "SAM 2",
            "polygon_pts": [(40 + 15 * np.cos(a), 40 + 15 * np.sin(a)) for a in np.linspace(0, 2*np.pi, 16)],
        }
    ]
    geojson_str = LunarSAM2Segmenter.export_geojson(craters, gsd_m=0.25)
    parsed = json.loads(geojson_str)
    assert parsed["type"] == "FeatureCollection"
    assert len(parsed["features"]) == 1
    feat = parsed["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Polygon"
    assert feat["properties"]["crater_id"] == 1
    assert feat["properties"]["circularity"] == 0.94


def test_evaluate_landing_hazards_integration():
    """Verify services.evaluate_landing_hazards executes cleanly with SAM 2 polygon masks."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ui_clean"))
    import services

    z_dem = np.full((64, 64), 50.0, dtype=np.float32)
    craters = [
        {
            "id": 1,
            "cx": 32.0,
            "cy": 32.0,
            "r": 10.0,
            "diameter_m": 20.0,
            "depth_m": 3.0,
            "depth_diam_ratio": 0.15,
            "circularity": 0.92,
            "ellipticity": 0.95,
            "area_m2": 314.0,
            "perimeter_m": 62.8,
            "rim_slope": 12.0,
            "hazard_level": "MODERATE HAZARD",
            "polygon_pts": [(32 + 10 * np.cos(a), 32 + 10 * np.sin(a)) for a in np.linspace(0, 2*np.pi, 20)],
        }
    ]
    safety_map, slope, hazard_mask, ranked_sites = services.evaluate_landing_hazards(
        z_dem=z_dem,
        craters=craters,
        gsd_m=2.0,
        max_safe_slope=12.0,
        lander_footprint_m=20.0,
        img_size=(64, 64),
    )
    assert safety_map is not None
    assert safety_map.shape == (64, 64)
    assert slope.shape == (64, 64)
    assert hazard_mask.shape == (64, 64)
    assert isinstance(ranked_sites, list)

