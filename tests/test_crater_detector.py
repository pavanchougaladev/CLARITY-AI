"""
Tests for LunarCraterDetector (YOLO11) and Gate 3 Crater Integration
"""

import os
import numpy as np
import pytest
import cv2

from src.features.crater_detector import LunarCraterDetector, _ULTRALYTICS_AVAILABLE
from src.evaluate.quality_gate import assess_quality

pytestmark = pytest.mark.skipif(
    not _ULTRALYTICS_AVAILABLE,
    reason="ultralytics not installed; install 'ultralytics>=8.3.0' to run crater detector tests",
)


def create_synthetic_lunar_patch(size=640):
    """Generate a synthetic lunar surface patch with 5 circular craters."""
    img = np.full((size, size), 128, dtype=np.uint8)
    # Add noise
    noise = np.random.normal(0, 8, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw craters: dark bowl with bright rim
    craters = [
        (150, 150, 40),
        (450, 150, 30),
        (320, 320, 50),
        (150, 480, 35),
        (480, 480, 45)
    ]
    for cx, cy, r in craters:
        # Dark inner shadow
        cv2.circle(img, (cx, cy), r, 60, -1)
        # Bright sunlit rim
        cv2.circle(img, (cx - int(r*0.2), cy - int(r*0.2)), r, 210, 3)
    return img


def test_crater_detector_initialization():
    """Verify detector initializes with valid fallback and device."""
    detector = LunarCraterDetector(model_path="non_existent_model.pt")
    assert detector is not None
    assert detector.device in ["mps", "cuda", "0", "cpu"]


def test_crater_detector_inference():
    """Verify detect() returns correct dictionary schema."""
    detector = LunarCraterDetector(conf=0.10)
    synth = create_synthetic_lunar_patch(640)
    detections = detector.detect(synth)
    
    assert isinstance(detections, list)
    if len(detections) > 0:
        first = detections[0]
        assert "box" in first
        assert "centroid" in first
        assert "diameter_px" in first
        assert "confidence" in first
        assert "class_name" in first
        assert len(first["centroid"]) == 2


def test_get_crater_centroids():
    """Verify centroid coordinate extraction."""
    detector = LunarCraterDetector(conf=0.10)
    synth = create_synthetic_lunar_patch(640)
    centroids = detector.get_crater_centroids(synth)
    assert isinstance(centroids, np.ndarray)
    if len(centroids) > 0:
        assert centroids.shape[1] == 2


def test_compute_crater_coverage_pass():
    """Verify Gate 3 passes when tie-points are distributed across craters."""
    detector = LunarCraterDetector()
    
    # Mock 10 detected craters
    mock_craters = [
        {
            "centroid": (float(i * 50 + 25), float(i * 50 + 25)),
            "diameter_px": 30.0,
            "confidence": 0.9,
            "class_name": "crater",
            "box": (float(i * 50), float(i * 50), float(i * 50 + 30), float(i * 50 + 30))
        }
        for i in range(10)
    ]
    
    # Tie points placed inside 5 of the 10 craters (50% coverage >= 15%)
    tie_points = np.array([
        [25.0, 25.0],
        [75.0, 75.0],
        [125.0, 125.0],
        [175.0, 175.0],
        [225.0, 225.0],
    ], dtype=np.float32)
    
    cov = detector.compute_crater_coverage(tie_points, mock_craters)
    assert cov["total_craters"] == 10
    assert cov["covered_craters"] == 5
    assert cov["coverage_ratio"] == 0.5
    assert cov["passed_gate_3"] is True


def test_compute_crater_coverage_fail():
    """Verify Gate 3 fails when tie-points are heavily clustered in only 1 crater."""
    detector = LunarCraterDetector()
    
    mock_craters = [
        {
            "centroid": (float(i * 60 + 20), float(i * 60 + 20)),
            "diameter_px": 25.0,
            "confidence": 0.9,
            "class_name": "crater",
            "box": (float(i * 60), float(i * 60), float(i * 60 + 25), float(i * 60 + 25))
        }
        for i in range(10)
    ]
    
    # All 20 tie-points clustered in the very first crater (10% coverage < 15%)
    tie_points = np.random.normal(loc=20.0, scale=2.0, size=(20, 2)).astype(np.float32)
    
    cov = detector.compute_crater_coverage(tie_points, mock_craters)
    assert cov["total_craters"] == 10
    assert cov["covered_craters"] == 1
    assert cov["coverage_ratio"] == 0.10
    assert cov["passed_gate_3"] is False


def test_compute_crater_diameters_m():
    """Verify metric diameter conversion using GSD."""
    detector = LunarCraterDetector()
    mock_craters = [
        {"diameter_px": 100.0, "confidence": 0.85, "class_name": "crater", "centroid": (50.0, 50.0), "box": (0, 0, 100, 100)}
    ]
    # GSD for Chandrayaan-2 OHRC is ~0.25 m/px
    scaled = detector.compute_crater_diameters_m(mock_craters, gsd_m=0.25)
    assert np.isclose(scaled[0]["diameter_m"], 25.0)


def test_quality_gate_physical_crater_assessment():
    """Verify assess_quality incorporates physical crater coverage telemetry."""
    res = assess_quality(
        rmse_pixels=0.45,
        coverage=0.65,
        inlier_ratio=0.75,
        inlier_count=80,
        total_matches=100,
        crater_coverage=0.40,
        crater_count=20
    )
    assert res["details"]["crater_coverage"] == 0.40
    assert res["details"]["crater_count"] == 20
    assert "LOW_CRATER_COVERAGE" not in res["reason_codes"]

    # Low crater coverage failure test
    res_low = assess_quality(
        rmse_pixels=0.45,
        coverage=0.65,
        inlier_ratio=0.75,
        inlier_count=80,
        total_matches=100,
        crater_coverage=0.08,
        crater_count=20
    )
    assert "LOW_CRATER_COVERAGE" in res_low["reason_codes"]
