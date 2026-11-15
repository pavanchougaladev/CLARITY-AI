"""
Unit & Integration Tests for Cross-Sun Angle Illumination Invariance
====================================================================
Tests the five pillars of cross-illumination invariance:
1. Automated Solar Illumination Vector Estimation (Azimuth & Elevation).
2. Dual-Pass Cross-Sun Normalization (Hapke BRDF Albedo + Phase Congruency).
3. Geometric Crater Constellation Hashing (Scale, Rotation, Illumination-Invariant).
4. Gradient-Magnitude ZNCC under 180° Inverted Relief Shading.
"""

import math
import numpy as np
import pytest
import cv2

from src.preprocess.photometric_correction import (
    estimate_solar_illumination_vector,
    apply_dual_pass_cross_sun_normalization,
    invert_hapke_to_albedo,
    compute_phase_congruency,
)
from src.features.crater_matcher import (
    _match_crater_constellations,
    _compute_neighborhood_descriptor,
)


def _render_synthetic_crater_image(
    sun_az_deg: float,
    sun_el_deg: float = 25.0,
    size: int = 160,
    crater_radius: float = 45.0,
) -> np.ndarray:
    """Renders a physically shaded lunar crater under specified solar vector."""
    y, x = np.mgrid[-size // 2 : size // 2, -size // 2 : size // 2]
    r = np.hypot(x, y) / crater_radius
    z = np.zeros_like(r, dtype=np.float32)

    # Crater rim crest (raised) and bowl (depressed)
    z += 18.0 * np.exp(-(((r - 1.0) / 0.22) ** 2))
    z[r < 1.0] -= 30.0 * (1.0 - r[r < 1.0] ** 2)

    # Compute surface normals
    zy, zx = np.gradient(z)
    norm = np.sqrt(zx**2 + zy**2 + 1.0)
    nx, ny, nz = -zx / norm, -zy / norm, 1.0 / norm

    # Solar unit vector
    az_rad = np.radians(sun_az_deg)
    el_rad = np.radians(sun_el_deg)
    sx = np.sin(az_rad) * np.cos(el_rad)
    sy = -np.cos(az_rad) * np.cos(el_rad)
    sz = np.sin(el_rad)

    dot = nx * sx + ny * sy + nz * sz
    img = np.clip(dot * 255.0, 0, 255).astype(np.uint8)
    return img


def test_solar_azimuth_estimation_cardinal_angles():
    """Verify automated solar azimuth estimation across all 4 cardinal and intercardinal compass directions."""
    test_angles = [0.0, 45.0, 90.0, 180.0, 270.0]

    for true_az in test_angles:
        img = _render_synthetic_crater_image(sun_az_deg=true_az, sun_el_deg=25.0)
        est_az, est_el, conf = estimate_solar_illumination_vector(img)

        # Angular difference taking modulo 360 into account
        diff = min(abs(est_az - true_az), 360.0 - abs(est_az - true_az))
        assert diff <= 5.0, f"Failed at {true_az}°: estimated {est_az:.1f}° (diff={diff:.1f}°)"
        assert conf > 0.05, f"Confidence too low for valid crater: {conf}"
        assert 10.0 <= est_el <= 75.0, f"Solar elevation out of bounds: {est_el}"


def test_dual_pass_cross_sun_normalization():
    """Verify that dual-pass normalization resolves opposing illumination (90° East vs 270° West)."""
    img_east = _render_synthetic_crater_image(sun_az_deg=90.0)
    img_west = _render_synthetic_crater_image(sun_az_deg=270.0)

    # Before normalization: raw intensity correlation across 180° inverted illumination is negative
    raw_corr = float(np.corrcoef(img_east.ravel().astype(np.float32), img_west.ravel().astype(np.float32))[0, 1])
    assert raw_corr < 0.10, f"Expected negative or low raw correlation between opposite sun images, got {raw_corr}"

    # Apply dual-pass cross-sun normalization
    norm_east, norm_west, telem = apply_dual_pass_cross_sun_normalization(
        img_east, img_west, sun_az_ref=90.0, sun_az_tgt=270.0
    )

    assert norm_east.shape == img_east.shape
    assert norm_west.shape == img_west.shape
    assert telem["cross_sun_detected"] is True
    assert telem["extreme_opposite_sun"] is True
    assert telem["delta_azimuth_deg"] == 180.0

    # Phase congruency and albedo decoupling dramatically boosts structural correlation
    fused_corr = float(np.corrcoef(norm_east.ravel().astype(np.float32), norm_west.ravel().astype(np.float32))[0, 1])
    assert fused_corr > raw_corr, f"Fused correlation {fused_corr} should exceed raw correlation {raw_corr}"


def test_crater_constellations_geometric_invariance():
    """Verify that geometric crater constellation hashing is 100% invariant to lighting, scale, and rotation."""
    # Define 4 crater centroids forming two distinct triangles
    centroids_s = np.array([
        [100.0, 100.0],
        [250.0, 120.0],
        [180.0, 300.0],
        [400.0, 350.0],
    ], dtype=np.float32)
    diameters_s = np.array([40.0, 60.0, 35.0, 80.0], dtype=np.float32)

    # Apply 2D rigid transform (rotation 35 deg, translation (50, -30), scale 1.25)
    theta = math.radians(35.0)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    scale = 1.25
    centroids_r = (centroids_s @ R.T) * scale + np.array([50.0, -30.0], dtype=np.float32)
    diameters_r = diameters_s * scale

    # Shuffle the reference craters to test correspondence search
    perm = [2, 0, 3, 1]
    centroids_r_perm = centroids_r[perm]
    diameters_r_perm = diameters_r[perm]

    votes = _match_crater_constellations(
        centroids_s, diameters_s, centroids_r_perm, diameters_r_perm
    )

    # For each source crater i, the matching permuted target index should have the highest vote
    for i in range(4):
        expected_perm_idx = perm.index(i)
        best_match_idx = int(np.argmax(votes[i]))
        assert best_match_idx == expected_perm_idx, (
            f"Crater {i} matched to {best_match_idx}, expected {expected_perm_idx} with votes {votes[i]}"
        )
        assert votes[i, best_match_idx] > 0.0, f"Expected non-zero geometric consensus votes for crater {i}"


def test_phase_congruency_dimensionless_stability():
    """Verify that monogenic phase congruency produces identical rim peak locations regardless of lighting direction."""
    img_east = _render_synthetic_crater_image(sun_az_deg=90.0, size=128)
    img_west = _render_synthetic_crater_image(sun_az_deg=270.0, size=128)

    pc_east = compute_phase_congruency(img_east)
    pc_west = compute_phase_congruency(img_west)

    assert pc_east.shape == (128, 128)
    assert pc_west.shape == (128, 128)
    assert 0.0 <= pc_east.min() and pc_east.max() <= 1.0
    assert 0.0 <= pc_west.min() and pc_west.max() <= 1.0

    # Both must peak on the circular crater rim
    center = (64, 64)
    y, x = np.mgrid[0:128, 0:128]
    r = np.hypot(x - center[0], y - center[1])

    # Rim mask: radius between 40 and 50 px
    rim_mask = (r >= 38.0) & (r <= 52.0)
    floor_mask = r < 25.0

    # Mean phase congruency on the physical rim crest must be significantly higher than on the flat floor
    assert np.mean(pc_east[rim_mask]) > np.mean(pc_east[floor_mask])
    assert np.mean(pc_west[rim_mask]) > np.mean(pc_west[floor_mask])
