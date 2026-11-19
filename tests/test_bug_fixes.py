import pytest
import numpy as np
import cv2
from src.registration.subpixel_refine import refine_subpixel
from src.features.classical_descriptors import phase_congruency_map
from src.preprocess.shadow_normalization import _homomorphic_filter, normalize_illumination
from src.matching.outlier_rejection import _multipass_refine
from src.mosaic import MosaicEngine
import sys
import os
sys.path.insert(0, os.path.abspath("client"))

from components.visualizations import interactive_match_plot


def test_subpixel_refine_non_square():
    """Verify subpixel refinement works with non-square rectangular patches (Hann window width/height order)."""
    h, w = 32, 64
    patch_s = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(patch_s, (w // 2, h // 2), 8, 255, -1)
    
    # Slight shift
    patch_r = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(patch_r, (w // 2 + 1, h // 2), 8, 255, -1)
    
    shift, response = refine_subpixel(patch_s, patch_r)
    assert isinstance(shift, tuple)
    assert len(shift) == 2
    assert response >= 0.0


def test_crater_hybrid_narrow_strip():
    """Aspect-preserved narrow strips (e.g. 512x61) must not crash the crater
    fallback sampler (regression: uniform(60, w-60) with high < low)."""
    import services

    for shape in [(512, 61), (61, 512), (512, 512)]:
        img = np.full(shape, 120, dtype=np.uint8)
        craters = services.detect_craters_hybrid(img, conf=0.18, gsd=0.25)
        assert len(craters) > 0, f"no craters produced for shape {shape}"
        h_img, w_img = shape
        for c in craters:
            assert 0 <= c["cx"] < w_img
            assert 0 <= c["cy"] < h_img
            assert 0 < c["r"] <= 0.5 * min(w_img, h_img)


def test_phase_congruency_constant_image():
    """Verify phase congruency map does not divide by zero on all-zero or constant images."""
    black_img = np.zeros((64, 64), dtype=np.uint8)
    pc = phase_congruency_map(black_img)
    assert pc.shape == (64, 64)
    assert not np.isnan(pc).any()
    assert pc.max() <= 1.0


def test_homomorphic_filter_black_image():
    """Verify homomorphic filtering handles black or uniform images properly without complex number artifacts."""
    black_img = np.zeros((64, 64), dtype=np.uint8)
    filtered = _homomorphic_filter(black_img)
    assert filtered.shape == (64, 64)
    assert filtered.dtype == np.uint8
    assert not np.isnan(filtered).any()


def test_multipass_refine_degenerate_points():
    """Verify _multipass_refine handles fewer than 4 points cleanly without crashing."""
    pts_s = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    pts_r = np.array([[11, 11], [21, 21], [31, 31]], dtype=np.float32)
    mask = np.array([True, True, True], dtype=bool)
    
    H, refined_mask = _multipass_refine(pts_s, pts_r, mask)
    assert H is None
    assert np.array_equal(refined_mask, mask)


def test_mosaic_bounds_clipping():
    """Verify MosaicEngine safety bounds clamp correctly on single and multiple images."""
    engine = MosaicEngine()
    img1 = np.ones((50, 50), dtype=np.uint8) * 100
    engine.add_image(img1, "img1")
    
    mosaic_out, meta = engine.compute_mosaic(base_idx=0)
    assert meta["success"] is True
    assert mosaic_out.shape == (50, 50)


def test_interactive_match_plot():
    """Verify interactive match plot generates figure cleanly without mask errors."""
    img1 = np.zeros((100, 100), dtype=np.uint8)
    img2 = np.zeros((100, 100), dtype=np.uint8)
    pts1 = np.array([[10, 10], [50, 50]], dtype=np.float32)
    pts2 = np.array([[15, 15], [55, 55]], dtype=np.float32)
    
    fig = interactive_match_plot(img1, img2, pts1, pts2)
    assert fig is not None
    assert len(fig.data) > 0


def test_upsampled_dft_fractional_recovery():
    """Verify upsampled DFT phase correlation recovers fractional shifts across [0, 1] without peak-locking bias."""
    from src.registration.subpixel_refine import upsampled_dft_phase_correlation
    size = 64
    y, x = np.mgrid[-32:32, -32:32]
    base = 120.0 - 80.0 * np.exp(-(x**2 + y**2) / 100.0) + 30.0 * np.exp(-((np.sqrt(x**2 + y**2) - 14.0)**2) / 10.0)

    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        F = np.fft.fft2(base)
        ky = np.fft.fftfreq(size).reshape(-1, 1)
        kx = np.fft.fftfreq(size).reshape(1, -1)
        shift_phase = np.exp(-2j * np.pi * (kx * s + ky * s))
        shifted = np.real(np.fft.ifft2(F * shift_phase))

        (est_dx, est_dy), resp = upsampled_dft_phase_correlation(base, shifted, upsample_factor=100)
        assert abs(est_dx - s) < 0.05
        assert abs(est_dy - s) < 0.05
        assert resp > 0.8

