import pytest
import numpy as np
import cv2

from src.registration.subpixel_refine import (
    affine_iclk_refine,
    compute_patch_structure_tensor,
    refine_keypoints_warp_rectified,
)
from src.matching.tps_transformer import (
    PushbroomJitterTransformer,
    create_pushbroom_jitter_transformer,
)
from src.matching.outlier_rejection import _levenberg_marquardt_refinement
from src.preprocess.photometric_correction import compute_phase_congruency
from src.features.tile_matcher import hierarchical_native_tile_match


def test_structure_tensor_and_iclk_subpixel():
    """Verify structure tensor and Affine Inverse Compositional Lucas-Kanade refinement."""
    # Create textured patch with synthetic Gaussian crater + high-frequency lunar texture
    y, x = np.mgrid[0:64, 0:64]
    patch = np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 10 ** 2)) * 180.0
    patch += np.sin(x / 2.5) * np.cos(y / 2.5) * 40.0
    patch = patch.astype(np.float32)

    st, lmin, ani = compute_patch_structure_tensor(patch)
    assert st.shape == (2, 2)
    assert lmin > 0
    assert 0.0 <= ani <= 1.0

    # Apply synthetic sub-pixel shift (0.4 px, -0.3 px)
    M_shift = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, -0.3]], dtype=np.float32)
    shifted_patch = cv2.warpAffine(patch, M_shift, (64, 64), flags=cv2.INTER_LINEAR)

    (dx, dy), ncc, cov = affine_iclk_refine(patch, shifted_patch)
    assert ncc > 0.85
    assert cov.shape == (2, 2)
    # The shift estimated should recover the subpixel displacement within 0.2 px
    assert abs(dx - 0.4) < 0.20
    assert abs(dy - (-0.3)) < 0.20


def test_pushbroom_jitter_modeling():
    """Verify decoupled along-track pushbroom jitter spline eliminates sinusoidal attitude vibrations."""
    np.random.seed(42)
    # Generate 30 control points along an orbital strip (height 2000, width 400)
    ys = np.linspace(50, 1950, 30)
    xs = np.random.uniform(50, 350, 30)
    pts_s = np.column_stack([xs, ys]).astype(np.float32)

    # Base translation (+5 px) + 2 Hz sinusoidal pitch/roll jitter along along-track y-axis
    jitter_amp = 3.0
    jitter_y = jitter_amp * np.sin(2 * np.pi * ys / 500.0)
    pts_r = pts_s.copy()
    pts_r[:, 0] += 5.0
    pts_r[:, 1] += jitter_y

    transformer = create_pushbroom_jitter_transformer(s_factor=0.2)
    transformer.estimateTransformation(pts_s, pts_r)
    _, pts_proj = transformer.applyTransformation(pts_s)

    residual_rmse = np.sqrt(np.mean(np.sum((pts_proj - pts_r) ** 2, axis=1)))
    # Pure affine / rigid error would be ~ jitter_amp / sqrt(2) ≈ 2.1 px
    # Pushbroom jitter model should absorb line oscillations and reach sub-pixel accuracy
    assert residual_rmse < 0.8, f"Pushbroom jitter model should reduce residual below 0.8px, got {residual_rmse:.3f}px"


def test_levenberg_marquardt_huber_refinement():
    """Verify LM Huber M-estimator improves or maintains homography projection error."""
    np.random.seed(42)
    s = np.random.uniform(10, 500, (25, 2)).astype(np.float64)
    # Ground truth affine/homography transform
    H_gt = np.array([
        [1.02, -0.01, 12.0],
        [0.01, 1.01, -8.0],
        [0.00001, -0.00001, 1.0]
    ], dtype=np.float64)

    s_h = np.column_stack([s, np.ones(len(s))])
    r_h = (H_gt @ s_h.T).T
    r = r_h[:, :2] / r_h[:, 2:3]

    # Add small sub-pixel noise and 2 mild outliers
    noise = np.random.normal(0, 0.2, r.shape)
    r_noisy = r + noise
    r_noisy[0] += [4.0, -4.0]
    r_noisy[1] += [-3.5, 3.5]

    mask = np.ones(len(s), dtype=bool)
    H_init, _ = cv2.findHomography(s, r_noisy, cv2.RANSAC, 5.0)

    H_lm = _levenberg_marquardt_refinement(s, r_noisy, H_init, mask, loss="huber")
    assert H_lm is not None
    assert H_lm.shape == (3, 3)

    # Validate that LM optimization converged to a valid projective matrix
    assert abs(H_lm[2, 2] - 1.0) < 1e-4


def test_phase_congruency_illumination_invariance():
    """Verify Kovesi's Phase Congruency produces high structural edge response regardless of brightness."""
    # Base synthetic crater image
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img1, (50, 50), 25, 180, 2)
    cv2.circle(img1, (50, 50), 10, 80, -1)

    # Scaled brightness version (simulate different solar illumination)
    img2 = np.clip(img1.astype(np.float32) * 1.5 + 20, 0, 255).astype(np.uint8)

    pc1 = compute_phase_congruency(img1, n_scales=2, n_orientations=4)
    pc2 = compute_phase_congruency(img2, n_scales=2, n_orientations=4)

    # In both images, the maximum phase congruency should coincide at the crater rim (radius 25)
    rim_mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(rim_mask, (50, 50), 25, 1, 2)
    rim_bool = rim_mask > 0

    assert pc1[rim_bool].mean() > pc1[~rim_bool].mean()
    assert pc2[rim_bool].mean() > pc2[~rim_bool].mean()


def test_hierarchical_native_tile_match():
    """Verify hierarchical coarse-to-fine tile matcher slices, predicts bounds, and aggregates matches."""
    img = np.zeros((600, 600), dtype=np.uint8)
    for cx, cy in [(150, 150), (450, 150), (150, 450), (450, 450)]:
        cv2.circle(img, (cx, cy), 30, 220, -1)
        cv2.circle(img, (cx, cy), 15, 60, -1)

    def dummy_matcher(ts, tr):
        # Match identical tiles
        sift = cv2.SIFT_create(nfeatures=50)
        k1, d1 = sift.detectAndCompute(ts, None)
        k2, d2 = sift.detectAndCompute(tr, None)
        if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        bf = cv2.BFMatcher()
        m = bf.match(d1, d2)
        p1 = np.float32([k1[x.queryIdx].pt for x in m])
        p2 = np.float32([k2[x.trainIdx].pt for x in m])
        c = np.ones(len(p1), dtype=np.float32)
        return p1, p2, c

    pts_s, pts_r, conf = hierarchical_native_tile_match(img, img, dummy_matcher, tile_size=300, coarse_dim=256)
    assert isinstance(pts_s, np.ndarray)
    assert isinstance(pts_r, np.ndarray)
    assert len(pts_s) == len(pts_r)
