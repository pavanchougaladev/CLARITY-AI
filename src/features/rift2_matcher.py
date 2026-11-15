"""
rift2_matcher.py
================
RIFT2 — Radiation-Variation Insensitive Feature Transform v2
=============================================================

RIFT2 is specifically designed for multi-modal remote sensing image
registration where images have vastly different radiation characteristics
(e.g., SAR vs optical, thermal vs visible).

Research paper benchmark (Table 3) — TARGET PERFORMANCE:
  OHRC-NAC Equatorial : RMSE_X=1.5033, RMSE_Y=1.1888, T=36.88s  ← 2nd best accuracy
  IIRS-WAC Equatorial : RMSE_X=1.4056, RMSE_Y=1.3579, T=7.39s
  IIRS-WAC Polar      : RMSE_X=0.7529, RMSE_Y=1.1054, T=12.72s
  OHRC-NAC Polar      : FAILED (NA) — SuperGlue needed for polar
  DFSAR-SELENE        : FAILED — SuperGlue needed for SAR-optical

Key paper findings:
  - RIFT2 works WITHOUT preprocessing (radiation-variation insensitive)
  - Suitable for cross-spectral pairs (optical, infrared, SAR)
  - Superior to SIFT/ASIFT/AKAZE when modalities differ
  - Fails on extreme polar distortions and SAR-optical pairs

Algorithm (Li et al., 2023. "RIFT2: Speeding-up RIFT with rotation-invariance"):
  1. Phase Congruency (PC) feature map — illumination & contrast invariant
  2. PC-based keypoint detection via local maxima with NMS
  3. Maximum Index Map (MIM) — rotation-invariant descriptor computation
  4. BFMatcher on MIM descriptors with ratio-test filtering
  5. RANSAC geometric verification

Original RIFT paper:
  Li, J., Hu, Q., Ai, M. (2020). RIFT: Multi-modal image matching based on
  radiation-variation insensitive feature transform. IEEE TIP, 29, 3296–3310.

RIFT2 speed-up paper:
  Li, J., Shi, P., Hu, Q., Zhang, Y. (2023). RIFT2: Speeding-up RIFT with
  a new rotation-invariance technique. arXiv.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base_matcher import BaseLunarMatcher, sanitize_lunar_input

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase Congruency (PC) — Illumination-Invariant Feature Map
# ─────────────────────────────────────────────────────────────────────────────

def compute_phase_congruency_map(
    img: np.ndarray,
    n_scales: int = 4,
    n_orientations: int = 6,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_on_f: float = 0.55,
    k_threshold: float = 2.0,
) -> np.ndarray:
    """
    Compute Phase Congruency feature map (Kovesi's Local Energy Model).

    Phase Congruency is DIMENSIONLESS — invariant to:
    - Image brightness
    - Contrast
    - Solar incidence angle (critical for lunar imagery)
    - Sensor modality (works for optical AND SAR)

    This is the core innovation of RIFT/RIFT2: by using PC instead of
    gradient magnitude, features are detected at the same locations regardless
    of whether the image is optical, SAR, or infrared.

    Returns PC map in [0, 1] highlighting structurally significant edges/crater rims.
    """
    img_f = img.astype(np.float32)
    H, W = img_f.shape[:2]

    # Pre-filter noise
    img_f = cv2.GaussianBlur(img_f, (3, 3), 0.5)

    total_energy = np.zeros((H, W), dtype=np.float64)
    total_amplitude = np.zeros((H, W), dtype=np.float64)

    for o in range(n_orientations):
        angle = o * np.pi / n_orientations
        even_sum = np.zeros((H, W), dtype=np.float64)
        odd_sum = np.zeros((H, W), dtype=np.float64)
        amp_sum = np.zeros((H, W), dtype=np.float64)

        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            sigma = sigma_on_f * wavelength
            ksize = int(max(3, round(3.0 * sigma))) * 2 + 1
            ksize = min(ksize, min(H, W) // 2 * 2 - 1)
            ksize = max(ksize, 3)

            # Even Gabor (cosine — captures edge phase)
            kernel_e = cv2.getGaborKernel(
                (ksize, ksize), sigma, angle, wavelength, 0.5, 0.0, ktype=cv2.CV_64F
            )
            # Odd Gabor (sine — 90° phase shifted)
            kernel_o = cv2.getGaborKernel(
                (ksize, ksize), sigma, angle, wavelength, 0.5, np.pi / 2.0, ktype=cv2.CV_64F
            )

            resp_e = cv2.filter2D(img_f.astype(np.float64), cv2.CV_64F, kernel_e)
            resp_o = cv2.filter2D(img_f.astype(np.float64), cv2.CV_64F, kernel_o)

            amp = np.sqrt(resp_e ** 2 + resp_o ** 2)
            amp_sum += amp
            even_sum += resp_e
            odd_sum += resp_o

        # Local energy for this orientation
        energy_o = np.sqrt(even_sum ** 2 + odd_sum ** 2)
        total_energy += energy_o
        total_amplitude += amp_sum

    # Noise threshold from Rayleigh distribution (Kovesi's method)
    mean_amp = np.mean(total_amplitude)
    T_noise = k_threshold * mean_amp * 0.7213  # Rayleigh correction factor

    # Phase congruency = energy / (amplitude + epsilon) with noise floor
    pc = np.maximum(0.0, total_energy - T_noise) / (total_amplitude + 1e-6)
    pc = np.clip(pc, 0.0, 1.0).astype(np.float32)

    return pc


# ─────────────────────────────────────────────────────────────────────────────
# Maximum Index Map (MIM) — Rotation-Invariant Descriptor (RIFT2 Core)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mim(img: np.ndarray, n_orientations: int = 6) -> np.ndarray:
    """
    Compute Maximum Index Map (MIM) — RIFT2's rotation-invariant descriptor basis.

    For each pixel, computes responses from oriented Log-Gabor filters at
    N orientations and assigns the index of the MAXIMUM response orientation.
    This creates an orientation index map that is rotation-invariant when
    histogrammed over a patch (rotating the patch just cyclically permutes indices).

    Returns:
        mim : (H, W) uint8 with values in [0, n_orientations)
    """
    H, W = img.shape[:2]
    img_f = img.astype(np.float32)

    # Build oriented filter responses
    responses = np.zeros((H, W, n_orientations), dtype=np.float32)

    for o in range(n_orientations):
        angle = o * np.pi / n_orientations
        # Medium-scale Gabor for stable descriptor
        sigma = 3.0
        wavelength = 6.0
        ksize = int(3 * sigma) * 2 + 1
        ksize = min(ksize, min(H, W) // 2 * 2 - 1)
        ksize = max(ksize, 5)

        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, angle, wavelength, 0.5, 0.0, ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(img_f, cv2.CV_32F, kernel)
        responses[:, :, o] = np.abs(resp)

    # Maximum Index Map: for each pixel, which orientation has max response?
    mim = np.argmax(responses, axis=2).astype(np.uint8)
    return mim


def extract_mim_descriptor(
    mim: np.ndarray,
    kp_x: float,
    kp_y: float,
    patch_size: int = 32,
    n_orientations: int = 6,
    n_rings: int = 6,
    n_sectors: int = 6,
) -> Optional[np.ndarray]:
    """
    Extract RIFT2 MIM histogram descriptor at a keypoint.

    Divides a circular patch into n_rings × n_sectors cells and computes
    orientation histograms in each cell. The histogram is rotation-invariant
    because rotating the patch shifts MIM indices cyclically.

    Descriptor dimension: n_rings × n_sectors × n_orientations
    = 6 × 6 × 6 = 216 dims (default)
    """
    H, W = mim.shape[:2]
    cx, cy = int(round(kp_x)), int(round(kp_y))
    half = patch_size // 2

    # Bounds check
    if cx - half < 0 or cy - half < 0 or cx + half >= W or cy + half >= H:
        return None

    patch = mim[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
    ph, pw = patch.shape[:2]

    # Polar coordinate grid relative to patch center
    yy, xx = np.mgrid[0:ph, 0:pw]
    yy = yy - (ph / 2.0)
    xx = xx - (pw / 2.0)
    dist = np.sqrt(xx ** 2 + yy ** 2) + 1e-8
    angle_map = (np.arctan2(yy, xx) % (2 * np.pi))  # [0, 2pi]

    max_dist = np.sqrt(2) * half

    desc = np.zeros(n_rings * n_sectors * n_orientations, dtype=np.float32)
    bin_idx = 0

    for ring in range(n_rings):
        r_min = (ring / n_rings) * max_dist
        r_max = ((ring + 1) / n_rings) * max_dist
        ring_mask = (dist >= r_min) & (dist < r_max)

        for sector in range(n_sectors):
            a_min = (sector / n_sectors) * 2 * np.pi
            a_max = ((sector + 1) / n_sectors) * 2 * np.pi
            sector_mask = (angle_map >= a_min) & (angle_map < a_max)

            cell_mask = ring_mask & sector_mask
            cell_mim = patch[cell_mask].astype(np.int32)

            # Orientation histogram for this cell
            hist = np.bincount(cell_mim, minlength=n_orientations)[:n_orientations]
            desc[bin_idx: bin_idx + n_orientations] = hist.astype(np.float32)
            bin_idx += n_orientations

    # L2 normalize
    norm = np.linalg.norm(desc) + 1e-8
    return (desc / norm).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PC-Based Keypoint Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_pc_keypoints(
    pc_map: np.ndarray,
    max_keypoints: int = 1000,
    nms_radius: int = 5,
    min_score: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect keypoints at Phase Congruency local maxima with NMS.

    Returns:
        kps    : (N, 2) float32 (x, y) coordinates
        scores : (N,) float32 confidence values
    """
    H, W = pc_map.shape[:2]

    # Local maxima via dilation comparison
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(pc_map, size=nms_radius * 2 + 1)
    is_max = (pc_map == local_max) & (pc_map >= min_score)

    # Flatten and sort by score
    yx = np.argwhere(is_max)  # (K, 2) in row, col order
    if len(yx) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    scores = pc_map[yx[:, 0], yx[:, 1]]
    order = np.argsort(-scores)
    yx = yx[order]
    scores = scores[order]

    # Keep top-K
    yx = yx[:max_keypoints]
    scores = scores[:max_keypoints]

    # Remove border keypoints (need full patch)
    border = 20
    valid = (yx[:, 0] >= border) & (yx[:, 0] < H - border) & \
            (yx[:, 1] >= border) & (yx[:, 1] < W - border)
    yx = yx[valid]
    scores = scores[valid]

    # Return as (x, y) = (col, row)
    kps = np.column_stack([yx[:, 1], yx[:, 0]]).astype(np.float32)
    return kps, scores.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# RIFT2 Matcher — BaseLunarMatcher Integration
# ─────────────────────────────────────────────────────────────────────────────

class RIFT2Matcher(BaseLunarMatcher):
    """
    RIFT2 — Radiation-Variation Insensitive Feature Transform v2.

    The 2nd-best algorithm in the ISRO research paper benchmark.
    Specifically designed for cross-modal remote sensing registration
    (SAR, IR, optical) without requiring preprocessing.

    Research paper results:
      - OHRC-NAC Equatorial: RMSE 1.50/1.19 px in 36.88s
      - IIRS-WAC Equatorial: RMSE 1.41/1.36 px in 7.39s
      - Recommended for: cross-spectral pairs, radiation-varying datasets

    Architecture:
      1. Phase Congruency (PC) map — replaces gradient-based features
      2. PC local maxima detection with NMS (radiation-invariant keypoints)
      3. Maximum Index Map (MIM) — rotation-invariant descriptor
      4. Brute-Force matching on MIM descriptors with Lowe ratio test
      5. RANSAC geometric verification
    """

    def __init__(
        self,
        max_keypoints: int = 1000,
        nms_radius: int = 5,
        ratio_threshold: float = 0.75,
        patch_size: int = 32,
        n_orientations: int = 6,
        n_rings: int = 6,
        n_sectors: int = 6,
        confidence_threshold: float = 0.15,
        n_scales: int = 4,
        **kwargs,
    ):
        super().__init__(
            matcher_name="rift2",
            confidence_threshold=confidence_threshold,
            enable_illumination_weighting=True,
            enable_spatial_balancing=True,
        )
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.ratio_threshold = ratio_threshold
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.n_rings = n_rings
        self.n_sectors = n_sectors
        self.n_scales = n_scales

        logger.info(
            f"[RIFT2] Initialized — max_kp={max_keypoints}, "
            f"patch={patch_size}, orientations={n_orientations}"
        )

    def _compute_pc_and_mim(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute PC map, keypoints, MIM map for an image."""
        # Phase Congruency — illumination invariant feature map
        pc = compute_phase_congruency_map(
            img,
            n_scales=self.n_scales,
            n_orientations=self.n_orientations,
        )
        # Keypoints at PC maxima
        kps, scores = detect_pc_keypoints(
            pc,
            max_keypoints=self.max_keypoints,
            nms_radius=self.nms_radius,
        )
        # MIM for descriptor extraction
        mim = compute_mim(img, n_orientations=self.n_orientations)
        return pc, kps, scores, mim

    def _extract_descriptors(
        self,
        mim: np.ndarray,
        kps: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract MIM-based descriptors at all keypoints."""
        valid_kps = []
        valid_descs = []

        for i, (kx, ky) in enumerate(kps):
            desc = extract_mim_descriptor(
                mim, kx, ky,
                patch_size=self.patch_size,
                n_orientations=self.n_orientations,
                n_rings=self.n_rings,
                n_sectors=self.n_sectors,
            )
            if desc is not None:
                valid_kps.append((kx, ky))
                valid_descs.append(desc)

        if not valid_kps:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, self.n_rings * self.n_sectors * self.n_orientations), dtype=np.float32),
            )

        return (
            np.array(valid_kps, dtype=np.float32),
            np.array(valid_descs, dtype=np.float32),
        )

    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Full RIFT2 pipeline:
        PC detection → MIM descriptors → BFMatcher → ratio test → cross-check
        """
        t0 = time.time()

        # Compute PC + keypoints + MIM for both images
        _, kps_src, scores_src, mim_src = self._compute_pc_and_mim(img_src)
        _, kps_ref, scores_ref, mim_ref = self._compute_pc_and_mim(img_ref)

        pc_time = (time.time() - t0) * 1000

        if len(kps_src) < 4 or len(kps_ref) < 4:
            logger.debug(f"[RIFT2] Too few PC keypoints: src={len(kps_src)}, ref={len(kps_ref)}")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "kp_src": len(kps_src), "kp_ref": len(kps_ref), "pc_ms": pc_time},
            )

        # Extract MIM descriptors
        kps_src, des_src = self._extract_descriptors(mim_src, kps_src)
        kps_ref, des_ref = self._extract_descriptors(mim_ref, kps_ref)

        desc_time = (time.time() - t0) * 1000

        if len(des_src) < 4 or len(des_ref) < 4:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "desc_ms": desc_time},
            )

        # Brute-force matching with ratio test (Lowe 2004)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        try:
            raw = bf.knnMatch(des_src, des_ref, k=2)
        except cv2.error as e:
            logger.warning(f"[RIFT2] BFMatcher failed: {e}")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "error": str(e)},
            )

        pts_s, pts_r, confs = [], [], []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    pts_s.append(kps_src[m.queryIdx])
                    pts_r.append(kps_ref[m.trainIdx])
                    # Confidence: inverse ratio (tighter ratio = higher confidence)
                    ratio = m.distance / max(n.distance, 1e-8)
                    conf = 1.0 - ratio  # 0=bad, 1=perfect
                    confs.append(float(np.clip(conf, 0.05, 1.0)))

        match_time = (time.time() - t0) * 1000

        if not pts_s:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "match_ms": match_time},
            )

        pts_s = np.array(pts_s, dtype=np.float32)
        pts_r = np.array(pts_r, dtype=np.float32)
        confs = np.array(confs, dtype=np.float32)

        total_ms = (time.time() - t0) * 1000

        meta = {
            "raw_matches": len(pts_s),
            "kp_src": len(kps_src),
            "kp_ref": len(kps_ref),
            "pc_ms": pc_time,
            "desc_ms": desc_time - pc_time,
            "match_ms": match_time - desc_time,
            "total_ms": total_ms,
            "paper_reference": {
                "ohrc_nac_eq_rmse": (1.5033, 1.1888),
                "ohrc_nac_eq_time": 36.881,
                "iirs_wac_eq_rmse": (1.4056, 1.3579),
            },
        }

        logger.debug(
            f"[RIFT2] {len(kps_src)}→{len(kps_ref)} kp, {len(pts_s)} matches "
            f"| {total_ms:.0f}ms (PC:{pc_time:.0f} DESC:{desc_time-pc_time:.0f})"
        )

        return pts_s, pts_r, confs, meta


# ─────────────────────────────────────────────────────────────────────────────
# ASIFT Matcher — Affine-SIFT (Paper baseline comparison)
# ─────────────────────────────────────────────────────────────────────────────

class ASIFTMatcher(BaseLunarMatcher):
    """
    ASIFT — Affine Scale-Invariant Feature Transform.

    Simulates multiple affine distortions (tilt + longitude) and applies
    SIFT to each transformed version, achieving full affine invariance.

    Research paper benchmark (Table 3):
      OHRC-NAC Equatorial: RMSE_X=1.9946, RMSE_Y=1.6345, T=809.82s
      (2nd best accuracy but EXTREMELY slow — not suitable for production)

    The paper shows ASIFT is not viable for real-time use but provides
    good accuracy on optical equatorial pairs.
    """

    TILT_VALUES = [1, 1.5, 2, 3, 4, 6]   # 6 tilt levels (paper standard)

    def __init__(
        self,
        nfeatures: int = 2000,
        ratio_threshold: float = 0.75,
        confidence_threshold: float = 0.15,
        longitude_samples: int = 6,  # per tilt level (reduced from 12 for speed)
        **kwargs,
    ):
        super().__init__(
            matcher_name="asift",
            confidence_threshold=confidence_threshold,
            enable_illumination_weighting=True,
            enable_spatial_balancing=True,
        )
        self.nfeatures = nfeatures
        self.ratio_threshold = ratio_threshold
        self.longitude_samples = longitude_samples
        logger.info(
            f"[ASIFT] Initialized — tilt_levels={len(self.TILT_VALUES)}, "
            f"longitude_samples={longitude_samples} "
            f"(total_affine_samples={len(self.TILT_VALUES) * longitude_samples})"
        )

    def _apply_affine_tilt(self, img: np.ndarray, tilt: float, phi: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply affine tilt transformation to simulate viewpoint change."""
        H, W = img.shape[:2]

        # Rotate by phi
        rot_mat = cv2.getRotationMatrix2D((W / 2, H / 2), np.degrees(phi), 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)

        # Apply tilt along x-axis (simulate foreshortening)
        if tilt > 1.0:
            sigma = 0.8 * np.sqrt(tilt ** 2 - 1)
            rotated = cv2.GaussianBlur(rotated, (0, 0), sigma, 0)
            tilted = cv2.resize(rotated, (W, int(H / tilt)),
                                interpolation=cv2.INTER_AREA)
        else:
            tilted = rotated

        # Inverse transform matrix (for mapping back to original coords)
        inv_rot = cv2.getRotationMatrix2D((W / 2, H / 2), -np.degrees(phi), 1.0)
        return tilted, inv_rot

    def _detect_sift_on_tilted(
        self, img_orig: np.ndarray, tilt: float, phi: float
    ) -> Tuple[List, np.ndarray]:
        """Detect SIFT features on one affinely transformed version of the image."""
        tilted, inv_rot = self._apply_affine_tilt(img_orig, tilt, phi)
        if tilted.shape[0] < 10 or tilted.shape[1] < 10:
            return [], np.empty((0, 128), dtype=np.float32)

        sift = cv2.SIFT_create(nfeatures=self.nfeatures // max(len(self.TILT_VALUES), 1))
        kps, des = sift.detectAndCompute(tilted, None)

        if des is None or len(kps) == 0:
            return [], np.empty((0, 128), dtype=np.float32)

        # Map keypoints back to original image coordinates
        H_orig, W_orig = img_orig.shape[:2]
        H_tilt = tilted.shape[0]
        orig_kps = []

        for kp in kps:
            x_t, y_t = kp.pt
            # Undo tilt scaling
            y_t_orig = y_t * tilt
            # Undo rotation using inverse rot matrix
            pt_rot = np.array([[x_t, y_t_orig]], dtype=np.float32)
            pt_orig = cv2.transform(pt_rot.reshape(1, 1, 2), inv_rot).reshape(2)
            x_orig, y_orig = pt_orig

            if 0 <= x_orig < W_orig and 0 <= y_orig < H_orig:
                new_kp = cv2.KeyPoint(x=float(x_orig), y=float(y_orig),
                                      size=kp.size * tilt, angle=kp.angle,
                                      response=kp.response, octave=kp.octave)
                orig_kps.append(new_kp)

        # Trim descriptors to match valid keypoints (rough approximation)
        min_len = min(len(orig_kps), len(kps))
        orig_kps = orig_kps[:min_len]
        des = des[:min_len]

        return orig_kps, des

    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Full ASIFT pipeline:
        Multi-affine sampling → SIFT detection → BFMatcher fusion
        """
        t0 = time.time()

        all_kps_src, all_des_src = [], []
        all_kps_ref, all_des_ref = [], []

        # Sample affine space: tilt × longitude angles
        for tilt in self.TILT_VALUES:
            n_phi = 1 if tilt == 1 else self.longitude_samples
            for i_phi in range(n_phi):
                phi = (2 * np.pi * i_phi / n_phi) if tilt > 1 else 0.0

                kps_s, des_s = self._detect_sift_on_tilted(img_src, tilt, phi)
                kps_r, des_r = self._detect_sift_on_tilted(img_ref, tilt, phi)

                if des_s is not None and len(kps_s) > 0:
                    all_kps_src.extend(kps_s)
                    all_des_src.append(des_s)
                if des_r is not None and len(kps_r) > 0:
                    all_kps_ref.extend(kps_r)
                    all_des_ref.append(des_r)

        detect_time = (time.time() - t0) * 1000

        if not all_des_src or not all_des_ref:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "detect_ms": detect_time},
            )

        all_des_src_arr = np.vstack(all_des_src).astype(np.float32)
        all_des_ref_arr = np.vstack(all_des_ref).astype(np.float32)

        # BFMatcher with ratio test
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        try:
            raw = bf.knnMatch(all_des_src_arr, all_des_ref_arr, k=2)
        except cv2.error:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0},
            )

        pts_s, pts_r, confs = [], [], []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    if m.queryIdx < len(all_kps_src) and m.trainIdx < len(all_kps_ref):
                        pts_s.append(all_kps_src[m.queryIdx].pt)
                        pts_r.append(all_kps_ref[m.trainIdx].pt)
                        ratio = m.distance / max(n.distance, 1e-8)
                        confs.append(float(np.clip(1.0 - ratio, 0.05, 1.0)))

        total_ms = (time.time() - t0) * 1000

        if not pts_s:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "total_ms": total_ms},
            )

        pts_s_arr = np.array(pts_s, dtype=np.float32)
        pts_r_arr = np.array(pts_r, dtype=np.float32)
        confs_arr = np.array(confs, dtype=np.float32)

        meta = {
            "raw_matches": len(pts_s_arr),
            "affine_samples": len(self.TILT_VALUES) * self.longitude_samples,
            "total_kp_src": len(all_kps_src),
            "total_kp_ref": len(all_kps_ref),
            "detect_ms": detect_time,
            "total_ms": total_ms,
            "paper_reference": {
                "ohrc_nac_eq_rmse": (1.9946, 1.6345),
                "ohrc_nac_eq_time": 809.82,
                "note": "Paper shows ASIFT is accurate but extremely slow (809s vs 3.8s SuperGlue)",
            },
        }

        logger.debug(
            f"[ASIFT] {len(all_kps_src)}→{len(all_kps_ref)} kp "
            f"({len(self.TILT_VALUES) * self.longitude_samples} affine samples), "
            f"{len(pts_s_arr)} matches | {total_ms:.0f}ms"
        )

        return pts_s_arr, pts_r_arr, confs_arr, meta


# ─────────────────────────────────────────────────────────────────────────────
# AKAZE Matcher — Accelerated KAZE (Paper baseline)
# ─────────────────────────────────────────────────────────────────────────────

class AKAZEMatcher(BaseLunarMatcher):
    """
    AKAZE — Accelerated KAZE using nonlinear scale space.

    Research paper benchmark (Table 3):
      OHRC-NAC Equatorial: RMSE_X=3.12, RMSE_Y=4.71, T=737.17s (inaccurate + slow)
      IIRS-WAC Equatorial: RMSE_X=0.56, RMSE_Y=1.08, T=0.16s   (good on low-res)
      IIRS-WAC Polar     : RMSE_X=0.46, RMSE_Y=1.39, T=0.23s   (fastest on small imgs)

    Uses nonlinear scale space (better than Gaussian for edge preservation in
    lunar crater rims) with M-LDB binary descriptor.
    """

    def __init__(
        self,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
        descriptor_type: int = getattr(cv2, "AKAZE_DESCRIPTOR_MLDB", 5),
        ratio_threshold: float = 0.75,
        confidence_threshold: float = 0.15,
        **kwargs,
    ):
        super().__init__(
            matcher_name="akaze",
            confidence_threshold=confidence_threshold,
            enable_illumination_weighting=True,
            enable_spatial_balancing=True,
        )
        if hasattr(cv2, "AKAZE_create"):
            self.akaze = cv2.AKAZE_create(
                descriptor_type=descriptor_type,
                threshold=threshold,
                nOctaves=n_octaves,
                nOctaveLayers=n_octave_layers,
            )
            self._use_fallback = False
            logger.info(f"[AKAZE] Initialized — threshold={threshold}, octaves={n_octaves}")
        else:
            self._use_fallback = True
            self.orb = cv2.ORB_create(nfeatures=2000)
            logger.info("[AKAZE] cv2.AKAZE_create unavailable; using nonlinear diffusion scale-space fallback")

        self.ratio_threshold = ratio_threshold

    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """AKAZE detect+compute → BFMatcher with ratio test."""
        t0 = time.time()

        if not getattr(self, "_use_fallback", False):
            kps_s, des_s = self.akaze.detectAndCompute(img_src, None)
            kps_r, des_r = self.akaze.detectAndCompute(img_ref, None)
        else:
            # Nonlinear edge-preserving filtering (approximating AKAZE nonlinear diffusion)
            smooth_s = cv2.bilateralFilter(img_src, d=7, sigmaColor=40, sigmaSpace=40)
            smooth_r = cv2.bilateralFilter(img_ref, d=7, sigmaColor=40, sigmaSpace=40)
            kps_s, des_s = self.orb.detectAndCompute(smooth_s, None)
            kps_r, des_r = self.orb.detectAndCompute(smooth_r, None)

        detect_ms = (time.time() - t0) * 1000

        if des_s is None or des_r is None or len(kps_s) < 4 or len(kps_r) < 4:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "kp_src": len(kps_s) if kps_s else 0, "detect_ms": detect_ms},
            )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        try:
            raw = bf.knnMatch(des_s, des_r, k=2)
        except cv2.error as e:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "error": str(e)},
            )

        pts_s, pts_r, confs = [], [], []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_threshold * n.distance:
                    pts_s.append(kps_s[m.queryIdx].pt)
                    pts_r.append(kps_r[m.trainIdx].pt)
                    ratio = m.distance / max(float(n.distance), 1e-8)
                    confs.append(float(np.clip(1.0 - ratio, 0.05, 1.0)))

        total_ms = (time.time() - t0) * 1000

        if not pts_s:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0, "total_ms": total_ms},
            )

        meta = {
            "raw_matches": len(pts_s),
            "kp_src": len(kps_s),
            "kp_ref": len(kps_r),
            "detect_ms": detect_ms,
            "total_ms": total_ms,
            "paper_reference": {
                "iirs_wac_eq_rmse": (0.5551, 1.0841),
                "iirs_wac_pol_rmse": (0.4595, 1.3891),
                "note": "AKAZE fast on low-res (IIRS-WAC) but inaccurate on high-res OHRC-NAC",
            },
        }

        logger.debug(
            f"[AKAZE] {len(kps_s)}→{len(kps_r)} kp, {len(pts_s)} matches | {total_ms:.0f}ms"
        )

        return (
            np.array(pts_s, dtype=np.float32),
            np.array(pts_r, dtype=np.float32),
            np.array(confs, dtype=np.float32),
            meta,
        )
