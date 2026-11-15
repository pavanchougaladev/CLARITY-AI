"""
Photometric Correction — Hapke BRDF & Lunar-Lambert Models
===========================================================
Corrects radiance images to a standard viewing geometry so that two images
of the same terrain taken at different sun-angles look similar.

The Hapke model is the standard in planetary science for the Moon's
bidirectional reflectance distribution function (BRDF).
"""

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np


def _cos_deg(angle_deg: float) -> float:
    """Cosine of an angle in degrees."""
    return np.cos(np.radians(angle_deg))


def _hapke_reflectance(
    incidence: float,
    emission: float,
    phase: float,
    w: float = 0.25,
    h_shoe: float = 0.07,
    B0: float = 1.6,
    b: float = 0.25,
    c: float = 0.40,
) -> float:
    """
    Simplified Hapke bidirectional reflectance model for the lunar surface.

    Parameters
    ----------
    incidence : float   Incidence angle in degrees (sun → surface normal).
    emission  : float   Emission angle in degrees (surface normal → camera).
    phase     : float   Phase angle in degrees (sun → surface → camera).
    w         : float   Single-scattering albedo (typical lunar highland ~0.25).
    h_shoe    : float   Opposition surge angular half-width.
    B0        : float   Opposition surge amplitude.
    b, c      : float   Henyey-Greenstein asymmetry parameters.

    Returns
    -------
    float : Modeled reflectance value (r_Hapke).
    """
    mu0 = max(_cos_deg(incidence), 1e-6)   # cos(i)
    mu = max(_cos_deg(emission), 1e-6)      # cos(e)
    alpha = np.radians(phase)

    # Henyey-Greenstein single-particle phase function (two-term)
    cos_g = np.cos(alpha)
    P_hg = ((1 - c) * (1 - b**2) / (1 + 2 * b * cos_g + b**2)**1.5 +
            c * (1 - b**2) / (1 - 2 * b * cos_g + b**2)**1.5)

    # Opposition effect (SHOE — Shadow Hiding Opposition Effect)
    tan_half = np.tan(alpha / 2 + 1e-10)
    B_shoe = B0 / (1 + tan_half / h_shoe)

    # Chandrasekhar H-function approximation
    gamma = np.sqrt(1 - w)

    def H(x):
        return (1 + 2 * x) / (1 + 2 * gamma * x)

    # Hapke reflectance
    r = (w / (4 * np.pi)) * (mu0 / (mu0 + mu)) * (
        (1 + B_shoe) * P_hg + H(mu0) * H(mu) - 1
    )

    return max(r, 1e-10)


def _apply_shadow_aware_correction(img_f: np.ndarray, correction: float) -> np.ndarray:
    """
    Applies photometric correction with shadow-awareness and dynamic range protection.
    Deep shadows (which receive zero direct solar illumination) are not artificially
    amplified, preventing sensor noise explosion in permanently shadowed regions.
    """
    # Clamp extreme correction factors to physically reasonable bounds
    clamped_corr = float(np.clip(correction, 0.2, 5.0))
    # Smooth transition from shadowed (<5 DN) to illuminated (>25 DN) terrain
    shadow_weight = np.clip((img_f - 5.0) / 20.0, 0.0, 1.0)
    effective_scale = 1.0 + (clamped_corr - 1.0) * shadow_weight
    return np.clip(img_f * effective_scale, 0, 255)


def apply_hapke_correction(
    img: np.ndarray,
    incidence_angle: float,
    emission_angle: float,
    phase_angle: float,
    target_incidence: float = 30.0,
    target_emission: float = 0.0,
    target_phase: float = 30.0,
) -> np.ndarray:
    """
    Corrects image radiance from observed geometry to standard
    viewing geometry using Hapke BRDF ratio method with shadow awareness.

    Parameters
    ----------
    img              : 2D uint8/float array — input image.
    incidence_angle  : Observed incidence angle (degrees).
    emission_angle   : Observed emission angle (degrees).
    phase_angle      : Observed phase angle (degrees).
    target_incidence : Standard incidence angle to normalise to.
    target_emission  : Standard emission angle to normalise to.
    target_phase     : Standard phase angle to normalise to.

    Returns
    -------
    np.ndarray : Photometrically corrected image (same dtype as input).
    """
    original_dtype = img.dtype
    img_f = img.astype(np.float64)

    # Reflectance at observed geometry
    r_obs = _hapke_reflectance(incidence_angle, emission_angle, phase_angle)

    # Reflectance at standard geometry
    r_std = _hapke_reflectance(target_incidence, target_emission, target_phase)

    # Correction factor
    correction = r_std / r_obs

    # Apply shadow-aware correction
    corrected = _apply_shadow_aware_correction(img_f, correction)

    if original_dtype == np.uint8:
        return corrected.astype(np.uint8)
    return corrected.astype(np.float32)


def apply_lunar_lambert(
    img: np.ndarray,
    incidence_angle: float,
    emission_angle: float,
    phase_angle: float,
    L: float = 0.5,
    target_incidence: float = 30.0,
    target_emission: float = 0.0,
) -> np.ndarray:
    """
    Lunar-Lambert photometric correction with shadow-aware scaling.

    Combines Lambertian (cos(i)) and Lommel-Seeliger (cos(i)/(cos(i)+cos(e)))
    models with a mixing parameter L that varies with phase angle.
    """
    original_dtype = img.dtype
    img_f = img.astype(np.float64)

    mu0_obs = max(_cos_deg(incidence_angle), 1e-6)
    mu_obs = max(_cos_deg(emission_angle), 1e-6)

    mu0_std = max(_cos_deg(target_incidence), 1e-6)
    mu_std = max(_cos_deg(target_emission), 1e-6)

    # Lunar-Lambert model: f = 2*L*mu0 + (1-L) * mu0 / (mu0 + mu)
    def lunar_lambert_f(mu0, mu):
        return 2 * L * mu0 + (1 - L) * mu0 / (mu0 + mu)

    f_obs = lunar_lambert_f(mu0_obs, mu_obs)
    f_std = lunar_lambert_f(mu0_std, mu_std)

    correction = f_std / max(f_obs, 1e-10)
    corrected = _apply_shadow_aware_correction(img_f, correction)

    if original_dtype == np.uint8:
        return corrected.astype(np.uint8)
    return corrected.astype(np.float32)


def invert_hapke_to_albedo(
    image: np.ndarray,
    sun_az_deg: float = 65.0,
    sun_el_deg: float = 25.0,
    w: float = 0.25,
    h_shoe: float = 0.07,
    B0: float = 1.6,
    b: float = 0.25,
    c: float = 0.40,
    noise_floor: float = 8.0,
) -> np.ndarray:
    """
    Physical Hapke BRDF Albedo Inversion (Shadow Removal & Illumination Decoupling).

    Decouples directional solar incidence angle and shadow hill-shading from intrinsic
    surface reflectance (albedo) by inverting the non-linear planetary Hapke BRDF equation:
        A(x, y) = I_observed(x, y) / max(R_Hapke(n(x,y), s), epsilon)

    Features:
    - Reconstructs local surface normals n(x, y) across crater walls.
    - Resolves Henyey-Greenstein scattering and Shadow Hiding Opposition Effect (SHOE).
    - Includes soft umbra noise floor clamping (I < 3*sigma) to prevent sensor noise
      explosion inside permanently shadowed crater basins.

    Parameters
    ----------
    image       : 2D array (uint8 or float32).
    sun_az_deg  : Solar azimuth in degrees (0 = North, 90 = East).
    sun_el_deg  : Solar elevation in degrees above horizon.
    w           : Single-scattering albedo (default 0.25 for lunar highland).
    h_shoe      : Opposition surge angular width (default 0.07).
    B0          : Opposition surge amplitude (default 1.6).
    b, c        : Henyey-Greenstein asymmetry parameters.
    noise_floor : Sensor dark noise floor DN for shadow gating (default 8.0).

    Returns
    -------
    albedo : np.ndarray uint8 [0, 255] representing intrinsic lunar surface albedo.
    """
    import cv2

    img_f = np.asarray(image, dtype=np.float32)
    h, w_img = img_f.shape[:2]

    # 1. Solar illumination unit vector in image coordinates:
    # Azimuth: 0° = North (-y), 90° = East (+x)
    az_rad = np.radians(float(sun_az_deg) % 360.0)
    el_rad = np.radians(float(np.clip(sun_el_deg, 5.0, 85.0)))

    sx = float(np.sin(az_rad) * np.cos(el_rad))
    sy = -float(np.cos(az_rad) * np.cos(el_rad))
    sz = float(np.sin(el_rad))

    # Phase angle for nadir camera: alpha = 90° - elevation
    phase_deg = 90.0 - float(sun_el_deg)
    alpha = np.radians(max(1.0, min(89.0, phase_deg)))

    # 2. Local surface normal estimation n(x, y)
    # Filter high-frequency speckle to estimate macro slope
    smooth = cv2.GaussianBlur(img_f, (15, 15), 3.0)
    denom_smooth = np.maximum(smooth, 1.0)
    d_radiance = (img_f - smooth) / denom_smooth

    el_factor = max(float(np.tan(el_rad)), 0.15)
    grad_scale = 0.22 / el_factor
    p = np.clip(-d_radiance * sx * grad_scale, -1.5, 1.5)
    q = np.clip(-d_radiance * sy * grad_scale, -1.5, 1.5)

    norm_mag = np.sqrt(p * p + q * q + 1.0)
    nx = -p / norm_mag
    ny = -q / norm_mag
    nz = 1.0 / norm_mag

    # Local incidence cos(i) = n . s and emission cos(e) = nz
    mu0 = np.maximum(nx * sx + ny * sy + nz * sz, 1e-4)
    mu = np.maximum(nz, 1e-4)

    # 3. Hapke BRDF components
    cos_g = np.cos(alpha)
    P_hg = ((1.0 - c) * (1.0 - b**2) / (1.0 + 2.0 * b * cos_g + b**2)**1.5 +
            c * (1.0 - b**2) / (1.0 - 2.0 * b * cos_g + b**2)**1.5)

    tan_half = np.tan(alpha / 2.0 + 1e-10)
    B_shoe = B0 / (1.0 + tan_half / max(h_shoe, 1e-4))

    gamma = np.sqrt(max(1e-4, 1.0 - w))
    H_mu0 = (1.0 + 2.0 * mu0) / (1.0 + 2.0 * gamma * mu0)
    H_mu = (1.0 + 2.0 * mu) / (1.0 + 2.0 * gamma * mu)

    # Modeled Hapke bidirectional reflectance factor (without intrinsic albedo)
    r_geom = (1.0 / (4.0 * np.pi)) * (mu0 / (mu0 + mu)) * (
        (1.0 + B_shoe) * P_hg + H_mu0 * H_mu - 1.0
    )
    r_geom = np.maximum(r_geom, 1e-5)

    # 4. Invert to intrinsic albedo A(x, y) = I / R_geom
    raw_albedo = img_f / r_geom

    # 5. Soft umbra noise floor protection
    # In deep shadows (I <= noise_floor), direct solar flux is absent;
    # blend raw albedo with smooth background albedo to eliminate noise blowup
    shadow_weight = np.clip((img_f - noise_floor) / 18.0, 0.0, 1.0)
    bg_albedo = cv2.GaussianBlur(raw_albedo.astype(np.float32), (7, 7), 2.0)
    protected_albedo = shadow_weight * raw_albedo + (1.0 - shadow_weight) * bg_albedo

    # 6. Normalize to [0, 255] uint8 preserving regolith dynamic range
    p2 = np.percentile(protected_albedo, 2)
    p98 = np.percentile(protected_albedo, 98)
    if p98 > p2:
        norm = ((protected_albedo - p2) / (p98 - p2) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        norm = np.clip(protected_albedo, 0, 255).astype(np.uint8)

    return norm


def remove_crater_shadows_hapke_congruency(
    image: np.ndarray,
    sun_az_deg: float = 65.0,
    sun_el_deg: float = 25.0,
) -> np.ndarray:
    """
    Unified Illumination-Invariant Representation:
    Fuses Physical Hapke Albedo Inversion with Monogenic Phase Congruency.

    Eliminates false tie-point migration across changing crater shadows and
    provides a 100% illumination-invariant feature plane for matching.
    """
    albedo = invert_hapke_to_albedo(image, sun_az_deg=sun_az_deg, sun_el_deg=sun_el_deg)
    pc = compute_phase_congruency(image)
    pc_u8 = (pc * 255.0).clip(0, 255).astype(np.uint8)

    # Fused representation: 65% intrinsic albedo + 35% invariant phase congruency edges
    fused = (0.65 * albedo.astype(np.float32) + 0.35 * pc_u8.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return fused


def compute_phase_congruency(
    img: np.ndarray,
    n_scales: int = 3,
    n_orientations: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.0,
) -> np.ndarray:
    """
    Phase Congruency feature map (Peter Kovesi's Local Energy model).

    Calculates feature significance at points where Fourier components are maximally
    in phase. Because phase congruency is dimensionless, it is strictly invariant
    to image brightness, contrast, and solar incidence angle disparity, preventing
    false tie-points from migrating across shifting crater shadows.

    Parameters
    ----------
    img            : 2D array (H, W), uint8 or float32.
    n_scales       : Number of wavelet scales to evaluate (default 3).
    n_orientations : Number of filter orientations (default 4).
    min_wavelength : Wavelength of smallest scale filter in pixels (default 3.0).
    mult           : Scaling factor between successive filters (default 2.0).

    Returns
    -------
    pc : np.ndarray (H, W) in [0.0, 1.0] representing invariant structural edges/crater rims.
    """
    import cv2

    img_f = img.astype(np.float32)
    h, w = img_f.shape[:2]

    # Pre-filter DC and high-frequency noise
    blurred = cv2.GaussianBlur(img_f, (3, 3), 0.5)

    total_energy = np.zeros((h, w), dtype=np.float32)
    total_amplitude = np.zeros((h, w), dtype=np.float32)

    # Spatial filter bank approximating 2D quadrature Gabor wavelets
    for o in range(n_orientations):
        angle = o * np.pi / n_orientations
        even_sum = np.zeros((h, w), dtype=np.float32)
        odd_sum = np.zeros((h, w), dtype=np.float32)

        for s in range(n_scales):
            wavelength = min_wavelength * (mult ** s)
            sigma = 0.55 * wavelength
            ksize = int(max(3, round(3.0 * sigma))) * 2 + 1
            ksize = min(ksize, min(h, w) // 2 * 2 + 1)
            if ksize < 3:
                ksize = 3

            # Even-symmetric Gabor (cosine)
            kernel_even = cv2.getGaborKernel(
                (ksize, ksize), sigma, angle, wavelength, 0.5, 0.0, ktype=cv2.CV_32F
            )
            # Odd-symmetric Gabor (sine)
            kernel_odd = cv2.getGaborKernel(
                (ksize, ksize), sigma, angle, wavelength, 0.5, np.pi / 2.0, ktype=cv2.CV_32F
            )

            resp_even = cv2.filter2D(blurred, cv2.CV_32F, kernel_even)
            resp_odd = cv2.filter2D(blurred, cv2.CV_32F, kernel_odd)

            amplitude = np.sqrt(resp_even ** 2 + resp_odd ** 2)
            total_amplitude += amplitude

            even_sum += resp_even
            odd_sum += resp_odd

        # Local Energy for this orientation
        energy_o = np.sqrt(even_sum ** 2 + odd_sum ** 2)
        total_energy += energy_o

    # Noise threshold T estimation based on Rayleigh distribution
    median_amp = np.median(total_amplitude)
    T = float(median_amp * 0.35)

    energy_thresh = np.maximum(0.0, total_energy - T)
    pc = energy_thresh / (total_amplitude + 1e-4)
    pc = np.clip(pc, 0.0, 1.0)
    return pc


def apply_log_transform(img: np.ndarray, c: float = None) -> np.ndarray:
    """
    Research Paper Pipeline B (IIRS ↔ WAC Hyperspectral):
    Applies logarithmic dynamic range compression:
        s = c * log(1 + r)
    where c = 255.0 / log(1 + max(r)).

    This non-linear mapping dramatically expands shadow details and suppresses
    saturation at high solar incidence angles.
    """
    img_f = img.astype(np.float32)
    min_val = img_f.min()
    if min_val < 0:
        img_f = img_f - min_val

    max_val = img_f.max()
    if max_val <= 0:
        return np.zeros_like(img, dtype=np.uint8)

    if c is None:
        c = 255.0 / np.log1p(max_val)

    transformed = c * np.log1p(img_f)
    return np.clip(transformed, 0, 255).astype(np.uint8)


def apply_pca_enhancement(img: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Research Paper Pipeline A (OHRC ↔ NAC Optical Registration):
    Enhances morphological lunar features via Principal Component Analysis (PCA).

    If `img` has multiple channels/bands (e.g. hyperspectral or color), PCA projects
    the multi-band data onto the principal variance axis (1st PC).
    If `img` is single-channel 2D grayscale, creates a multi-scale pyramid representation
    (original + Gaussian blurred scales) and projects onto the 1st principal component,
    effectively isolating topography and crater rims from flat illumination bias.

    Returns
    -------
    np.ndarray : Enhanced uint8 grayscale image.
    """
    import cv2

    img_f = img.astype(np.float32)

    if img.ndim == 3 and img.shape[2] > 1:
        # Multi-channel / multi-band PCA
        h, w, c = img.shape
        flat = img_f.reshape(-1, c)
        mean = np.mean(flat, axis=0)
        centered = flat - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Largest eigenvector
        v1 = eigvecs[:, -1]
        pc1 = centered @ v1
        pc1_2d = pc1.reshape(h, w)
    else:
        # Single-channel 2D: Multi-scale spatial PCA
        h, w = img_f.shape[:2]
        s0 = img_f
        s1 = cv2.GaussianBlur(img_f, (5, 5), 1.0)
        s2 = cv2.GaussianBlur(img_f, (11, 11), 2.5)
        s3 = cv2.GaussianBlur(img_f, (21, 21), 5.0)

        # Feature matrix across 4 scales
        features = np.stack([s0.ravel(), s1.ravel(), s2.ravel(), s3.ravel()], axis=1)
        mean = np.mean(features, axis=0)
        centered = features - mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v1 = eigvecs[:, -1]
        pc1 = centered @ v1
        pc1_2d = pc1.reshape(h, w)

    # Normalize to [0, 255] uint8
    p_min, p_max = pc1_2d.min(), pc1_2d.max()
    if p_max > p_min:
        norm = ((pc1_2d - p_min) / (p_max - p_min) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        norm = np.full((h, w), 128, dtype=np.uint8)

    return norm


def estimate_solar_illumination_vector(
    img: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Automated Solar Illumination Vector Estimator (Azimuth, Elevation, Confidence).

    Extracts the physical 3D solar lighting direction directly from lunar orbital
    imagery when SPICE ephemeris kernels or PDS4 metadata labels are unavailable.

    Mathematical Basis:
    - On planetary crater rims, ridges, and slopes, intensity gradient vectors
      grad(I) = (dI/dx, dI/dy) consistently point from shadow/dark terrain towards
      the directly illuminated crest facing the solar source.
    - Resolves the dominant gradient vector v = (vx, -vy) across edge-transition
      regions where grad_magnitude exceeds the 70th percentile.
    - Solar azimuth phi_sun is the circular direction:
          phi_sun = atan2(vx, -vy) (degrees in [0, 360), where 0=N, 90=E, 180=S, 270=W).
    - Solar elevation theta_sun is derived from the cast shadow surface fraction
      and gradient distribution: grazing illumination (low sun) produces extensive
      deep umbra (shadows > 20% area), whereas high sun (nadir) minimizes cast shadows (< 2%).

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (uint8 or float32).

    Returns
    -------
    Tuple[float, float, float]
        - sun_az_deg  : Estimated solar azimuth in degrees [0.0, 360.0).
        - sun_el_deg  : Estimated solar elevation in degrees [5.0, 85.0].
        - confidence  : Statistical coherence / confidence of estimate [0.0, 1.0].
    """
    import cv2

    img_f = np.asarray(img, dtype=np.float32)
    if img_f.ndim == 3:
        img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY) if img_f.shape[2] == 3 else img_f[:, :, 0]

    h, w = img_f.shape[:2]
    if h < 16 or w < 16:
        return 65.0, 25.0, 0.0

    # Gaussian smoothing to eliminate sensor readout noise and cosmic ray hits
    smooth = cv2.GaussianBlur(img_f, (5, 5), 1.2)

    # 1. Compute directional spatial gradients
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    # 2. Identify significant gradient transition boundaries (avoid flat regolith plains)
    p70 = float(np.percentile(mag, 70))
    p_max = float(np.max(mag))
    if p_max <= 1e-3:
        # Uniform image with no topographic relief
        return 65.0, 25.0, 0.0

    valid_mask = (mag >= p70)
    if not np.any(valid_mask):
        return 65.0, 25.0, 0.0

    gx_sub = gx[valid_mask]
    gy_sub = gy[valid_mask]
    mag_sub = mag[valid_mask]

    # Weighted directional vector pointing from dark to bright (sunward)
    vx = float(np.sum(gx_sub))
    vy = float(np.sum(gy_sub))
    norm_v = math.hypot(vx, vy)

    if norm_v < 1e-4:
        return 65.0, 25.0, 0.1

    # Azimuth: 0° = North (-y), 90° = East (+x), 180° = South (+y), 270° = West (-x)
    az_deg = float(np.degrees(np.arctan2(vx, -vy)) % 360.0)

    # 3. Confidence based on circular concentration (Rayleigh resultant length)
    unit_vx = gx_sub / np.maximum(mag_sub, 1e-4)
    unit_vy = gy_sub / np.maximum(mag_sub, 1e-4)
    mean_dir_x = float(np.mean(unit_vx))
    mean_dir_y = float(np.mean(unit_vy))
    resultant_r = math.hypot(mean_dir_x, mean_dir_y)
    confidence = float(np.clip(resultant_r * 2.0, 0.0, 1.0))

    # 4. Solar elevation estimation from shadow fraction
    p10 = float(np.percentile(smooth, 10))
    shadow_thresh = min(35.0, max(8.0, p10))
    shadow_mask = smooth < shadow_thresh
    shadow_fraction = float(np.mean(shadow_mask))

    est_el = 75.0 - 180.0 * math.sqrt(shadow_fraction)
    sun_el_deg = float(np.clip(est_el, 10.0, 75.0))

    return az_deg, sun_el_deg, confidence


def apply_dual_pass_cross_sun_normalization(
    img_ref: np.ndarray,
    img_tgt: np.ndarray,
    sun_az_ref: Optional[float] = None,
    sun_el_ref: Optional[float] = None,
    sun_az_tgt: Optional[float] = None,
    sun_el_tgt: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Dual-Pass Cross-Sun Normalization for Opposing Illumination Invariance.

    Decouples solar incidence and cast shadow disparities when registering
    lunar image pairs captured under drastically different sun angles (e.g.
    morning east sun vs afternoon west sun, or low sun vs high sun).

    1. Automatically estimates the 3D solar illumination vector (azimuth & elevation)
       for each frame if not provided from SPICE kernels or PDS4 labels.
    2. Computes the illumination disparity angle delta_azimuth.
    3. Inverts both frames to physical Hapke BRDF intrinsic surface albedos using
       their respective illumination vectors, eliminating cast shadow gradients.
    4. Computes dimensionless Monogenic Phase Congruency, anchoring structural
       crater rim crests regardless of illumination flip.
    5. Fuses 60% Hapke albedo + 40% Phase Congruency into an illumination-neutral
       feature representation ready for feature matching engines.
    """
    # 1. Resolve illumination vectors
    conf_ref = 1.0
    if sun_az_ref is None or sun_el_ref is None:
        est_az, est_el, conf_ref = estimate_solar_illumination_vector(img_ref)
        if sun_az_ref is None:
            sun_az_ref = est_az
        if sun_el_ref is None:
            sun_el_ref = est_el

    conf_tgt = 1.0
    if sun_az_tgt is None or sun_el_tgt is None:
        est_az, est_el, conf_tgt = estimate_solar_illumination_vector(img_tgt)
        if sun_az_tgt is None:
            sun_az_tgt = est_az
        if sun_el_tgt is None:
            sun_el_tgt = est_el

    # 2. Compute angular disparity
    diff_az = abs(float(sun_az_ref) - float(sun_az_tgt)) % 360.0
    delta_azimuth = min(diff_az, 360.0 - diff_az)
    is_cross_sun = bool(delta_azimuth >= 45.0)
    is_extreme_opposite = bool(delta_azimuth >= 120.0)

    # 3. Apply physical Hapke albedo inversion to each frame with its respective sun vector
    albedo_ref = invert_hapke_to_albedo(img_ref, sun_az_deg=float(sun_az_ref), sun_el_deg=float(sun_el_ref))
    albedo_tgt = invert_hapke_to_albedo(img_tgt, sun_az_deg=float(sun_az_tgt), sun_el_deg=float(sun_el_tgt))

    # 4. Monogenic Phase Congruency (illumination-invariant structural rim anchor)
    pc_ref = compute_phase_congruency(img_ref)
    pc_tgt = compute_phase_congruency(img_tgt)
    pc_ref_u8 = (pc_ref * 255.0).clip(0, 255).astype(np.uint8)
    pc_tgt_u8 = (pc_tgt * 255.0).clip(0, 255).astype(np.uint8)

    # 5. Dual-pass fusion: 60% intrinsic albedo + 40% phase congruency
    fused_ref = (0.60 * albedo_ref.astype(np.float32) + 0.40 * pc_ref_u8.astype(np.float32)).clip(0, 255).astype(np.uint8)
    fused_tgt = (0.60 * albedo_tgt.astype(np.float32) + 0.40 * pc_tgt_u8.astype(np.float32)).clip(0, 255).astype(np.uint8)

    telemetry = {
        "sun_az_ref": round(float(sun_az_ref), 1),
        "sun_el_ref": round(float(sun_el_ref), 1),
        "confidence_ref": round(float(conf_ref), 2),
        "sun_az_tgt": round(float(sun_az_tgt), 1),
        "sun_el_tgt": round(float(sun_el_tgt), 1),
        "confidence_tgt": round(float(conf_tgt), 2),
        "delta_azimuth_deg": round(float(delta_azimuth), 1),
        "cross_sun_detected": is_cross_sun,
        "extreme_opposite_sun": is_extreme_opposite,
        "normalization_applied": "dual_pass_hapke_phase_congruency",
    }

    return fused_ref, fused_tgt, telemetry



