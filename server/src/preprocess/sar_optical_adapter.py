"""
SAR-Optical Cross-Modal Adapter (DFSAR ↔ SELENE / TMC / OHRC)
=============================================================
Implements Paper Pipeline C for SAR-to-Optical Lunar Registration.
Dual-Frequency SAR (DFSAR) contains multiplicative speckle noise and
fundamentally distinct active microwave scattering physics compared to
passive optical sensors (SELENE TC / Chandrayaan-2 TMC).

Key Stages:
1. Enhanced Lee Filter / Adaptive Speckle Suppression
2. Radiometric Log-Ratio dynamic range normalization
3. Phase Congruency structural gradient alignment
"""

import logging
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from .photometric_correction import compute_phase_congruency

logger = logging.getLogger("clarity.preprocess.sar_optical_adapter")


def lee_speckle_filter(
    img: np.ndarray,
    kernel_size: int = 5,
    enl: float = 4.0,
) -> np.ndarray:
    """
    Enhanced Lee Filter for multiplicative SAR speckle noise reduction.

    In SAR imagery:
        I(x, y) = R(x, y) * N(x, y)
    where R is true radar reflectivity and N is speckle noise with variance:
        var_n = 1.0 / ENL (Equivalent Number of Looks)

    Filter formula:
        R_hat = local_mean + W * (I - local_mean)
        W = max(0, (local_var - noise_var) / local_var)
        noise_var = (local_mean^2) / ENL

    Parameters
    ----------
    img : np.ndarray
        Grayscale SAR image (uint8 or float).
    kernel_size : int
        Window size for local statistics (must be odd, default: 5).
    enl : float
        Equivalent Number of Looks (Chandrayaan-2 DFSAR default: 4.0).
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    img_f = img.astype(np.float32)

    local_mean = cv2.blur(img_f, (k, k))
    local_sq_mean = cv2.blur(img_f ** 2, (k, k))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-6)

    # Noise variance under multiplicative speckle model
    noise_var = (local_mean ** 2) / float(max(1.0, enl))

    # Weight factor W
    weight = np.clip((local_var - noise_var) / (local_var + 1e-10), 0.0, 1.0)

    filtered = local_mean + weight * (img_f - local_mean)

    if np.issubdtype(img.dtype, np.integer):
        return np.clip(np.round(filtered), 0, 255).astype(img.dtype)
    return filtered.astype(img.dtype)


def sar_log_ratio_normalization(
    sar_img: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Converts raw SAR radar backscatter intensity into optical-comparable dynamic range:
        I_norm = c * log1p(I_sar)
    Multiplicative radar speckle becomes additive, and dynamic range spikes are compressed.
    """
    img_f = sar_img.astype(np.float32)
    min_v = img_f.min()
    if min_v < 0:
        img_f = img_f - min_v

    max_v = img_f.max()
    if max_v <= 0:
        return np.zeros_like(sar_img, dtype=np.uint8)

    c = 255.0 / np.log1p(max_v)
    log_img = c * np.log1p(img_f)

    if gamma != 1.0:
        norm_01 = log_img / 255.0
        log_img = (np.power(norm_01, gamma) * 255.0).clip(0, 255)

    return np.clip(log_img, 0, 255).astype(np.uint8)


def preprocess_sar_optical_pair(
    sar_img: np.ndarray,
    optical_img: np.ndarray,
    kernel_size: int = 5,
    enl: float = 4.0,
    extract_pc_structure: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Paper Pipeline C: End-to-end preprocessing for DFSAR ↔ SELENE/Optical.

    Parameters
    ----------
    sar_img : np.ndarray
        DFSAR SAR radar image.
    optical_img : np.ndarray
        SELENE / TMC / OHRC optical panchromatic reference image.
    kernel_size : int
        Window size for SAR Lee speckle filtering.
    enl : float
        Equivalent Number of Looks.
    extract_pc_structure : bool
        If True, applies Phase Congruency structural edge extraction to bridge
        the modal physics difference between radar backscatter and optical reflectance.

    Returns
    -------
    (sar_out, optical_out, telemetry_steps)
    """
    steps: Dict[str, Any] = {}

    # Convert to 2D grayscale
    def _to_gray(im):
        if im.ndim == 3:
            return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY if im.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        return im

    s_gray = _to_gray(sar_img)
    o_gray = _to_gray(optical_img)

    # 1. SAR Lee Speckle Reduction
    sar_despeckled = lee_speckle_filter(s_gray, kernel_size=kernel_size, enl=enl)
    steps["sar_despeckled"] = sar_despeckled

    # 2. Radiometric Log-ratio normalization
    sar_log = sar_log_ratio_normalization(sar_despeckled)
    steps["sar_log_normalized"] = sar_log

    # 3. Optical contrast normalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    opt_clahe = clahe.apply(_to_gray(optical_img).astype(np.uint8))
    steps["optical_clahe"] = opt_clahe

    # 4. Phase Congruency structural enhancement (optional, invariant to illumination & modality)
    if extract_pc_structure:
        sar_pc = compute_phase_congruency(sar_log)
        opt_pc = compute_phase_congruency(opt_clahe)
        sar_out = (sar_pc * 255.0).clip(0, 255).astype(np.uint8)
        opt_out = (opt_pc * 255.0).clip(0, 255).astype(np.uint8)
        steps["sar_phase_congruency"] = sar_out
        steps["optical_phase_congruency"] = opt_out
    else:
        sar_out = sar_log
        opt_out = opt_clahe

    return sar_out, opt_out, steps
