"""
Research Paper Preprocessing Pipeline Router (ISRO / Chandrayaan-2)
==================================================================
Implements Pipelines A, B, and C as rigorously established in:
"Comparative Analysis of Image Registration Algorithms for Chandrayaan-2 Lunar Datasets" (2025)

- Pipeline A: OHRC ↔ NAC (High-Resolution Optical Registration)
  1. 8-bit normalization
  2. Contrast Limited Adaptive Histogram Equalization (CLAHE)
  3. Image Inversion (turns dark crater bottoms into bright anchor features)
  4. Morphological Dilation (expands crater rims and bridges broken edges)
  5. Principal Component Analysis (PCA) feature variance enhancement

- Pipeline B: IIRS ↔ WAC (Hyperspectral-to-Panchromatic Registration)
  1. Band selection (if multi-band input)
  2. Histogram Matching (CDF alignment across sensor modalities)
  3. Shadow Normalization (Multi-scale CLAHE + Wallis filtering)
  4. Logarithmic Transformation (dynamic range shadow recovery)

- Pipeline C: DFSAR ↔ SELENE (SAR-to-Optical Cross-Modal Registration)
  1. SAR Speckle Filter (Adaptive Lee / Bilateral speckle suppression)
  2. Radiometric Log-ratio normalization (squashes extreme backscatter spikes)
  3. Phase Congruency structural edge extraction
"""

import logging
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from .shadow_normalization import (
    normalize_illumination,
    apply_image_inversion,
    apply_morphological_dilation,
)
from .photometric_correction import (
    apply_pca_enhancement,
    apply_log_transform,
    compute_phase_congruency,
)
from .histogram_utils import apply_histogram_matching

logger = logging.getLogger("clarity.preprocess.pipeline_router")


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Normalize input image to uint8 [0, 255]."""
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if img.shape[-1] == 3 else cv2.COLOR_RGBA2GRAY)
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img[0]
    
    img_f = img.astype(np.float32)
    min_v, max_v = img_f.min(), img_f.max()
    if max_v > min_v:
        return ((img_f - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)
    return np.zeros_like(img, dtype=np.uint8)


class SensorModalityRouter:
    """
    Automated pre-processing pipeline router for lunar remote sensing modalities.
    """

    @classmethod
    def run_pipeline_a(
        cls,
        img: np.ndarray,
        clip_limit: float = 3.0,
        dilation_ksize: int = 3,
        use_pca: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute Paper Pipeline A (OHRC ↔ NAC Optical Registration).
        Steps: 8-bit norm -> CLAHE -> Inversion -> Dilation -> PCA
        """
        steps = {}
        # 1. 8-bit normalization
        u8 = _to_u8(img)
        steps["normalized_8bit"] = u8

        # 2. CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        clahe_out = clahe.apply(u8)
        steps["clahe"] = clahe_out

        # 3. Image Inversion
        inverted = apply_image_inversion(clahe_out)
        steps["inversion"] = inverted

        # 4. Morphological Dilation
        dilated = apply_morphological_dilation(inverted, kernel_size=dilation_ksize)
        steps["dilation"] = dilated

        # 5. PCA Enhancement
        if use_pca:
            final = apply_pca_enhancement(dilated)
            steps["pca"] = final
        else:
            final = dilated

        return final, steps

    @classmethod
    def run_pipeline_b(
        cls,
        src_img: np.ndarray,
        ref_img: np.ndarray,
        log_constant: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute Paper Pipeline B (IIRS ↔ WAC Hyperspectral Registration).
        Steps: 8-bit norm -> Histogram Matching -> Shadow Normalization -> Log Transform
        """
        steps = {}
        s_u8 = _to_u8(src_img)
        r_u8 = _to_u8(ref_img)
        steps["src_norm"] = s_u8
        steps["ref_norm"] = r_u8

        # 1. Histogram Matching (match source hyperspectral to optical reference)
        hist_matched_src = apply_histogram_matching(s_u8, r_u8)
        steps["hist_matched_src"] = hist_matched_src

        # 2. Shadow Normalization (Wallis + multi-scale CLAHE)
        norm_src = normalize_illumination(hist_matched_src, use_wallis=True)
        norm_ref = normalize_illumination(r_u8, use_wallis=True)
        steps["norm_src"] = norm_src
        steps["norm_ref"] = norm_ref

        # 3. Logarithmic Transformation
        final_src = apply_log_transform(norm_src, c=log_constant)
        final_ref = apply_log_transform(norm_ref, c=log_constant)
        steps["log_src"] = final_src
        steps["log_ref"] = final_ref

        return final_src, final_ref, steps

    @classmethod
    def run_pipeline_c(
        cls,
        sar_img: np.ndarray,
        optical_img: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute Paper Pipeline C (DFSAR ↔ SELENE SAR-Optical Registration).
        Steps: SAR Speckle filter -> Log-ratio -> Phase Congruency structural enhancement
        """
        from .sar_optical_adapter import preprocess_sar_optical_pair
        return preprocess_sar_optical_pair(sar_img, optical_img)

    @classmethod
    def route_and_preprocess(
        cls,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        modality_or_pair: str = "auto",
        sensor_type_src: str = "auto",
        sensor_type_ref: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Automatically selects and executes the appropriate pipeline based on sensor pair.

        Parameters
        ----------
        modality_or_pair : str
            'pipeline_a' / 'ohrc_nac'       -> Pipeline A (Optical)
            'pipeline_b' / 'iirs_wac'       -> Pipeline B (Hyperspectral)
            'pipeline_c' / 'dfsar_selene'   -> Pipeline C (SAR-Optical)
            'auto'                          -> Infer from sensor types or image statistics
        """
        mode = modality_or_pair.lower()
        src_s = sensor_type_src.lower()
        ref_s = sensor_type_ref.lower()

        # Check explicit sensor indicators
        is_sar = (
            "dfsar" in [src_s, ref_s] or
            "sar" in [src_s, ref_s] or
            mode in ["pipeline_c", "dfsar_selene", "sar_optical"]
        )
        is_hyperspectral = (
            "iirs" in [src_s, ref_s] or
            mode in ["pipeline_b", "iirs_wac", "hyperspectral"]
        )
        is_optical = (
            "ohrc" in [src_s, ref_s] or
            "nac" in [src_s, ref_s] or
            mode in ["pipeline_a", "ohrc_nac", "optical"]
        )

        telemetry: Dict[str, Any] = {}

        if is_sar:
            telemetry["pipeline_selected"] = "Pipeline C (DFSAR ↔ SELENE SAR-Optical)"
            out_s, out_r, steps = cls.run_pipeline_c(img_src, img_ref)
            telemetry["steps"] = steps
            return out_s, out_r, telemetry

        elif is_hyperspectral:
            telemetry["pipeline_selected"] = "Pipeline B (IIRS ↔ WAC Hyperspectral)"
            out_s, out_r, steps = cls.run_pipeline_b(img_src, img_ref)
            telemetry["steps"] = steps
            return out_s, out_r, telemetry

        elif is_optical:
            telemetry["pipeline_selected"] = "Pipeline A (OHRC ↔ NAC Optical)"
            out_s, steps_s = cls.run_pipeline_a(img_src)
            out_r, steps_r = cls.run_pipeline_a(img_ref)
            telemetry["steps_src"] = steps_s
            telemetry["steps_ref"] = steps_r
            return out_s, out_r, telemetry

        else:
            # Standard adaptive illumination normalization
            telemetry["pipeline_selected"] = "Standard Adaptive Illumination (Multi-Scale CLAHE + Wallis)"
            u8_s = _to_u8(img_src)
            u8_r = _to_u8(img_ref)
            out_s = normalize_illumination(u8_s, use_wallis=True)
            out_r = normalize_illumination(u8_r, use_wallis=True)
            return out_s, out_r, telemetry
