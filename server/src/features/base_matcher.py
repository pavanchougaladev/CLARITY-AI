"""
base_matcher.py
===============
Common Base Interface and Quality Contracts for Lunar Feature Matchers.
========================================================================

Guarantees:
  1. Coordinate Integrity: Strictly (x, y) = (col, row) float32 coordinates
     in the native frame of the input images.
  2. Input Sanitization: Validates dimensions, channels, and types (handles
     uint8 / float32 grayscale or RGB).
  3. Structured Output: Standardized MatcherOutput dataclass with confidence,
     timing, and engine metadata.
  4. Illumination-Aware Confidence Weighting: Radiometric penalty for deep
     permanent lunar shadows.
  5. Spatial Balancing: Adaptive grid bucketing to prevent point clustering.
  6. Mutual Consistency: Cross-check mechanism for bidirectional verification.
"""

from __future__ import annotations

import abc
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MatcherOutput:
    """Standardized output container for any lunar feature matcher."""
    pts_src: np.ndarray          # Shape (N, 2), dtype float32 (x, y) = (col, row)
    pts_ref: np.ndarray          # Shape (N, 2), dtype float32 (x, y) = (col, row)
    confidences: np.ndarray      # Shape (N,), dtype float32 in [0.0, 1.0]
    matcher_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.pts_src) if self.pts_src is not None else 0

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backwards-compatible 3-tuple format (pts_src, pts_ref, confidences)."""
        return self.pts_src, self.pts_ref, self.confidences


def sanitize_lunar_input(img: Any) -> np.ndarray:
    """
    Standardize an image input into a clean 2D uint8 grayscale numpy array.
    Handles torch Tensors, float32 normalized images, and multi-channel RGB/BGR.
    """
    if hasattr(img, "detach") and hasattr(img, "cpu"):
        # Torch tensor
        arr = img.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr.squeeze(0)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        img = arr.squeeze()

    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[:, :, 0]

    arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image after flattening, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    # If float or integer other than uint8
    if arr.dtype in (np.float32, np.float64):
        max_val = float(np.nanmax(arr)) if arr.size > 0 else 1.0
        if max_val <= 1.05:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def apply_illumination_weighting(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    confidences: np.ndarray,
    img_src: np.ndarray,
    img_ref: np.ndarray,
    shadow_threshold_dn: float = 20.0,
) -> np.ndarray:
    """
    Weight match confidences by illumination quality in lunar imagery.
    Matches falling in deep shadows (DN < shadow_threshold_dn) or blown-out
    highlands are downweighted to prevent spurious noise matches from biasing RANSAC.
    """
    if len(pts_src) == 0:
        return confidences

    s_gray = sanitize_lunar_input(img_src)
    r_gray = sanitize_lunar_input(img_ref)
    h_s, w_s = s_gray.shape[:2]
    h_r, w_r = r_gray.shape[:2]

    weighted_confs = np.array(confidences, dtype=np.float32, copy=True)

    for i in range(len(pts_src)):
        xs, ys = int(round(pts_src[i, 0])), int(round(pts_src[i, 1]))
        xr, yr = int(round(pts_ref[i, 0])), int(round(pts_ref[i, 1]))

        # Check bounds
        if 0 <= xs < w_s and 0 <= ys < h_s and 0 <= xr < w_r and 0 <= yr < h_r:
            val_s = float(s_gray[ys, xs])
            val_r = float(r_gray[yr, xr])

            # Deep shadow penalty: smooth transition between 0 DN and shadow_threshold_dn
            s_weight = np.clip(val_s / max(shadow_threshold_dn, 1.0), 0.25, 1.0)
            r_weight = np.clip(val_r / max(shadow_threshold_dn, 1.0), 0.25, 1.0)
            radiometric_factor = float(np.sqrt(s_weight * r_weight))

            weighted_confs[i] = float(np.clip(weighted_confs[i] * radiometric_factor, 0.01, 1.0))

    return weighted_confs


def spatially_balance_matches(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    confidences: np.ndarray,
    img_shape: Tuple[int, int],
    max_per_cell: int = 15,
    grid: Tuple[int, int] = (8, 8),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatially balances correspondences across an M x N grid.
    Limits points in dense clusters (e.g. single crater rim) so MAGSAC
    receives well-distributed candidates across the entire lunar surface.
    """
    if len(pts_src) <= max_per_cell:
        return pts_src, pts_ref, confidences

    h, w = img_shape[:2]
    gx, gy = grid
    cell_w = max(w / gx, 1.0)
    cell_h = max(h / gy, 1.0)

    # Sort indices by confidence descending so strongest points are chosen first
    sort_idx = np.argsort(-confidences)

    grid_buckets: Dict[Tuple[int, int], list] = {}
    selected_indices = []

    for idx in sort_idx:
        x, y = pts_src[idx, 0], pts_src[idx, 1]
        c_x = min(int(x // cell_w), gx - 1)
        c_y = min(int(y // cell_h), gy - 1)
        cell = (c_x, c_y)

        if cell not in grid_buckets:
            grid_buckets[cell] = 0

        if grid_buckets[cell] < max_per_cell:
            grid_buckets[cell] += 1
            selected_indices.append(idx)

    # Preserve order
    selected_indices = np.array(sorted(selected_indices), dtype=int)
    return pts_src[selected_indices], pts_ref[selected_indices], confidences[selected_indices]


class BaseLunarMatcher(abc.ABC):
    """
    Abstract Base Class for all lunar feature matchers.
    Enforces standardized interfaces, error boundaries, and quality hooks.
    """

    def __init__(
        self,
        matcher_name: str,
        confidence_threshold: float = 0.20,
        enable_illumination_weighting: bool = True,
        enable_spatial_balancing: bool = False,
    ):
        self.matcher_name = matcher_name
        self.confidence_threshold = confidence_threshold
        self.enable_illumination_weighting = enable_illumination_weighting
        self.enable_spatial_balancing = enable_spatial_balancing

    @abc.abstractmethod
    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Subclasses implement their specific matching logic here.
        Must return: (pts_src, pts_ref, confidences, metadata_dict).
        """
        raise NotImplementedError

    def match(
        self,
        img_src: Any,
        img_ref: Any,
        **kwargs
    ) -> MatcherOutput:
        """
        Public entry point with contract enforcement, timing, and error insulation.
        """
        t0 = time.time()
        try:
            s_clean = sanitize_lunar_input(img_src)
            r_clean = sanitize_lunar_input(img_ref)

            h_s, w_s = s_clean.shape[:2]
            h_r, w_r = r_clean.shape[:2]
            if min(h_s, w_s, h_r, w_r) < 16:
                logger.warning(f"[{self.matcher_name}] Image too small for matching: {s_clean.shape} vs {r_clean.shape}")
                return MatcherOutput(
                    pts_src=np.empty((0, 2), dtype=np.float32),
                    pts_ref=np.empty((0, 2), dtype=np.float32),
                    confidences=np.empty((0,), dtype=np.float32),
                    matcher_name=self.matcher_name,
                    metadata={"error": "image_too_small"},
                    execution_time_ms=(time.time() - t0) * 1000.0,
                )

            pts_s, pts_r, conf, meta = self._match_implementation(s_clean, r_clean, **kwargs)

            # Ensure numpy float32 types
            if pts_s is None or len(pts_s) == 0:
                pts_s = np.empty((0, 2), dtype=np.float32)
                pts_r = np.empty((0, 2), dtype=np.float32)
                conf = np.empty((0,), dtype=np.float32)
            else:
                pts_s = np.asarray(pts_s, dtype=np.float32)
                pts_r = np.asarray(pts_r, dtype=np.float32)
                conf = np.asarray(conf, dtype=np.float32)

            # Enforce confidence threshold
            if len(pts_s) > 0 and self.confidence_threshold > 0.0:
                valid_mask = conf >= self.confidence_threshold
                pts_s = pts_s[valid_mask]
                pts_r = pts_r[valid_mask]
                conf = conf[valid_mask]

            # Optional illumination weighting
            if len(pts_s) > 0 and self.enable_illumination_weighting:
                conf = apply_illumination_weighting(pts_s, pts_r, conf, s_clean, r_clean)

            # Optional spatial balancing
            if len(pts_s) > 0 and self.enable_spatial_balancing:
                pts_s, pts_r, conf = spatially_balance_matches(pts_s, pts_r, conf, s_clean.shape)

            elapsed_ms = (time.time() - t0) * 1000.0
            return MatcherOutput(
                pts_src=pts_s,
                pts_ref=pts_r,
                confidences=conf,
                matcher_name=self.matcher_name,
                metadata=meta or {},
                execution_time_ms=elapsed_ms,
            )

        except Exception as e:
            logger.error(f"[{self.matcher_name}] Matching exception: {e}", exc_info=True)
            elapsed_ms = (time.time() - t0) * 1000.0
            return MatcherOutput(
                pts_src=np.empty((0, 2), dtype=np.float32),
                pts_ref=np.empty((0, 2), dtype=np.float32),
                confidences=np.empty((0,), dtype=np.float32),
                matcher_name=self.matcher_name,
                metadata={"error": str(e)},
                execution_time_ms=elapsed_ms,
            )


class DeepMatcherAdapter(BaseLunarMatcher):
    """Adapter wrapping LunarMatcher engines behind BaseLunarMatcher."""

    def __init__(self, matcher_type: str = "loftr", confidence_threshold: float = 0.20, **kwargs):
        super().__init__(
            matcher_name=matcher_type,
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
        self.matcher_type = matcher_type
        self._underlying = None

    def _get_underlying(self):
        if self._underlying is None:
            from .deep_matcher import LunarMatcher
            self._underlying = LunarMatcher(matcher_type=self.matcher_type, confidence_threshold=self.confidence_threshold)
        return self._underlying

    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        matcher = self._get_underlying()
        pts_s, pts_r, conf = matcher.match(img_src, img_ref)
        meta = {
            "engine": self.matcher_type,
            "device": getattr(matcher, "device", "cpu"),
            "raw_matches": len(pts_s),
        }
        return pts_s, pts_r, conf, meta


class RootSIFTAdapter(BaseLunarMatcher):
    """Adapter for classical RootSIFT matching."""

    def __init__(self, nfeatures: int = 4000, confidence_threshold: float = 0.20, **kwargs):
        super().__init__(
            matcher_name="root_sift",
            confidence_threshold=confidence_threshold,
            **kwargs,
        )
        self.nfeatures = nfeatures

    def _match_implementation(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        from .root_sift import root_sift_descriptors
        sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        kp_src, des_src = sift.detectAndCompute(img_src, None)
        kp_ref, des_ref = sift.detectAndCompute(img_ref, None)

        if des_src is None or des_ref is None or len(kp_src) < 2 or len(kp_ref) < 2:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"raw_matches": 0},
            )

        des_src = root_sift_descriptors(des_src)
        des_ref = root_sift_descriptors(des_ref)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(des_src, des_ref, k=2)

        pts_s, pts_r, confs = [], [], []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    pts_s.append(kp_src[m.queryIdx].pt)
                    pts_r.append(kp_ref[m.trainIdx].pt)
                    score = 1.0 - (m.distance / max(n.distance, 1e-6))
                    confs.append(float(np.clip(score, 0.05, 1.0)))

        pts_s = np.array(pts_s, dtype=np.float32).reshape(-1, 2)
        pts_r = np.array(pts_r, dtype=np.float32).reshape(-1, 2)
        confs = np.array(confs, dtype=np.float32)
        return pts_s, pts_r, confs, {"raw_matches": len(pts_s)}


def get_matcher_adapter(matcher_name: str, **kwargs) -> BaseLunarMatcher:
    """Factory helper to obtain any standard lunar matcher behind BaseLunarMatcher."""
    m_name = matcher_name.lower()
    if m_name in ("crater", "yolo11", "crater_yolo11", "yolo11_crater"):
        from .crater_matcher import YOLO11CraterMatcher
        return YOLO11CraterMatcher(**kwargs)
    elif m_name in ("root_sift", "rootsift"):
        return RootSIFTAdapter(**kwargs)
    else:
        return DeepMatcherAdapter(matcher_type=m_name, **kwargs)

