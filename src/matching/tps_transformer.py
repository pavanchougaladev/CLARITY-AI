"""
tps_transformer.py
==================
Thin-Plate Spline (TPS) Non-Rigid Distortion & Jitter Transformer
==================================================================
Provides sub-pixel pushbroom sensor jitter modeling and non-rigid distortion
correction across lunar orbital imaging lines.

Supports:
1. OpenCV ThinPlateSplineShapeTransformer (when available in cv2 build).
2. Pure Python / Scipy Thin-Plate Spline fallback via RBFInterpolator
   with bi-harmonic / thin-plate spline kernel (r^2 * ln(r)), guaranteeing
   robust non-rigid jitter correction on macOS Apple Silicon, headless Linux,
   and minimal container environments.
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScipyThinPlateSplineTransformer:
    """
    Drop-in replacement for cv2.ThinPlateSplineShapeTransformer using SciPy.
    Models spacecraft pushbroom sinusoidal line jitter with sub-pixel residual precision.
    """

    def __init__(self, smoothing: float = 0.0):
        self.smoothing = smoothing
        self.src_pts: Optional[np.ndarray] = None
        self.ref_pts: Optional[np.ndarray] = None
        self._rbf_forward = None
        self._rbf_inverse = None

    def estimateTransformation(
        self,
        src_points: np.ndarray,
        ref_points: np.ndarray,
        matches: Optional[list] = None,
    ) -> bool:
        """
        Estimate non-rigid TPS transformation mapping src_points to ref_points.
        Supports shapes (1, N, 2) or (N, 2).
        """
        from scipy.interpolate import RBFInterpolator

        s = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
        r = np.asarray(ref_points, dtype=np.float64).reshape(-1, 2)

        if len(s) < 4:
            raise ValueError(f"TPS estimation requires at least 4 control points, got {len(s)}")

        self.src_pts = s
        self.ref_pts = r

        # Forward displacement: src -> ref
        disp_fwd = r - s
        self._rbf_forward = RBFInterpolator(
            s, disp_fwd,
            kernel="thin_plate_spline",
            smoothing=self.smoothing,
        )

        # Inverse displacement: ref -> src (for backward image warping)
        disp_inv = s - r
        self._rbf_inverse = RBFInterpolator(
            r, disp_inv,
            kernel="thin_plate_spline",
            smoothing=self.smoothing,
        )
        return True

    def applyTransformation(self, pts: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Apply estimated TPS to points.
        Returns (0.0, transformed_pts) preserving the input tensor shape.
        """
        orig_shape = pts.shape
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self._rbf_forward is None:
            return 0.0, pts

        disp = self._rbf_forward(p)
        transformed = (p + disp).astype(np.float32)
        return 0.0, transformed.reshape(orig_shape)

    def warpImage(self, img: np.ndarray, output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Warp source image into reference geometry using backward TPS interpolation.
        Supports explicit output_shape (h_ref, w_ref) and high-fidelity bicubic interpolation.
        """
        if self._rbf_inverse is None:
            return img.copy()

        if output_shape is not None:
            h, w = output_shape
        else:
            h, w = img.shape[:2]

        # Fast meshgrid generation for reference image coordinate frame
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float64)

        disp = self._rbf_inverse(grid_pts)
        src_x = (grid_pts[:, 0] + disp[:, 0]).reshape(h, w).astype(np.float32)
        src_y = (grid_pts[:, 1] + disp[:, 1]).reshape(h, w).astype(np.float32)

        return cv2.remap(img, src_x, src_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


_NATIVE_CV2_TPS = getattr(cv2, "createThinPlateSplineShapeTransformer", None)


class PushbroomJitterTransformer:
    """
    Decoupled Along-Track / Cross-Track Spline Model for Orbital Pushbroom Sensors.

    Orbital pushbroom line-scanners (e.g. Chandrayaan-2 TMC-2 / OHRC) acquire images
    line-by-line where line index y represents orbital flight time (t = y / v_orbit).
    Spacecraft attitude vibrations (pitch, roll, yaw jitter) modulate coordinates along
    the time axis (y), while cross-track geometry is dictated by the linear CCD array (x).

    This transformer:
    1. Fits a robust global baseline homography / affine model.
    2. Isolates residual displacements along the temporal scanline axis.
    3. Fits regularized cubic smoothing B-splines along y to model high-frequency jitter
       without boundary oscillations (Runge's phenomenon).
    """

    def __init__(self, s_factor: float = 0.5):
        self.s_factor = s_factor
        self.H_base = None
        self._spline_x = None
        self._spline_y = None
        self._inv_spline_x = None
        self._inv_spline_y = None
        self.H_inv = None

    def estimateTransformation(
        self,
        src_points: np.ndarray,
        ref_points: np.ndarray,
        matches: Optional[list] = None,
    ) -> bool:
        from scipy.interpolate import UnivariateSpline

        s = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
        r = np.asarray(ref_points, dtype=np.float64).reshape(-1, 2)

        if len(s) < 4:
            raise ValueError(f"Pushbroom jitter model requires at least 4 points, got {len(s)}")

        # 1. Fit robust global projective base
        H, _ = cv2.findHomography(s, r, cv2.RANSAC, 3.0)
        if H is None or H.shape != (3, 3):
            H, _ = cv2.estimateAffinePartial2D(s, r)
            if H is not None:
                H = np.vstack([H, [0.0, 0.0, 1.0]])
            else:
                H = np.eye(3, dtype=np.float64)

        self.H_base = H
        try:
            self.H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.H_inv = np.eye(3, dtype=np.float64)

        # 2. Forward predicted points from base H
        s_homo = np.column_stack([s, np.ones(len(s), dtype=np.float64)])
        pred_h = (self.H_base @ s_homo.T).T
        pred_xy = pred_h[:, :2] / np.maximum(np.abs(pred_h[:, 2:3]), 1e-12)

        # Residuals relative to base model
        res_x = r[:, 0] - pred_xy[:, 0]
        res_y = r[:, 1] - pred_xy[:, 1]

        # 3. Fit 1D cubic splines along the along-track dimension (y)
        # Sort by y to ensure monotonically increasing knots
        y_coords = s[:, 1]
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        rx_sorted = res_x[sort_idx]
        ry_sorted = res_y[sort_idx]

        # Deduplicate consecutive identical y-coordinates
        _, unique_idx = np.unique(y_sorted, return_index=True)
        if len(unique_idx) >= 6:
            yu = y_sorted[unique_idx]
            rx_u = rx_sorted[unique_idx]
            ry_u = ry_sorted[unique_idx]

            smoothing = float(len(yu) * self.s_factor)
            try:
                self._spline_x = UnivariateSpline(yu, rx_u, k=3, s=smoothing)
                self._spline_y = UnivariateSpline(yu, ry_u, k=3, s=smoothing)
            except Exception:
                self._spline_x = None
                self._spline_y = None
        else:
            self._spline_x = None
            self._spline_y = None

        return True

    def applyTransformation(self, pts: np.ndarray) -> Tuple[float, np.ndarray]:
        orig_shape = pts.shape
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if self.H_base is None:
            return 0.0, pts

        p_homo = np.column_stack([p, np.ones(len(p), dtype=np.float64)])
        pred_h = (self.H_base @ p_homo.T).T
        pred_xy = pred_h[:, :2] / np.maximum(np.abs(pred_h[:, 2:3]), 1e-12)

        if self._spline_x is not None and self._spline_y is not None:
            jit_x = self._spline_x(p[:, 1])
            jit_y = self._spline_y(p[:, 1])
            # Clamp jitter corrections to avoid edge extrapolation divergence
            jit_x = np.clip(jit_x, -15.0, 15.0)
            jit_y = np.clip(jit_y, -15.0, 15.0)
            pred_xy[:, 0] += jit_x
            pred_xy[:, 1] += jit_y

        transformed = pred_xy.astype(np.float32)
        return 0.0, transformed.reshape(orig_shape)

    def warpImage(self, img: np.ndarray, output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if self.H_base is None:
            return img.copy()
        if output_shape is not None:
            h, w = output_shape
        else:
            h, w = img.shape[:2]

        # If we have splines, apply base homography plus inverse jitter remap
        if self._spline_x is not None and self._spline_y is not None:
            try:
                grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
                H_inv = np.linalg.inv(self.H_base)
                denom = H_inv[2, 0] * grid_x + H_inv[2, 1] * grid_y + H_inv[2, 2]
                denom = np.where(np.abs(denom) < 1e-7, 1e-7, denom)
                src_x0 = (H_inv[0, 0] * grid_x + H_inv[0, 1] * grid_y + H_inv[0, 2]) / denom
                src_y0 = (H_inv[1, 0] * grid_x + H_inv[1, 1] * grid_y + H_inv[1, 2]) / denom

                jit_x = self._spline_x(src_y0.ravel()).reshape(h, w)
                jit_y = self._spline_y(src_y0.ravel()).reshape(h, w)
                remap_x = (src_x0 - jit_x).astype(np.float32)
                remap_y = (src_y0 - jit_y).astype(np.float32)

                return cv2.remap(img, remap_x, remap_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            except Exception as e_jit:
                logger.warning(f"Pushbroom jitter remap fallback: {e_jit}")

        return cv2.warpPerspective(img, self.H_base, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def create_thin_plate_spline_transformer():
    """Factory creating an OpenCV or Scipy Thin-Plate Spline shape transformer."""
    if _NATIVE_CV2_TPS is not None:
        try:
            return _NATIVE_CV2_TPS()
        except Exception:
            pass
    return ScipyThinPlateSplineTransformer()


def create_pushbroom_jitter_transformer(s_factor: float = 0.5):
    """Factory creating a decoupled along-track / cross-track pushbroom jitter spline transformer."""
    return PushbroomJitterTransformer(s_factor=s_factor)


# Transparently polyfill cv2 if absent so existing callers and tests work everywhere
if _NATIVE_CV2_TPS is None:
    cv2.createThinPlateSplineShapeTransformer = create_thin_plate_spline_transformer


