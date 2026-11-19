"""
Cross-Sun Illumination-Invariant Matching
=========================================
Lunar image pairs acquired at different solar azimuth/elevation angles can
have almost no overlap in raw intensity space: a crater floor that is a deep
shadow in the morning pass can be a bright highlight in the afternoon pass.
Intensity descriptors (plain SIFT, LoFTR on radiance, phase correlation on
DN patches) then either fail outright or return a handful of clustered
matches, and the quality gate correctly rejects the result.

What *is* stable across illumination is the surface relief signature: the
gradient-magnitude field peaks on crater rims, boulder edges and scarps
regardless of which side is lit. This module builds several
gradient-magnitude channels and runs the existing RootSIFT matcher on each,
then fuses the correspondences with a spatial non-maximum suppression pass.

The matcher is injected by the caller (``src.pipeline.sift_match``) so this
module stays free of circular imports and remains unit-testable.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..preprocess.shadow_normalization import normalize_illumination

logger = logging.getLogger(__name__)

MatcherFn = Callable[
    [np.ndarray, np.ndarray, Optional[dict]],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def to_gray(img: np.ndarray) -> np.ndarray:
    """Return a 2D uint8 grayscale view of an image."""
    if img is None:
        raise ValueError("cross-sun matcher received None image")
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
        elif arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def gradient_magnitude_image(
    img: np.ndarray,
    gain: float = 90.0,
    ksize: int = 3,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Build a contrast-normalized gradient-magnitude image.

    The mean magnitude is forced to 1.0 before applying ``gain`` so the same
    gain produces comparable contrast on both frames even when their absolute
    DN ranges differ. Gain values around 60-120 were empirically stable for
    cross-sun lunar pairs; the exact value mainly affects SIFT keypoint
    density, not the geometric result.
    """
    gray = to_gray(img).astype(np.float32)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), float(blur_sigma))
    k = int(ksize) | 1
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = np.hypot(gx, gy)
    mag /= max(float(np.mean(mag)), 1e-6)
    return np.clip(mag * float(gain), 0.0, 255.0).astype(np.uint8)


def merge_correspondences(
    sets: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tolerance_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse several correspondence sets while suppressing duplicates.

    A candidate is a duplicate when both its source and reference endpoints
    are within ``tolerance_px`` of an already accepted pair. Higher-confidence
    candidates are admitted first, so fusion keeps the strongest evidence in
    each spatial neighbourhood.
    """
    candidates: List[Tuple[float, float, float, float, float]] = []
    for pts_s, pts_r, conf in sets:
        if pts_s is None or pts_r is None or len(pts_s) == 0:
            continue
        n = min(len(pts_s), len(pts_r))
        conf = np.asarray(conf, dtype=np.float32) if conf is not None and len(conf) else np.ones(n, np.float32)
        for i in range(n):
            c = float(conf[i]) if i < len(conf) else 0.5
            candidates.append(
                (c, float(pts_s[i, 0]), float(pts_s[i, 1]), float(pts_r[i, 0]), float(pts_r[i, 1]))
            )

    if not candidates:
        empty2 = np.empty((0, 2), dtype=np.float32)
        return empty2, empty2.copy(), np.empty((0,), dtype=np.float32)

    candidates.sort(key=lambda item: -item[0])
    accepted: List[Tuple[float, float, float, float, float]] = []
    tol2 = float(tolerance_px) ** 2
    for cand in candidates:
        _, sx, sy, rx, ry = cand
        duplicate = False
        for acc in accepted:
            _, asx, asy, arx, ary = acc
            if (sx - asx) ** 2 + (sy - asy) ** 2 <= tol2 and (rx - arx) ** 2 + (ry - ary) ** 2 <= tol2:
                duplicate = True
                break
        if not duplicate:
            accepted.append(cand)

    pts_src = np.array([[a[1], a[2]] for a in accepted], dtype=np.float32)
    pts_ref = np.array([[a[3], a[4]] for a in accepted], dtype=np.float32)
    conf = np.array([a[0] for a in accepted], dtype=np.float32)
    return pts_src, pts_ref, conf


def cross_sun_channels(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    gains: Sequence[float] = (60.0, 90.0, 120.0),
    include_illumination_normalized: bool = True,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Return the gradient-magnitude channel pairs used for cross-sun matching."""
    gray_src = to_gray(img_src)
    gray_ref = to_gray(img_ref)

    channels: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for gain in gains:
        channels.append(
            (
                f"grad_raw_g{gain:g}",
                gradient_magnitude_image(gray_src, gain=gain),
                gradient_magnitude_image(gray_ref, gain=gain),
            )
        )

    if include_illumination_normalized:
        norm_src = normalize_illumination(gray_src.copy())
        norm_ref = normalize_illumination(gray_ref.copy())
        channels.append(
            (
                "grad_wallis_g90",
                gradient_magnitude_image(norm_src, gain=90.0),
                gradient_magnitude_image(norm_ref, gain=90.0),
            )
        )
    return channels


def match_cross_sun_gradient(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    matcher_fn: MatcherFn,
    config: Optional[dict] = None,
    gains: Sequence[float] = (60.0, 90.0, 120.0),
    include_illumination_normalized: bool = True,
    dedupe_tolerance_px: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Match two images using illumination-invariant gradient channels.

    Parameters
    ----------
    img_src, img_ref : np.ndarray
        Grayscale (or colour) images after resolution normalization.
    matcher_fn : callable
        ``(src, ref, config) -> (pts_src, pts_ref, confidences)`` matcher,
        normally the pipeline RootSIFT two-tier matcher.
    config : dict, optional
        Passed through to ``matcher_fn``.
    gains : sequence of float
        Gradient gain values to sample. More channels improve robustness but
        cost one SIFT extraction per channel.
    include_illumination_normalized : bool
        Also match on the gradient of the Wallis/CLAHE normalized images.

    Returns
    -------
    (pts_src, pts_ref, confidences, stats)
    """
    stats: Dict[str, object] = {
        "engine": "cross_sun_gradient_rootsift",
        "channels": [],
        "raw_matches_per_channel": {},
        "raw_matches": 0,
        "dedupe_tolerance_px": float(dedupe_tolerance_px),
    }

    sets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for name, chan_src, chan_ref in cross_sun_channels(
        img_src, img_ref, gains=gains, include_illumination_normalized=include_illumination_normalized
    ):
        try:
            pts_s, pts_r, conf = matcher_fn(chan_src, chan_ref, config)
        except Exception as exc:  # a single channel must never sink the fallback
            logger.warning(f"[cross-sun] channel {name} failed: {exc}")
            pts_s, pts_r, conf = np.empty((0, 2)), np.empty((0, 2)), np.empty((0,))

        n = len(pts_s) if pts_s is not None else 0
        stats["raw_matches_per_channel"][name] = int(n)
        stats["channels"].append(name)
        if n > 0:
            sets.append((np.asarray(pts_s, np.float32), np.asarray(pts_r, np.float32), conf))

    pts_src, pts_ref, confidences = merge_correspondences(sets, tolerance_px=dedupe_tolerance_px)
    stats["raw_matches"] = int(len(pts_src))
    stats["channels_with_matches"] = int(len(sets))
    return pts_src, pts_ref, confidences, stats
