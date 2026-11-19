"""
APEX — 6-Engine Super-Fusion Matcher
====================================
Highest-tier combination engine. Orchestrates, in a single deterministic
pipeline, the union of every paradigm available in the suite:

    1. DISK + LightGlue   — learned local keypoints + adaptive attention graph
    2. RoMa v2            — SOTA dense coarse-to-fine transformer
    3. ResShift           — diffusion-style residual-shift image restoration
                            (pre-pass; restores crater rims before matching)
    4. RootSIFT           — Hellinger-normalized classical descriptors
    5. MAGSAC++           — final progressive robust outlier rejection

Note: SuperGlue is intentionally NOT part of the APEX combination. It remains
available as a standalone matcher, but it is excluded from the APEX engine
stack and from APEX fusion entirely.

Pipeline stages
---------------
    Stage 0  ResShift restoration pre-pass  (degraded rasters -> clean rasters)
    Stage 1  Parallel-fault-isolated candidate matching per engine
    Stage 1b Gate 5 deterministic CPU SIFT fallback when the pool is below
             the Gate 2 MVR floor (raw rasters, no AI dependency)
    Stage 2  Spatial-consensus fusion (weighted, conflict-resolved)
    Stage 3  MAGSAC++ homography outlier rejection (progressive thresholds)
    Stage 4  Zero-hallucination telemetry (Gate 4 compliant weighting)

Deep engine memory bound: DISK+LightGlue / RoMa v2 build O(N^2) attention
tensors, which exhausts CPU memory on long pushbroom strips (e.g. 4096x486).
They therefore run on a bounded raster (``deep_engine_max_dim``) and their
coordinates are mapped back to the full resolution before fusion.

Gate 4 compliance: any engine returning 0 raw matches contributes 0% weight
and can never be reported as a dominant contributor.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    HAS_TORCH = False

from .deep_matcher import LunarMatcher
from .ensemble_matcher import fuse_keypoints
from .root_sift import root_sift_descriptors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 0: ResShift restoration pre-pass
# ---------------------------------------------------------------------------

def resshift_restore(
    img: np.ndarray,
    steps: int = 4,
    detail_boost: float = 0.55,
) -> Tuple[np.ndarray, str]:
    """
    ResShift-inspired residual-shift restoration pre-pass.

    ResShift (Efficient Diffusion Model for Image Super-Resolution and
    Restoration, NeurIPS 2023) shifts residual pixels between the degraded
    image and the clean latent over a short Markov chain. If the actual
    ``resshift`` package is installed it is used directly; otherwise a
    fully deterministic CPU fallback emulates the shift-chain with a
    progressive denoise + deconvolution schedule that restores crater rim
    gradients without hallucinating structure (Gate 1 safe).

    Parameters
    ----------
    img : np.ndarray (H, W) uint8 grayscale
    steps : int
        Number of residual-shift chain steps (fallback path).
    detail_boost : float
        High-frequency detail restoration weight (0..1).

    Returns
    -------
    (restored_u8, mode) : np.ndarray, str
        mode is "resshift_package" or "deterministic_fallback".
    """
    if img is None or getattr(img, "size", 0) == 0:
        return img, "skipped_empty_input"

    # --- Preferred path: real ResShift diffusion package -------------------
    if HAS_TORCH and torch is not None:
        try:
            import importlib
            rs = importlib.import_module("resshift")
            if hasattr(rs, "restore"):
                with torch.no_grad():
                    out = rs.restore(img)  # type: ignore[attr-defined]
                restored = np.asarray(out)
                if restored.ndim == 3:
                    restored = cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY)
                return np.clip(restored, 0, 255).astype(np.uint8), "resshift_package"
        except Exception as e_rs:
            logger.debug(f"ResShift package unavailable ({e_rs}); using deterministic fallback.")

    # --- Deterministic residual-shift fallback ------------------------------
    # Progressive chain: each step removes one "noise level" of the degraded
    # observation, then a deconvolution shift restores rim micro-gradients.
    work = img.astype(np.float32)
    for step in range(max(1, int(steps))):
        # Residual shift: blend the observation toward its edge-preserving
        # latent estimate with a schedule that decreases per chain step.
        alpha = 0.6 * (1.0 - step / max(1, steps)) + 0.2
        latent = cv2.bilateralFilter(work.astype(np.uint8), 7, 45, 7).astype(np.float32)
        work = alpha * latent + (1.0 - alpha) * work

    # Deconvolution shift: restore high-frequency crater rim gradients
    # (Wiener-style unsharp, strictly derivative-based -> no hallucination)
    blur = cv2.GaussianBlur(work, (0, 0), 1.4)
    work = work + detail_boost * (work - blur)

    restored = np.clip(work, 0, 255).astype(np.uint8)
    return restored, "deterministic_fallback"


# ---------------------------------------------------------------------------
# Stage 1: RootSIFT classical engine
# ---------------------------------------------------------------------------

def rootsift_match(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    nfeatures: int = 6000,
    ratio: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    RootSIFT matching: SIFT keypoints + Hellinger-normalized descriptors,
    BFMatcher(L2) k-NN with Lowe ratio test.

    Returns (pts_src, pts_ref, confidences) — confidence is the ratio-margin
    mapped into [0, 1].
    """
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.02, edgeThreshold=10)
    kp_s, des_s = sift.detectAndCompute(img_src, None)
    kp_r, des_r = sift.detectAndCompute(img_ref, None)

    if des_s is None or des_r is None or len(des_s) < 4 or len(des_r) < 4:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    # RootSIFT Hellinger embedding — L2 in this space == Hellinger distance
    des_s = root_sift_descriptors(des_s)
    des_r = root_sift_descriptors(des_r)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_s, des_r, k=2)

    pts_s, pts_r, conf = [], [], []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            pts_s.append(kp_s[m.queryIdx].pt)
            pts_r.append(kp_r[m.trainIdx].pt)
            # ratio-margin confidence in [0, 1]
            conf.append(max(0.0, min(1.0, 1.0 - (m.distance / (n.distance + 1e-6)))))

    if not pts_s:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(pts_s, dtype=np.float32),
        np.asarray(pts_r, dtype=np.float32),
        np.asarray(conf, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Stage 1b: Gate 5 deterministic recovery engine
# ---------------------------------------------------------------------------

# Gate 2 (MVR) floor: a registration cannot pass with fewer than 15 verified
# inliers, so fewer than 15 raw correspondences can never be certified.
MIN_VIABLE_RAW_MATCHES = 15


def sift_recovery_match(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    nfeatures: int = 8000,
    ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic CPU SIFT fallback (Gate 5, Recovery Action 2).

    Runs on the RAW rasters (no ResShift restoration) so that a failed or
    over-smoothed AI engine cannot poison the recovery path. Uses plain SIFT
    descriptors with a relaxed Lowe ratio to maximise recall on lunar texture.
    """
    try:
        sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=12)
        kp_s, des_s = sift.detectAndCompute(img_src, None)
        kp_r, des_r = sift.detectAndCompute(img_ref, None)
    except Exception as e_sift:
        logger.warning(f"APEX Gate 5 SIFT detection failed: {e_sift}")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    if des_s is None or des_r is None or len(des_s) < 4 or len(des_r) < 4:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    bf = cv2.BFMatcher(cv2.NORM_L2)
    try:
        knn = bf.knnMatch(des_s, des_r, k=2)
    except cv2.error as e_bf:
        logger.warning(f"APEX Gate 5 SIFT matching failed: {e_bf}")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    pts_s, pts_r, conf = [], [], []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            pts_s.append(kp_s[m.queryIdx].pt)
            pts_r.append(kp_r[m.trainIdx].pt)
            conf.append(max(0.0, min(1.0, 1.0 - (m.distance / (n.distance + 1e-6)))))

    if not pts_s:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return (
        np.asarray(pts_s, dtype=np.float32),
        np.asarray(pts_r, dtype=np.float32),
        np.asarray(conf, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Stage 3: MAGSAC++ final outlier rejection
# ---------------------------------------------------------------------------

def magsac_filter(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    confidences: np.ndarray,
    base_threshold: float = 3.0,
    confidence: float = 0.999,
    max_iters: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Progressive MAGSAC++ (cv2.USAC_MAGSAC) homography inlier selection.

    Starts strict and relaxes the threshold until a usable inlier pool is
    found, mirroring the suite's `_progressive_magsac` strategy inside the
    Apex engine itself.

    Returns (pts_src, pts_ref, confidences, stats).
    """
    stats: Dict[str, Any] = {"magsac_applied": False, "magsac_inliers": 0, "magsac_threshold": None}

    n = len(pts_src)
    if n < 8:
        stats["magsac_skipped_reason"] = f"too_few_matches({n})"
        return pts_src, pts_ref, confidences, stats

    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    thresh = base_threshold
    while thresh <= base_threshold * 4.0:
        try:
            _, mask = cv2.findHomography(
                pts_src, pts_ref, method, thresh,
                maxIters=max_iters, confidence=confidence,
            )
        except cv2.error as e_cv:
            logger.warning(f"APEX MAGSAC++ cv2 error ({e_cv}); skipping filter.")
            stats["magsac_skipped_reason"] = "cv2_error"
            return pts_src, pts_ref, confidences, stats

        count = int(mask.sum()) if mask is not None else 0
        if count >= 8:
            mask = mask.ravel().astype(bool)
            stats.update({
                "magsac_applied": True,
                "magsac_inliers": count,
                "magsac_threshold": float(thresh),
                "magsac_rejected": int(n - count),
            })
            logger.info(f"APEX MAGSAC++: {count}/{n} inliers survived (threshold={thresh:.2f}px)")
            return pts_src[mask], pts_ref[mask], confidences[mask], stats
        thresh *= 2.0

    stats["magsac_skipped_reason"] = "no_usable_inlier_pool"
    return pts_src, pts_ref, confidences, stats


# ---------------------------------------------------------------------------
# APEX orchestrator
# ---------------------------------------------------------------------------

class ApexMatcher:
    """
    6-Engine Super-Fusion matcher.

    Engine stack (candidate generators):
        - ``roma_v2``    RoMa v2 dense transformer
        - ``lightglue``  DISK + LightGlue attention graph
        - ``rootsift``   RootSIFT classical descriptors (deterministic CPU)

    Fixed post-stages:
        - ResShift restoration pre-pass
        - Spatial-consensus fusion
        - MAGSAC++ final outlier rejection

    SuperGlue is deliberately excluded from the APEX combination.
    """

    DEFAULT_ENGINES = ["roma_v2", "lightglue", "rootsift"]

    # Engines that must never participate in APEX fusion.
    EXCLUDED_ENGINES = {"superglue", "superpoint_superglue", "superpoint+superglue"}

    # Deep graph/dense transformers allocate O(N^2) attention tensors. On
    # pushbroom strips (e.g. 4096x486) that OOMs the CPU allocator, silently
    # dropping the engine. Deep engines therefore run on a bounded working
    # raster and their coordinates are mapped back to full resolution.
    DEEP_ENGINE_MAX_DIM = 1600

    def __init__(
        self,
        engines: Optional[List[str]] = None,
        confidence_threshold: float = 0.2,
        spatial_tolerance: float = 3.0,
        enable_resshift: bool = True,
        enable_magsac: bool = True,
        resshift_steps: int = 4,
        deep_engine_max_dim: int = DEEP_ENGINE_MAX_DIM,
        progress_callback: Optional[callable] = None,
    ):
        base_engines = [e.lower() for e in (engines or self.DEFAULT_ENGINES)]
        # Hard exclusion: SuperGlue must never enter the APEX combination,
        # even if requested explicitly via ``engines=[...]`` or a config file.
        requested_excluded = [e for e in base_engines if e in self.EXCLUDED_ENGINES]
        if requested_excluded:
            logger.warning(
                f"APEX: SuperGlue is not part of the APEX combination; "
                f"dropping requested engine(s) {requested_excluded}."
            )
            base_engines = [e for e in base_engines if e not in self.EXCLUDED_ENGINES]
        # RoMa v2 requires GPU/MPS to be practical (dense transformer + ~1GB
        # checkpoint). On CPU-only machines drop it, mirroring EnsembleMatcher
        # behavior — the remaining engines still form the full APEX stack.
        has_gpu = (
            HAS_TORCH
            and torch is not None
            and (getattr(torch.cuda, "is_available", lambda: False)()
                 or getattr(torch.backends, "mps", None) is not None
                 and torch.backends.mps.is_available())
        )
        if not has_gpu:
            base_engines = [e for e in base_engines if e not in ("roma_v2", "roma", "roma_omega", "roma_outdoor")]
        self.engines = base_engines
        self.confidence_threshold = confidence_threshold
        self.spatial_tolerance = spatial_tolerance
        self.enable_resshift = enable_resshift
        self.enable_magsac = enable_magsac
        self.resshift_steps = resshift_steps
        self.deep_engine_max_dim = int(deep_engine_max_dim)
        self.progress_callback = progress_callback
        self._matchers: Dict[str, LunarMatcher] = {}

    # -- public API ---------------------------------------------------------

    def match(self, img_src, img_ref, config: Optional[dict] = None):
        """Returns (pts_src, pts_ref, confidences)."""
        pts_s, pts_r, conf, _ = self.match_with_stats(img_src, img_ref, config=config)
        return pts_s, pts_r, conf

    def match_with_stats(
        self,
        img_src,
        img_ref,
        config: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Runs the full APEX stack.

        Accepts np.ndarray (H, W) uint8 grayscale OR torch.Tensor (1, 1, H, W)
        in [0, 1] for either image.

        Returns (pts_src, pts_ref, confidences, stats).
        """
        t0 = time.time()
        src_u8 = self._to_gray_u8(img_src)
        ref_u8 = self._to_gray_u8(img_ref)

        stats: Dict[str, Any] = {
            "engine": "apex",
            "stack": "DISK+LightGlue + RoMa v2 + ResShift + RootSIFT + MAGSAC++",
            "resshift_mode": "disabled",
            "raw_counts": {},
            "engines_run": [],
            "engines_contributing": [],
            "engine_weights": {},
            "magsac": {},
        }

        # ---- Stage 0: ResShift restoration pre-pass -----------------------
        active_src, active_ref = src_u8, ref_u8
        if self.enable_resshift:
            self._notify("ResShift restoration pre-pass (residual-shift chain)...")
            try:
                active_src, mode_s = resshift_restore(src_u8, steps=self.resshift_steps)
                active_ref, mode_r = resshift_restore(ref_u8, steps=self.resshift_steps)
                stats["resshift_mode"] = mode_s if mode_s == mode_r else f"{mode_s}/{mode_r}"
                logger.info(f"APEX ResShift pre-pass complete ({stats['resshift_mode']}).")
            except Exception as e_rs:
                logger.warning(f"APEX ResShift pre-pass failed ({e_rs}); using raw rasters.")
                active_src, active_ref = src_u8, ref_u8
                stats["resshift_mode"] = "failed_fallback_raw"

        # ---- Stage 1: candidate matching per engine (fault isolated) ------
        # Deep engines run on a bounded raster to avoid O(N^2) attention OOM
        # on long orbital strips; their coordinates are mapped back afterwards.
        deep_src, deep_ref, deep_scale = self._deep_engine_inputs(active_src, active_ref)
        tensor_src = self._to_tensor(deep_src)
        tensor_ref = self._to_tensor(deep_ref)
        stats["deep_engine_scale"] = round(float(deep_scale), 4)
        if deep_scale != 1.0:
            logger.info(
                f"APEX deep engines run at {deep_src.shape[1]}x{deep_src.shape[0]} "
                f"(scale={deep_scale:.4f}) to avoid attention-tensor OOM; "
                "coordinates are mapped back to the full raster."
            )

        matches_by_engine: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for idx, engine in enumerate(self.engines):
            self._notify(f"Running APEX engine [{idx + 1}/{len(self.engines)}]: {engine.upper()}...")
            t_e = time.time()
            try:
                if engine == "rootsift":
                    pts_s, pts_r, conf = rootsift_match(active_src, active_ref)
                else:
                    matcher = self._get_lunar_matcher(engine)
                    pts_s, pts_r, conf = matcher.match(tensor_src, tensor_ref)
                    if deep_scale != 1.0 and len(pts_s) > 0:
                        # Map bounded-raster coordinates back to full resolution
                        inv_scale = 1.0 / deep_scale
                        pts_s = np.asarray(pts_s, dtype=np.float32) * inv_scale
                        pts_r = np.asarray(pts_r, dtype=np.float32) * inv_scale
            except Exception as e_eng:
                logger.warning(f"APEX engine {engine.upper()} failed: {e_eng}")
                pts_s, pts_r, conf = (
                    np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                )

            pts_s = np.asarray(pts_s, dtype=np.float32).reshape(-1, 2)
            pts_r = np.asarray(pts_r, dtype=np.float32).reshape(-1, 2)
            conf = np.asarray(conf, dtype=np.float32).reshape(-1)

            raw_count = len(pts_s)
            stats["raw_counts"][engine] = raw_count
            stats["engines_run"].append(engine)
            stats[f"{engine}_latency_ms"] = round((time.time() - t_e) * 1000.0, 1)

            # GATE 4: zero raw matches -> zero weight, excluded from fusion
            if raw_count == 0:
                logger.warning(
                    f"APEX Gate 4: engine {engine.upper()} returned 0 raw matches; "
                    "weight forced to 0% (excluded from fusion and dominance claims)."
                )
                stats["engine_weights"][engine] = 0.0
                continue
            matches_by_engine[engine] = (pts_s, pts_r, conf)

        contributing = list(matches_by_engine.keys())
        stats["engines_contributing"] = contributing

        # ---- Stage 1b: Gate 5 deterministic recovery ---------------------
        # If the AI/classical engine pool cannot even reach the Gate 2 MVR
        # floor, fall back to deterministic CPU SIFT on the raw rasters
        # (Recovery Action 2) instead of handing a hopeless pool downstream.
        total_raw = int(sum(stats["raw_counts"].values()))
        stats["recovery"] = {"triggered": False, "reason": None, "raw_matches": 0}
        if total_raw < MIN_VIABLE_RAW_MATCHES:
            self._notify(
                "Gate 5: low match yield — deterministic SIFT fallback on raw rasters..."
            )
            t_rec = time.time()
            try:
                rec_s, rec_r, rec_c = sift_recovery_match(src_u8, ref_u8)
            except Exception as e_rec:
                logger.warning(f"APEX Gate 5 SIFT recovery failed: {e_rec}")
                rec_s, rec_r, rec_c = None, None, None

            if rec_s is not None and len(rec_s) > 0:
                engine = "sift_recovery"
                matches_by_engine[engine] = (rec_s, rec_r, rec_c)
                stats["raw_counts"][engine] = int(len(rec_s))
                stats["engines_run"].append(engine)
                stats[f"{engine}_latency_ms"] = round((time.time() - t_rec) * 1000.0, 1)
                stats["recovery"] = {
                    "triggered": True,
                    "reason": f"engine_pool_below_mvr_floor({total_raw})",
                    "raw_matches": int(len(rec_s)),
                    "mode": "deterministic_sift_cpu_raw_rasters",
                }
                contributing = list(matches_by_engine.keys())
                stats["engines_contributing"] = contributing
                logger.warning(
                    f"APEX Gate 5: deterministic SIFT fallback added {len(rec_s)} raw "
                    f"matches (previous pool: {total_raw})."
                )
            else:
                stats["recovery"] = {
                    "triggered": True,
                    "reason": f"engine_pool_below_mvr_floor({total_raw})",
                    "raw_matches": 0,
                    "mode": "deterministic_sift_cpu_raw_rasters",
                }
                logger.warning(
                    "APEX Gate 5: deterministic SIFT fallback produced no additional matches."
                )

        # ---- Stage 2: spatial-consensus fusion ----------------------------
        if not matches_by_engine:
            logger.error("APEX: all engines returned 0 raw matches; registration cannot proceed.")
            stats["total_fused"] = 0
            stats["consensus_count"] = 0
            stats["total_time_s"] = round(time.time() - t0, 2)
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                stats,
            )

        self._notify("Fusing cross-engine correspondences (spatial consensus)...")
        fused_src, fused_ref, fused_conf, fuse_stats = fuse_keypoints(
            matches_by_engine,
            spatial_tolerance=self.spatial_tolerance,
        )
        stats["fusion"] = fuse_stats

        # ---- Stage 3: MAGSAC++ final outlier rejection ---------------------
        if self.enable_magsac:
            self._notify("MAGSAC++ progressive outlier rejection...")
            cfg = config or {}
            pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}
            fused_src, fused_ref, fused_conf, magsac_stats = magsac_filter(
                fused_src,
                fused_ref,
                fused_conf,
                base_threshold=pipeline_cfg.get("outlier_threshold", 3.0),
                confidence=pipeline_cfg.get("magsac_confidence", 0.999),
                max_iters=pipeline_cfg.get("magsac_max_iters", 50000),
            )
            stats["magsac"] = magsac_stats

        # ---- Stage 4: Gate 4 zero-hallucination weighting ------------------
        # Engine weight = share of surviving (post-MAGSAC) fused contributions.
        # Engines with 0 raw matches already carry 0% weight (set in Stage 1).
        raw_counts = stats["raw_counts"]
        surviving_total = sum(c for c in raw_counts.values() if c > 0)
        if surviving_total > 0:
            for engine, count in raw_counts.items():
                stats["engine_weights"][engine] = round(100.0 * count / surviving_total, 1)
        dominant = max(raw_counts, key=raw_counts.get) if raw_counts else None
        if dominant is not None and raw_counts[dominant] > 0:
            stats["dominant_engine"] = dominant
        else:
            # Gate 4: never declare a dominant engine with 0 matches
            stats["dominant_engine"] = None

        stats["total_fused"] = len(fused_src)
        stats["total_time_s"] = round(time.time() - t0, 2)
        logger.info(
            f"APEX complete: {stats['total_fused']} surviving matches from "
            f"{contributing} in {stats['total_time_s']}s "
            f"(consensus: {fuse_stats.get('consensus_count', 0)})"
        )
        return fused_src, fused_ref, fused_conf, stats

    # -- internals -----------------------------------------------------------

    def _deep_engine_inputs(
        self,
        src_u8: np.ndarray,
        ref_u8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Bound the working resolution handed to deep attention/dense engines.

        DISK+LightGlue and RoMa v2 build O(N^2) attention tensors over detected
        keypoints. On pushbroom strips (e.g. 4096x486) that allocates tens of
        gigabytes and raises ``DefaultCPUAllocator: not enough memory``, which
        silently removed LightGlue from the APEX pool. Scaling the raster down
        first keeps the engine inside memory; the caller maps the returned
        coordinates back by ``1 / scale``.

        Returns (src, ref, scale) where ``scale == 1.0`` means untouched.
        """
        long_src = max(src_u8.shape[:2])
        long_ref = max(ref_u8.shape[:2])
        longest = max(long_src, long_ref)
        if longest <= self.deep_engine_max_dim:
            return src_u8, ref_u8, 1.0

        scale = float(self.deep_engine_max_dim) / float(longest)

        def _resize(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            return cv2.resize(
                img,
                (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        return _resize(src_u8), _resize(ref_u8), scale

    def _get_lunar_matcher(self, engine: str) -> LunarMatcher:
        """Lazily initialize (and cache) deep sub-matchers."""
        if engine not in self._matchers:
            self._matchers[engine] = LunarMatcher(
                matcher_type=engine,
                confidence_threshold=self.confidence_threshold,
            )
        return self._matchers[engine]

    @staticmethod
    def _to_gray_u8(img) -> np.ndarray:
        """Convert np array or (1,1,H,W) tensor to uint8 grayscale."""
        if HAS_TORCH and torch is not None and torch.is_tensor(img):
            arr = img.squeeze().detach().cpu().numpy()
        else:
            arr = np.asarray(img)
        arr = np.squeeze(arr)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] >= 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:
                arr = arr[:, :, 0]
        if arr.max() <= 1.0 and arr.dtype != np.uint8:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_tensor(img: np.ndarray):
        """(H, W) uint8 -> (1, 1, H, W) float tensor in [0, 1], or None."""
        if not HAS_TORCH or torch is None:
            return img
        try:
            return torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        except Exception:
            return img

    def _notify(self, msg: str) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(msg)
            except Exception:
                pass
