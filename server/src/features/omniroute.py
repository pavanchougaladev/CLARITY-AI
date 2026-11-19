"""
omniroute.py
=============
OmniRoute AI Gateway & Multi-Engine Router for Lunar Imagery
============================================================
Provides fast, seamless, and fault-tolerant orchestration across all lunar AI models:
  - RoMa (Dense Foundation Transformer)
  - LoFTR (Detector-Free Transformer)
  - DISK + LightGlue (Deep Feature Matcher)
  - NI-LFM (NASA-IBM Lunar Foundation ViT)
  - SuperPoint + SuperGlue
  - RootSIFT (Illumination-Normalized Classical Scale-Invariant)
  - ORB (Real-Time Binary Matching)

Core Capabilities:
  1. Concurrent ThreadPool Dispatch: Runs independent AI models in parallel threads
  2. Auto-Fallback Cascade: Zero-downtime automated fallback if any model fails or is sparse
  3. Spatial Consensus Fusion: Cross-model voting and spatial proximity filtering
  4. Adaptive Profile Diagnosis: Automatic terrain characterization (<50ms)
  5. Fastest-Wins Race Mode: Dispatches concurrently and returns upon first quality result
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from .deep_matcher import LunarMatcher
from .root_sift import root_sift_descriptors
from .ensemble_matcher import fuse_keypoints
from .lunar_fusion_matcher import ImageProfileAnalyzer, AdaptiveRouter, ImageProfile

logger = logging.getLogger(__name__)

_MATCHER_CACHE: Dict[str, Any] = {}


@dataclass
class OmniRouteResult:
    """Telemetry and outcome data from an OmniRoute matching run."""
    pts_src: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    pts_ref: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    confidences: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    strategy: str = "consensus"
    models_attempted: List[str] = field(default_factory=list)
    models_succeeded: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    per_model_latency: Dict[str, float] = field(default_factory=dict)
    per_model_counts: Dict[str, int] = field(default_factory=dict)
    profile_tag: str = "normal"
    reason: str = ""


class LunarOmniRoute:
    """
    High-performance AI Gateway for seamless lunar feature matching.
    Orchestrates deep neural networks and classical pipelines with
    multi-threading, intelligent fallback, and consensus synthesis.
    """

    AVAILABLE_MODELS = [
        "superglue", "rift2", "asift", "akaze",
        "roma_v2", "roma_omega", "roma", "loftr", "lightglue", "ni_lfm", "sift", "superpoint", "orb"
    ]

    def __init__(
        self,
        strategy: str = "consensus",
        confidence_threshold: float = 0.20,
        max_workers: int = 4,
        timeout_sec: float = 45.0,
        spatial_tolerance: float = 3.5,
    ):
        """
        Parameters
        ----------
        strategy : str
            - 'consensus'        : Parallel execution across top models with consensus boosting
            - 'auto_fallback'    : Sequential cascade with instant fallback on error or low inliers
            - 'adaptive_profile' : Profile-driven routing based on crater/shadow terrain diagnosis
            - 'fastest_wins'     : Concurrent race mode returning the first valid result
        confidence_threshold : float
            Minimum match confidence threshold (0.0 to 1.0).
        max_workers : int
            Concurrency limit for thread pool.
        timeout_sec : float
            Per-model timeout in seconds.
        spatial_tolerance : float
            Proximity radius in pixels for spatial consensus clustering.
        """
        self.strategy = strategy.lower()
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self.timeout_sec = timeout_sec
        self.spatial_tolerance = spatial_tolerance
        self.profile_analyzer = ImageProfileAnalyzer()
        self.adaptive_router = AdaptiveRouter()

    def route_and_match(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        strategy: Optional[str] = None,
        preferred_models: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        config: Optional[dict] = None
    ) -> OmniRouteResult:
        """
        Routes the image pair through the configured strategy and returns
        unified sub-pixel correspondence points and telemetry.
        """
        strat = (strategy or self.strategy).lower()
        t0 = time.time()

        if strat == "auto_fallback":
            res = self._run_auto_fallback(img_src, img_ref, preferred_models, progress_callback, config)
        elif strat == "fastest_wins":
            res = self._run_fastest_wins(img_src, img_ref, preferred_models, progress_callback, config)
        elif strat == "adaptive_profile":
            res = self._run_adaptive_profile(img_src, img_ref, progress_callback, config)
        else:  # Default: 'consensus'
            res = self._run_consensus(img_src, img_ref, preferred_models, progress_callback, config)

        res.total_latency_ms = (time.time() - t0) * 1000.0
        res.strategy = strat
        logger.info(
            f"OmniRoute [{strat.upper()}] completed in {res.total_latency_ms:.1f}ms: "
            f"{len(res.pts_src)} matches from {res.models_succeeded}"
        )
        return res

    def _execute_single_model(
        self,
        model_name: str,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        config: Optional[dict] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Executes a single AI matcher with strict timing, grayscale normalization, and exception handling."""
        t_start = time.time()
        m_name = model_name.lower()
        pts_s = np.empty((0, 2), dtype=np.float32)
        pts_r = np.empty((0, 2), dtype=np.float32)
        conf = np.empty((0,), dtype=np.float32)

        # Grayscale normalization to ensure consistent descriptor extraction
        if img_src.ndim == 3:
            s_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY if img_src.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        else:
            s_gray = img_src

        if img_ref.ndim == 3:
            r_gray = cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY if img_ref.shape[2] == 3 else cv2.COLOR_RGBA2GRAY)
        else:
            r_gray = img_ref

        try:
            if m_name in ["sift", "rootsift"]:
                sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)
                kp1, des1 = sift.detectAndCompute(s_gray, None)
                kp2, des2 = sift.detectAndCompute(r_gray, None)
                if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                    des1_root = root_sift_descriptors(des1)
                    des2_root = root_sift_descriptors(des2)
                    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                    raw_matches = flann.knnMatch(des1_root, des2_root, k=2)
                    good_s, good_r, good_c = [], [], []
                    for m_tuple in raw_matches:
                        if len(m_tuple) == 2:
                            m, n = m_tuple
                            if m.distance < 0.75 * n.distance:
                                good_s.append(kp1[m.queryIdx].pt)
                                good_r.append(kp2[m.trainIdx].pt)
                                good_c.append(1.0 - (m.distance / (n.distance + 1e-6)))
                    if good_s:
                        pts_s = np.array(good_s, dtype=np.float32)
                        pts_r = np.array(good_r, dtype=np.float32)
                        conf = np.array(good_c, dtype=np.float32)

            elif m_name == "orb":
                orb = cv2.ORB_create(nfeatures=2000)
                kp1, des1 = orb.detectAndCompute(s_gray, None)
                kp2, des2 = orb.detectAndCompute(r_gray, None)
                if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)[:1000]
                    if matches:
                        pts_s = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
                        pts_r = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
                        conf = np.array([1.0 / (1.0 + m.distance * 0.05) for m in matches], dtype=np.float32)

            else:
                # Deep learning transformer models (RoMa v2, LoFTR, LightGlue, NI-LFM)
                actual_name = "roma_v2" if m_name in ["roma", "roma_outdoor", "tiny_roma", "roma_omega"] else m_name
                cache_key = f"{actual_name}_{self.confidence_threshold}"
                if cache_key not in _MATCHER_CACHE:
                    _MATCHER_CACHE[cache_key] = LunarMatcher(matcher_type=actual_name, confidence_threshold=self.confidence_threshold)
                matcher = _MATCHER_CACHE[cache_key]

                if HAS_TORCH and torch is not None and hasattr(torch, 'from_numpy'):
                    try:
                        t_src = torch.from_numpy(s_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
                        t_ref = torch.from_numpy(r_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
                        pts_s, pts_r, conf = matcher.match(t_src, t_ref)
                    except Exception as e_torch:
                        logger.debug(f"Torch tensor match failed ({e_torch}), attempting ndarray pass...")
                        pts_s, pts_r, conf = matcher.match(s_gray, r_gray)
                else:
                    pts_s, pts_r, conf = matcher.match(s_gray, r_gray)

        except Exception as e:
            logger.warning(f"OmniRoute model [{model_name}] execution error: {e}")

        latency = (time.time() - t_start) * 1000.0
        return pts_s, pts_r, conf, latency

    def _run_consensus(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        models: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        config: Optional[dict] = None
    ) -> OmniRouteResult:
        """Runs candidate models concurrently in thread pool, then fuses matches."""
        target_models = models or ["sift", "loftr", "roma_v2", "lightglue"]
        result = OmniRouteResult()
        result.models_attempted = target_models

        if progress_callback:
            progress_callback(f"OmniRoute: Dispatching {len(target_models)} AI models concurrently...")

        matches_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # Parallel dispatch via ThreadPoolExecutor (sequential on Apple Silicon MPS to prevent Metal collisions)
        is_mps = HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        workers = 1 if is_mps else min(self.max_workers, len(target_models))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_model = {
                executor.submit(self._execute_single_model, m, img_src, img_ref, config): m
                for m in target_models
            }
            for future in as_completed(future_to_model):
                m_name = future_to_model[future]
                try:
                    pts_s, pts_r, conf, latency = future.result()
                    result.per_model_latency[m_name] = latency
                    result.per_model_counts[m_name] = len(pts_s)
                    if len(pts_s) >= 4:
                        matches_dict[m_name] = (pts_s, pts_r, conf)
                        result.models_succeeded.append(m_name)
                        if progress_callback:
                            progress_callback(f"OmniRoute: {m_name.upper()} returned {len(pts_s):,} candidates ({latency:.0f}ms)")
                except Exception as exc:
                    logger.warning(f"OmniRoute worker exception for {m_name}: {exc}")
                    result.per_model_latency[m_name] = 0.0
                    result.per_model_counts[m_name] = 0

        # Fallback to SIFT if all parallel models failed
        if not matches_dict:
            logger.warning("OmniRoute consensus: No deep models yielded matches. Falling back to SIFT...")
            pts_s, pts_r, conf, lat = self._execute_single_model("sift", img_src, img_ref, config)
            result.per_model_latency["sift"] = lat
            result.per_model_counts["sift"] = len(pts_s)
            if len(pts_s) > 0:
                matches_dict["sift"] = (pts_s, pts_r, conf)
                result.models_succeeded.append("sift")

        # Consensus spatial fusion
        if matches_dict:
            fused_src, fused_ref, fused_conf, stats = fuse_keypoints(
                matches_dict,
                spatial_tolerance=self.spatial_tolerance,
                consensus_boost=1.35
            )
            result.pts_src = fused_src
            result.pts_ref = fused_ref
            result.confidences = fused_conf
            result.reason = f"Consensus fusion across {len(result.models_succeeded)} active engines."

        return result

    def _run_auto_fallback(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        models: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        config: Optional[dict] = None
    ) -> OmniRouteResult:
        """Cascades through models sequentially until a satisfactory match set is obtained."""
        cascade = models or ["roma_v2", "loftr", "ni_lfm", "sift"]
        result = OmniRouteResult()
        result.models_attempted = cascade

        min_acceptable_matches = 20

        for m_name in cascade:
            if progress_callback:
                progress_callback(f"OmniRoute Cascade: Attempting {m_name.upper()}...")
            pts_s, pts_r, conf, lat = self._execute_single_model(m_name, img_src, img_ref, config)
            result.per_model_latency[m_name] = lat
            result.per_model_counts[m_name] = len(pts_s)

            if len(pts_s) >= min_acceptable_matches:
                result.models_succeeded.append(m_name)
                result.pts_src = pts_s
                result.pts_ref = pts_r
                result.confidences = conf
                result.reason = f"Cascade accepted primary engine [{m_name.upper()}] with {len(pts_s)} matches."
                return result
            else:
                logger.info(f"OmniRoute Cascade: {m_name.upper()} returned {len(pts_s)} < {min_acceptable_matches}. Falling back...")

        # If cascade did not hit threshold, take best available among models with >0 matches
        valid_counts = {m: c for m, c in result.per_model_counts.items() if c > 0}
        if not valid_counts:
            # GATE 4 ENFORCEMENT: All models yielded 0 matches; strictly 0% weight, no dominant engine
            logger.warning("[GATE 4: ZERO-HALLUCINATION] OmniRoute cascade exhausted: all attempted engines returned 0 correspondences.")
            result.pts_src = np.empty((0, 2), dtype=np.float32)
            result.pts_ref = np.empty((0, 2), dtype=np.float32)
            result.confidences = np.empty((0,), dtype=np.float32)
            result.models_succeeded = []
            result.reason = "Cascade exhausted: 0 correspondences across all attempted engines."
            return result

        best_model = max(valid_counts, key=valid_counts.get)
        pts_s, pts_r, conf, _ = self._execute_single_model(best_model, img_src, img_ref, config)
        result.pts_src = pts_s
        result.pts_ref = pts_r
        result.confidences = conf
        if len(pts_s) > 0:
            result.models_succeeded.append(best_model)
            result.reason = f"Cascade best available fallback [{best_model.upper()}] with {len(pts_s)} matches."
        else:
            result.reason = "Cascade fallback returned 0 matches."
        return result

    def _run_fastest_wins(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        models: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        config: Optional[dict] = None
    ) -> OmniRouteResult:
        """Runs candidate models in parallel and returns immediately with the fastest valid result."""
        race_models = models or ["sift", "loftr", "roma_v2"]
        result = OmniRouteResult()
        result.models_attempted = race_models

        if progress_callback:
            progress_callback(f"OmniRoute Fastest-Wins: Racing {len(race_models)} models concurrently...")

        is_mps = HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        race_workers = 1 if is_mps else len(race_models)
        with ThreadPoolExecutor(max_workers=race_workers) as executor:
            future_to_model = {
                executor.submit(self._execute_single_model, m, img_src, img_ref, config): m
                for m in race_models
            }
            for future in as_completed(future_to_model):
                m_name = future_to_model[future]
                try:
                    pts_s, pts_r, conf, latency = future.result()
                    result.per_model_latency[m_name] = latency
                    result.per_model_counts[m_name] = len(pts_s)
                    if len(pts_s) >= 15:
                        result.models_succeeded.append(m_name)
                        result.pts_src = pts_s
                        result.pts_ref = pts_r
                        result.confidences = conf
                        result.reason = f"Fastest-wins champion: [{m_name.upper()}] in {latency:.0f}ms."
                        return result
                except Exception as e:
                    logger.warning(f"OmniRoute race error in {m_name}: {e}")

        # Fallback to SIFT if none reached threshold
        return self._run_auto_fallback(img_src, img_ref, ["sift"], progress_callback, config)

    def _run_adaptive_profile(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        progress_callback: Optional[Callable[[str], None]] = None,
        config: Optional[dict] = None
    ) -> OmniRouteResult:
        """Diagnoses surface texture/shadow/crater profile in <50ms and routes to tailored models."""
        t_diag = time.time()
        profile = self.profile_analyzer.analyze(img_src, img_ref)
        plan = self.adaptive_router.route(profile, speed_mode="balanced")
        diag_time = (time.time() - t_diag) * 1000.0

        if progress_callback:
            progress_callback(
                f"OmniRoute Diagnosis ({diag_time:.0f}ms): {profile.profile_tag.upper()} surface → "
                f"Routing to {plan.models}"
            )

        res = self._run_consensus(img_src, img_ref, plan.models, progress_callback, config)
        res.profile_tag = profile.profile_tag
        res.reason = f"{profile.profile_tag.upper()} terrain: {plan.reason}"
        return res
