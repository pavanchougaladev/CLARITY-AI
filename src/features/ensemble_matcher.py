"""
Ensemble Matcher — Multi-Model Hybrid Feature Fusion
=====================================================
Combines multiple complementary feature matchers (RoMa, LoFTR, DISK+LightGlue, SIFT)
into a unified correspondence set.

Core principles:
1. Diversity: Combines dense transformers (RoMa), semi-dense patch matchers (LoFTR),
   graph-refined keypoints (LightGlue), and classical multiscale gradients (SIFT).
2. Spatial Consensus: Detects when multiple independent models agree on a correspondence;
   boosts confidence for consensus matches.
3. Conflict Resolution: Discards or downweights ambiguous pairs where models disagree
   on the destination coordinates for the same source region.
4. Coverage Maximization: Retains non-overlapping valid keypoints across crater rims,
   plains, and shadowed terrains.
5. Fault Isolation: Gracefully tolerates individual model OOM or failure.
"""

import logging
import time
import numpy as np
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False
from typing import List, Tuple, Dict, Any, Optional

from .deep_matcher import LunarMatcher
from .tile_matcher import tile_match, create_sift_tile_matcher

logger = logging.getLogger(__name__)


def fuse_keypoints(
    matches_by_model: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    spatial_tolerance: float = 3.0,
    consensus_boost: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fuses candidate correspondences from multiple matching models with spatial
    consensus boosting and conflict rejection.

    Parameters
    ----------
    matches_by_model : dict
        Mapping of model_name -> (pts_src, pts_ref, confidences).
    spatial_tolerance : float
        Max Euclidean distance in pixels to consider two source points
        as referring to the same physical feature. Default: 3.0 px.
    consensus_boost : float
        Confidence bonus added when 2+ models agree on reference target.

    Returns
    -------
    fused_pts_src : np.ndarray (M, 2)
    fused_pts_ref : np.ndarray (M, 2)
    fused_conf    : np.ndarray (M,)
    stats         : dict containing counts per model and consensus stats.
    """
    all_src_list = []
    all_ref_list = []
    all_conf_list = []
    all_origin_list = []

    model_counts = {}
    for model_name, (pts_s, pts_r, conf) in matches_by_model.items():
        count = len(pts_s)
        model_counts[model_name] = count
        if count == 0:
            continue
        all_src_list.append(pts_s)
        all_ref_list.append(pts_r)
        
        # Ensure confidence is a 1D float array in [0, 1]
        c = np.asarray(conf, dtype=np.float32)
        if c.ndim == 0 or len(c) != count:
            c = np.ones(count, dtype=np.float32) * 0.5
        c = np.clip(c, 0.0, 1.0)
        all_conf_list.append(c)
        all_origin_list.extend([model_name] * count)

    if not all_src_list:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {"model_counts": model_counts, "total_raw": 0, "consensus_count": 0, "total_fused": 0},
        )

    all_src = np.vstack(all_src_list).astype(np.float32)
    all_ref = np.vstack(all_ref_list).astype(np.float32)
    all_conf = np.concatenate(all_conf_list).astype(np.float32)
    N = len(all_src)

    if N < 2:
        return (
            all_src,
            all_ref,
            all_conf,
            {"model_counts": model_counts, "total_raw": N, "consensus_count": 0, "total_fused": N},
        )

    # Spatial clustering on source points using grid bucketing (fast O(N) spatial index)
    grid_cell_size = spatial_tolerance
    grid = {}
    for i in range(N):
        gx = int(np.floor(all_src[i, 0] / grid_cell_size))
        gy = int(np.floor(all_src[i, 1] / grid_cell_size))
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)

    visited = np.zeros(N, dtype=bool)
    fused_src = []
    fused_ref = []
    fused_conf = []
    consensus_count = 0

    # Sort indices by confidence descending so strongest points anchor clusters
    sorted_order = np.argsort(-all_conf)

    for idx in sorted_order:
        if visited[idx]:
            continue

        gx = int(np.floor(all_src[idx, 0] / grid_cell_size))
        gy = int(np.floor(all_src[idx, 1] / grid_cell_size))

        # Check current and 8 neighboring grid cells
        cluster_indices = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor_key = (gx + dx, gy + dy)
                if neighbor_key in grid:
                    for cand_idx in grid[neighbor_key]:
                        if not visited[cand_idx]:
                            dist_src = np.linalg.norm(all_src[idx] - all_src[cand_idx])
                            if dist_src <= spatial_tolerance:
                                cluster_indices.append(cand_idx)

        if not cluster_indices:
            cluster_indices = [idx]

        for ci in cluster_indices:
            visited[ci] = True

        if len(cluster_indices) == 1:
            # Single-model detection — keep as-is
            i = cluster_indices[0]
            fused_src.append(all_src[i])
            fused_ref.append(all_ref[i])
            fused_conf.append(all_conf[i])
        else:
            # Multiple candidate points within tolerance in source space
            c_src = all_src[cluster_indices]
            c_ref = all_ref[cluster_indices]
            c_conf = all_conf[cluster_indices]

            # Check if reference points also agree (consensus) or diverge (conflict)
            ref_anchor = c_ref[0]  # Reference from highest-confidence point
            ref_dists = np.linalg.norm(c_ref - ref_anchor, axis=1)
            agree_mask = ref_dists <= spatial_tolerance

            if np.sum(agree_mask) > 1:
                # Consensus match! Multiple models agree
                consensus_count += 1
                agree_src = c_src[agree_mask]
                agree_ref = c_ref[agree_mask]
                agree_conf = c_conf[agree_mask]

                weights = agree_conf / (np.sum(agree_conf) + 1e-8)
                weighted_src = np.sum(agree_src * weights[:, None], axis=0)
                weighted_ref = np.sum(agree_ref * weights[:, None], axis=0)

                # Consensus confidence boost
                boosted_conf = min(1.0, float(np.max(agree_conf) + consensus_boost))

                fused_src.append(weighted_src)
                fused_ref.append(weighted_ref)
                fused_conf.append(boosted_conf)
            else:
                # Disagreement or single consistent point: keep highest-confidence match
                best_i = np.argmax(c_conf)
                fused_src.append(c_src[best_i])
                fused_ref.append(c_ref[best_i])
                fused_conf.append(c_conf[best_i])

    fused_src = np.array(fused_src, dtype=np.float32)
    fused_ref = np.array(fused_ref, dtype=np.float32)
    fused_conf = np.array(fused_conf, dtype=np.float32)

    stats = {
        "model_counts": model_counts,
        "total_raw": N,
        "total_fused": len(fused_src),
        "consensus_count": consensus_count,
    }
    logger.info(
        f"Ensemble Fusion: {N} raw matches across {len(matches_by_model)} models "
        f"-> {len(fused_src)} fused matches ({consensus_count} consensus agreement boosts)"
    )
    return fused_src, fused_ref, fused_conf, stats


class EnsembleMatcher:
    """
    Multi-model ensemble matcher executing an orchestrated set of deep and
    classical feature matching algorithms.

    Supported model types
    ---------------------
    ``roma``     — RoMa dense transformer (CVPR 2024 SOTA)
    ``loftr``    — Local Feature Transformer (CVPR 2021)
    ``sift``     — Classical SIFT with tile-based coverage
    ``ni_lfm``   — NASA-IBM Lunar Foundation Model (domain-adapted ViT-Base,
                   trained on 20 lunar modalities incl. vis / dtm / slope)
    ``lightglue`` / ``disk_lightglue`` — DISK + LightGlue keypoint matcher
    ``tiny_roma`` — Lightweight RoMa variant
    """

    # Default ensemble: RoMa v2 + LoFTR + SIFT  (fast, proven on Chandrayaan data)
    # Add "ni_lfm" here to enable the NASA-IBM backbone as a 4th model.
    DEFAULT_MODELS = ["roma_v2", "loftr", "sift"]

    def __init__(
        self,
        models: Optional[List[str]] = None,
        confidence_threshold: float = 0.2,
        spatial_tolerance: float = 3.0,
    ):
        base_models = list(models or self.DEFAULT_MODELS)
        has_cuda = torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
        if not has_cuda:
            base_models = ["lightglue" if m in ["roma_v2", "roma_omega"] else m for m in base_models]
        self.models = base_models
        self.confidence_threshold = confidence_threshold
        self.spatial_tolerance = spatial_tolerance

    def match(
        self,
        img_src_np: np.ndarray,
        img_ref_np: np.ndarray,
        config: Optional[dict] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
        """
        Executes ensemble matching across all configured models and fuses the results.

        Parameters
        ----------
        img_src_np : np.ndarray (H, W) uint8 grayscale
        img_ref_np : np.ndarray (H, W) uint8 grayscale
        config     : dict, optional pipeline config
        progress_callback : callable, optional

        Returns
        -------
        pts_src          : np.ndarray (M, 2)
        pts_ref          : np.ndarray (M, 2)
        confidences      : np.ndarray (M,)
        models_succeeded : list of str
        stats            : dict
        """
        config = config or {}
        matches_by_model = {}
        models_succeeded = []

        # Prepare normalized tensors for deep models
        tensor_src, tensor_ref = None, None
        if HAS_TORCH and torch is not None:
            try:
                tensor_src = torch.from_numpy(img_src_np).float().unsqueeze(0).unsqueeze(0) / 255.0
                tensor_ref = torch.from_numpy(img_ref_np).float().unsqueeze(0).unsqueeze(0) / 255.0
            except Exception:
                pass

        total_models = len(self.models)
        for idx, m_type in enumerate(self.models):
            m_type = m_type.lower()
            logger.info(f"Ensemble running sub-matcher [{idx+1}/{total_models}]: {m_type.upper()}...")
            if progress_callback:
                progress_callback(f"Running {m_type.upper()} in Ensemble...")

            t0 = time.time()
            try:
                if m_type == "sift":
                    # SIFT with tile-based coverage
                    sift_tile_fn = create_sift_tile_matcher(config)
                    pts_s, pts_r, conf = tile_match(
                        img_src_np,
                        img_ref_np,
                        matcher_fn=sift_tile_fn,
                        grid=(2, 2),
                        overlap=0.25,
                        confidence_threshold=0.1,
                    )
                else:
                    # Deep matchers: RoMa, Tiny-RoMa, LoFTR, LightGlue, NI-LFM
                    matcher = LunarMatcher(
                        matcher_type=m_type,
                        confidence_threshold=self.confidence_threshold,
                    )
                    input_s = tensor_src if tensor_src is not None else img_src_np
                    input_r = tensor_ref if tensor_ref is not None else img_ref_np
                    pts_s, pts_r, conf = matcher.match(input_s, input_r)

                elapsed = time.time() - t0
                if len(pts_s) > 0:
                    matches_by_model[m_type] = (pts_s, pts_r, conf)
                    models_succeeded.append(m_type)
                    logger.info(
                        f"Ensemble sub-matcher {m_type.upper()} succeeded: "
                        f"{len(pts_s)} matches in {elapsed:.2f}s"
                    )
                else:
                    logger.warning(
                        f"Ensemble sub-matcher {m_type.upper()} returned 0 matches ({elapsed:.2f}s)"
                    )
            except Exception as e:
                logger.warning(f"Ensemble sub-matcher {m_type.upper()} failed: {e}")

        # If none of the preferred models succeeded, try a final classical SIFT fallback
        if not matches_by_model:
            logger.warning("All primary ensemble models failed; attempting fallback SIFT...")
            try:
                sift_tile_fn = create_sift_tile_matcher(config)
                pts_s, pts_r, conf = tile_match(
                    img_src_np,
                    img_ref_np,
                    matcher_fn=sift_tile_fn,
                    grid=(2, 2),
                    overlap=0.25,
                    confidence_threshold=0.1,
                )
                if len(pts_s) > 0:
                    matches_by_model["sift_fallback"] = (pts_s, pts_r, conf)
                    models_succeeded.append("sift")
            except Exception as e:
                logger.error(f"Fallback SIFT in ensemble also failed: {e}")

        # Fuse candidate correspondences
        fused_src, fused_ref, fused_conf, stats = fuse_keypoints(
            matches_by_model,
            spatial_tolerance=self.spatial_tolerance,
        )

        return fused_src, fused_ref, fused_conf, models_succeeded, stats
