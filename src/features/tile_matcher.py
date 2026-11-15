"""
Tile-Based Feature Matching — Spatial Coverage Maximizer
==========================================================
Divides source and reference images into overlapping tiles,
matches each tile pair independently, then aggregates all
matches into a unified set with full spatial coverage.

This directly addresses the low-coverage problem where whole-image
matching clusters tie-points in only one region (9–37% coverage).
"""

import numpy as np
import cv2
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False
import logging
from typing import Tuple
from .root_sift import root_sift_descriptors

logger = logging.getLogger(__name__)


def tile_match(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    matcher_fn,
    grid: Tuple[int, int] = (3, 3),
    overlap: float = 0.25,
    min_tile_size: int = 128,
    confidence_threshold: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tile-based matching for improved spatial coverage.

    Divides both images into an NxN grid of overlapping tiles,
    runs the matcher on each corresponding tile pair, maps coordinates
    back to original image space, and de-duplicates overlapping matches.

    Parameters
    ----------
    img_src           : Grayscale source image (H, W) uint8.
    img_ref           : Grayscale reference image (H, W) uint8.
    matcher_fn        : Callable(tile_src, tile_ref) -> (pts_src, pts_ref, confs).
    grid              : (rows, cols) number of tiles.
    overlap           : Fractional overlap between adjacent tiles (0.0–0.5).
    min_tile_size     : Minimum tile dimension in pixels.
    confidence_threshold : Minimum confidence to keep a match.

    Returns
    -------
    (pts_src, pts_ref, confidences) in original image coordinates.
    """
    h_s, w_s = img_src.shape[:2]
    h_r, w_r = img_ref.shape[:2]

    if grid is None:
        # Auto-compute optimal grid so tiles fall in the 256-768px sweet spot
        max_img_dim = max(h_s, w_s)
        grid_dim = max(2, min(5, max_img_dim // 512))
        aspect = h_s / max(w_s, 1)
        if aspect > 3.0:
            # Tall strip: more rows, fewer columns
            rows = max(grid_dim, max(2, int(aspect * 1.5)))
            cols = max(1, grid_dim // 2)
        elif aspect < 0.33:
            # Wide strip: more columns, fewer rows
            rows = max(1, grid_dim // 2)
            cols = max(grid_dim, max(2, int((1.0 / aspect) * 1.5)))
        else:
            rows = grid_dim
            cols = grid_dim
    else:
        rows, cols = grid
        # Auto-adjust if image is an extreme strip and default (3,3) is used
        aspect = h_s / max(w_s, 1)
        if aspect > 3.0 and grid == (3, 3):
            cols = 1
            rows = max(1, int(aspect))
        elif aspect < 0.33 and grid == (3, 3):
            rows = 1
            cols = max(1, int(1.0 / aspect))
            
    overlap = np.clip(overlap, 0.0, 0.5)

    # Adaptive minimum tile size: scale down for smaller images, floor at 32px
    max_grid_dim = max(rows, cols)
    min_dim = min(min(h_s, w_s), min(h_r, w_r))
    adaptive_min_tile = max(32, min_dim // (max_grid_dim * 2))
    effective_min_tile = min(min_tile_size, adaptive_min_tile)

    # Calculate tile dimensions for source image
    tile_h_s = h_s / rows
    tile_w_s = w_s / cols
    pad_h_s = int(tile_h_s * overlap)
    pad_w_s = int(tile_w_s * overlap)

    # Calculate tile dimensions for reference image
    tile_h_r = h_r / rows
    tile_w_r = w_r / cols
    pad_h_r = int(tile_h_r * overlap)
    pad_w_r = int(tile_w_r * overlap)

    all_pts_src = []
    all_pts_ref = []
    all_confs = []

    total_tiles = rows * cols
    matched_tiles = 0

    # Create tile-level CLAHE for local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for r in range(rows):
        for c in range(cols):
            # Source tile bounds with overlap padding
            y0_s = max(int(r * tile_h_s) - pad_h_s, 0)
            y1_s = min(int((r + 1) * tile_h_s) + pad_h_s, h_s)
            x0_s = max(int(c * tile_w_s) - pad_w_s, 0)
            x1_s = min(int((c + 1) * tile_w_s) + pad_w_s, w_s)

            # Reference tile bounds with overlap padding
            y0_r = max(int(r * tile_h_r) - pad_h_r, 0)
            y1_r = min(int((r + 1) * tile_h_r) + pad_h_r, h_r)
            x0_r = max(int(c * tile_w_r) - pad_w_r, 0)
            x1_r = min(int((c + 1) * tile_w_r) + pad_w_r, w_r)

            # Skip tiles that are too small
            if (y1_s - y0_s) < effective_min_tile or (x1_s - x0_s) < effective_min_tile:
                continue
            if (y1_r - y0_r) < effective_min_tile or (x1_r - x0_r) < effective_min_tile:
                continue

            tile_src = img_src[y0_s:y1_s, x0_s:x1_s]
            tile_ref = img_ref[y0_r:y1_r, x0_r:x1_r]

            # Per-tile local contrast enhancement for difficult lunar lighting
            if tile_src.dtype == np.uint8 and tile_src.shape[0] >= 16 and tile_src.shape[1] >= 16:
                tile_src_proc = clahe.apply(tile_src)
            else:
                tile_src_proc = tile_src
            if tile_ref.dtype == np.uint8 and tile_ref.shape[0] >= 16 and tile_ref.shape[1] >= 16:
                tile_ref_proc = clahe.apply(tile_ref)
            else:
                tile_ref_proc = tile_ref

            try:
                pts_s, pts_r, confs = matcher_fn(tile_src_proc, tile_ref_proc)

                if len(pts_s) == 0:
                    continue

                # Filter by confidence threshold
                if len(confs) > 0:
                    keep = confs >= confidence_threshold
                    pts_s = pts_s[keep]
                    pts_r = pts_r[keep]
                    confs = confs[keep]

                if len(pts_s) == 0:
                    continue

                # Map tile-local coordinates back to full-image coordinates
                pts_s_global = pts_s.copy()
                pts_s_global[:, 0] += x0_s
                pts_s_global[:, 1] += y0_s

                pts_r_global = pts_r.copy()
                pts_r_global[:, 0] += x0_r
                pts_r_global[:, 1] += y0_r

                all_pts_src.append(pts_s_global)
                all_pts_ref.append(pts_r_global)
                all_confs.append(confs)
                matched_tiles += 1

            except Exception as e:
                logger.debug(f"Tile ({r},{c}) matching failed: {e}")
                continue

    logger.info(f"Tile matching: {matched_tiles}/{total_tiles} tiles produced matches")

    if not all_pts_src:
        return np.array([]), np.array([]), np.array([])

    pts_src_all = np.vstack(all_pts_src).astype(np.float32)
    pts_ref_all = np.vstack(all_pts_ref).astype(np.float32)
    confs_all = np.concatenate(all_confs).astype(np.float32)

    # De-duplicate matches from overlapping tiles
    pts_src_all, pts_ref_all, confs_all = _deduplicate_matches(
        pts_src_all, pts_ref_all, confs_all, radius=3.0
    )

    logger.info(f"Tile matching total: {len(pts_src_all)} unique matches after dedup")
    return pts_src_all, pts_ref_all, confs_all


def _deduplicate_matches(
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    confs: np.ndarray,
    radius: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate matches from overlapping tiles using a KDTree for O(n log n)
    complexity instead of a naive O(n²) nested loop.

    When two matches are within `radius` pixels in both source and reference,
    keep the one with higher confidence.
    """
    if len(pts_src) <= 1:
        return pts_src, pts_ref, confs

    # Sort by confidence descending so we always prefer the higher-confidence match
    order = np.argsort(-confs)
    pts_src = pts_src[order]
    pts_ref = pts_ref[order]
    confs = confs[order]

    try:
        from scipy.spatial import KDTree  # O(n log n) spatial lookup
        tree_src = KDTree(pts_src)
        tree_ref = KDTree(pts_ref)

        keep = np.ones(len(pts_src), dtype=bool)
        for i in range(len(pts_src)):
            if not keep[i]:
                continue
            # Find neighbours in source space
            neighbours_src = tree_src.query_ball_point(pts_src[i], r=radius)
            for j in neighbours_src:
                if j <= i or not keep[j]:
                    continue
                # Also require proximity in reference space
                if np.linalg.norm(pts_ref[i] - pts_ref[j]) < radius:
                    keep[j] = False  # j has lower confidence (already sorted)

    except ImportError:
        # scipy not available — fall back to vectorised numpy approach (still O(n²)
        # but avoids Python loop overhead via broadcasting for modest n)
        keep = np.ones(len(pts_src), dtype=bool)
        for i in range(len(pts_src)):
            if not keep[i]:
                continue
            dists_src = np.linalg.norm(pts_src[i + 1:] - pts_src[i], axis=1)
            dists_ref = np.linalg.norm(pts_ref[i + 1:] - pts_ref[i], axis=1)
            dup_mask = (dists_src < radius) & (dists_ref < radius)
            dup_indices = np.where(dup_mask)[0] + i + 1
            keep[dup_indices] = False

    return pts_src[keep], pts_ref[keep], confs[keep]


def create_sift_tile_matcher(config: dict = None):
    """
    Returns a matcher function compatible with tile_match()
    that uses SIFT + BFMatcher internally with two-tier ratio test.
    
    Tier 1: strict ratio (0.75) — high-confidence matches
    Tier 2: relaxed ratio (0.82) on remaining unmatched features,
            kept only if they pass local affine consistency with nearest primary matches.
    """
    config = config or {}
    nfeatures = config.get('sift_fallback', {}).get('nfeatures', 8000)
    ratio_thresh = config.get('sift_fallback', {}).get('ratio_threshold', 0.75)
    ratio_thresh_secondary = config.get('sift_fallback', {}).get('ratio_threshold_secondary', 0.82)

    def _local_affine_consistent(kp_query, kp_train, primary_kp1, primary_kp2, k_check=3):
        """Check if a secondary match is consistent in scale/orientation with nearest primary matches."""
        if len(primary_kp1) < k_check:
            return True  # Not enough primaries to check, accept by default

        # Find k nearest primary keypoints in source space
        q_pt = np.array(kp_query.pt)
        dists = np.array([np.linalg.norm(q_pt - np.array(pk.pt)) for pk in primary_kp1])
        nearest_idx = np.argsort(dists)[:k_check]

        # Check scale ratio consistency
        scale_ratio = kp_train.size / max(kp_query.size, 1e-6)
        neighbor_ratios = []
        for ni in nearest_idx:
            nr = primary_kp2[ni].size / max(primary_kp1[ni].size, 1e-6)
            neighbor_ratios.append(nr)
        if not neighbor_ratios:
            return True
        median_ratio = float(np.median(neighbor_ratios))
        if median_ratio > 0 and abs(scale_ratio / median_ratio - 1.0) > 0.5:
            return False  # Scale inconsistent

        # Check orientation consistency
        angle_diff = (kp_train.angle - kp_query.angle) % 360
        neighbor_angles = []
        for ni in nearest_idx:
            na = (primary_kp2[ni].angle - primary_kp1[ni].angle) % 360
            neighbor_angles.append(na)
        if neighbor_angles:
            median_angle = float(np.median(neighbor_angles))
            angle_dev = min(abs(angle_diff - median_angle), 360 - abs(angle_diff - median_angle))
            if angle_dev > 30:  # More than 30° deviation from neighbors
                return False

        return True

    def _match(tile_src: np.ndarray, tile_ref: np.ndarray):
        sift = cv2.SIFT_create(nfeatures=nfeatures)
        kp1, des1 = sift.detectAndCompute(tile_src, None)
        kp2, des2 = sift.detectAndCompute(tile_ref, None)
        des1 = root_sift_descriptors(des1)  # Upgrade to RootSIFT (Hellinger metric)
        des2 = root_sift_descriptors(des2)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return np.array([]), np.array([]), np.array([])

        # Adaptive ratio: relax when match pool is small (low-texture maria)
        n_candidates = min(len(des1), len(des2))
        adaptive_ratio = ratio_thresh if n_candidates >= 500 else min(ratio_thresh + 0.05, 0.85)
        adaptive_ratio_secondary = ratio_thresh_secondary if n_candidates >= 500 else min(ratio_thresh_secondary + 0.03, 0.88)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_fwd = bf.knnMatch(des1, des2, k=2)
        raw_bwd = bf.knnMatch(des2, des1, k=2)

        # === TIER 1: Primary strict ratio test ===
        fwd_pass = {}
        for m_list in raw_fwd:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < adaptive_ratio * n.distance:
                    fwd_pass[m.queryIdx] = m

        bwd_pass = {}
        for m_list in raw_bwd:
            if len(m_list) == 2:
                m, n = m_list
                if m.distance < adaptive_ratio * n.distance:
                    bwd_pass[m.queryIdx] = m

        # Cross-check: keep only mutual nearest neighbours (primary matches)
        primary_good = [
            fwd_pass[i] for i in fwd_pass
            if fwd_pass[i].trainIdx in bwd_pass
            and bwd_pass[fwd_pass[i].trainIdx].trainIdx == i
        ]

        # Track which query/train indices are already matched
        matched_query = {m.queryIdx for m in primary_good}
        matched_train = {m.trainIdx for m in primary_good}

        # Build primary keypoint lists for affine consistency check
        primary_kp1 = [kp1[m.queryIdx] for m in primary_good]
        primary_kp2 = [kp2[m.trainIdx] for m in primary_good]

        # === TIER 2: Secondary relaxed ratio test on remaining unmatched features ===
        secondary_good = []
        if len(primary_good) >= 3:  # Need enough primaries for consistency check
            fwd_pass_2 = {}
            for m_list in raw_fwd:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.queryIdx not in matched_query and m.trainIdx not in matched_train:
                        if m.distance < adaptive_ratio_secondary * n.distance:
                            fwd_pass_2[m.queryIdx] = m

            bwd_pass_2 = {}
            for m_list in raw_bwd:
                if len(m_list) == 2:
                    m, n = m_list
                    if m.queryIdx not in matched_train and m.trainIdx not in matched_query:
                        if m.distance < adaptive_ratio_secondary * n.distance:
                            bwd_pass_2[m.queryIdx] = m

            # Cross-check for secondary matches
            secondary_candidates = [
                fwd_pass_2[i] for i in fwd_pass_2
                if fwd_pass_2[i].trainIdx in bwd_pass_2
                and bwd_pass_2[fwd_pass_2[i].trainIdx].trainIdx == i
            ]

            # Local affine consistency filter for secondary matches
            for m in secondary_candidates:
                if _local_affine_consistent(kp1[m.queryIdx], kp2[m.trainIdx], primary_kp1, primary_kp2):
                    secondary_good.append(m)

            if secondary_good:
                logger.debug(f"SIFT two-tier: {len(primary_good)} primary + {len(secondary_good)} secondary matches")

        good = primary_good + secondary_good

        if len(good) < 4:
            return np.array([]), np.array([]), np.array([])

        pts_src = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_ref = np.float32([kp2[m.trainIdx].pt for m in good])

        # Ratio-test based confidence: more distinctive matches get higher scores.
        # For each match m, find nearest competitor distance from knn list.
        # Confidence = 1 - (m.distance / competitor_distance), clipped to [0, 1].
        # Fall back to linear normalization if knn info unavailable.
        fwd_knn_map = {}
        for m_list in raw_fwd:
            if len(m_list) == 2:
                m, n = m_list
                fwd_knn_map[m.queryIdx] = (m.distance, n.distance)

        confidences_list = []
        for m in good:
            if m.queryIdx in fwd_knn_map:
                d1, d2 = fwd_knn_map[m.queryIdx]
                if d2 > 1e-6:
                    conf = float(np.clip(1.0 - d1 / d2, 0.0, 1.0))
                else:
                    conf = 0.5
            else:
                # Secondary match: use distance directly
                max_dist_val = max(m.distance for m in good) + 1e-6
                conf = float(np.clip(1.0 - m.distance / max_dist_val, 0.0, 1.0))
            confidences_list.append(conf)
        confidences = np.array(confidences_list, dtype=np.float32)

        return pts_src, pts_ref, confidences

    return _match


def create_deep_tile_matcher(matcher_type: str = "loftr", device: str = "cpu"):
    """
    Returns a matcher function compatible with tile_match()
    that uses a deep neural matcher (LoFTR / LightGlue) internally.
    Guarded against missing torch: returns empty arrays when torch is unavailable.
    """
    from .deep_matcher import LunarMatcher
    matcher = LunarMatcher(matcher_type=matcher_type)

    def _match(tile_src: np.ndarray, tile_ref: np.ndarray):
        if not matcher._available:
            return np.array([]), np.array([]), np.array([])

        if not HAS_TORCH or torch is None:
            logger.warning(
                f"create_deep_tile_matcher [{matcher_type}]: torch not available — "
                "returning empty matches for this tile."
            )
            return np.array([]), np.array([]), np.array([])

        try:
            # Ensure input is 2D uint8 grayscale
            if tile_src.ndim == 3:
                tile_src = cv2.cvtColor(tile_src, cv2.COLOR_BGR2GRAY)
            if tile_ref.ndim == 3:
                tile_ref = cv2.cvtColor(tile_ref, cv2.COLOR_BGR2GRAY)
            if tile_src.dtype != np.uint8:
                tile_src = np.clip(tile_src, 0, 255).astype(np.uint8)
            if tile_ref.dtype != np.uint8:
                tile_ref = np.clip(tile_ref, 0, 255).astype(np.uint8)

            tensor_src = torch.from_numpy(
                np.ascontiguousarray(tile_src)
            ).float().unsqueeze(0).unsqueeze(0) / 255.0
            tensor_ref = torch.from_numpy(
                np.ascontiguousarray(tile_ref)
            ).float().unsqueeze(0).unsqueeze(0) / 255.0

            pts_s, pts_r, confs = matcher.match(tensor_src, tensor_ref)
            # Ensure well-shaped outputs even if matcher returns raw empty arrays
            if pts_s is None or len(pts_s) == 0:
                return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32)
            return (
                np.asarray(pts_s, dtype=np.float32).reshape(-1, 2),
                np.asarray(pts_r, dtype=np.float32).reshape(-1, 2),
                np.asarray(confs, dtype=np.float32).ravel(),
            )
        except Exception as e:
            logger.warning(f"create_deep_tile_matcher [{matcher_type}] error: {e}")
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    return _match


def hierarchical_native_tile_match(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    matcher_fn,
    tile_size: int = 512,
    overlap: float = 0.20,
    confidence_threshold: float = 0.25,
    coarse_dim: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hierarchical Coarse-to-Fine Native-Resolution Tile Matching.

    Enables deep neural matching at 100% native uncompressed sensor resolution:
    1. Computes a fast coarse global homography H_0 on downscaled thumbnails.
    2. Slices the full-resolution source image into (tile_size x tile_size) patches.
    3. Projects each tile's bounding envelope into the reference image using H_0.
    4. Crops the geometrically corresponding reference window and matches at 1:1 scale.
    5. Re-projects local tile coordinates to full-resolution global coordinates.

    Eliminates the quantization error and blurring caused by downscaling multi-megapixel
    lunar products to fit GPU memory.
    """
    h_s, w_s = img_src.shape[:2]
    h_r, w_r = img_ref.shape[:2]

    # Step 1: Coarse global alignment
    scale_s = coarse_dim / max(h_s, w_s)
    scale_r = coarse_dim / max(h_r, w_r)

    thumb_s = cv2.resize(img_src, (int(w_s * scale_s), int(h_s * scale_s)), interpolation=cv2.INTER_AREA)
    thumb_r = cv2.resize(img_ref, (int(w_r * scale_r), int(h_r * scale_r)), interpolation=cv2.INTER_AREA)

    sift_coarse = cv2.SIFT_create(nfeatures=1000)
    kp_s, desc_s = sift_coarse.detectAndCompute(thumb_s, None)
    kp_r, desc_r = sift_coarse.detectAndCompute(thumb_r, None)

    H_coarse = None
    if desc_s is not None and desc_r is not None and len(kp_s) >= 4 and len(kp_r) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_matches = bf.knnMatch(desc_s, desc_r, k=2)
        good = [m for m, n in raw_matches if m.distance < 0.8 * n.distance] if raw_matches else []
        if len(good) >= 4:
            pts_s_th = np.float32([kp_s[m.queryIdx].pt for m in good])
            pts_r_th = np.float32([kp_r[m.trainIdx].pt for m in good])
            # Scale coordinates up to full resolution
            pts_s_full = pts_s_th / scale_s
            pts_r_full = pts_r_th / scale_r
            H_coarse, _ = cv2.findHomography(pts_s_full, pts_r_full, cv2.RANSAC, 5.0)

    # Step 2: Tile iteration over source image
    stride = int(tile_size * (1.0 - overlap))
    all_pts_src = []
    all_pts_ref = []
    all_confs = []

    y_steps = list(range(0, max(1, h_s - tile_size // 2), stride))
    x_steps = list(range(0, max(1, w_s - tile_size // 2), stride))

    margin = int(tile_size * 0.25)

    for y0_s in y_steps:
        y1_s = min(y0_s + tile_size, h_s)
        for x0_s in x_steps:
            x1_s = min(x0_s + tile_size, w_s)

            tile_src = img_src[y0_s:y1_s, x0_s:x1_s]
            if tile_src.shape[0] < 64 or tile_src.shape[1] < 64:
                continue

            # Determine predicted reference window
            if H_coarse is not None:
                corners_src = np.array([
                    [x0_s, y0_s], [x1_s, y0_s], [x1_s, y1_s], [x0_s, y1_s]
                ], dtype=np.float32).reshape(-1, 1, 2)
                try:
                    corners_ref = cv2.perspectiveTransform(corners_src, H_coarse).reshape(-1, 2)
                    rx0 = max(0, int(np.floor(corners_ref[:, 0].min())) - margin)
                    rx1 = min(w_r, int(np.ceil(corners_ref[:, 0].max())) + margin)
                    ry0 = max(0, int(np.floor(corners_ref[:, 1].min())) - margin)
                    ry1 = min(h_r, int(np.ceil(corners_ref[:, 1].max())) + margin)
                except Exception:
                    rx0, rx1 = 0, w_r
                    ry0, ry1 = 0, h_r
            else:
                # Direct spatial correspondence fallback
                rx0 = max(0, int(x0_s * (w_r / w_s)) - margin)
                rx1 = min(w_r, int(x1_s * (w_r / w_s)) + margin)
                ry0 = max(0, int(y0_s * (h_r / h_s)) - margin)
                ry1 = min(h_r, int(y1_s * (h_r / h_s)) + margin)

            if (rx1 - rx0) < 64 or (ry1 - ry0) < 64:
                continue

            tile_ref = img_ref[ry0:ry1, rx0:rx1]

            try:
                p_s, p_r, conf = matcher_fn(tile_src, tile_ref)
                if len(p_s) > 0:
                    keep = conf >= confidence_threshold
                    p_s = p_s[keep]
                    p_r = p_r[keep]
                    conf = conf[keep]

                if len(p_s) > 0:
                    # Global coordinates
                    p_s[:, 0] += x0_s
                    p_s[:, 1] += y0_s
                    p_r[:, 0] += rx0
                    p_r[:, 1] += ry0

                    all_pts_src.append(p_s)
                    all_pts_ref.append(p_r)
                    all_confs.append(conf)
            except Exception as e_t:
                logger.debug(f"Tile matching failed at ({x0_s}, {y0_s}): {e_t}")

    if not all_pts_src:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    cat_src = np.vstack(all_pts_src).astype(np.float32)
    cat_ref = np.vstack(all_pts_ref).astype(np.float32)
    cat_conf = np.concatenate(all_confs).astype(np.float32)

    # De-duplicate matches from overlapping margins
    return _deduplicate_matches(cat_src, cat_ref, cat_conf, radius=3.0)

