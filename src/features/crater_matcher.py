"""
ISRO LunarFusion — YOLO11 Lunar Crater Landmark Matcher
======================================================
Provides physical macroscopic landmark matching across lunar image pairs
by detecting craters using fine-tuned YOLO11, computing photometric patch
descriptors, enforcing scale/morphological invariants, performing
crater-neighborhood graph matching, and producing rotation-compensated
anchored tie-points (crater centroids + cardinal rim points).

Advantages for Lunar Registration:
1. Physical Grounding: Tie-points are anchored to actual geological crater basins.
2. Invariant to Regolith Micro-Texture: When fine dust lack keypoints, macro craters persist.
3. Gate 3 Compliance: Naturally guarantees high spatial distribution across crater landscapes.
4. Zero-Hallucination: If 0 craters are detected, returns empty matches, allowing Gate 4
   to set weight to 0%.
5. Crater-Neighborhood Graph Matching: Resolves ambiguous craters and estimates scene
   rotation before synthesizing oriented rim points.
"""

from __future__ import annotations

import logging
import os
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from .crater_detector import LunarCraterDetector, DEFAULT_MODEL_PATH
from .base_matcher import BaseLunarMatcher, MatcherOutput, sanitize_lunar_input

logger = logging.getLogger(__name__)


def _compute_neighborhood_descriptor(
    centroids: np.ndarray,
    idx: int,
    k_neighbors: int = 3
) -> Optional[np.ndarray]:
    """
    Compute a rotation-, scale-, and translation-invariant descriptor
    for a crater based on the relative distances and angles to its k nearest neighbors.
    """
    n = len(centroids)
    if n < 3:
        return None

    c_i = centroids[idx]
    dists = np.linalg.norm(centroids - c_i, axis=1)
    # Exclude self
    order = np.argsort(dists)[1:min(k_neighbors + 1, n)]
    if len(order) < 2:
        return None

    p1 = centroids[order[0]]
    p2 = centroids[order[1]]
    d1 = dists[order[0]]
    d2 = dists[order[1]]

    if d1 < 1e-4 or d2 < 1e-4:
        return None

    ratio = min(d1, d2) / max(d1, d2)
    # Angle between vectors (c_i -> p1) and (c_i -> p2)
    v1 = p1 - c_i
    v2 = p2 - c_i
    cos_angle = np.dot(v1, v2) / (d1 * d2)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_rad = math.acos(cos_angle)

    return np.array([ratio, angle_rad], dtype=np.float32)


def _match_crater_constellations(
    centroids_s: np.ndarray,
    diameters_s: np.ndarray,
    centroids_r: np.ndarray,
    diameters_r: np.ndarray,
    ratio_tol: float = 0.16,
    angle_tol_deg: float = 20.0,
) -> np.ndarray:
    """
    Matches crater triangles across frames using pure geometric constellation hashing.
    Completely independent of pixel intensity, shadows, or sun elevation/azimuth.
    Returns geometric consensus voting matrix of shape (N_src, N_ref).
    """
    n_s = len(centroids_s)
    n_r = len(centroids_r)
    votes = np.zeros((n_s, n_r), dtype=np.float32)

    if n_s < 3 or n_r < 3:
        return votes

    # Build triplets for source
    triplets_s = []
    for i in range(n_s):
        dists = np.linalg.norm(centroids_s - centroids_s[i], axis=1)
        nn = np.argsort(dists)[1:min(6, n_s)]
        for a in range(len(nn)):
            for b in range(a + 1, len(nn)):
                j, k = nn[a], nn[b]
                pts = centroids_s[[i, j, k]]
                d01 = float(np.linalg.norm(pts[0] - pts[1]))  # opposite to k (idx 2)
                d12 = float(np.linalg.norm(pts[1] - pts[2]))  # opposite to i (idx 0)
                d20 = float(np.linalg.norm(pts[2] - pts[0]))  # opposite to j (idx 1)
                pairs = sorted([(d12, i), (d20, j), (d01, k)])
                sides = [p[0] for p in pairs]
                canon_idx = [p[1] for p in pairs]
                if sides[0] < 5.0 or sides[2] < 1e-3:
                    continue
                r1 = sides[0] / sides[2]
                r2 = sides[1] / sides[2]
                cos_ang = (sides[0]**2 + sides[1]**2 - sides[2]**2) / max(2.0 * sides[0] * sides[1], 1e-4)
                ang = math.acos(float(np.clip(cos_ang, -1.0, 1.0)))
                triplets_s.append({"canon_idx": canon_idx, "r1": r1, "r2": r2, "ang": ang})

    # Build triplets for reference
    triplets_r = []
    for i in range(n_r):
        dists = np.linalg.norm(centroids_r - centroids_r[i], axis=1)
        nn = np.argsort(dists)[1:min(6, n_r)]
        for a in range(len(nn)):
            for b in range(a + 1, len(nn)):
                j, k = nn[a], nn[b]
                pts = centroids_r[[i, j, k]]
                d01 = float(np.linalg.norm(pts[0] - pts[1]))
                d12 = float(np.linalg.norm(pts[1] - pts[2]))
                d20 = float(np.linalg.norm(pts[2] - pts[0]))
                pairs = sorted([(d12, i), (d20, j), (d01, k)])
                sides = [p[0] for p in pairs]
                canon_idx = [p[1] for p in pairs]
                if sides[0] < 5.0 or sides[2] < 1e-3:
                    continue
                r1 = sides[0] / sides[2]
                r2 = sides[1] / sides[2]
                cos_ang = (sides[0]**2 + sides[1]**2 - sides[2]**2) / max(2.0 * sides[0] * sides[1], 1e-4)
                ang = math.acos(float(np.clip(cos_ang, -1.0, 1.0)))
                triplets_r.append({"canon_idx": canon_idx, "r1": r1, "r2": r2, "ang": ang})

    rad_tol = math.radians(angle_tol_deg)

    for ts in triplets_s:
        for tr in triplets_r:
            if abs(ts["r1"] - tr["r1"]) < ratio_tol and abs(ts["r2"] - tr["r2"]) < ratio_tol:
                if abs(ts["ang"] - tr["ang"]) < rad_tol:
                    for vi, vj in zip(ts["canon_idx"], tr["canon_idx"]):
                        votes[vi, vj] += 1.0

    max_v = float(np.max(votes))
    if max_v > 0:
        votes = votes / max_v

    return votes


class YOLO11CraterMatcher(BaseLunarMatcher):
    """
    Crater Landmark Matcher powered by fine-tuned Ultralytics YOLO11
    and Crater-Neighborhood Graph Matching.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf: float = 0.20,
        min_similarity: float = 0.35,
        include_rim_points: bool = True,
        patch_size: int = 48,
        device: Optional[str] = None,
        k_graph_neighbors: int = 3,
    ):
        super().__init__(
            matcher_name="yolo11_crater",
            confidence_threshold=conf,
            enable_illumination_weighting=False,
            enable_spatial_balancing=False,
        )
        self.conf = conf
        self.min_similarity = min_similarity
        self.include_rim_points = include_rim_points
        self.patch_size = patch_size
        self.k_graph_neighbors = k_graph_neighbors
        self.detector = LunarCraterDetector(
            model_path=model_path,
            conf=conf,
            device=device,
        )

    def _extract_crater_patch(
        self,
        img: np.ndarray,
        centroid: Tuple[float, float],
        diameter: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract and normalize intensity and gradient-magnitude ROI patches around a detected crater."""
        h, w = img.shape[:2]
        cx, cy = centroid
        half_w = max(int(round(diameter * 0.65)), 8)

        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_w))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_w))

        # Pad if box extends beyond image bounds
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded = cv2.copyMakeBorder(
                img, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REFLECT_101
            )
            crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
        else:
            crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Resize to fixed patch size
        resized = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_AREA)
        crop_f = resized.astype(np.float32)

        # 1. Standardize intensity for zero-mean normalized cross correlation (ZNCC)
        std = float(np.std(crop_f))
        if std < 1e-4:
            return None
        norm_patch = (crop_f - float(np.mean(crop_f))) / std

        # 2. Standardize gradient magnitude (strictly invariant to sun angle direction)
        gx = cv2.Sobel(crop_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(crop_f, cv2.CV_32F, 0, 1, ksize=3)
        g_mag = np.hypot(gx, gy)
        g_std = float(np.std(g_mag))
        norm_grad = (g_mag - float(np.mean(g_mag))) / max(g_std, 1e-4)

        return {
            "intensity": norm_patch,
            "grad_mag": norm_grad,
        }

    def _match_implementation(
        self,
        src_gray: np.ndarray,
        ref_gray: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Step 1: Detect craters in both frames
        craters_src = [c for c in self.detector.detect(src_gray) if c["class_name"].lower() == "crater"]
        craters_ref = [c for c in self.detector.detect(ref_gray) if c["class_name"].lower() == "crater"]

        if len(craters_src) == 0 or len(craters_ref) == 0:
            logger.info(f"YOLO11 Matcher: 0 craters found (src={len(craters_src)}, ref={len(craters_ref)})")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"matched_craters": 0, "rotation_deg": 0.0},
            )

        # Step 2: Extract normalized crater ROI patches (intensity + gradient magnitude)
        patches_src = [
            self._extract_crater_patch(src_gray, c["centroid"], c["diameter_px"])
            for c in craters_src
        ]
        patches_ref = [
            self._extract_crater_patch(ref_gray, c["centroid"], c["diameter_px"])
            for c in craters_ref
        ]

        n_s = len(craters_src)
        n_r = len(craters_ref)
        centroids_s = np.array([c["centroid"] for c in craters_src], dtype=np.float32)
        centroids_r = np.array([c["centroid"] for c in craters_ref], dtype=np.float32)
        diameters_s = np.array([c["diameter_px"] for c in craters_src], dtype=np.float32)
        diameters_r = np.array([c["diameter_px"] for c in craters_ref], dtype=np.float32)

        # Precompute illumination-invariant constellation geometric consensus
        constellation_votes = _match_crater_constellations(
            centroids_s, diameters_s, centroids_r, diameters_r
        )

        # Precompute neighborhood graph descriptors
        graph_desc_s = [_compute_neighborhood_descriptor(centroids_s, i, self.k_graph_neighbors) for i in range(n_s)]
        graph_desc_r = [_compute_neighborhood_descriptor(centroids_r, j, self.k_graph_neighbors) for j in range(n_r)]

        sim_matrix = np.zeros((n_s, n_r), dtype=np.float32)

        # Step 3: Compute pairwise appearance + morphological + graph similarity
        for i in range(n_s):
            p_s = patches_src[i]
            if p_s is None:
                continue
            d_s = craters_src[i]["diameter_px"]
            c_s_score = craters_src[i]["confidence"]
            g_s = graph_desc_s[i]

            for j in range(n_r):
                p_r = patches_ref[j]
                if p_r is None:
                    continue
                d_r = craters_ref[j]["diameter_px"]
                c_r_score = craters_ref[j]["confidence"]
                g_r = graph_desc_r[j]

                # Scale ratio penalty: craters should have plausible relative diameters
                scale_ratio = d_s / max(d_r, 1e-3)
                if scale_ratio < 0.25 or scale_ratio > 4.0:
                    continue
                scale_sim = float(np.exp(-abs(np.log(scale_ratio))))

                # Illumination-Resilient Normalized Cross Correlation
                # 1. Raw intensity ZNCC
                zncc_raw = float(np.mean(p_s["intensity"] * p_r["intensity"]))
                # 2. Gradient magnitude ZNCC (Strictly illumination-invariant)
                zncc_grad = float(np.mean(p_s["grad_mag"] * p_r["grad_mag"]))

                # Under opposing sun angles (morning vs afternoon), raw ZNCC flips negative.
                # Effective ZNCC blends absolute raw and gradient magnitude correlation:
                zncc_eff = max(zncc_raw, abs(zncc_raw) * 0.85, zncc_grad)

                geo_score = float(constellation_votes[i, j])
                if zncc_eff < 0.12 and geo_score < 0.25:
                    continue

                # Geometric constellation boost
                graph_bonus = 1.0 + 0.60 * geo_score
                if g_s is not None and g_r is not None:
                    delta_ratio = abs(g_s[0] - g_r[0])
                    delta_angle = abs(g_s[1] - g_r[1])
                    if delta_ratio < 0.25 and delta_angle < math.radians(30):
                        graph_bonus += 0.25
                    elif delta_ratio > 0.6 or delta_angle > math.radians(75):
                        graph_bonus *= 0.80

                det_score = float(np.sqrt(c_s_score * c_r_score))
                sim_matrix[i, j] = zncc_eff * scale_sim * det_score * graph_bonus

        # Step 4: Mutual nearest-neighbor matching
        matched_pairs: List[Tuple[int, int, float]] = []
        for i in range(n_s):
            j_best = int(np.argmax(sim_matrix[i, :]))
            best_sim = sim_matrix[i, j_best]
            if best_sim < self.min_similarity:
                continue

            # Mutual check: crater j_best must also select crater i as top match
            i_best = int(np.argmax(sim_matrix[:, j_best]))
            if i_best == i:
                matched_pairs.append((i, j_best, best_sim))

        if not matched_pairs:
            logger.info("YOLO11 Matcher: No mutually consistent crater pairs survived.")
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                {"matched_craters": 0, "rotation_deg": 0.0},
            )

        # Step 5: Estimate global scene rotation angle theta from matched crater constellation
        estimated_theta = 0.0
        if len(matched_pairs) >= 2:
            pair_angles = []
            for p_idx in range(len(matched_pairs)):
                for q_idx in range(p_idx + 1, len(matched_pairs)):
                    i1, j1, _ = matched_pairs[p_idx]
                    i2, j2, _ = matched_pairs[q_idx]

                    v_src = centroids_s[i2] - centroids_s[i1]
                    v_ref = centroids_r[j2] - centroids_r[j1]

                    if np.linalg.norm(v_src) > 10.0 and np.linalg.norm(v_ref) > 10.0:
                        ang_src = math.atan2(float(v_src[1]), float(v_src[0]))
                        ang_ref = math.atan2(float(v_ref[1]), float(v_ref[0]))
                        d_ang = (ang_ref - ang_src) % (2.0 * math.pi)
                        if d_ang > math.pi:
                            d_ang -= 2.0 * math.pi
                        pair_angles.append(d_ang)

            if pair_angles:
                estimated_theta = float(np.median(pair_angles))
                logger.info(f"YOLO11 Crater Graph: Estimated scene rotation = {math.degrees(estimated_theta):.1f}°")

        # Step 6: Synthesize tie-points (centroids + rotation-compensated cardinal rim points)
        pts_s_list = []
        pts_r_list = []
        conf_list = []

        cardinal_angles = [
            -math.pi / 2.0,  # North
            math.pi / 2.0,   # South
            0.0,             # East
            math.pi,         # West
        ]

        for i, j, score in matched_pairs:
            c_s = craters_src[i]
            c_r = craters_ref[j]
            cx_s, cy_s = c_s["centroid"]
            cx_r, cy_r = c_r["centroid"]
            r_s = c_s["diameter_px"] / 2.0
            r_r = c_r["diameter_px"] / 2.0

            # 1. Central tie-point (crater center)
            pts_s_list.append([cx_s, cy_s])
            pts_r_list.append([cx_r, cy_r])
            conf_list.append(score)

            # 2. Rotation-compensated cardinal rim points
            # Only synthesize rim points when >= 2 craters support rotation orientation
            if self.include_rim_points and len(matched_pairs) >= 2 and r_s >= 6.0 and r_r >= 6.0:
                rim_conf = score * 0.90
                for phi in cardinal_angles:
                    # In source image: angle is phi
                    x_rim_s = cx_s + r_s * math.cos(phi)
                    y_rim_s = cy_s + r_s * math.sin(phi)

                    # In reference image: angle is phi + estimated_theta
                    x_rim_r = cx_r + r_r * math.cos(phi + estimated_theta)
                    y_rim_r = cy_r + r_r * math.sin(phi + estimated_theta)

                    pts_s_list.append([x_rim_s, y_rim_s])
                    pts_r_list.append([x_rim_r, y_rim_r])
                    conf_list.append(rim_conf)

        pts_s_arr = np.array(pts_s_list, dtype=np.float32)
        pts_r_arr = np.array(pts_r_list, dtype=np.float32)
        confs_arr = np.array(conf_list, dtype=np.float32)

        meta = {
            "matched_craters": len(matched_pairs),
            "rotation_deg": float(math.degrees(estimated_theta)),
            "craters_detected_src": n_s,
            "craters_detected_ref": n_r,
        }

        logger.info(
            f"YOLO11 Matcher: {len(matched_pairs)} matched craters → "
            f"{len(pts_s_arr)} landmark tie-points (theta={math.degrees(estimated_theta):.1f}°, mean_conf={float(np.mean(confs_arr)):.2f})"
        )
        return pts_s_arr, pts_r_arr, confs_arr, meta

    def match(
        self,
        img_src: Any,
        img_ref: Any,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backwards-compatible tuple return matching LunarMatcher interface."""
        out: MatcherOutput = super().match(img_src, img_ref, **kwargs)
        return out.as_tuple()


def match_craters_yolo11(
    img_src: Union[torch.Tensor, np.ndarray],
    img_ref: Union[torch.Tensor, np.ndarray],
    config: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard entry-point function for YOLO11 Crater Landmark Matching.
    Compatible with LunarFusionMatcher._run_single_matcher() and pipeline.py.
    """
    cfg = config or {}
    model_path = cfg.get("yolo11_model_path", DEFAULT_MODEL_PATH)
    conf = float(cfg.get("yolo11_conf", 0.20))
    min_sim = float(cfg.get("yolo11_min_similarity", 0.35))
    rim_pts = bool(cfg.get("yolo11_rim_points", True))

    matcher = YOLO11CraterMatcher(
        model_path=model_path,
        conf=conf,
        min_similarity=min_sim,
        include_rim_points=rim_pts,
    )
    return matcher.match(img_src, img_ref)
