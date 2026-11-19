"""
Dual Stereo Pair Pipeline — 4-Image Terrain Mapping
=====================================================
Orchestrates two independent stereo pair registrations (Pair A and Pair B)
and fuses their resulting Digital Elevation Models (DEMs) into one unified
high-resolution lunar surface terrain map.

Pipeline Steps:
1. Co-registers Pair A (src1 + ref1) using multi-engine fusion (RoMa v2, LoFTR, SIFT, etc.).
2. Co-registers Pair B (src2 + ref2) using multi-engine fusion.
3. Generates 3D photogrammetric DEMs for Pair A and Pair B.
4. Fuses DEM A and DEM B using distance-transform weighted feathering in overlapping regions.
5. Computes combined Quality Control (QC), surface hillshade, and unified statistics.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, Tuple

from src.pipeline import run_registration_pipeline
from src.features.dem_builder import (
    generate_3d_dem_from_pipeline,
    fuse_dem_pair,
    evaluate_dem_qc,
    compute_hillshade,
    get_camera_model,
)

logger = logging.getLogger(__name__)


def run_dual_pair_pipeline(
    src1_path: str,
    ref1_path: str,
    src2_path: str,
    ref2_path: str,
    config: Optional[dict] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Execute 4-image dual stereo pair pipeline.

    Parameters
    ----------
    src1_path : str
        Path to source image for Pair A.
    ref1_path : str
        Path to reference image for Pair A.
    src2_path : str
        Path to source image for Pair B.
    ref2_path : str
        Path to reference image for Pair B.
    config : dict, optional
        Pipeline configuration settings.
    progress_callback : callable, optional
        Callback function receiving (progress_pct: float, message: str).

    Returns
    -------
    dict
        Dual pair results including Pair A, Pair B, and fused DEM topography.
    """
    def _notify(pct: float, msg: str):
        logger.info(f"[{pct*100:.0f}%] {msg}")
        if progress_callback is not None:
            progress_callback(pct, msg)

    cfg = config or {}

    _notify(0.05, "Starting 4-Image Dual Stereo Pair Pipeline...")

    # --- 1. Register Stereo Pair A ---
    _notify(0.10, "Registering Stereo Pair A (Image 1 + Image 2)...")
    img_src1, img_ref1, reg_a = run_registration_pipeline(src1_path, ref1_path, matcher_type="fusion")
    num_inl_a = reg_a.get('num_inliers', reg_a.get('inliers', 0))
    _notify(0.35, f"Pair A Registered: {num_inl_a} inliers, RMSE={reg_a.get('rmse', 0.0):.3f}px")

    # --- 2. Register Stereo Pair B ---
    _notify(0.40, "Registering Stereo Pair B (Image 3 + Image 4)...")
    img_src2, img_ref2, reg_b = run_registration_pipeline(src2_path, ref2_path, matcher_type="fusion")
    num_inl_b = reg_b.get('num_inliers', reg_b.get('inliers', 0))
    _notify(0.65, f"Pair B Registered: {num_inl_b} inliers, RMSE={reg_b.get('rmse', 0.0):.3f}px")

    sensor_type = cfg.get("stereo", {}).get("sensor", "tmc")

    # --- 3. Build DEM for Pair A ---
    _notify(0.70, "Generating 3D DEM for Stereo Pair A...")
    dem_res_a = generate_3d_dem_from_pipeline(
        img_src=img_src1,
        img_ref=img_ref1,
        registration_results=reg_a,
        camera_sensor=sensor_type,
        config=cfg,
    )

    # --- 4. Build DEM for Pair B ---
    _notify(0.80, "Generating 3D DEM for Stereo Pair B...")
    dem_res_b = generate_3d_dem_from_pipeline(
        img_src=img_src2,
        img_ref=img_ref2,
        registration_results=reg_b,
        camera_sensor=sensor_type,
        config=cfg,
    )

    # --- 5. Fuse DEM A and DEM B ---
    _notify(0.90, "Fusing Stereo Pair A & B into Unified Terrain Model...")
    dem_a = dem_res_a["dem_m"]
    dem_b = dem_res_b["dem_m"]
    tex_a = dem_res_a.get("texture")
    tex_b = dem_res_b.get("texture")

    fused_dem_m, fused_texture = fuse_dem_pair(
        dem1=dem_a,
        dem2=dem_b,
        tex1=tex_a,
        tex2=tex_b,
    )

    cam = get_camera_model(cfg)
    fused_qc = evaluate_dem_qc(fused_dem_m, gsd_m=cam.gsd_m)
    fused_hillshade = compute_hillshade(
        fused_dem_m,
        gsd_m=cam.gsd_m,
        azimuth_deg=float(reg_a.get("sun_azimuth", 315.0)),
        altitude_deg=float(reg_a.get("sun_elevation", 45.0)),
    )

    _notify(1.0, "Dual Stereo Pair 4-Image Terrain Fusion Complete!")

    return {
        "status": "completed",
        "pair_a": {
            "registration": reg_a,
            "dem_results": dem_res_a,
        },
        "pair_b": {
            "registration": reg_b,
            "dem_results": dem_res_b,
        },
        "fused_dem": {
            "dem_m": fused_dem_m,
            "texture": fused_texture,
            "hillshade": fused_hillshade,
            "qc": fused_qc,
            "relief_span_m": float(fused_dem_m.max() - fused_dem_m.min()) if fused_dem_m.size > 0 else 0.0,
            "elevation_min_m": float(fused_dem_m.min()) if fused_dem_m.size > 0 else 0.0,
            "elevation_max_m": float(fused_dem_m.max()) if fused_dem_m.size > 0 else 0.0,
        },
    }
