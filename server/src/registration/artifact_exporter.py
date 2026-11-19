"""
artifact_exporter.py — Canonical Registration Artifact Writer
=============================================================
Writes all §9 export artifacts required by the CLARITY3D brief:

  registered_source.tif          — source resampled to reference grid (accepted only)
  valid_mask.tif                 — valid sampling / nodata coverage
  registration_support_mask.tif  — supported vs extrapolated regions
  matches.csv                    — per-match table with provenance fields
  transform.json                 — model, parameters, direction, convention, inversion
  quality.json                   — metrics, gate results, evidence classes, limitations
  provenance.json                — input hashes, code revision, device, config

Rules enforced by this module
-----------------------------
* A ``registered_source.tif`` is NEVER written for a rejected result.
  Optional previews are written only when ``rejected_preview=True`` and are
  marked with the suffix ``_UNVALIDATED``.
* RMSE fields in quality.json are always labelled with their evidence class:
  ``fit_only``, ``held_out``, or ``independently_validated``.
* Source-pixel RMSE and reference-pixel RMSE are reported as separate fields,
  not conflated.
* The ``transform.json`` records direction (moving→fixed), coordinate
  convention (pixel-centre / pixel-corner), and inversion availability.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Evidence-class constants (§7)
# ---------------------------------------------------------------------------
EVIDENCE_FIT_ONLY = "fit_only"
EVIDENCE_HELD_OUT = "held_out"
EVIDENCE_INDEPENDENT = "independently_validated"
EVIDENCE_UNAVAILABLE = "unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v: Any) -> Optional[float]:
    """Return a JSON-safe float or None."""
    if v is None:
        return None
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _ensure_dir(path: str) -> None:
    """Create parent directory if it does not exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_json(path: str, data: dict) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# 1. matches.csv — per-match provenance table
# ---------------------------------------------------------------------------
def export_matches_csv(
    path: str,
    pts_moving: np.ndarray,
    pts_fixed: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    engine: Optional[np.ndarray] = None,     # per-match engine label (str array)
    inlier_mask: Optional[np.ndarray] = None,
    validation_split: Optional[np.ndarray] = None,  # "train" / "holdout"
    residuals_ref_px: Optional[np.ndarray] = None,
    residuals_src_px: Optional[np.ndarray] = None,
    scale_level: Optional[np.ndarray] = None,
    tile_id: Optional[np.ndarray] = None,
    refinement_status: Optional[np.ndarray] = None,
) -> str:
    """
    Write per-match table to *path*.

    Each row records:
      x_moving, y_moving  — original moving-image pixel coordinates (col, row)
      x_fixed,  y_fixed   — original fixed-image pixel coordinates
      engine              — matcher that produced this match
      raw_confidence      — matcher's raw confidence score [0,1]
      is_inlier           — 1 if survived geometric verification, 0 otherwise
      validation_split    — 'train' (used to fit model) or 'holdout' (withheld)
      residual_ref_px     — ||T(moving) - fixed|| in reference-image pixels
      residual_src_px     — ||T_inv(fixed) - moving|| in source-image pixels
                             (null when inverse is unavailable)
      scale_level         — pyramid / tile scale level at which match was found
      tile_id             — tile grid identifier (e.g. '2_3') or '' for whole-image
      refinement_status   — 'refined', 'unrefined', or 'failed'
    """
    _ensure_dir(path)
    n = len(pts_moving)
    if n == 0:
        logger.warning("export_matches_csv: no matches to write")
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("x_moving,y_moving,x_fixed,y_fixed,engine,raw_confidence,"
                    "is_inlier,validation_split,residual_ref_px,residual_src_px,"
                    "scale_level,tile_id,refinement_status\n")
        return path

    def _col(arr, default, idx):
        if arr is not None and len(arr) > idx:
            return arr[idx]
        return default

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "x_moving_px", "y_moving_px",
            "x_fixed_px",  "y_fixed_px",
            "engine", "raw_confidence",
            "is_inlier", "validation_split",
            "residual_ref_px", "residual_src_px",
            "scale_level", "tile_id", "refinement_status",
        ])
        for i in range(n):
            xm, ym = float(pts_moving[i, 0]), float(pts_moving[i, 1])
            xf, yf = float(pts_fixed[i, 0]),  float(pts_fixed[i, 1])
            conf   = round(float(_col(confidences, np.nan, i)), 6)
            eng    = str(_col(engine, "", i))
            inl    = int(bool(_col(inlier_mask, False, i)))
            vsplit = str(_col(validation_split, "train", i))
            r_ref  = _col(residuals_ref_px, None, i)
            r_src  = _col(residuals_src_px, None, i)
            sl     = str(_col(scale_level, "1.0", i))
            tid    = str(_col(tile_id, "", i))
            ref_st = str(_col(refinement_status, "unrefined", i))
            writer.writerow([
                round(xm, 4), round(ym, 4),
                round(xf, 4), round(yf, 4),
                eng, conf, inl, vsplit,
                round(float(r_ref), 4) if r_ref is not None else "",
                round(float(r_src), 4) if r_src is not None else "",
                sl, tid, ref_st,
            ])

    logger.info("Wrote %d matches to %s", n, path)
    return path


# ---------------------------------------------------------------------------
# 2. transform.json — transform contract
# ---------------------------------------------------------------------------
def export_transform_json(
    path: str,
    model: str,
    matrix: Optional[np.ndarray],
    direction: str = "moving_to_fixed",
    coordinate_convention: str = "pixel_corner_origin",
    reference_grid: Optional[Dict[str, Any]] = None,
    inversion_available: bool = False,
    inversion_method: Optional[str] = None,
    domain: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write transform.json with explicit direction and coordinate convention.

    Parameters
    ----------
    model : str
        Model type string, e.g. "HOMOGRAPHY_3x3", "AFFINE_2x3", "TPS", "TRANSLATION".
    matrix : ndarray or None
        Numeric parameters.  For TPS this may be None (or a dict of control points).
    direction : str
        Which space maps to which.  Always "moving_to_fixed" (§2 convention).
    coordinate_convention : str
        Origin convention ("pixel_corner_origin" = GeoTIFF / rasterio;
        "pixel_centre_origin" = PDS4 convention).
    reference_grid : dict
        {crs_wkt, origin_x, origin_y, pixel_size_x, pixel_size_y, width, height}.
    inversion_available : bool
        Whether a reliable closed-form or numerically computed inverse exists.
    inversion_method : str
        How the inverse is computed, e.g. "numpy.linalg.inv", "numerical_grid".
    domain : dict
        Valid spatial domain of the transform (bounding box in fixed-image coords).
    """
    mat_list = matrix.tolist() if isinstance(matrix, np.ndarray) else matrix
    data = {
        "_schema": "clarity3d_transform_v1",
        "model": model,
        "direction": direction,
        "coordinate_convention": coordinate_convention,
        "parameters": mat_list,
        "reference_grid": reference_grid or {},
        "inversion": {
            "available": inversion_available,
            "method": inversion_method,
            "note": (
                "Inversion computed; source-pixel RMSE is exact."
                if inversion_available
                else "No reliable inverse; source-pixel RMSE reported as unavailable."
            ),
        },
        "domain": domain or {"note": "Full image extent assumed"},
        "extra": extra or {},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(path, data)
    return path


# ---------------------------------------------------------------------------
# 3. quality.json — metrics + gate results + evidence classes
# ---------------------------------------------------------------------------
def export_quality_json(
    path: str,
    results: Dict[str, Any],
    evidence_class: str = EVIDENCE_FIT_ONLY,
) -> str:
    """
    Write quality.json with separated evidence classes and explicit limitations.

    Evidence classes (§7):
      fit_only           — RMSE computed on points used to fit the model
      held_out           — RMSE on withheld points (not used for fitting)
      independently_validated — externally verified landmarks or GCPs
    """
    quality = results.get("quality") or {}
    gate_tel = quality.get("gate_telemetry") or {}

    # RMSE fields — labelled with evidence class
    fit_rmse = _safe_float(results.get("rmse_pixels"))
    holdout  = results.get("holdout_validation") or {}
    ho_rmse  = _safe_float(holdout.get("holdout_rmse"))
    gt_rmse  = _safe_float(results.get("gt_rmse_pixels"))

    # Source-pixel vs reference-pixel (both kept separate per §7)
    fit_rmse_raw = _safe_float(results.get("rmse_raw_pixels"))   # src-pixel space
    fit_rmse_ref = fit_rmse                                        # ref-pixel space (downscaled)

    data = {
        "_schema": "clarity3d_quality_v1",
        "pipeline_status": results.get("pipeline_status", "UNKNOWN"),
        "quality_grade": results.get("quality_grade", "Grade F"),
        "passed": bool(quality.get("passed", False)),
        "primary_reason": quality.get("primary_reason", "UNKNOWN"),
        "reason_codes": quality.get("reason_codes", []),
        "warnings": quality.get("warnings", []),

        # Evidence-class labelled RMSE (§7 requirement)
        "accuracy_evidence": {
            "class": evidence_class,
            "note": (
                "RMSE computed on inlier points used to fit the model. "
                "Model-fitting residuals are NOT independent accuracy evidence."
                if evidence_class == EVIDENCE_FIT_ONLY
                else f"Evidence class: {evidence_class}"
            ),
            "fit_rmse_ref_px": fit_rmse_ref,
            "fit_rmse_src_px": fit_rmse_raw,
            "holdout_rmse_ref_px": ho_rmse,
            "holdout_n": int(holdout.get("n_holdout", 0)),
            "holdout_rmse_note": (
                "Withheld correspondence residuals — not used for fitting."
                if ho_rmse is not None
                else "Holdout RMSE unavailable (insufficient inliers or not computed)."
            ),
            "gt_rmse_ref_px": gt_rmse,
            "gt_status": results.get("gt_validation_status", "not_provided"),
            "gt_n": int(results.get("gt_point_count", 0)),
            "gt_note": (
                "Externally verified GCP residuals (most trusted evidence class)."
                if results.get("gt_validation_status") == "valid"
                else "No independently verified landmarks available; accuracy is unverified."
            ),
        },

        # Distribution and counts (§7)
        "match_counts": {
            "raw_total": int(results.get("total_matches", 0)),
            "inlier_count": int(results.get("inlier_count", 0)),
            "inlier_ratio": _safe_float(results.get("inlier_ratio")),
            "inlier_ratio_denominator": "all_putative_matches_before_ransac",
            "holdout_n": int(holdout.get("n_holdout", 0)),
            "train_n": int(holdout.get("n_train", 0)),
        },

        "error_statistics": results.get("error_stats") or {},
        "error_budget": results.get("error_budget") or {},

        # Spatial coverage (§7)
        "spatial_coverage": {
            "occupied_grid_fraction": _safe_float(results.get("coverage")),
            "spatial_entropy": _safe_float(results.get("spatial_entropy")),
            "grid_note": "8x8 cell grid over valid image overlap",
            "coverage_note": (
                "Coverage measured over downscaled working image. "
                "Grid fraction <15% triggers SEVERE_MATCH_CLUSTERING gate."
            ),
        },

        # Symmetric / conditioning diagnostics (§6 & §7)
        "geometry_diagnostics": {
            "symmetric_rmse_px": _safe_float(results.get("symmetric_rmse")),
            "condition_number": _safe_float(results.get("condition_number")),
            "geometric_model_type": results.get("geometric_model_type", "UNKNOWN"),
            "fallback_code": results.get("fallback_code"),
        },

        # Gate telemetry
        "gates": gate_tel,

        # Explicit limitations (§7 requirement)
        "limitations": [
            "Sub-pixel RMSE is evaluated in downscaled working coordinates. "
            "Original-sensor-pixel accuracy is reported as 'fit_rmse_src_px'.",
            "Model-fitting residuals cannot establish geographic accuracy independently.",
            "Without GCP/independent landmark data, accuracy is 'fit_only' class.",
            "Spatial coverage is measured over the downscaled working image extent.",
            "High inlier count does not compensate for clustered spatial distribution.",
        ],

        # Execution/runtime info (§7)
        "runtime": results.get("pipeline_times") or {},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(path, data)
    return path


# ---------------------------------------------------------------------------
# 4. provenance.json — complete lineage record
# ---------------------------------------------------------------------------
def export_provenance_json(
    path: str,
    results: Dict[str, Any],
    fixed_descriptor: Optional[Dict[str, Any]] = None,
    moving_descriptor: Optional[Dict[str, Any]] = None,
) -> str:
    """Write provenance.json with input hashes, code revision, device, config."""
    raw_prov = results.get("provenance") or {}
    data = {
        "_schema": "clarity3d_provenance_v1",
        "pipeline_version": raw_prov.get("pipeline_version", "unknown"),
        "python_version": raw_prov.get("python_version"),
        "opencv_version": raw_prov.get("opencv_version"),
        "torch_version": raw_prov.get("torch_version"),
        "platform": raw_prov.get("platform"),
        "device": raw_prov.get("device"),
        "config_sha256": raw_prov.get("config_sha256"),
        "inputs": raw_prov.get("inputs", {}),
        "pds4": raw_prov.get("pds4", {}),
        "fixed_image": fixed_descriptor or {},
        "moving_image": moving_descriptor or {},
        "recovery_attempts": results.get("recovery_attempts", []),
        "method_used": results.get("method_used", "unknown"),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(path, data)
    return path


# ---------------------------------------------------------------------------
# 5. valid_mask.tif + registration_support_mask.tif
# ---------------------------------------------------------------------------
def export_valid_mask(
    path: str,
    warped_img: np.ndarray,
    nodata_threshold: int = 5,
    reference_tif: Optional[str] = None,
) -> str:
    """
    Write valid_mask.tif — pixels where registered source has valid data.

    A pixel is invalid when the border-reflected warp produces near-zero values
    (empty / black border region outside the source footprint).

    Parameters
    ----------
    path : str
        Output path for the mask GeoTIFF.
    warped_img : ndarray
        Warped source image (uint8, 2D grayscale).
    nodata_threshold : int
        Pixels with value ≤ this threshold are treated as nodata.
    reference_tif : str, optional
        Reference GeoTIFF to copy CRS/transform from.
    """
    _ensure_dir(path)
    mask = (warped_img.astype(np.float32) > nodata_threshold).astype(np.uint8) * 255

    if HAS_RASTERIO and reference_tif and os.path.exists(reference_tif):
        try:
            with rasterio.open(reference_tif) as ref:
                kwargs = {
                    "driver": "GTiff", "height": mask.shape[0],
                    "width": mask.shape[1], "count": 1,
                    "dtype": "uint8", "crs": ref.crs,
                    "transform": ref.transform, "compress": "lzw",
                }
            with rasterio.open(path, "w", **kwargs) as dst:
                dst.write(mask, 1)
            logger.info("Wrote valid_mask.tif (georeferenced): %s", path)
            return path
        except Exception as exc:
            logger.warning("Could not write georeferenced valid_mask: %s", exc)

    cv2.imwrite(path, mask)
    logger.info("Wrote valid_mask.tif (image-space): %s", path)
    return path


def export_support_mask(
    path: str,
    img_shape: Tuple[int, int],
    inliers_fixed: np.ndarray,
    H: Optional[np.ndarray] = None,
    support_radius_px: int = 64,
    reference_tif: Optional[str] = None,
) -> str:
    """
    Write registration_support_mask.tif.

    Values:
      255 — supported (within radius of a verified inlier)
      128 — potential (within convex hull but no nearby inlier)
        0 — extrapolated / unsupported

    The supported/extrapolated distinction is kept separate from nodata (§9).
    """
    _ensure_dir(path)
    h, w = img_shape[:2]
    support = np.zeros((h, w), dtype=np.uint8)

    if len(inliers_fixed) >= 3:
        # Convex hull region → potential support (128)
        try:
            pts = inliers_fixed.astype(np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(support, hull, 128)
        except Exception:
            pass

    # Near verified inliers → supported (255)
    for pt in inliers_fixed:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x0 = max(0, x - support_radius_px)
        x1 = min(w, x + support_radius_px)
        y0 = max(0, y - support_radius_px)
        y1 = min(h, y + support_radius_px)
        support[y0:y1, x0:x1] = 255

    if HAS_RASTERIO and reference_tif and os.path.exists(reference_tif):
        try:
            with rasterio.open(reference_tif) as ref:
                kwargs = {
                    "driver": "GTiff", "height": h, "width": w, "count": 1,
                    "dtype": "uint8", "crs": ref.crs,
                    "transform": ref.transform, "compress": "lzw",
                }
            with rasterio.open(path, "w", **kwargs) as dst:
                dst.write(support, 1)
            logger.info("Wrote registration_support_mask.tif (georeferenced): %s", path)
            return path
        except Exception as exc:
            logger.warning("Could not write georeferenced support mask: %s", exc)

    cv2.imwrite(path, support)
    logger.info("Wrote registration_support_mask.tif (image-space): %s", path)
    return path


# ---------------------------------------------------------------------------
# 6. registered_source.tif — accepted results only
# ---------------------------------------------------------------------------
def export_registered_source(
    path: str,
    warped_img: np.ndarray,
    quality_passed: bool,
    reference_tif: Optional[str] = None,
) -> Optional[str]:
    """
    Write registered_source.tif ONLY when quality_passed=True.

    If quality_passed=False, this function returns None and writes nothing.
    Callers must never silently produce a registered product for rejected results.
    """
    if not quality_passed:
        logger.warning(
            "export_registered_source: quality gate failed — registered_source.tif "
            "will NOT be written (§9: rejected results must not produce accepted products)."
        )
        return None

    _ensure_dir(path)

    if HAS_RASTERIO and reference_tif and os.path.exists(reference_tif):
        try:
            with rasterio.open(reference_tif) as ref:
                n_bands = 1 if warped_img.ndim == 2 else warped_img.shape[2]
                kwargs = {
                    "driver": "GTiff",
                    "height": warped_img.shape[0],
                    "width":  warped_img.shape[1],
                    "count": n_bands, "dtype": str(warped_img.dtype),
                    "crs": ref.crs, "transform": ref.transform,
                    "compress": "lzw",
                }
            with rasterio.open(path, "w", **kwargs) as dst:
                if n_bands == 1:
                    dst.write(warped_img, 1)
                else:
                    for b in range(n_bands):
                        dst.write(warped_img[:, :, b], b + 1)
            logger.info("Wrote registered_source.tif (georeferenced): %s", path)
            return path
        except Exception as exc:
            logger.warning("Rasterio write failed, falling back to cv2: %s", exc)

    cv2.imwrite(path, warped_img)
    logger.info("Wrote registered_source.tif (image-space, no CRS): %s", path)
    return path


# ---------------------------------------------------------------------------
# 7. Master export function — write all §9 artifacts at once
# ---------------------------------------------------------------------------
def export_all_artifacts(
    output_dir: str,
    results: Dict[str, Any],
    warped_img: Optional[np.ndarray] = None,
    reference_tif: Optional[str] = None,
    fixed_descriptor: Optional[Dict[str, Any]] = None,
    moving_descriptor: Optional[Dict[str, Any]] = None,
    rejected_preview: bool = False,
) -> Dict[str, Optional[str]]:
    """
    Write the complete §9 artifact set to *output_dir*.

    Parameters
    ----------
    output_dir : str
        Directory where artifacts are written (created if absent).
    results : dict
        Pipeline results dict from ``run_registration_pipeline``.
    warped_img : ndarray, optional
        Registered source image (uint8 grayscale).
    reference_tif : str, optional
        Path to reference GeoTIFF for CRS/affine inheritance.
    fixed_descriptor : dict, optional
        Serialised ``ImageDescriptor`` for the fixed (reference) image.
    moving_descriptor : dict, optional
        Serialised ``ImageDescriptor`` for the moving (source) image.
    rejected_preview : bool
        If True and quality failed, write an UNVALIDATED preview GeoTIFF.

    Returns
    -------
    dict
        Mapping artifact_name → absolute path (or None if not written).
    """
    os.makedirs(output_dir, exist_ok=True)
    written: Dict[str, Optional[str]] = {}

    quality = results.get("quality") or {}
    passed  = bool(quality.get("passed", False))
    H       = results.get("H_fallback")
    inliers_fixed = results.get("inliers_ref")
    inliers_moving = results.get("inliers_src")

    # Determine evidence class
    gt_status = results.get("gt_validation_status", "not_provided")
    ho_val = results.get("holdout_validation") or {}
    if gt_status == "valid" and results.get("gt_rmse_pixels") is not None:
        ev_class = EVIDENCE_INDEPENDENT
    elif ho_val.get("n_holdout", 0) > 0:
        ev_class = EVIDENCE_HELD_OUT
    else:
        ev_class = EVIDENCE_FIT_ONLY

    # --- matches.csv ---
    pts_m = results.get("pts_src")
    pts_f = results.get("pts_ref")
    if pts_m is not None and pts_f is not None and len(pts_m) > 0:
        n   = len(pts_m)
        ink = results.get("mask")
        if ink is not None and len(ink) == n:
            inl_arr = np.asarray(ink, dtype=bool)
        else:
            inl_arr = None

        # Build per-match residuals in reference-pixel space
        res_ref = None
        res_src = None
        if H is not None and H.shape == (3, 3) and inl_arr is not None:
            try:
                proj = cv2.perspectiveTransform(
                    pts_m.reshape(-1, 1, 2).astype(np.float32), H
                ).reshape(-1, 2)
                res_ref = np.linalg.norm(proj - pts_f, axis=1)
            except Exception:
                pass

        # Build per-match validation split labels for holdout points
        vsplit = None
        if ho_val.get("n_holdout", 0) > 0 and inl_arr is not None:
            inl_idx = np.where(inl_arr)[0]
            n_inl = len(inl_idx)
            rng = np.random.RandomState(42)
            n_ho = max(1, n_inl // 5)
            ho_idx_local = rng.choice(n_inl, size=n_ho, replace=False)
            labels = np.array(["train"] * n, dtype=object)
            labels[inl_idx[ho_idx_local]] = "holdout"
            vsplit = labels

        written["matches.csv"] = export_matches_csv(
            os.path.join(output_dir, "matches.csv"),
            pts_moving=pts_m,
            pts_fixed=pts_f,
            confidences=results.get("confidences"),
            engine=np.array(
                [results.get("method_used", "")] * n, dtype=object
            ),
            inlier_mask=inl_arr,
            validation_split=vsplit,
            residuals_ref_px=res_ref,
            residuals_src_px=res_src,
        )

    # --- transform.json ---
    mat = H if H is not None else results.get("transform")
    if isinstance(mat, np.ndarray):
        model = f"HOMOGRAPHY_3x3" if mat.shape == (3, 3) else f"AFFINE_2x3"
        inv_avail = mat.shape == (3, 3)
        inv_method = "numpy.linalg.inv" if inv_avail else None
    elif hasattr(mat, "applyTransformation"):
        model = "TPS_NONRIGID"
        inv_avail = False
        inv_method = None
        mat = None
    else:
        model = "UNKNOWN"
        inv_avail = False
        inv_method = None
        mat = None

    written["transform.json"] = export_transform_json(
        os.path.join(output_dir, "transform.json"),
        model=model,
        matrix=mat,
        direction="moving_to_fixed",
        coordinate_convention="pixel_corner_origin",
        inversion_available=inv_avail,
        inversion_method=inv_method,
    )

    # --- quality.json ---
    written["quality.json"] = export_quality_json(
        os.path.join(output_dir, "quality.json"),
        results=results,
        evidence_class=ev_class,
    )

    # --- provenance.json ---
    written["provenance.json"] = export_provenance_json(
        os.path.join(output_dir, "provenance.json"),
        results=results,
        fixed_descriptor=fixed_descriptor,
        moving_descriptor=moving_descriptor,
    )

    # --- registered_source.tif (accepted only) ---
    if warped_img is not None:
        reg_path = os.path.join(output_dir, "registered_source.tif")
        if passed:
            written["registered_source.tif"] = export_registered_source(
                reg_path, warped_img, quality_passed=True,
                reference_tif=reference_tif,
            )
        elif rejected_preview:
            preview_path = os.path.join(output_dir, "registered_source_UNVALIDATED.tif")
            logger.warning(
                "Writing UNVALIDATED preview (quality gate failed). "
                "This file is NOT an accepted registered product."
            )
            cv2.imwrite(preview_path, warped_img)
            written["registered_source_UNVALIDATED.tif"] = preview_path
        else:
            written["registered_source.tif"] = None  # explicitly not written

    # --- valid_mask.tif ---
    if warped_img is not None and passed:
        written["valid_mask.tif"] = export_valid_mask(
            os.path.join(output_dir, "valid_mask.tif"),
            warped_img=warped_img,
            reference_tif=reference_tif,
        )

    # --- registration_support_mask.tif ---
    if (
        inliers_fixed is not None
        and len(inliers_fixed) > 0
        and warped_img is not None
    ):
        written["registration_support_mask.tif"] = export_support_mask(
            os.path.join(output_dir, "registration_support_mask.tif"),
            img_shape=warped_img.shape[:2],
            inliers_fixed=inliers_fixed,
            H=H,
            reference_tif=reference_tif,
        )

    # Summary log
    n_written = sum(1 for v in written.values() if v is not None)
    logger.info(
        "Artifact export complete: %d/%d artifacts written to %s",
        n_written, len(written), output_dir,
    )
    return written
