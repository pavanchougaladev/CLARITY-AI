"""
FastAPI Background Registration Worker
======================================
ISRO LunarFusion Co-Registration Suite — Execution Pipeline
"""

import os
import time
import json
import logging
import threading
from typing import Optional
import numpy as np

from .state import jobs, cancellation_events, jobs_lock
from src.pipeline import run_registration_pipeline
from src.result_schema import RegistrationTelemetry
from src.registration.export import export_geotiff, export_asp_compatible_tiepoints
from src.reporting.pdf_report import generate_pdf_report

logger = logging.getLogger(__name__)


def process_registration(
    job_id: str,
    src_path: str,
    ref_path: str,
    matcher: str = "loftr",
    max_dim: int = 2048,
    use_tps: bool = True,
    speed_mode: str = "balanced",
    cancellation_token: Optional[threading.Event] = None,
    job_dir: Optional[str] = None,
    pipeline_mode: Optional[str] = None,
    sensor_pair: Optional[str] = None,
    region_type: Optional[str] = None,
):
    """
    Execution worker for registration pipeline.
    Maintains compatibility with test_backend_frontend_contract.py while
    adding cooperative cancellation, artifact persistence, and progress tracking.
    """
    if cancellation_token is None:
        cancellation_token = cancellation_events.get(job_id)

    try:
        with jobs_lock:
            if job_id not in jobs:
                jobs[job_id] = {}
            if cancellation_token and cancellation_token.is_set():
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id]["error"] = "Job cancelled prior to processing"
                return
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["start_time"] = time.time()
            jobs[job_id]["progress"] = {
                "step": 0,
                "total": 8,
                "message": "Starting registration pipeline",
            }

        t0 = time.time()

        def _on_progress(step: int, total: int, msg: str):
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["progress"] = {"step": step, "total": total, "message": msg}

        pipeline_kwargs = {
            "use_tps": use_tps,
            "max_dim": max_dim,
            "matcher_type": matcher,
            "progress_callback": _on_progress,
            "cancellation_token": cancellation_token,
            "pipeline_mode": pipeline_mode,
            "sensor_pair": sensor_pair,
            "region_type": region_type,
        }
        if matcher in ("fusion", "lunar_fusion"):
            pipeline_kwargs["speed_mode"] = speed_mode

        _, _, results = run_registration_pipeline(src_path, ref_path, **pipeline_kwargs)

        # Check cancellation before serializing
        if cancellation_token and cancellation_token.is_set():
            with jobs_lock:
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id]["error"] = "Registration cancelled by user"
            return

        telemetry = RegistrationTelemetry.from_results(results)
        clean_results = telemetry.to_dict()
        clean_results["quality"] = {
            "grade": telemetry.quality_grade,
            "passed": telemetry.quality_passed,
            "confidence": telemetry.quality_confidence,
            "warnings": telemetry.warnings,
            "reason_codes": telemetry.reason_codes,
            "primary_reason": telemetry.primary_reason,
        }
        clean_results["provenance"] = results.get("provenance", {})
        clean_results["provenance_chain"] = results.get("provenance_chain")
        clean_results["recovery_attempts"] = results.get("recovery_attempts", [])
        clean_results["method_used"] = results.get("method_used", telemetry.method)
        clean_results["scale_src"] = float(results.get("scale_src", 1.0))
        clean_results["scale_ref"] = float(results.get("scale_ref", 1.0))
        clean_results["inliers_src"] = (
            results["inliers_src"].tolist()
            if isinstance(results.get("inliers_src"), np.ndarray)
            else list(results.get("inliers_src", []))
        )
        clean_results["inliers_ref"] = (
            results["inliers_ref"].tolist()
            if isinstance(results.get("inliers_ref"), np.ndarray)
            else list(results.get("inliers_ref", []))
        )
        clean_results["confidences"] = (
            results["confidences"].tolist()
            if isinstance(results.get("confidences"), np.ndarray)
            else list(results.get("confidences", []))
        )
        if isinstance(results.get("H_fallback"), np.ndarray):
            clean_results["H_fallback"] = results["H_fallback"].tolist()
        clean_results["error_stats"] = results.get("error_stats", {})
        clean_results["omniroute_stats"] = results.get("omniroute_stats")
        clean_results["lunar_fusion_stats"] = results.get("lunar_fusion_stats")
        clean_results["self_healed"] = bool(results.get("self_healed", False))

        # Generate and save downloadable artifacts if job_dir is provided
        artifacts_manifest = {}
        if job_dir and os.path.exists(job_dir):
            try:
                # 1. Canonical §9 Artifact Exporter
                from src.registration.artifact_exporter import export_all_artifacts

                warped_img = results.get("img_src_warped")
                canonical = export_all_artifacts(
                    output_dir=job_dir,
                    results=results,
                    warped_img=warped_img,
                    reference_tif=ref_path if ref_path.lower().endswith((".tif", ".tiff")) else None,
                )
                for art_name, art_path in canonical.items():
                    if art_path and os.path.exists(art_path):
                        artifacts_manifest[art_name] = art_path

                # 2. Legacy Warped GeoTIFF
                if results.get("quality", {}).get("passed", False) and warped_img is not None:
                    tif_path = os.path.join(job_dir, "warped_registered.tif")
                    export_geotiff(warped_img, tif_path, reference_tif=ref_path)
                    if os.path.exists(tif_path):
                        artifacts_manifest["geotiff"] = tif_path

                # 3. ASP-compatible tie-points CSV
                inliers_s = results.get("inliers_src")
                inliers_r = results.get("inliers_ref")
                confs = results.get("confidences")
                if inliers_s is not None and len(inliers_s) > 0:
                    pts_s_arr = np.asarray(inliers_s, dtype=np.float32)
                    pts_r_arr = np.asarray(inliers_r, dtype=np.float32)
                    conf_arr = (
                        np.asarray(confs, dtype=np.float32)
                        if confs is not None
                        else np.ones(len(pts_s_arr), dtype=np.float32)
                    )
                    csv_path = os.path.join(job_dir, "tiepoints_asp.csv")
                    export_asp_compatible_tiepoints(pts_s_arr, pts_r_arr, conf_arr, csv_path)
                    if os.path.exists(csv_path):
                        artifacts_manifest["tiepoints"] = csv_path

                # 4. PDF Certification Report
                pdf_path = os.path.join(job_dir, "certification_report.pdf")
                generate_pdf_report(results, src_path, ref_path, pdf_path)
                if os.path.exists(pdf_path):
                    artifacts_manifest["pdf"] = pdf_path

                # 5. JSON Telemetry
                json_path = os.path.join(job_dir, "telemetry.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(clean_results, f, indent=2, default=str)
                artifacts_manifest["telemetry"] = json_path
            except Exception as e_art:
                logger.warning(f"Error persisting artifacts for job {job_id}: {e_art}")

        with jobs_lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["results"] = clean_results
            jobs[job_id]["artifacts"] = list(artifacts_manifest.keys())
            jobs[job_id]["artifact_paths"] = artifacts_manifest
            jobs[job_id]["elapsed_seconds"] = round(time.time() - t0, 2)
            jobs[job_id]["progress"] = {
                "step": 8,
                "total": 8,
                "message": "Registration completed successfully",
            }

    except InterruptedError:
        with jobs_lock:
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["error"] = "Registration pipeline cancelled by client"
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            logger.error(f"Job {job_id} failed with error: {e}", exc_info=True)
