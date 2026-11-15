"""
Registration Job Submission Endpoints
=====================================
ISRO LunarFusion Co-Registration Suite — Execution Ingest
"""

import os
import uuid
import time
import shutil
import logging
import asyncio
import threading
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, Query, HTTPException, BackgroundTasks

from ..config import (
    ALLOWED_MATCHERS,
    JOB_TIMEOUT_SECONDS,
)
from ..state import (
    jobs,
    cancellation_events,
    executor,
    job_semaphore,
    jobs_lock,
    SCRATCH_ROOT,
    _sanitize_filename,
    _validate_upload,
    _save_upload,
    _evict_old_jobs,
)
from ..worker import process_registration

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Registration"])


async def _managed_worker(
    job_id: str,
    src_path: str,
    ref_path: str,
    matcher: str,
    max_dim: int,
    use_tps: bool,
    speed_mode: str,
    cancellation_token: threading.Event,
    job_dir: str,
    pipeline_mode: Optional[str] = None,
    sensor_pair: Optional[str] = None,
    region_type: Optional[str] = None,
):
    """Asynchronous worker orchestrating bounded semaphore and execution timeout."""
    async with job_semaphore:
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    process_registration,
                    job_id,
                    src_path,
                    ref_path,
                    matcher,
                    max_dim,
                    use_tps,
                    speed_mode,
                    cancellation_token,
                    job_dir,
                    pipeline_mode,
                    sensor_pair,
                    region_type,
                ),
                timeout=JOB_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            cancellation_token.set()
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = f"Job exceeded execution timeout of {JOB_TIMEOUT_SECONDS:.0f} seconds"
            logger.warning(f"Job {job_id} cancelled due to timeout ({JOB_TIMEOUT_SECONDS}s).")
        except Exception as e:
            with jobs_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = str(e)


@router.post("/api/v1/register", status_code=202)
async def register_images(
    background_tasks: BackgroundTasks,
    source_image: Optional[UploadFile] = File(None),
    reference_image: Optional[UploadFile] = File(None),
    source_path: Optional[str] = Form(None),
    reference_path: Optional[str] = Form(None),
    matcher: Optional[str] = Form(None),
    matcher_query: Optional[str] = Query(None, alias="matcher"),
    max_dim: Optional[int] = Form(2048),
    use_tps: Optional[bool] = Form(True),
    speed_mode: Optional[str] = Form("balanced"),
    pipeline_mode: Optional[str] = Form(None),
    sensor_pair: Optional[str] = Form(None),
    region_type: Optional[str] = Form(None),
):
    """
    Submit an asynchronous registration job.
    Accepts either direct file uploads or disk paths (for high-speed local processing).
    Returns a job ID that can be used to poll progress, download artifacts, or cancel.
    """
    raw_matcher = matcher or matcher_query or "loftr"
    selected_matcher = raw_matcher.lower().strip()
    if selected_matcher not in ALLOWED_MATCHERS:
        raise HTTPException(status_code=422, detail=f"Unsupported matcher: {selected_matcher}")

    _evict_old_jobs()
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(SCRATCH_ROOT, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)

    if source_path and reference_path and os.path.exists(source_path) and os.path.exists(reference_path):
        src_path = os.path.abspath(source_path)
        ref_path = os.path.abspath(reference_path)
        src_filename = os.path.basename(source_path)
        ref_filename = os.path.basename(reference_path)
    elif source_image is not None and reference_image is not None:
        src_ext = _validate_upload(source_image)
        ref_ext = _validate_upload(reference_image)
        src_path = os.path.join(job_dir, f"source{src_ext}")
        ref_path = os.path.join(job_dir, f"reference{ref_ext}")
        try:
            await _save_upload(source_image, src_path)
            await _save_upload(reference_image, ref_path)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        src_filename = _sanitize_filename(source_image.filename or "")
        ref_filename = _sanitize_filename(reference_image.filename or "")
    else:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Must provide either source_image & reference_image files, or valid source_path & reference_path on disk.",
        )

    eff_max_dim = int(max_dim) if max_dim else 2048
    eff_use_tps = bool(use_tps) if use_tps is not None else True
    eff_speed = str(speed_mode) if speed_mode else "balanced"

    cancellation_token = threading.Event()
    cancellation_events[job_id] = cancellation_token

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "job_id": job_id,
            "submitted_at": time.time(),
            "src_file": src_filename,
            "ref_file": ref_filename,
            "matcher": selected_matcher,
            "job_dir": job_dir,
            "pipeline_mode": pipeline_mode,
            "sensor_pair": sensor_pair,
            "region_type": region_type,
            "progress": {"step": 0, "total": 8, "message": "Queued for processing"},
        }

    # Dispatch managed worker asynchronously
    background_tasks.add_task(
        _managed_worker,
        job_id,
        src_path,
        ref_path,
        selected_matcher,
        eff_max_dim,
        eff_use_tps,
        eff_speed,
        cancellation_token,
        job_dir,
        pipeline_mode,
        sensor_pair,
        region_type,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/api/v1/jobs/{job_id}",
        "cancel_url": f"/api/v1/jobs/{job_id}/cancel",
        "submitted_at": jobs[job_id]["submitted_at"],
        "matcher": selected_matcher,
    }
