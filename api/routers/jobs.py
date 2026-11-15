"""
Job Polling, Cancellation, Artifact Retrieval & Deletion Endpoints
==================================================================
ISRO LunarFusion Co-Registration Suite — Job Lifecycle
"""

import os
import time
import shutil
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse

from ..state import jobs, cancellation_events, jobs_lock
from ..dependencies import require_admin

router = APIRouter(tags=["Jobs"])


@router.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll registration job progress, execution metrics, and results."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = dict(jobs[job_id])

    # Omit raw scratch directory from public JSON response
    job.pop("job_dir", None)
    return job


@router.post("/api/v1/jobs/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str):
    """Signal cooperative cancellation to an active or queued job."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        status = jobs[job_id].get("status")
        if status in ("completed", "failed", "cancelled"):
            return {
                "job_id": job_id,
                "status": status,
                "message": f"Job already terminated with status {status}",
            }

        jobs[job_id]["status"] = "cancelling"
        if job_id in cancellation_events:
            cancellation_events[job_id].set()

    return {"job_id": job_id, "status": "cancelling", "message": "Cancellation signal transmitted"}


@router.delete("/api/v1/jobs/{job_id}", status_code=200)
async def delete_job(job_id: str, user: dict = Depends(require_admin)):
    """Delete a completed, cancelled, or failed job and purge its scratch files."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        if jobs[job_id].get("status") == "processing":
            raise HTTPException(status_code=409, detail="Cannot delete a job that is actively processing; cancel first")
        job = jobs.pop(job_id)
        cancellation_events.pop(job_id, None)

    if job and "job_dir" in job and os.path.exists(job["job_dir"]):
        try:
            shutil.rmtree(job["job_dir"], ignore_errors=True)
        except Exception:
            pass

    return {"deleted": job_id}


@router.get("/api/v1/jobs/{job_id}/artifacts/{artifact_type}")
async def download_artifact(job_id: str, artifact_type: str):
    """
    Download verified output artifacts: 'geotiff', 'tiepoints', 'pdf', 'telemetry',
    'matches_csv', 'transform_json', 'quality_json', or 'provenance_json'.
    """
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = jobs[job_id]

    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (current status: {job.get('status')})")

    paths = job.get("artifact_paths", {})
    if artifact_type not in paths or not os.path.exists(paths[artifact_type]):
        available = list(paths.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{artifact_type}' not available. Available: {available}",
        )

    file_path = paths[artifact_type]
    media_types = {
        "geotiff": "image/tiff",
        "tiepoints": "text/csv",
        "matches_csv": "text/csv",
        "pdf": "application/pdf",
        "telemetry": "application/json",
        "transform_json": "application/json",
        "quality_json": "application/json",
        "provenance_json": "application/json",
    }
    return FileResponse(
        file_path,
        media_type=media_types.get(artifact_type, "application/octet-stream"),
        filename=os.path.basename(file_path),
    )


@router.get("/api/v1/jobs")
async def list_jobs(limit: int = 50):
    """List recent registration jobs ordered chronologically."""
    with jobs_lock:
        summary = [
            {
                "job_id": jid,
                "status": j.get("status"),
                "submitted_at": j.get("submitted_at"),
                "src_file": j.get("src_file"),
                "ref_file": j.get("ref_file"),
                "elapsed_seconds": (
                    j.get("elapsed_seconds")
                    if j.get("elapsed_seconds") is not None
                    else round(time.time() - j.get("submitted_at", time.time()), 2)
                ),
                "progress": j.get("progress"),
                "has_results": "results" in j,
            }
            for jid, j in sorted(
                jobs.items(),
                key=lambda x: x[1].get("submitted_at", 0),
                reverse=True,
            )
        ]
    return {"count": len(summary[:limit]), "jobs": summary[:limit]}
