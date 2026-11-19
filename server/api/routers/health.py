"""
Health Check & Root Status Endpoints
====================================
ISRO LunarFusion Co-Registration Suite — Health Monitor
"""

from fastapi import APIRouter
from ..config import MAX_CONCURRENT_JOBS, JOB_TIMEOUT_SECONDS
from ..state import jobs, jobs_lock

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {
        "message": "ISRO Lunar Image Registration Pipeline API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
        "max_concurrent_workers": MAX_CONCURRENT_JOBS,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
    }


@router.get("/health")
@router.get("/api/v1/health")
def health():
    with jobs_lock:
        active = sum(1 for j in jobs.values() if j.get("status") in ("queued", "processing"))
        total = len(jobs)
    return {
        "status": "healthy",
        "service": "lunar-registration-pipeline",
        "active_jobs": active,
        "total_jobs": total,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
    }
