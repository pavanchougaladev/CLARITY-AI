"""
API In-Memory State & Synchronization Primitives
================================================
ISRO LunarFusion Co-Registration Suite — Job Registry
"""

import os
import shutil
import tempfile
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any
from fastapi import UploadFile, HTTPException

from .config import (
    MAX_CONCURRENT_JOBS,
    MAX_JOBS,
    MAX_UPLOAD_BYTES,
    UPLOAD_CHUNK_BYTES,
    ALLOWED_EXTENSIONS,
    API_WORKER_POOL,
    AUTH_ENABLED,
)
from src.auth.jwt_handler import decode_access_token

# Scratch directory root for job isolation
SCRATCH_ROOT = os.path.join(tempfile.gettempdir(), "isro_lunar_api_scratch")
os.makedirs(SCRATCH_ROOT, exist_ok=True)

# In-memory stores
jobs: Dict[str, Any] = {}
cancellation_events: Dict[str, threading.Event] = {}

# Concurrency primitives
if API_WORKER_POOL == "process":
    executor = ProcessPoolExecutor(max_workers=MAX_CONCURRENT_JOBS)
else:
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="LunarRegWorker")

job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
jobs_lock = threading.Lock()


def _sanitize_filename(filename: str) -> str:
    """Strip path traversal components and unsafe characters."""
    base = os.path.basename(filename or "")
    clean = "".join(c for c in base if c.isalnum() or c in "._-")
    return clean or "upload.bin"


def _evict_old_jobs():
    """Remove oldest jobs when store reaches capacity, cleaning scratch dirs."""
    with jobs_lock:
        if len(jobs) >= MAX_JOBS:
            ordered = sorted(jobs.keys(), key=lambda jid: jobs[jid].get("submitted_at", 0))
            evict_count = max(1, len(jobs) - MAX_JOBS + 1)
            for jid in ordered[:evict_count]:
                job = jobs.pop(jid, None)
                cancellation_events.pop(jid, None)
                if job and "job_dir" in job and os.path.exists(job["job_dir"]):
                    try:
                        shutil.rmtree(job["job_dir"], ignore_errors=True)
                    except Exception:
                        pass


def _validate_upload(upload: UploadFile) -> str:
    """Validate uploaded file extension."""
    filename = _sanitize_filename(upload.filename or "")
    extension = os.path.splitext(filename)[1].lower()
    if not filename or extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"Unsupported image type. Allowed: {allowed}")
    return extension


async def _save_upload(upload: UploadFile, destination: str) -> int:
    """Stream an uploaded file to disk with strict size caps."""
    bytes_written = 0
    with open(destination, "wb") as f:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                f.close()
                if os.path.exists(destination):
                    os.remove(destination)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowable size of {MAX_UPLOAD_BYTES // (1024*1024)} MB",
                )
            f.write(chunk)
    return bytes_written
