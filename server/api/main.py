"""
ISRO Lunar Image Registration Pipeline — Scalable Modular REST API Gateway
==========================================================================
Modular FastAPI architecture with dedicated routers, authentication, bounded
concurrency, cooperative cancellation, timeout handling, and quality telemetry.
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT, AUTH_ENABLED
from .state import (
    jobs,
    cancellation_events,
    executor,
    jobs_lock,
    job_semaphore,
    SCRATCH_ROOT,
    _sanitize_filename,
    _validate_upload,
    _save_upload,
    _evict_old_jobs,
)
from .worker import process_registration
from .dependencies import (
    LoginRequest,
    TripletValidationRequest,
    get_current_user,
    require_admin,
)

# Modular Routers
from .routers.health import router as health_router
from .routers.auth import router as auth_router
from .routers.registration import router as registration_router
from .routers.jobs import router as jobs_router
from .routers.triplet import router as triplet_router
from .routers.export import router as export_router

logger = logging.getLogger(__name__)

# FastAPI Application Factory / Instance
app = FastAPI(
    title="CLARITY* — Chandrayaan Lunar AI Registration & Image Transformation system and 3D lunar mapping API",
    description="REST API for automated sub-pixel image co-registration and 3D lunar mapping",
    version="2.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Modular Routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(registration_router)
app.include_router(jobs_router)
app.include_router(triplet_router)
app.include_router(export_router)

# Compatibility exports for test suites
__all__ = [
    "app",
    "jobs",
    "cancellation_events",
    "executor",
    "process_registration",
    "_sanitize_filename",
    "_validate_upload",
    "_save_upload",
    "_evict_old_jobs",
    "jobs_lock",
    "job_semaphore",
    "SCRATCH_ROOT",
    "LoginRequest",
    "TripletValidationRequest",
    "get_current_user",
    "require_admin",
    "AUTH_ENABLED",
]

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
