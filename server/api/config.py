"""
FastAPI Server Configuration & Environment Bindings
===================================================
ISRO LunarFusion Co-Registration Suite — API Gateway
"""

import os

# Server Networking
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Upload and concurrency constraints
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".img"}
ALLOWED_MATCHERS = {
    "loftr", "lightglue", "sift", "rootsift", "roma", "roma_v2", "roma_omega",
    "tiny_roma", "ni_lfm", "ensemble", "fusion", "lunar_fusion", "omniroute", "omni",
    "yolo11", "yolo", "yolo11_crater",
    "superglue", "superpoint_superglue", "rift2", "asift", "akaze",
}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))  # 512MB
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
JOB_TIMEOUT_SECONDS = float(os.getenv("JOB_TIMEOUT_SECONDS", "900.0"))
MAX_JOBS = int(os.getenv("MAX_JOBS", "200"))

# Worker Execution Pool: "thread" or "process"
API_WORKER_POOL = os.getenv("API_WORKER_POOL", "thread").lower()

# Authentication & RBAC
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1", "yes")

# CORS Origins
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    if origin.strip()
]
