"""
UI Configuration and Environment Settings
========================================
ISRO LunarFusion Co-Registration Suite — UI Frontend Config
"""

import os

# Base Directories
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(WORKSPACE_ROOT, "data"))
OUTPUTS_DIR = os.path.abspath(os.path.join(WORKSPACE_ROOT, "outputs"))

# Backend REST API Endpoint
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://127.0.0.1:8000")

# Security & Operations Defaults
DEFAULT_CALLSIGN = os.environ.get("ISRO_DEFAULT_CALLSIGN", "MISSION-SCIENTIST")
DEFAULT_CLEARANCE = os.environ.get("ISRO_DEFAULT_CLEARANCE", "Level 4 — Flight Director & Sub-Pixel Lead")

# Processing Defaults
DEFAULT_MAX_DIM = int(os.environ.get("MAX_PROCESSING_DIM", "2048"))
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("BACKEND_TIMEOUT_SECONDS", "900"))
DEFAULT_STORAGE_FILE = "registration_history.json"
