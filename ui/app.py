"""
CLARITY* — Chandrayaan Lunar AI Registration & Image Transformation system and 3D lunar mapping
Production Web Interface Gateway
===============================================================================================
Production-grade scientific dashboard for Chandrayaan-1 / Chandrayaan-2
heterogeneous image co-registration, photogrammetry export, 3D mapping, and mosaic creation.

Architectural Design:
- Decoupled into specialized views (Single Scene, Batch, Mosaic, History, 3D Topography, YOLO11)
- Isolated services for state management, REST API client, and safe raster loading
- Reusable UI styling and visualization canvas components
- 100% backward-compatible interface contracts with existing automated test suites
"""

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Configuration and Paths
from ui.config import (
    DATA_DIR,
    WORKSPACE_ROOT,
    OUTPUTS_DIR,
    BACKEND_API_URL,
    DEFAULT_TIMEOUT_SECONDS,
)
from ui.services.state import (
    init_session_state,
    login,
    logout,
    set_active_registration,
    get_active_registration,
)
from ui.services.api_client import (
    is_backend_online,
    format_backend_results,
    list_backend_jobs,
    cancel_backend_job,
    delete_backend_job,
    fetch_backend_artifact,
    trigger_backend_blender_export,
    fetch_backend_job,
    submit_registration_to_backend,
    _format_backend_results,
    _list_backend_jobs,
    _cancel_backend_job,
    _delete_backend_job,
    _fetch_backend_artifact,
    _trigger_backend_blender_export,
    _fetch_backend_job,
    _submit_registration_to_backend,
    _is_backend_online,
)
from ui.services.image_loader import (
    get_safe_preview,
    find_archive_metadata,
    metadata_for_archive_pair,
    save_uploaded_orbital_image,
    save_uploaded_metadata,
    sha256_file,
    discover_browse_images,
    get_cached_crater_detector,
    _find_archive_metadata,
    _metadata_for_archive_pair,
    _save_uploaded_orbital_image,
    _save_uploaded_metadata,
    _sha256_file,
)
from ui.components.styles import apply_custom_theme, render_metric_card
from ui.components.visualizations import (
    create_checkerboard,
    compute_seam_alignment_continuity,
    create_hd_match_canvas,
    interactive_match_plot,
    normalize_to_rgb8,
    generate_dem_preview,
)
from ui.views.header import render_cosmic_header
from ui.views.sidebar import render_sidebar
from ui.views.tab_single import render_single_registration_tab, render_registration_results
from ui.views.tab_batch import render_batch_tab
from ui.views.tab_mosaic import render_mosaic_tab
from ui.views.tab_history import render_history_tab, get_cached_history
from ui.views.tab_3d import render_3d_tab
from ui.views.tab_yolo import render_yolo_tab

# Core Algorithmic Pipeline Symbols
from src.history import HistoryManager
from src.config import get_config


# Global Managers
history_mgr = HistoryManager()
cfg = get_config()


def main() -> None:
    """Main UI Application Flow orchestrating header, sidebar, and tab views."""
    # ── Page Configuration ────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="CLARITY* — Chandrayaan Lunar AI Registration & Image Transformation system and 3D lunar mapping",
        page_icon="🌔",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── State Initialization & Theme Injection ────────────────────────────────────
    init_session_state()
    apply_custom_theme()

    available_images = discover_browse_images(DATA_DIR)

    # 1. 3D Cosmic Space Intro & Mission Authentication Gateway
    render_cosmic_header()

    # 2. Sidebar Navigation & Global Pipeline Parameters
    pipeline_config = render_sidebar(dataset_count=len(available_images))

    # Temporarily override in-memory pipeline config with sidebar selections
    _cfg_pipeline = cfg.setdefault("pipeline", {})
    _cfg_pipeline["enable_subpixel"] = pipeline_config.get("enable_subpixel", True)

    # 3. Application Workflow Navigation Tabs
    tab_single, tab_batch, tab_mosaic, tab_history, tab_3d, tab_yolo = st.tabs([
        "Single Scene Registration",
        "Batch Processing",
        "Mosaic Builder",
        "Mission History",
        "3D Topography & DEM",
        "🌔 YOLO11 Crater & Hazard AI",
    ])

    with tab_single:
        render_single_registration_tab(available_images, history_mgr, cfg, pipeline_config)

    with tab_batch:
        render_batch_tab()

    with tab_mosaic:
        render_mosaic_tab(available_images)

    with tab_history:
        render_history_tab(history_mgr)

    with tab_3d:
        render_3d_tab(available_images)

    with tab_yolo:
        render_yolo_tab(available_images)

    # 4. Aerospace Mission Footer
    st.markdown("---")
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 6px; font-size: 0.82rem; color: #94A3B8; flex-wrap: wrap; gap: 8px;">
        <div>
            <strong style="color:#38BDF8; font-size:0.88rem;">CLARITY*</strong> — Chandrayaan Lunar AI Registration &amp; Image Transformation System | LoFTR • RoMa v2 • LightGlue • MAGSAC++ • NASA ASP
        </div>
        <div>
            <span style="color: #CBD5E1;">Autonomous Planetary Mapping Suite</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


# ── Full Backward-Compatibility Exports for Automated Test Suites ──────────────
__all__ = [
    "create_checkerboard",
    "compute_seam_alignment_continuity",
    "generate_dem_preview",
    "interactive_match_plot",
    "render_registration_results",
    "_format_backend_results",
    "_list_backend_jobs",
    "_cancel_backend_job",
    "_delete_backend_job",
    "_fetch_backend_artifact",
    "_trigger_backend_blender_export",
    "_find_archive_metadata",
    "get_safe_preview",
    "discover_browse_images",
    "BACKEND_API_URL",
    "DATA_DIR",
    "_save_uploaded_metadata",
    "_save_uploaded_orbital_image",
    "_sha256_file",
    "_metadata_for_archive_pair",
    "_is_backend_online",
    "_fetch_backend_job",
    "_submit_registration_to_backend",
    "get_cached_history",
]
