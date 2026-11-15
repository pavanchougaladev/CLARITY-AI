"""
State Management Service for Streamlit Session State
====================================================
ISRO LunarFusion Co-Registration Suite — UI State Store
"""

import streamlit as st
from typing import Optional, Dict, Any

try:
    from ui.config import DEFAULT_CALLSIGN, DEFAULT_CLEARANCE
except (ImportError, ValueError):
    from ..config import DEFAULT_CALLSIGN, DEFAULT_CLEARANCE


def init_session_state() -> None:
    """Initialize all core session state keys with deterministic defaults."""
    default_keys: Dict[str, Any] = {
        "active_results": None,
        "active_img_src": None,
        "active_img_ref": None,
        "active_src_path": None,
        "active_ref_path": None,
        "last_backend_job_id": None,
        "dual_pair_results": None,
        "authenticated": False,
        "officer_callsign": DEFAULT_CALLSIGN,
        "clearance_tier": DEFAULT_CLEARANCE,
        "show_moon_warp_overlay": False,
        "dem_task": None,
        "active_mesh_obj_path": None,
        "active_dem_geotiff_path": None,
        "crater_annotations": None,
    }

    for key, default_val in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

    if st.session_state.get("dem_task") is None:
        try:
            from src.features.dem_builder import BackgroundDEMTask
            st.session_state.dem_task = BackgroundDEMTask()
        except Exception:
            st.session_state.dem_task = None

    # Synchronize optional URL query parameters for direct deep-linking
    _sync_query_params()


def _sync_query_params() -> None:
    """Sync authentication and officer parameters from URL query params if provided."""
    try:
        qp = st.query_params
        qp_auth = qp.get("auth", "").lower() in ["true", "1", "granted"]
        qp_officer = qp.get("officer", "")
        qp_tier = qp.get("tier", "")

        if qp_auth:
            st.session_state.authenticated = True
        if qp_officer:
            st.session_state.officer_callsign = qp_officer
        if qp_tier:
            st.session_state.clearance_tier = qp_tier
    except Exception:
        pass


def login(callsign: str, clearance: str) -> None:
    """Authenticate session and trigger cosmic portal animation."""
    st.session_state.authenticated = True
    st.session_state.officer_callsign = callsign or DEFAULT_CALLSIGN
    st.session_state.clearance_tier = clearance or DEFAULT_CLEARANCE
    st.session_state.show_moon_warp_overlay = True


def logout() -> None:
    """Invalidate session and clear authorization credentials."""
    st.session_state.authenticated = False
    try:
        st.query_params.clear()
    except Exception:
        pass


def set_active_registration(
    results: Dict[str, Any],
    src_path: Optional[str] = None,
    ref_path: Optional[str] = None,
    img_src: Optional[Any] = None,
    img_ref: Optional[Any] = None,
) -> None:
    """Store the latest active registration telemetry and rendered pair."""
    st.session_state.active_results = results
    if src_path is not None:
        st.session_state.active_src_path = src_path
    if ref_path is not None:
        st.session_state.active_ref_path = ref_path
    if img_src is not None:
        st.session_state.active_img_src = img_src
    if img_ref is not None:
        st.session_state.active_img_ref = img_ref


def get_active_registration() -> Optional[Dict[str, Any]]:
    """Retrieve active registration results if present."""
    return st.session_state.get("active_results", None)
