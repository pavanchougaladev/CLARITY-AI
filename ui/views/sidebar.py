"""
Sidebar Navigation and Global Pipeline Parameters View
======================================================
ISRO LunarFusion Co-Registration Suite — UI Sidebar
Renders mission controls, clearance verification, physical optics toggles, and backend health indicators.
"""

import os
import streamlit as st
import urllib.request
from typing import Dict, Any

try:
    from ui.config import BACKEND_API_URL
    from ui.services.state import logout
except (ImportError, ValueError):
    from ..config import BACKEND_API_URL
    from ..services.state import logout


def render_sidebar(dataset_count: int = 0) -> Dict[str, Any]:
    """Render sidebar controls and return active configuration options."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:12px 10px; margin-bottom:14px; background:rgba(16,26,54,0.65); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.10); border-radius:12px; box-shadow:0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.12);">
            <div style="display:inline-flex; align-items:center; justify-content:center; gap:8px;">
                <span style="font-size:1.3rem;">🌔</span>
                <span style="font-family:'Inter',sans-serif; font-size:1.25rem; font-weight:800; letter-spacing:0.04em; color:#FFFFFF;">CLARITY<span style="color:#38BDF8;">*</span></span>
            </div>
            <div style="font-size:0.70rem; font-weight:500; color:#94A3B8; margin-top:4px; line-height:1.35;">
                Chandrayaan Lunar AI Registration &amp; 3D Mapping
            </div>
        </div>
        """, unsafe_allow_html=True)

        callsign = st.session_state.get("officer_callsign", "MISSION-SCIENTIST")
        tier = st.session_state.get("clearance_tier", "Level 4 — Flight Director & Sub-Pixel Lead")
        is_auth = st.session_state.get("authenticated", False)

        st.markdown(f"""
        <div style="background:rgba(14,23,48,0.55); backdrop-filter:blur(12px); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:12px 14px; margin-bottom:14px; box-shadow:0 4px 16px rgba(0,0,0,0.3);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:0.68rem; font-weight:600; color:{'#34D399' if is_auth else '#38BDF8'}; letter-spacing:0.05em; display:flex; align-items:center; gap:6px; font-family:'JetBrains Mono',monospace;">
                    <span style="width:6px; height:6px; border-radius:50%; background:{'#34D399' if is_auth else '#38BDF8'}; box-shadow:0 0 6px {'#34D399' if is_auth else '#38BDF8'}; display:inline-block;"></span> {'AUTHENTICATED' if is_auth else 'STANDBY'}
                </span>
                <span style="font-size:0.65rem; color:#64748B; font-family:'JetBrains Mono',monospace;">SIH-26166</span>
            </div>
            <div style="font-family:'Inter',sans-serif; font-size:0.86rem; font-weight:700; color:#FFFFFF;">
                👨‍🚀 {callsign}
            </div>
            <div style="font-size:0.70rem; color:#94A3B8; margin-top:2px;">
                🛡️ {tier.split('—')[0].strip()}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_auth:
            if st.button("🚪 Logout / Re-enter Gateway", key="btn_logout_sidebar", width="stretch"):
                logout()
                st.rerun()

        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.08);">
            <span style="font-size:1.1rem;">⚙️</span>
            <span style="font-family:'Inter',sans-serif; font-weight:600; font-size:0.92rem; color:#E8ECF5;">Pipeline Parameters</span>
        </div>
        """, unsafe_allow_html=True)

        use_tps = st.toggle(
            "Pushbroom jitter correction (TPS)",
            value=True,
            help="Applies non-rigid Thin-Plate Spline surface transformation to model spacecraft attitude drift"
        )

        max_dim = st.slider(
            "Max processing dimension (px)",
            min_value=512, max_value=4096, value=2048, step=256,
            help="Higher resolution yields more tie-points across fine crater boundaries"
        )

        with st.expander("Physical optical models", expanded=False):
            enable_subpixel = st.checkbox("Sub-pixel parabolic peak refinement", value=True, help="Quadratic sub-pixel peak interpolation for sub-0.05px tie-point precision")
            enable_pyramid = st.checkbox("Multi-scale pyramid matching (0.5x, 1.0x, 1.5x)", value=True, help="Extracts tie-points hierarchically across both macro-crater rims and micro-regolith textures")
            enable_photo = st.checkbox("Hapke BRDF illumination correction", value=False, help="Normalizes severe sun-illumination and shadow angles")

        st.markdown("---")

        # Linked Backend REST API Status
        try:
            req = urllib.request.Request(f"{BACKEND_API_URL}/api/v1/health", headers={"User-Agent": "ISRO-UI"})
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                backend_badge = '<span style="color:#34D399; font-weight:600; font-family:\'JetBrains Mono\',monospace;">Online</span>' if resp.status == 200 else f'<span style="color:#FBBF24; font-weight:600; font-family:\'JetBrains Mono\',monospace;">HTTP {resp.status}</span>'
        except Exception:
            backend_badge = '<span style="color:#94A3B8; font-weight:600; font-family:\'JetBrains Mono\',monospace;">Local Embedded</span>'

        st.markdown(f"""
        <div style="background:rgba(14,23,48,0.50); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px 14px; font-size:0.78rem;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8; font-weight:500;">REST Service</span>
                <span>{backend_badge}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:#94A3B8; font-weight:500;">Mission Library</span>
                <span style="color:#38BDF8; font-weight:600; font-family:'JetBrains Mono',monospace;">{dataset_count} Datasets</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94A3B8; font-weight:500;">Acceleration</span>
                <span style="color:#A78BFA; font-weight:600; font-family:'JetBrains Mono',monospace;">PyTorch MPS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-top:20px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.08);">
            <div style="font-size:0.82rem; color:#38BDF8; font-weight:700; letter-spacing:0.04em;">CLARITY*</div>
            <div style="font-size:0.68rem; color:#94A3B8; margin-top:2px; line-height:1.35;">Chandrayaan Lunar AI Registration &amp; 3D Mapping</div>
            <div style="font-size:0.72rem; color:#E8ECF5; font-weight:600; margin-top:4px;">College Capstone Project</div>
            <div style="font-size:0.70rem; color:#CBD5E1; margin-top:2px;">Lunar Co-Registration &amp; 3D Mapping</div>
            <div style="font-size:0.66rem; color:#64748B; margin-top:2px;">Chandrayaan-1 / 2 Payload Analysis</div>
        </div>
        """, unsafe_allow_html=True)

    return {
        "use_tps": use_tps,
        "max_dim": max_dim,
        "enable_subpixel": enable_subpixel,
        "enable_pyramid": enable_pyramid,
        "enable_photo": enable_photo,
    }
