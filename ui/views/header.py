"""
Header and Aerospace Authentication Gateway View
===============================================
ISRO LunarFusion Co-Registration Suite — UI Header View
Renders Aerospace Telemetry Header, Mission Authentication Console, and Pipeline Gateway.
"""

import os
import streamlit as st

try:
    from ui.services.state import login
except (ImportError, ValueError):
    from ..services.state import login


def render_cosmic_header() -> None:
    """Render aerospace stage background, mission gateway, and authentication banner."""
    cosmic_stage_path = os.path.join(os.path.dirname(__file__), "..", "static", "cosmic_stage.html")
    cosmic_stage_html = ""
    try:
        if os.path.exists(cosmic_stage_path):
            with open(cosmic_stage_path, "r", encoding="utf-8") as f:
                cosmic_stage_html = f.read()
    except Exception as err:
        cosmic_stage_html = f"<!-- Error loading cosmic_stage.html: {err} -->"

    if cosmic_stage_html:
        st.html(cosmic_stage_html)

    # Mission Control Direct Authentication Console
    if not st.session_state.get("authenticated", False):
        with st.container():
            st.markdown("""
            <div style="background:rgba(16,26,54,0.65); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:12px 18px; margin: 4px 0 16px 0; box-shadow:0 8px 32px -4px rgba(2,6,23,0.55), inset 0 1px 0 rgba(255,255,255,0.12);">
                <div style="font-size:0.75rem; font-weight:700; color:#38BDF8; letter-spacing:0.06em; display:flex; align-items:center; gap:8px;">
                    <span>🛡️</span> MISSION CONTROL DIRECT AUTHENTICATION CONSOLE
                </div>
            </div>
            """, unsafe_allow_html=True)
            c_mc1, c_mc2, c_mc3, c_mc4 = st.columns([1.6, 2.0, 1.4, 1.1])
            with c_mc1:
                st_officer = st.text_input("👨‍🚀 Mission Callsign", value=st.session_state.officer_callsign, key="st_officer_input")
            with c_mc2:
                st_tier = st.selectbox("🛡️ Clearance Role", [
                    "Level 4 — Flight Director & Sub-Pixel Lead",
                    "Level 3 — Photogrammetry Payload Specialist",
                    "Level 2 — Orbital DEM Research Analyst",
                    "Guest Scientist — Fast Pipeline Access"
                ], key="st_tier_select")
            with c_mc3:
                st.write("")
                st.write("")
                if st.button("Authenticate & Enter", type="primary", key="st_btn_auth_enter", width="stretch"):
                    login(st_officer, st_tier)
                    st.query_params["auth"] = "true"
                    st.query_params["officer"] = st_officer
                    st.query_params["tier"] = st_tier
                    st.rerun()
            with c_mc4:
                st.write("")
                st.write("")
                if st.button("Direct Pipeline ↓", key="st_btn_direct_pipe", width="stretch"):
                    st.session_state.authenticated = True
                    st.query_params["auth"] = "true"
                    st.rerun()

    # Target Anchor for Smooth Scroll Transition to Pipeline
    callsign = st.session_state.get("officer_callsign", "MISSION-SCIENTIST")
    tier = st.session_state.get("clearance_tier", "Level 4")
    st.html(f"""
    <div id="pipeline-registration-section" style="padding-top: 6px; margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(14,23,48,0.65); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:12px 18px; margin-bottom:14px; box-shadow:0 8px 32px -4px rgba(2,6,23,0.55), inset 0 1px 0 rgba(255,255,255,0.12);">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:1.3rem;">🛰️</span>
                <div>
                    <div style="font-size:0.92rem; font-weight:700; color:#FFFFFF; letter-spacing:0.02em; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <span style="color:#38BDF8; font-weight:800;">CLARITY*</span>
                        <span style="color:rgba(255,255,255,0.25);">|</span>
                        <span>CHANDRAYAAN LUNAR AI REGISTRATION &amp; 3D MAPPING</span>
                        <span style="background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.30); border-radius:6px; font-size:0.68rem; padding:2px 8px; color:#34D399; font-weight:600; font-family:'JetBrains Mono',monospace;">AUTH: {callsign}</span>
                    </div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-top:2px;">Planetary Geodesy &amp; Photogrammetry Suite · Clearance: {tier}</div>
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
                <span style="display:inline-flex; align-items:center; gap:6px; background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.30); border-radius:20px; padding:4px 12px; font-size:0.72rem; font-weight:600; color:#34D399; font-family:'JetBrains Mono',monospace;">
                    <span style="width:6px; height:6px; border-radius:50%; background:#34D399; box-shadow:0 0 8px #34D399; display:inline-block;"></span> System Ready
                </span>
                <button onclick="window.scrollToCosmicGateway ? window.scrollToCosmicGateway() : (document.getElementById('isro-cosmic-gateway') && document.getElementById('isro-cosmic-gateway').scrollIntoView({{behavior:'smooth'}}))" class="floating-hud-btn" style="position:static; padding:5px 12px; font-size:0.74rem;">
                    Mission Top ↑
                </button>
            </div>
        </div>
    </div>

    <!-- Floating Navigation HUD for fast jump -->
    <div class="floating-cosmic-nav">
        <button onclick="window.scrollToPipelineRegistration ? window.scrollToPipelineRegistration() : (document.getElementById('pipeline-registration-section') && document.getElementById('pipeline-registration-section').scrollIntoView({{behavior:'smooth'}}))" class="floating-hud-btn" title="Jump to CLARITY* Pipeline">
            🛰️ Pipeline ↓
        </button>
        <button onclick="window.scrollToCosmicGateway ? window.scrollToCosmicGateway() : (document.getElementById('isro-cosmic-gateway') && document.getElementById('isro-cosmic-gateway').scrollIntoView({{behavior:'smooth'}}))" class="floating-hud-btn" title="Return to Mission Top">
            Mission Top ↑
        </button>
    </div>
    """)
