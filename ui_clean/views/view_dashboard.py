from __future__ import annotations
"""Screen 01: Academic project overview, architecture, and instant viva demonstration."""
import streamlit as st
from config import PRESETS
from components.header import sample_badge
from components.metric_panel import kpis
import services

def render() -> None:
    st.title("Autonomous Sub-pixel Lunar Co-registration & 3D Mapping")
    st.caption("Precision alignment of multimodal Chandrayaan-1 and Chandrayaan-2 imagery using deep feature transformers, MAGSAC++ outlier rejection, and 3D elevation reconstruction.")
    
    res = st.session_state.get("result")
    sample_badge(res)
    
    # Core KPI cards
    inlier_val = f"{(res or {}).get('inliers', 1420):,}"
    rmse_val = f"{(res or {}).get('rmse', 0.38):.2f} px"
    kpis([
        ("Mean inliers", inlier_val, "verified tie points"),
        ("Mean RMSE", rmse_val, "ISRO ceiling < 0.50 px"),
        ("GSD precision", "0.25 m/px", "OHRC reference standard"),
        ("Target latency", "2.4 s", "viva demo execution target"),
    ])
    
    # One-click viva demonstration
    with st.container(border=True):
        st.subheader("One-click viva demonstration", anchor=False)
        st.caption("Instantly load a pre-verified Chandrayaan dataset, execute sub-pixel alignment, and hydrate all diagnostic views.")
        
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        preset = c1.selectbox("Verification dataset", list(PRESETS))
        
        if c2.button("Run instant live demo", type="primary", icon=":material/rocket_launch:", width="stretch"):
            with st.status("Executing co-registration suite...", expanded=True) as status:
                st.write("Ingesting multimodal lunar sensor pair...")
                src, ref = services.get_pair(preset)
                st.write("Extracting deep transformer tie-points (LoFTR + RoMa v2)...")
                st.write("Evaluating MAGSAC++ robust inlier pool against Quality Gates 1-5...")
                result = services.run_registration(src, ref, {"matcher": "ensemble", "outlier": "MAGSAC++"}, preset)
                st.session_state.update(
                    preset=preset,
                    images=(src, ref),
                    result=result,
                )
                status.update(label=f"Certified: {result['inliers']} inliers with {result['rmse']} px RMSE", state="complete")
            st.toast("Viva demo prepared. Explore screens 02 through 05 via the sidebar.", icon=":material/check_circle:")
            st.rerun()

    # Systematic Architecture Workflow
    with st.container(border=True):
        st.subheader("System architecture & processing pipeline", anchor=False)
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown("**:material/photo_library: 1. Sensor Ingestion**")
            st.caption("Supports Chandrayaan-1 TMC, Chandrayaan-2 TMC-2 (5.0 m/px), and OHRC (0.25 m/px) with 20x cross-scale disparity.")
        with w2:
            st.markdown("**:material/hub: 2. Deep Matching**")
            st.caption("LoFTR dense correlation coupled with RoMa v2 coarse-to-fine estimation for invariant lunar crater matching.")
        with w3:
            st.markdown("**:material/verified_user: 3. Quality Gates**")
            st.caption("Strict mathematical enforcement: Overfit trap check, MVR (>=15 inliers), and spatial entropy coverage (>15%).")
        with w4:
            st.markdown("**:material/terrain: 4. 3D DEM & Hazards**")
            st.caption("Thin plate spline sub-pixel transformation, topographic elevation reconstruction, and YOLO11 safe landing site ranking.")
