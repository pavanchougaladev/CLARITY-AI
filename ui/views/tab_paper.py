"""
Research Benchmark & Paper Replication View
===========================================
ISRO LunarFusion Co-Registration Suite — UI Benchmark View
"""

import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List

from scripts.paper_benchmark_replication import get_paper_reference_dataframe


@st.cache_data
def get_cached_paper_reference_dataframe():
    """Cached accessor for research paper Table 3 baseline records."""
    return get_paper_reference_dataframe()


def render_paper_tab() -> None:
    """Render the Research Benchmark & Paper Replication view."""
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(11,22,40,0.98)); border: 1px solid rgba(56,189,248,0.3); border-radius: 14px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
        <div>
            <span class="telemetry-pill" style="background:rgba(56,189,248,0.15); border-color:rgba(56,189,248,0.4); color:#38BDF8; font-weight:700;">
                ISRO / SAC RESEARCH REPLICATION SUITE • CHANDRAYAAN-2
            </span>
            <h2 style="font-family:'Inter',sans-serif; font-size:1.5rem; font-weight:700; color:#FFFFFF; margin:8px 0 4px 0; letter-spacing:-0.015em;">
                📊 Comparative Analysis of Lunar Image Registration Algorithms
            </h2>
            <div style="font-size:0.85rem; color:#94A3B8;">
                Empirical verification of <b>SuperGlue, RIFT2, ASIFT, AKAZE, and SIFT</b> across multi-sensor lunar datasets: <b>OHRC ↔ NAC, IIRS ↔ WAC, DFSAR ↔ SELENE</b>.
            </div>
        </div>
        <div style="text-align:right;">
            <span style="font-size:0.75rem; color:#A0C0E0; font-family:'JetBrains Mono',monospace;">ISRO BENCHMARK SPEC</span><br>
            <span style="font-family:'Inter',sans-serif; font-size:1.05rem; font-weight:700; color:#38BDF8;">Table 3 Certified</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Executive Summary KPI Cards ──
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🥇 Best Overall Algorithm</div>
        <div class="metric-value" style="color:#06D6A0;">SuperGlue</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Total RMSE <b>0.847 px</b> (3.81s)</div>
    </div>
    """, unsafe_allow_html=True)
with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🛡️ Polar & SAR Reliability</div>
        <div class="metric-value" style="color:#38BDF8;">100%</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Only SuperGlue succeeded in all 6/6 tests</div>
    </div>
    """, unsafe_allow_html=True)
with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">🥈 Cross-Modal Runner-Up</div>
        <div class="metric-value" style="color:#A78BFA;">RIFT2</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">Phase Congruency + MIM descriptor</div>
    </div>
    """, unsafe_allow_html=True)
with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">⚠️ Classical Matcher Trap</div>
        <div class="metric-value" style="color:#EF4444;">0 Matches</div>
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:4px;">SIFT, ASIFT, AKAZE failed on Polar & SAR</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ── Sub-tabs for Paper Results, Live Replication, and Preprocessing Architecture ──
p_tab1, p_tab2, p_tab3 = st.tabs([
    "📋 Official Research Table 3 Data",
    "⚡ Live Algorithm Benchmark on Active Scene",
    "⚙️ Preprocessing Pipelines (A, B, C Architecture)"
])

with p_tab1:
    st.markdown("##### 🛰️ Chandrayaan-2 Empirical Registration Benchmark (Table 3)")
    st.caption("Quantitative telemetry recorded across 6 planetary remote sensing pairs in equatorial and polar environments.")

    df_bench = get_cached_paper_reference_dataframe()

    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        sensor_filter = st.selectbox(
            "Filter Sensor Modality Pair:",
            ["All Sensor Pairs", "OHRC-NAC (Optical)", "IIRS-WAC (Hyperspectral)", "DFSAR-SELENE (SAR-Optical)"],
            index=0
        )
    with f_col2:
        region_filter = st.selectbox(
            "Filter Lunar Region:",
            ["All Regions", "Equatorial", "Polar"],
            index=0
        )
    with f_col3:
        algo_filter = st.multiselect(
            "Filter Algorithms:",
            options=["SuperGlue", "RIFT2", "ASIFT", "AKAZE", "SIFT"],
            default=["SuperGlue", "RIFT2", "ASIFT", "AKAZE", "SIFT"]
        )

    # Apply filtering
    filtered_df = df_bench.copy()
    if "OHRC" in sensor_filter:
        filtered_df = filtered_df[filtered_df["sensor_pair"] == "OHRC-NAC"]
    elif "IIRS" in sensor_filter:
        filtered_df = filtered_df[filtered_df["sensor_pair"] == "IIRS-WAC"]
    elif "DFSAR" in sensor_filter:
        filtered_df = filtered_df[filtered_df["sensor_pair"] == "DFSAR-SELENE"]

    if region_filter != "All Regions":
        filtered_df = filtered_df[filtered_df["region"] == region_filter]

    if algo_filter:
        filtered_df = filtered_df[filtered_df["algorithm"].isin(algo_filter)]

    # Render styled dataframe
    def _highlight_superglue(row):
        if row["algorithm"] == "SuperGlue":
            return ['background-color: rgba(6, 214, 160, 0.15); color: #06D6A0; font-weight: 700;'] * len(row)
        elif row["status"] == "Failed":
            return ['background-color: rgba(239, 68, 68, 0.12); color: #FCA5A5;'] * len(row)
        return [''] * len(row)

    st.dataframe(
        filtered_df.style.apply(_highlight_superglue, axis=1),
        column_config={
            "sensor_pair": st.column_config.TextColumn("Sensor Pair", width="medium"),
            "region": st.column_config.TextColumn("Region", width="small"),
            "algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
            "rmse_x": st.column_config.NumberColumn("RMSE_X (px)", format="%.4f"),
            "rmse_y": st.column_config.NumberColumn("RMSE_Y (px)", format="%.4f"),
            "total_rmse": st.column_config.NumberColumn("Total RMSE (px)", format="%.4f"),
            "time_s": st.column_config.NumberColumn("Runtime (s)", format="%.2f"),
            "matches": st.column_config.NumberColumn("Inlier Matches"),
            "status": st.column_config.TextColumn("Quality Gate Status")
        },
        hide_index=True,
        width="stretch"
    )

    st.info(
        "📌 **Key Paper Insights from Table 3:**\n"
        "- **SuperGlue Dominance:** Outperforms all algorithms in all modalities and regions, achieving sub-pixel accuracy (0.847 px on OHRC-NAC) in under 4 seconds.\n"
        "- **Classical Polar Failure:** SIFT, ASIFT, AKAZE, and RIFT2 all yield **0 matches** in polar regions due to severe illumination gradients and lack of gradient constancy.\n"
        "- **SAR Cross-Modal Physics:** Only SuperGlue (0.91 px RMSE) and RIFT2 (2.10 px RMSE) succeed on DFSAR ↔ SELENE. SIFT and AKAZE fail completely due to multiplicative speckle noise."
    )

with p_tab2:
    st.markdown("##### ⚡ Live Algorithm Benchmark on Current Lunar Pair")
    st.caption("Executes the research paper's 4 primary algorithms in real-time on your currently selected lunar scene, reporting exact RMSE, inlier counts, and Gate 1-5 validation status.")

    active_src_arr = st.session_state.get('active_img_src')
    active_ref_arr = st.session_state.get('active_img_ref')

    if active_src_arr is None or active_ref_arr is None:
        # Fallback to safe preview if not registered yet
        if src_path_final and ref_path_final:
            from src.pipeline import robust_read, smart_downscale
            try:
                s_r = robust_read(src_path_final, max_dim=768)
                r_r = robust_read(ref_path_final, max_dim=768)
                active_src_arr, _ = smart_downscale(s_r, max_dim=768)
                active_ref_arr, _ = smart_downscale(r_r, max_dim=768)
            except Exception:
                pass

    if active_src_arr is not None and active_ref_arr is not None:
        b_col1, b_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with b_col1:
            benchmark_algos = st.multiselect(
                "Select Algorithms to Benchmark:",
                options=["superglue", "rift2", "akaze", "sift"],
                default=["superglue", "rift2", "akaze", "sift"],
                format_func=lambda x: {
                    "superglue": "🌟 SuperGlue (Graph Neural Network)",
                    "rift2": "⚡ RIFT2 (Phase Congruency)",
                    "akaze": "🔹 AKAZE (Nonlinear Scale Space)",
                    "sift": "🟢 SIFT (Scale-Invariant Feature Transform)"
                }.get(x, x.upper())
            )
        with b_col2:
            run_live_benchmark = st.button("🚀 Run Live Benchmark", type="primary", width="stretch")

        if run_live_benchmark and benchmark_algos:
            from scripts.paper_benchmark_replication import run_benchmark_on_images
            with st.spinner("⏳ Running comparative benchmark across selected algorithms..."):
                df_live = run_benchmark_on_images(active_src_arr, active_ref_arr, algorithms=benchmark_algos)
                st.session_state["live_paper_benchmark_results"] = df_live

        if "live_paper_benchmark_results" in st.session_state:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("###### 📊 Benchmark Evaluation Telemetry:")
            res_df = st.session_state["live_paper_benchmark_results"]
            st.dataframe(
                res_df,
                column_config={
                    "algorithm": st.column_config.TextColumn("Algorithm"),
                    "matches": st.column_config.NumberColumn("Verified Inliers"),
                    "rmse_x": st.column_config.NumberColumn("RMSE_X (px)", format="%.4f"),
                    "rmse_y": st.column_config.NumberColumn("RMSE_Y (px)", format="%.4f"),
                    "total_rmse": st.column_config.NumberColumn("Total RMSE (px)", format="%.4f"),
                    "time_s": st.column_config.NumberColumn("Time (s)", format="%.2f"),
                    "status": st.column_config.TextColumn("Quality Gate Status"),
                },
                hide_index=True,
                width="stretch"
            )
    else:
        st.info("💡 Load or select a pair of lunar scenes in Tab 1 to run the live comparative benchmark.")

with p_tab3:
    st.markdown("##### ⚙️ Chandrayaan-2 Preprocessing Pipelines Architecture")
    st.caption("Detailed architectural specifications of the three specialized radiometric and morphological pipelines defined in the research paper.")

    pipe_a_col, pipe_b_col, pipe_c_col = st.columns(3)
    with pipe_a_col:
        st.markdown("""
        <div style="background:rgba(56,189,248,0.06); border:1px solid rgba(56,189,248,0.25); border-radius:12px; padding:16px; height:100%;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:1.3rem;">🛰️</span>
                <strong style="color:#38BDF8; font-size:1.0rem;">Pipeline A</strong>
            </div>
            <div style="font-size:0.78rem; font-weight:700; color:#E2E8F0; margin-bottom:6px;">OHRC ↔ NAC (Optical High-Res)</div>
            <p style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
                Designed for sub-meter panchromatic optical sensors (0.25m OHRC and 0.5m LROC NAC).
            </p>
            <ol style="font-size:0.75rem; color:#CBD5E1; padding-left:18px; margin-top:8px; line-height:1.6;">
                <li><b>8-Bit Normalization:</b> Rescales 16-bit PDS4 DN values.</li>
                <li><b>CLAHE:</b> Local contrast expansion in maria/highlands.</li>
                <li><b>Image Inversion:</b> Converts crater floors into positive highlights.</li>
                <li><b>Morphological Dilation:</b> Bridges broken crater rims.</li>
                <li><b>PCA Enhancement:</b> Maximizes topographical variance.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with pipe_b_col:
        st.markdown("""
        <div style="background:rgba(167,139,250,0.06); border:1px solid rgba(167,139,250,0.25); border-radius:12px; padding:16px; height:100%;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:1.3rem;">🌈</span>
                <strong style="color:#A78BFA; font-size:1.0rem;">Pipeline B</strong>
            </div>
            <div style="font-size:0.78rem; font-weight:700; color:#E2E8F0; margin-bottom:6px;">IIRS ↔ WAC (Hyperspectral)</div>
            <p style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
                Cross-modal alignment between 250-band IIRS infrared spectra and wide-angle optical albedo.
            </p>
            <ol style="font-size:0.75rem; color:#CBD5E1; padding-left:18px; margin-top:8px; line-height:1.6;">
                <li><b>Band Selection:</b> Isolates high-SNR absorption bands.</li>
                <li><b>Histogram Matching:</b> Synchronizes empirical CDFs across sensors.</li>
                <li><b>Shadow Normalization:</b> Multi-scale Wallis filter eliminates solar bias.</li>
                <li><b>Logarithmic Transform:</b> Expands low-light dynamic range.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with pipe_c_col:
        st.markdown("""
        <div style="background:rgba(6,214,160,0.06); border:1px solid rgba(6,214,160,0.25); border-radius:12px; padding:16px; height:100%;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-size:1.3rem;">📡</span>
                <strong style="color:#06D6A0; font-size:1.0rem;">Pipeline C</strong>
            </div>
            <div style="font-size:0.78rem; font-weight:700; color:#E2E8F0; margin-bottom:6px;">DFSAR ↔ SELENE (SAR-to-Optical)</div>
            <p style="font-size:0.75rem; color:#94A3B8; line-height:1.4;">
                Bridges active L-band/S-band radar backscatter and passive optical photogrammetry.
            </p>
            <ol style="font-size:0.75rem; color:#CBD5E1; padding-left:18px; margin-top:8px; line-height:1.6;">
                <li><b>Enhanced Lee Filter:</b> Multiplicative speckle noise suppression (ENL=4).</li>
                <li><b>Log-Ratio Normalization:</b> Compresses extreme dB radar spikes.</li>
                <li><b>Phase Congruency:</b> Illumination- and amplitude-invariant edge geometry.</li>
                <li><b>GNN / MIM Feature Pairing:</b> SuperGlue / RIFT2 structural matching.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; font-size: 0.82rem; color: #CBD5E1; flex-wrap: wrap; gap: 8px;">
<div>
    <strong style="color:#38BDF8; font-size:0.9rem;">CLARITY*</strong> — Chandrayaan Lunar AI Registration &amp; Image Transformation system and 3D lunar mapping | ISRO SAC — SIH26166  |  LoFTR • RoMa v2 • LightGlue • MAGSAC++ • NASA ASP
</div>
<div>
    <b>Academic Project:</b> <span style="color: #F8FAFC; font-weight: 600;">Autonomous Planetary Mapping Suite</span>
</div>
</div>
""", unsafe_allow_html=True)
