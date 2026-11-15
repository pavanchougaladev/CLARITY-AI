"""
YOLO11 AI Lunar Landmark & Hazard Intelligence View
===================================================
ISRO LunarFusion Co-Registration Suite — UI Crater & Hazard AI View
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List
from PIL import Image
import glob

try:
    from ui.config import DATA_DIR
    from ui.services.image_loader import get_cached_crater_detector
except (ImportError, ValueError):
    from ..config import DATA_DIR
    from ..services.image_loader import get_cached_crater_detector

from src.pipeline import robust_read


def render_yolo_tab(available_images: list = None) -> None:
    """Render the YOLO11 AI Crater & Hazard detection interface."""
    st.markdown("""
    <div style="background: rgba(16, 26, 54, 0.65); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 14px; padding: 18px 22px; margin-bottom: 20px;
                backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); box-shadow: 0 8px 32px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.12);">
        <div style="display: flex; align-items: center; gap: 14px;">
            <div style="font-size: 2.2rem; filter: drop-shadow(0 0 12px rgba(56, 189, 248, 0.4));">🌔</div>
            <div>
                <h2 style="margin: 0; font-size: 1.3rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.015em;">
                    YOLO11 Lunar Landmark &amp; Hazard Intelligence Suite
                </h2>
                <p style="margin: 4px 0 0 0; font-size: 0.85rem; color: #CBD5E1;">
                    Sub-20ms Crater Detection, Morphometric Characterization &amp; Physical Gate 3 Telemetry (Fine-Tuned on Chandrayaan-2 OHRC)
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Telemetry Badge Row
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown("""
        <div style="background: rgba(14, 23, 48, 0.50); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Engine Architecture</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">YOLO11 Nano (2.6M)</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div style="background: rgba(14, 23, 48, 0.50); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Inference Latency</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #34D399; font-family: 'JetBrains Mono', monospace;">~16.6 ms (MPS)</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div style="background: rgba(14, 23, 48, 0.50); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Validation mAP@50</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FBBF24; font-family: 'JetBrains Mono', monospace;">0.583 (58.3% mAP)</div>
        </div>
        """, unsafe_allow_html=True)
    with b4:
        st.markdown("""
        <div style="background: rgba(14, 23, 48, 0.50); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px;">
            <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em;">Gate 3 Authority</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #A78BFA; font-family: 'JetBrains Mono', monospace;">Physical Anchors</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Input Image Selection
    yolo_col_ctl1, yolo_col_ctl2, yolo_col_ctl3 = st.columns([2, 1, 1])
    with yolo_col_ctl1:
        # Find available lunar images across data/
        ch2_images = sorted(glob.glob("data/chandrayan-2 dataset/*_b_brw_*.png")) + sorted(glob.glob("data/lunar_crater_yolo/images/val/*.jpg"))
        sample_images = ch2_images if ch2_images else sorted(glob.glob("data/**/*.png", recursive=True))
        image_options = ["Custom Upload"] + [os.path.basename(p) for p in sample_images[:40]]
        selected_img_name = st.selectbox("Lunar Target Scene", image_options, index=1 if len(image_options) > 1 else 0)

    with yolo_col_ctl2:
        yolo_conf = st.slider("Detection Confidence", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    with yolo_col_ctl3:
        yolo_gsd = st.number_input("Pixel GSD (m/px)", min_value=0.05, max_value=10.0, value=0.25, step=0.05)

    input_yolo_img = None
    if selected_img_name == "Custom Upload":
        yolo_file = st.file_uploader("Upload Lunar Image (PNG / TIF / IMG)", type=["png", "jpg", "jpeg", "tif", "tiff", "img"])
        if yolo_file is not None:
            saved_yolo = _save_uploaded_orbital_image(yolo_file)
            raw = robust_read(saved_yolo, max_dim=1024)
            if raw is not None:
                input_yolo_img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw
    else:
        # Resolve full path from sample_images
        resolved_path = next((p for p in sample_images if os.path.basename(p) == selected_img_name), None)
        if resolved_path and os.path.exists(resolved_path):
            raw = robust_read(resolved_path, max_dim=1024)
            if raw is not None:
                input_yolo_img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 else raw

    if input_yolo_img is not None:
        # Crop or resize for fast interactive analysis if full image is huge
        ih, iw = input_yolo_img.shape[:2]
        if ih > 1024 or iw > 1024:
            st.caption(f"ℹ️ Scene resolution: {iw}x{ih} px. Extracting center 1024x1024 region for real-time analysis.")
            cy, cx = ih // 2, iw // 2
            input_yolo_img = input_yolo_img[max(0, cy-512):min(ih, cy+512), max(0, cx-512):min(iw, cx+512)]

        st.write("")
        c_y1, c_y2 = st.columns([3, 1], vertical_alignment="bottom")
        with c_y1:
            run_yolo = st.button("🚀 Detect Lunar Craters & Hazards (YOLO11)", type="primary", icon=":material/radar:", width="stretch", key="btn_run_yolo")
        with c_y2:
            clear_yolo = st.button("🔄 Reset Detections", key="btn_clear_yolo", width="stretch")
            if clear_yolo:
                st.session_state.pop('yolo_last_res', None)
                st.session_state.pop('yolo_last_img', None)
                st.rerun()

        # Run inference if button clicked or already cached for this image
        should_run = run_yolo or (st.session_state.get('yolo_last_img') == selected_img_name and 'yolo_last_res' in st.session_state)
        if should_run:
            try:
                detector = get_cached_crater_detector("models/yolo11_lunar_crater.pt")
                if detector is None:
                    st.error("YOLO11 Lunar Crater model could not be loaded.")
                else:
                    if run_yolo or 'yolo_last_res' not in st.session_state or st.session_state.get('yolo_last_img') != selected_img_name:
                        with st.spinner("Running YOLO11 sub-20ms landmark detection..."):
                            t_start = time.perf_counter()
                            detections = detector.detect(input_yolo_img, conf=yolo_conf)
                            latency_ms = (time.perf_counter() - t_start) * 1000
                            scaled_detections = detector.compute_crater_diameters_m(detections, gsd_m=yolo_gsd)
                            annotated_img = detector.draw_craters(input_yolo_img, scaled_detections)
                            st.session_state['yolo_last_res'] = {
                                'detections': detections,
                                'scaled_detections': scaled_detections,
                                'annotated_img': annotated_img,
                                'latency_ms': latency_ms,
                                'device': detector.device.upper()
                            }
                            st.session_state['yolo_last_img'] = selected_img_name

                    y_res = st.session_state.get('yolo_last_res')
                    if y_res:
                        detections = y_res['detections']
                        scaled_detections = y_res['scaled_detections']
                        annotated_img = y_res['annotated_img']
                        latency_ms = y_res['latency_ms']
                        device_name = y_res.get('device', 'MPS')

                        c_vis, c_stats = st.columns([3, 2])
                        with c_vis:
                            st.subheader("🎯 Crater & Hazard Map (YOLO11)")
                            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), width="stretch")
                            st.caption(f"⚡ Inference executed in {latency_ms:.1f} ms on {device_name} • {len(detections)} craters mapped.")

                        with c_stats:
                            st.subheader("📊 Terrain Morphometry")
                            m1, m2 = st.columns(2)
                            with m1:
                                st.metric("Craters Mapped", f"{len(scaled_detections)}")
                            with m2:
                                diameters = [d["diameter_m"] for d in scaled_detections]
                                avg_d = np.mean(diameters) if diameters else 0.0
                                st.metric("Avg Diameter", f"{avg_d:.1f} m")

                            m3, m4 = st.columns(2)
                            with m3:
                                max_d = np.max(diameters) if diameters else 0.0
                                st.metric("Largest Crater", f"{max_d:.1f} m")
                            with m4:
                                min_d = np.min(diameters) if diameters else 0.0
                                st.metric("Smallest Crater", f"{min_d:.1f} m")

                            if diameters:
                                fig_hist = go.Figure()
                                fig_hist.add_trace(go.Histogram(
                                    x=diameters,
                                    nbinsx=20,
                                    marker_color="#38BDF8",
                                    opacity=0.85
                                ))
                                fig_hist.update_layout(
                                    title="Crater Diameter Distribution (meters)",
                                    xaxis_title="Diameter (m)",
                                    yaxis_title="Count",
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    height=220,
                                    paper_bgcolor="rgba(11, 15, 25, 0)",
                                    plot_bgcolor="rgba(11, 15, 25, 0)",
                                    font=dict(color="#CBD5E1")
                                )
                                st.plotly_chart(fig_hist, width="stretch")

                            # Export CSV button
                            if scaled_detections:
                                import pandas as pd
                                csv_records = [
                                    {
                                        "id": i + 1,
                                        "centroid_x": d["centroid"][0],
                                        "centroid_y": d["centroid"][1],
                                        "diameter_px": d["diameter_px"],
                                        "diameter_m": d["diameter_m"],
                                        "confidence": d["confidence"]
                                    }
                                    for i, d in enumerate(scaled_detections)
                                ]
                                df_craters = pd.DataFrame(csv_records)
                                csv_data = df_craters.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Export Crater Catalog (CSV)",
                                    data=csv_data,
                                    file_name="lunar_crater_catalog_yolo11.csv",
                                    mime="text/csv",
                                    width="stretch"
                                )

            except Exception as e:
                st.error(f"Failed to run YOLO11 Crater Detection: {e}")
        else:
            # Preview before inference
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            c_prev1, c_prev2 = st.columns([3, 2])
            with c_prev1:
                st.subheader("🖼️ Selected Lunar Scene Preview")
                st.image(cv2.cvtColor(input_yolo_img, cv2.COLOR_BGR2RGB), caption=f"Target: {selected_img_name} ({input_yolo_img.shape[1]}x{input_yolo_img.shape[0]} px)", width="stretch")
            with c_prev2:
                st.subheader("💡 Landmark Detection Ready")
                st.info(
                    "Click **'🚀 Detect Lunar Craters & Hazards (YOLO11)'** above to trigger real-time AI landmark mapping.\n\n"
                    "- ⚡ **Inference Speed:** <20 ms on Apple Silicon MPS / CUDA\n"
                    "- 📐 **Physical Diameter Extraction:** Uses GSD (Ground Sampling Distance) to calculate crater diameters in meters\n"
                    "- 🛡️ **Gate 3 Authority:** Enforces crater coverage distribution to prevent homography clustering trap\n"
                    "- 📊 **Export Catalog:** Download detections as structured CSV with centroid coordinates and morphometry"
                )
    else:
        st.info("Select a lunar scene or upload an image to run the YOLO11 landmark detector.")

    # ── Tab Footer ─────────────────────────────────────────────────────────────
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

