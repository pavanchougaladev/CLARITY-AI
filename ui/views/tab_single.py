"""
Single Scene Co-Registration & Results Inspector View
=====================================================
ISRO LunarFusion Co-Registration Suite — UI Single Scene View
"""

import os
import sys
import time
import tempfile
import hashlib
import glob
import json
import traceback
import requests
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from typing import Optional, Any, Dict, List, Tuple

try:
    from ui.config import DATA_DIR, WORKSPACE_ROOT, BACKEND_API_URL, DEFAULT_TIMEOUT_SECONDS
    from ui.services.api_client import (
        submit_registration_to_backend,
        format_backend_results,
        is_backend_online,
        fetch_backend_job,
        list_backend_jobs,
        cancel_backend_job,
        delete_backend_job,
        fetch_backend_artifact,
        _format_backend_results,
        _submit_registration_to_backend,
        _fetch_backend_job,
        _is_backend_online,
        _list_backend_jobs,
        _cancel_backend_job,
        _delete_backend_job,
        _fetch_backend_artifact,
    )
    from ui.services.image_loader import (
        get_safe_preview,
        find_archive_metadata,
        metadata_for_archive_pair,
        save_uploaded_orbital_image,
        save_uploaded_metadata,
        sha256_file,
        _find_archive_metadata,
        _metadata_for_archive_pair,
        _save_uploaded_orbital_image,
        _save_uploaded_metadata,
        _sha256_file,
    )
    from ui.components.visualizations import (
        create_checkerboard,
        compute_seam_alignment_continuity,
        create_hd_match_canvas,
        interactive_match_plot,
        normalize_to_rgb8,
        generate_dem_preview,
    )
except (ImportError, ValueError):
    from ..config import DATA_DIR, WORKSPACE_ROOT, BACKEND_API_URL, DEFAULT_TIMEOUT_SECONDS
    from ..services.api_client import (
        submit_registration_to_backend,
        format_backend_results,
        is_backend_online,
        fetch_backend_job,
        list_backend_jobs,
        cancel_backend_job,
        delete_backend_job,
        fetch_backend_artifact,
        _format_backend_results,
        _submit_registration_to_backend,
        _fetch_backend_job,
        _is_backend_online,
        _list_backend_jobs,
        _cancel_backend_job,
        _delete_backend_job,
        _fetch_backend_artifact,
    )
    from ..services.image_loader import (
        get_safe_preview,
        find_archive_metadata,
        metadata_for_archive_pair,
        save_uploaded_orbital_image,
        save_uploaded_metadata,
        sha256_file,
        _find_archive_metadata,
        _metadata_for_archive_pair,
        _save_uploaded_orbital_image,
        _save_uploaded_metadata,
        _sha256_file,
    )
    from ..components.visualizations import (
        create_checkerboard,
        compute_seam_alignment_continuity,
        create_hd_match_canvas,
        interactive_match_plot,
        normalize_to_rgb8,
        generate_dem_preview,
    )
from src.pipeline import run_registration_pipeline, robust_read
from src.registration.export import (
    export_asp_compatible_tiepoints,
    export_geotiff,
    export_asp_match_file,
)
from src.reporting.pdf_report import generate_pdf_report


def render_registration_results(results: dict, img_src: np.ndarray, img_ref: np.ndarray, src_path: str, ref_path: str):
    """Render scientific telemetry, visual inspection tabs, and export tools."""
    # ── Step 3: Scientific Telemetry & Quality Assessment ──
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">03</span>
            <div>
                <div class="step-title">Scientific Telemetry &amp; Quality Assessment</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Empirical validation metrics across heterogeneous crater distributions.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Matcher Engine", results.get('method_used', 'Unknown'))
        m2.metric("Putative Matches", f"{results.get('total_matches', 0):,}")
        m3.metric("Verified Inliers", f"{results.get('inlier_count', 0):,}", f"{results.get('inlier_ratio', 0):.1%} retained")
        m4.metric("Spatial Coverage", f"{results.get('coverage', 0):.1%}")
        rmse_val = results.get('rmse_pixels')
        w_rmse_val = results.get('weighted_rmse')
        if rmse_val is not None:
            rmse_display = f"{rmse_val:.3f} px"
            w_rmse_display = f"Weighted: {w_rmse_val:.3f} px" if w_rmse_val is not None else ""
        else:
            rmse_display = "Rejected"
            w_rmse_display = "Overfit / Floor Gate"
        m5.metric("Sub-Pixel RMSE", rmse_display, w_rmse_display)

        if results.get('mission_grade_certified'):
            st.success(
                f"Mission-grade certification: geographic overlap verified with "
                f"{results.get('gt_point_count', 0)} independent control points "
                f"(GT-RMSE {float(results.get('gt_rmse_pixels') or 0.0):.3f} px)."
            )
        elif results.get('metadata_validation', {}).get('status') not in (None, 'not_requested', 'not_available'):
            st.warning(
                f"Geographic validation: {results.get('metadata_validation', {}).get('status', 'unknown')}. "
                "This run is not mission-grade certified."
            )

        gt_rmse_val = results.get('gt_rmse_pixels')
        gt_status = results.get('gt_validation_status', 'not_provided')
        if gt_status == 'valid' and gt_rmse_val is not None:
            st.metric("Independent GT-RMSE", f"{gt_rmse_val:.3f} px", f"{results.get('gt_point_count', 0)} control points")
        else:
            st.caption(f"Independent GT-RMSE: {gt_status.replace('_', ' ')}")

        omniroute_stats = results.get('omniroute_stats')
        if omniroute_stats:
            strat_label = {
                "consensus": "🚀 Consensus Boost (Parallel ThreadPool)",
                "auto_fallback": "⚡ Auto-Fallback Cascade (Zero Downtime)",
                "fastest_wins": "🏁 Fastest-Wins Race Mode",
                "adaptive_profile": "🧭 Adaptive Terrain Diagnosis",
            }.get(omniroute_stats.get("strategy", "").lower(), omniroute_stats.get("strategy", "").upper())

            models_succeeded = omniroute_stats.get("models_succeeded", [])
            total_lat = omniroute_stats.get("latency_ms", 0.0)
            per_model_lat = omniroute_stats.get("per_model_latency", {})
            per_model_cnt = omniroute_stats.get("per_model_counts", {})
            reason = omniroute_stats.get("reason", "")
            profile_tag = omniroute_stats.get("profile_tag", "normal") or "normal"

            st.markdown(f"""
            <div class="lunar-fusion-card" style="border-color: rgba(56, 189, 248, 0.4); background: linear-gradient(135deg, rgba(8, 20, 44, 0.9) 0%, rgba(14, 34, 68, 0.95) 100%);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                        <span style="font-size:1.4rem;">⚡</span>
                        <div>
                            <div style="font-family:'Inter',sans-serif; font-weight:700; color:#38BDF8; font-size:1.02rem;">
                                OmniRoute Multi-AI Gateway Telemetry
                            </div>
                            <div style="font-size:0.78rem; color:#CBD5E1;">
                                Concurrent ThreadPool Dispatch · Zero-Downtime Cascade · Seamless Fault Isolation
                            </div>
                        </div>
                    </div>
                    <span style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.4); color:#7DD3FC; font-weight:600; font-size:0.8rem; padding:4px 14px; border-radius:20px;">
                        {strat_label}
                    </span>
                </div>
                <div style="font-size:0.85rem; color:#E2E8F0; margin-bottom:12px; line-height:1.4;">
                    <strong>Routing Outcome:</strong> {reason}
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-bottom:12px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Gateway Latency</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#38BDF8; font-family:'JetBrains Mono',monospace;">{total_lat:.1f} ms</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Active Engines</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#34D399; font-family:'JetBrains Mono',monospace;">{len(models_succeeded)} Online</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Fault Isolation</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#A78BFA; font-family:'JetBrains Mono',monospace;">Zero Downtime</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Terrain Tag</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#FBBF24; font-family:'JetBrains Mono',monospace;">{profile_tag.upper()}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if per_model_lat:
                cols = st.columns(len(per_model_lat))
                label_map = {
                    "sift": "RootSIFT (Two-Tier)",
                    "rootsift": "RootSIFT (Two-Tier)",
                    "roma_v2": "RoMa v2",
                    "roma_omega": "RoMa v2",
                    "roma": "RoMa v2",
                    "loftr": "LoFTR",
                    "yolo11": "YOLO11 (Crater)",
                    "lightglue": "DISK+LightGlue",
                    "ni_lfm": "NI-LFM",
                    "orb": "ORB"
                }
                for c, (m, lat) in zip(cols, per_model_lat.items()):
                    m_label = label_map.get(m, m.upper())
                    n_matches = per_model_cnt.get(m, 0)
                    is_ok = m in models_succeeded
                    status_badge = "✅" if is_ok else "⚡"
                    c.metric(f"{status_badge} {m_label}", f"{n_matches:,} matches", f"{lat:.1f} ms")

        fusion_stats = results.get('fusion_stats')
        if fusion_stats:
            tag_emoji = {
                "crater": "🌋 Crater Rim Dominance",
                "shadow": "🌑 Shadow Regolith Zone",
                "low_texture": "🌊 Low-Texture Plains",
                "cross_sensor": "🛰️ Cross-Sensor Divergence",
                "small": "🔍 High-Res Micro-Patch",
                "normal": "🌔 Standard Lunar Surface",
            }.get(fusion_stats.get("profile_tag"), "🌔 Lunar Surface")

            st.markdown(f"""
            <div class="lunar-fusion-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span style="font-size:1.3rem;">🌙</span>
                        <div>
                            <div style="font-family:'Inter',sans-serif; font-weight:700; color:#38BDF8; font-size:0.98rem; display:flex; align-items:center; gap:8px;">
                                LunarFusion Adaptive AI Intelligence
                                <span style="background:rgba(56,189,248,0.15); border:1px solid rgba(56,189,248,0.4); color:#38BDF8; font-size:0.68rem; font-weight:700; padding:2px 8px; border-radius:12px;">⚡ Simultaneous ThreadPool</span>
                            </div>
                            <div style="font-size:0.78rem; color:#CBD5E1;">
                                Concurrent CPU/GPU multi-architecture routing · RootSIFT baseline · 10K MAGSAC++
                            </div>
                        </div>
                    </div>
                    <span style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.35); color:#38BDF8; font-weight:600; font-size:0.78rem; padding:4px 12px; border-radius:20px; font-family:'JetBrains Mono',monospace;">
                        {tag_emoji}
                    </span>
                </div>
                <div style="font-size:0.84rem; color:#E2E8F0; margin-bottom:12px; line-height:1.4;">
                    <strong>AI Decision:</strong> {fusion_stats.get('reason', '')}
                </div>
                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap:10px;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Texture Index</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#38BDF8; font-family:'JetBrains Mono',monospace;">{fusion_stats.get('texture_avg', 0):.3f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Shadow Fraction</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#FBBF24; font-family:'JetBrains Mono',monospace;">{fusion_stats.get('shadow_fraction', 0):.1%}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Appearance Gap</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#A78BFA; font-family:'JetBrains Mono',monospace;">{fusion_stats.get('hist_distance', 0):.3f}</div>
                    </div>
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:8px 12px;">
                        <div style="font-size:0.7rem; color:#CBD5E1; text-transform:uppercase; font-weight:600;">Unified Inliers</div>
                        <div style="font-size:0.95rem; font-weight:700; color:#34D399; font-family:'JetBrains Mono',monospace;">{fusion_stats.get('total_fused', 0):,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            models_used = fusion_stats.get("models_used", [])
            weights = fusion_stats.get("weights", {})
            counts = fusion_stats.get("per_model_counts", {})
            if models_used:
                cols = st.columns(len(models_used))
                engine_label_map = {
                    "roma_v2": "RoMa v2", "roma_omega": "RoMa v2", "roma": "RoMa v2",
                    "loftr": "LoFTR", "yolo11": "YOLO11 (Crater)",
                    "sift": "RootSIFT (Two-Tier)", "rootsift": "RootSIFT (Two-Tier)",
                    "lightglue": "DISK+LightGlue", "ni_lfm": "NI-LFM",
                    "tiny_roma": "Tiny-RoMa", "orb": "ORB"
                }
                for c, m in zip(cols, models_used):
                    m_label = engine_label_map.get(m, m.upper())
                    n = counts.get(m, 0)
                    w = 0.0 if n == 0 else float(weights.get(m, 0.0))
                    delta_txt = "Weight: 0% (Excluded)" if n == 0 else f"Weight: {w:.0%}"
                    c.metric(f"Engine: {m_label}", f"{n:,} raw", delta_txt)

        quality = results.get('quality', {})
        if quality:
            grade = quality.get('grade', '?')
            confidence = quality.get('confidence', 0)
            grade_emoji = {'A': '🟢', 'B': '🟢', 'C': '🟡', 'D': '🟠', 'F': '🔴'}.get(grade, '⚪')

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            qm1, qm2, qm3 = st.columns(3)
            qm1.metric("Quality Grade", f"{grade_emoji} Grade {grade}", f"Confidence: {confidence:.0%}")
            qm2.metric("Registration Gate", "✅ PASSED (Photogrammetry Grade)" if quality.get('passed', False) else "❌ FAILED QUALITY GATE")
            w_rmse_val = results.get('weighted_rmse')
            metric_is_valid = quality.get('passed', False) and results.get('inlier_count', 0) >= 15 and w_rmse_val is not None
            qm3.metric(
                "Weighted RMSE",
                f"{w_rmse_val:.3f} px" if metric_is_valid else "Rejected",
                "Confidence-weighted deviation" if metric_is_valid else "Quality gate failed",
            )

            auto_rec = results.get('auto_recovery') or quality.get('details', {}).get('auto_recovery')
            if auto_rec:
                st.info(f"⚡ **{auto_rec}**")

            warnings = quality.get('warnings', [])
            if warnings:
                for w in warnings:
                    if auto_rec and auto_rec in w:
                        continue
                    st.warning(f"⚠️ {w}")

            # ── Direct Scientific Telemetry & Quality Report Download Card ──
            eff_src = src_path if (src_path and os.path.exists(src_path)) else "source_lunar.png"
            eff_ref = ref_path if (ref_path and os.path.exists(ref_path)) else "reference_lunar.png"
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_pdf.close()
            if generate_pdf_report(results, eff_src, eff_ref, tmp_pdf.name) and os.path.exists(tmp_pdf.name):
                with open(tmp_pdf.name, "rb") as f:
                    pdf_bytes = f.read()
                st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
                card_col1, card_col2 = st.columns([3, 1.2], vertical_alignment="center")
                with card_col1:
                    fusion_note = " · Includes LunarFusion Adaptive AI Multi-Engine Telemetry & Weights" if results.get('fusion_stats') else ""
                    st.markdown(f"""
                    <div style="font-size:0.86rem; color:#E2E8F0; line-height:1.4;">
                        <strong>📑 Certified Scientific Telemetry &amp; Quality Assessment Report (PDF)</strong><br>
                        <span style="font-size:0.78rem; color:#CBD5E1;">Grade {grade} Quality Gate · Sub-Pixel Error Percentiles (P90–P99) · 8×8 Crater Grid Coverage{fusion_note}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with card_col2:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"ISRO_Scientific_Report_{grade}_{int(time.time())}.pdf",
                        mime="application/pdf",
                        type="primary",
                        icon=":material/picture_as_pdf:",
                        width="stretch"
                    )

    # ── Step 4: Verification Suite ──
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">04</span>
            <div>
                <div class="step-title">Photogrammetric Alignment Inspection Suite</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Visual confirmation of crater rim continuity, sub-pixel disparity, and non-linear warp.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        vis_tab1, vis_tab2, vis_tab3, vis_tab4, vis_tab5 = st.tabs([
            "Checkerboard",
            "Alpha Blend",
            "Error Difference",
            "Match Inspector",
            "3D Reconstruction"
        ])

        img_warped = results.get('img_src_warped', None)
        if img_warped is None and img_src is not None and img_ref is not None:
            h, w = img_ref.shape[:2]
            img_warped = cv2.warpPerspective(img_src, results.get('H_fallback', np.eye(3)), (w, h))

        with vis_tab1:
            if img_warped is not None and img_ref is not None:
                h_c, w_c = img_ref.shape[:2]
                w_aligned = img_warped if img_warped.shape[:2] == (h_c, w_c) else cv2.resize(img_warped, (w_c, h_c), interpolation=cv2.INTER_CUBIC)

                chk_c1, chk_c2, chk_c3, chk_c4 = st.columns([1.5, 1.2, 1.0, 1.0], vertical_alignment="center")
                with chk_c1:
                    chk_view = st.radio(
                        "Inspection Mode",
                        ["Checkerboard Grid", "Horizontal Split Wipe", "Vertical Split Wipe", "Edge Continuity Overlay"],
                        horizontal=True,
                        key="chk_mode_radio"
                    )
                with chk_c2:
                    chk_density = st.slider("Grid Divisions (Tiles)", 4, 64, 16, step=4, key="chk_density_slider")
                with chk_c3:
                    chk_seams = st.toggle("Seam Reticles", value=True, key="chk_seam_toggle")
                with chk_c4:
                    seam_color_choice = st.selectbox(
                        "Reticle Color",
                        ["Neon Cyan", "Emerald Green", "Solar Amber", "Pure White", "Crimson Red"],
                        index=0,
                        key="chk_color_choice"
                    )

                seam_color_map = {
                    "Neon Cyan": (0, 240, 255),
                    "Emerald Green": (0, 255, 160),
                    "Solar Amber": (255, 184, 0),
                    "Pure White": (255, 255, 255),
                    "Crimson Red": (255, 50, 70),
                }
                active_seam_rgb = seam_color_map.get(seam_color_choice, (0, 240, 255))

                # Quantitative Sub-Pixel Seam Continuity Telemetry
                seam_metrics = compute_seam_alignment_continuity(w_aligned, img_ref, tiles=chk_density)
                coherence_val = seam_metrics["coherence_pct"]
                mean_seam_disp = seam_metrics["mean_seam_disparity_px"]
                is_subpx_locked = seam_metrics["is_subpixel_locked"]

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 10px 14px; margin: 8px 0 12px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; gap: 18px; align-items: center;">
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Seam Boundary Coherence</span>
                            <div style="font-size:0.92rem; font-weight:800; color:{'#34D399' if coherence_val >= 90 else '#FBBF24'}; font-family:'JetBrains Mono',monospace;">{coherence_val:.1f}%</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Mean Seam Disparity</span>
                            <div style="font-size:0.92rem; font-weight:800; color:#38BDF8; font-family:'JetBrains Mono',monospace;">{mean_seam_disp:.2f} px</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Photogrammetric Phase Lock</span>
                            <div style="font-size:0.82rem; font-weight:700; color:{'#34D399' if is_subpx_locked else '#F87171'};">{'🔒 SUB-PIXEL LOCKED' if is_subpx_locked else '⚠️ DIVERGENCE DETECTED'}</div>
                        </div>
                    </div>
                    <span style="font-size:0.75rem; color:#CBD5E1;">Grid: <strong>{chk_density}×{chk_density}</strong> ({max(h_c//chk_density, 1)}×{max(w_c//chk_density, 1)} px/tile)</span>
                </div>
                """, unsafe_allow_html=True)

                if chk_view == "Checkerboard Grid":
                    checker = create_checkerboard(
                        w_aligned, img_ref,
                        tiles=chk_density,
                        show_seams=chk_seams,
                        seam_color=active_seam_rgb,
                        contrast_boost=True
                    )
                    st.image(checker, caption=f"Checkerboard Verification ({chk_density}×{chk_density}) — High-contrast boundaries verify sub-pixel crater rim continuity crossing tiles.", width="stretch")

                elif chk_view == "Horizontal Split Wipe":
                    wipe_x = st.slider("Curtain Divider Position (Left: Warped Source | Right: Reference Anchor)", 0, 100, 50, key="wipe_slider_x")
                    cut_x = int(w_c * (wipe_x / 100.0))
                    i1_rgb = normalize_to_rgb8(w_aligned, clahe_enhance=True)
                    i2_rgb = normalize_to_rgb8(img_ref, clahe_enhance=True)
                    wipe_img = np.zeros_like(i1_rgb)
                    wipe_img[:, :cut_x] = i1_rgb[:, :cut_x]
                    wipe_img[:, cut_x:] = i2_rgb[:, cut_x:]
                    if chk_seams and 0 < cut_x < w_c:
                        wipe_img[:, max(0, cut_x - 1):min(w_c, cut_x + 2)] = np.array(active_seam_rgb, dtype=np.uint8)
                    st.image(wipe_img, caption=f"Horizontal Split Wipe (at {wipe_x}% width) — Drag divider to verify sub-pixel rim alignment across boundary.", width="stretch")

                elif chk_view == "Vertical Split Wipe":
                    wipe_y = st.slider("Curtain Divider Position (Top: Warped Source | Bottom: Reference Anchor)", 0, 100, 50, key="wipe_slider_y")
                    cut_y = int(h_c * (wipe_y / 100.0))
                    i1_rgb = normalize_to_rgb8(w_aligned, clahe_enhance=True)
                    i2_rgb = normalize_to_rgb8(img_ref, clahe_enhance=True)
                    wipe_img = np.zeros_like(i1_rgb)
                    wipe_img[:cut_y, :] = i1_rgb[:cut_y, :]
                    wipe_img[cut_y:, :] = i2_rgb[cut_y:, :]
                    if chk_seams and 0 < cut_y < h_c:
                        wipe_img[max(0, cut_y - 1):min(h_c, cut_y + 2), :] = np.array(active_seam_rgb, dtype=np.uint8)
                    st.image(wipe_img, caption=f"Vertical Split Wipe (at {wipe_y}% height) — Drag divider to verify along-track sub-pixel rim continuity.", width="stretch")

                elif chk_view == "Edge Continuity Overlay":
                    g1 = cv2.cvtColor(w_aligned, cv2.COLOR_BGR2GRAY) if w_aligned.ndim == 3 else w_aligned
                    g2 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if img_ref.ndim == 3 else img_ref
                    e1 = cv2.Canny(g1, 40, 120)
                    e2 = cv2.Canny(g2, 40, 120)
                    edge_canvas = np.zeros((h_c, w_c, 3), dtype=np.uint8)
                    edge_canvas[e1 > 0, 1] = 255  # Green for Warped Source
                    edge_canvas[e2 > 0, 0] = 255  # Red for Reference
                    edge_canvas[e2 > 0, 2] = 255  # Blue for Reference (Magenta)
                    # Overlap is Green + Magenta = White (255, 255, 255)
                    st.image(edge_canvas, caption="Sub-Pixel Edge Continuity — Green: Warped Source | Magenta: Reference Anchor | White: Coincident Crater Rims (Sub-pixel locked).", width="stretch")

        with vis_tab2:
            if img_warped is not None and img_ref is not None:
                alpha = st.slider("Source Layer Opacity", 0.0, 1.0, 0.5, step=0.05, key="vis_alpha_slider")
                _h, _w = img_ref.shape[:2]
                _warped_blend = img_warped if img_warped.shape[:2] == (_h, _w) else cv2.resize(img_warped, (_w, _h))
                blend = cv2.addWeighted(_warped_blend, alpha, img_ref, 1.0 - alpha, 0)
                st.image(blend, caption="Continuous Alpha Blend — Slide opacity to confirm crater rim alignment.", width="stretch")

        with vis_tab3:
            if img_warped is not None and img_ref is not None:
                _h, _w = img_ref.shape[:2]
                _w_aligned = img_warped if img_warped.shape[:2] == (_h, _w) else cv2.resize(img_warped, (_w, _h))
                diff = cv2.absdiff(_w_aligned, img_ref)
                diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
                st.image(diff_colored, caption="Absolute Pixel Difference — Uniform deep blue confirms high geometric alignment.", width="stretch")

        with vis_tab4:
            inlier_count = results.get('inlier_count', 0)
            inliers_s = results.get('inliers_src')
            inliers_r = results.get('inliers_ref')
            has_matches = (
                inlier_count > 0 
                and inliers_s is not None 
                and len(inliers_s) > 0 
                and img_src is not None 
                and img_ref is not None
            )

            if has_matches:
                # ── Controls Header Bar ──
                ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.6, 1.2, 1.0, 1.2], vertical_alignment="center")
                
                with ctrl_col1:
                    vis_mode_sel = st.segmented_control(
                        "Inspector Mode",
                        options=["🖼️ Ultra-HD Canvas", "🔍 Interactive Vector (Plotly)"],
                        default="🖼️ Ultra-HD Canvas",
                        key="match_inspector_mode",
                        label_visibility="collapsed"
                    )
                    vis_mode = vis_mode_sel or "🖼️ Ultra-HD Canvas"

                with ctrl_col2:
                    density_option = st.selectbox(
                        "Tie-Point Density",
                        options=[
                            "100 Points (Clutter-Free)",
                            "250 Points (Balanced)",
                            "500 Points (High Density)",
                            "1000 Points (Dense Grid)",
                            "All Verified Inliers"
                        ],
                        index=1,
                        key="match_density_select"
                    )
                    if "100 Points" in density_option:
                        max_draw = 100
                    elif "250 Points" in density_option:
                        max_draw = 250
                    elif "500 Points" in density_option:
                        max_draw = 500
                    elif "1000 Points" in density_option:
                        max_draw = 1000
                    else:
                        max_draw = 0  # 0 indicates all points

                with ctrl_col3:
                    thick_opt = st.selectbox(
                        "Line Width",
                        options=["1 px (Fine)", "2 px (Standard HD)", "3 px (High Contrast)", "4 px (Bold)"],
                        index=1,
                        key="match_thickness_select"
                    )
                    thickness = int(thick_opt.split()[0])

                with ctrl_col4:
                    palette_opt = st.selectbox(
                        "Color Palette",
                        options=["Neon Aurora (Cyan/Mint)", "ISRO Amber-Gold", "Disparity Heatmap"],
                        index=0,
                        key="match_palette_select"
                    )
                    if "Gold" in palette_opt:
                        palette_code = "gold"
                    elif "Heatmap" in palette_opt:
                        palette_code = "heatmap"
                    else:
                        palette_code = "aurora"

                # Enhancement Controls
                c_opt1, c_opt2 = st.columns([2.5, 1.5], vertical_alignment="center")
                with c_opt1:
                    enhance_contrast = st.checkbox(
                        "🌓 Enhance Lunar Crater Contrast (Adaptive CLAHE)",
                        value=True,
                        help="Amplifies micro-crater topography and regolith texture clarity for HD inspection",
                        key="match_clahe_toggle"
                    )
                with c_opt2:
                    st.markdown(f"""
                    <div style="text-align:right; font-size:0.78rem; color:#CBD5E1; font-family:'JetBrains Mono',monospace;">
                        Verified Inliers: <strong style="color:#06D6A0;">{inlier_count:,}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)

                if vis_mode == "🖼️ Ultra-HD Canvas":
                    # Ultra-HD Canvas Mode
                    hd_canvas = create_hd_match_canvas(
                        img_src=img_src,
                        img_ref=img_ref,
                        pts_src=inliers_s,
                        pts_ref=inliers_r,
                        inliers_mask=results.get('inliers_mask'),
                        max_lines=max_draw if max_draw > 0 else None,
                        line_thickness=thickness,
                        color_palette=palette_code,
                        enhance_contrast=enhance_contrast,
                        target_canvas_height=1280
                    )
                    st.image(
                        hd_canvas,
                        caption=f"ISRO Ultra-HD Photogrammetric Match Inspector — Sub-Pixel Antialiased Reticle Overlay ({hd_canvas.shape[1]}×{hd_canvas.shape[0]} px)",
                        width="stretch"
                    )

                    # Export / Download HD Canvas
                    ok, png_buf = cv2.imencode(".png", cv2.cvtColor(hd_canvas, cv2.COLOR_RGB2BGR))
                    if ok:
                        c_dl1, c_dl2 = st.columns([3, 1.2], vertical_alignment="center")
                        with c_dl1:
                            st.caption(f"Full-resolution composite canvas ready for photogrammetry documentation and GIS review ({hd_canvas.shape[1]}×{hd_canvas.shape[0]} px).")
                        with c_dl2:
                            st.download_button(
                                "📥 Download HD Canvas (PNG)",
                                data=png_buf.tobytes(),
                                file_name=f"ISRO_HD_Match_Canvas_{inlier_count}inliers_{int(time.time())}.png",
                                mime="image/png",
                                icon=":material/download:",
                                width="stretch"
                            )
                else:
                    # Interactive Vector Mode (Plotly)
                    fig = interactive_match_plot(
                        img_src=img_src,
                        img_ref=img_ref,
                        pts_src=inliers_s,
                        pts_ref=inliers_r,
                        inliers_mask=results.get('inliers_mask'),
                        max_display_matches=max_draw if max_draw > 0 else 10000,
                        line_width=float(thickness),
                        color_palette=palette_code,
                        enhance_contrast=enhance_contrast
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption("💡 Click and drag to box-zoom into any crater region. Double-click to reset view. Hover over reticle keypoints to inspect exact coordinates and disparity vectors.")
            else:
                st.warning("No verified matches available for inspection. Run registration first or adjust quality threshold.")

        with vis_tab5:
            if img_warped is not None and img_ref is not None:
                st.markdown("""
                <div style="font-size:0.86rem; color:#CBD5E1; margin-bottom:12px;">
                    Calibrated 3D photogrammetric elevation surface synthesized from co-registered stereo pair. 
                    Supports <strong>100% Full-Size Native Mesh Render</strong>, metric triangulation ($Z = \\frac{f \\cdot B}{d \\cdot \\text{pitch}}$), and draped high-resolution lunar photographic texture.
                </div>
                """, unsafe_allow_html=True)

                c_v5_1, c_v5_2, c_v5_3, c_v5_4 = st.columns([1.5, 1.2, 1.0, 1.2], vertical_alignment="center")
                with c_v5_1:
                    res_choice = st.selectbox(
                        "3D Map Resolution",
                        ["Full-Size 100% Native", "High-Definition 512px", "Interactive Fast 256px"],
                        index=0,
                        key="v5_res_choice"
                    )
                with c_v5_2:
                    tex_choice = st.selectbox(
                        "Surface Texture / Shading",
                        [
                            "Lunar Texture (Photorealistic)",
                            "Lunar Highlands (Albedo)",
                            "Spectral Relief",
                            "Viridis",
                            "Plasma",
                            "Turbo",
                            "Terrain",
                            "Cividis"
                        ],
                        index=0,
                        key="v5_tex_choice"
                    )
                with c_v5_3:
                    v_exagg = st.slider("Relief Scale", 0.5, 4.0, 1.8, 0.1, key="v5_exagg_slider")
                with c_v5_4:
                    sensor_choice = st.selectbox(
                        "Sensor Optics",
                        [
                            "Chandrayaan-2 TMC-2 (f=140mm, B=93.3km)",
                            "Chandrayaan-2 OHRC (f=2200mm, B=10km)",
                            "Chandrayaan-1 TMC (f=120mm)"
                        ],
                        index=0,
                        key="v5_sensor_choice"
                    )

                s_code = "ohrc" if "OHRC" in sensor_choice else "tmc"

                with st.spinner("Rendering calibrated 100% full-size 3D lunar surface..."):
                    dem_fig, dem_raster, dem_tex, dem_qc = generate_dem_preview(
                        img_warped=img_warped,
                        img_ref=img_ref,
                        resolution_mode=res_choice,
                        vertical_exaggeration=v_exagg,
                        colorscale=tex_choice,
                        sensor_name=s_code,
                        inliers_src=results.get('inliers_src'),
                        inliers_ref=results.get('inliers_ref'),
                    )

                # Telemetry Bar — Full Matched Map Status
                grid_badge = "100% Native" if dem_qc['is_full_size'] else f"{dem_qc['scale_pct']}% Res"
                grid_txt = f"{dem_qc['dim_w']} × {dem_qc['dim_h']} px ({grid_badge})"
                extent_pct = dem_qc.get('extent_pct', 100)
                n_anchors = dem_qc.get('n_anchors', 0)
                crop_km_w = dem_qc.get('crop_km_w', 0.0)
                crop_km_h = dem_qc.get('crop_km_h', 0.0)
                extent_km_txt = f"{crop_km_w:.2f} × {crop_km_h:.2f} km"
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 8px; padding: 10px 14px; margin: 10px 0 14px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div style="display: flex; gap: 20px; align-items: center;">
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Resolution Grid</span>
                            <div style="font-size:0.88rem; font-weight:800; color:#38BDF8; font-family:'JetBrains Mono',monospace;">{grid_txt}</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Matched Area Coverage</span>
                            <div style="font-size:0.88rem; font-weight:800; color:{'#34D399' if extent_pct >= 80 else '#FBBF24'}; font-family:'JetBrains Mono',monospace;">{extent_pct}% · {extent_km_txt}</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Tie-Point Anchors</span>
                            <div style="font-size:0.88rem; font-weight:800; color:#818CF8; font-family:'JetBrains Mono',monospace;">{n_anchors} inliers</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Relief Span (ΔZ)</span>
                            <div style="font-size:0.88rem; font-weight:800; color:#34D399; font-family:'JetBrains Mono',monospace;">{dem_qc['relief_span_m']:.1f} m</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Elevation Bounds</span>
                            <div style="font-size:0.82rem; font-weight:700; color:#F8FAFC;">{dem_qc['min_el_m']:.0f} m to {dem_qc['max_el_m']:.0f} m</div>
                        </div>
                        <div style="height:24px; width:1px; background:rgba(255,255,255,0.1);"></div>
                        <div>
                            <span style="font-size:0.68rem; color:#94A3B8; text-transform:uppercase; font-weight:700;">Metric GSD</span>
                            <div style="font-size:0.88rem; font-weight:800; color:#A78BFA; font-family:'JetBrains Mono',monospace;">{dem_qc['gsd_m']:.1f} m/px</div>
                        </div>
                    </div>
                    <span style="background: rgba(6, 214, 160, 0.18); border: 1px solid #06D6A0; color: #06D6A0; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 14px;">
                        {'🗺️ FULL MATCHED MAP' if extent_pct >= 80 else ('🟢 100% FULL-SIZE RENDER' if dem_qc['is_full_size'] else '⚡ FAST PREVIEW')}
                    </span>
                </div>
                """, unsafe_allow_html=True)


                st.plotly_chart(dem_fig, width="stretch")

                # Scientific 3D Product Exports Card
                st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
                c_dl1, c_dl2, c_dl3 = st.columns(3)
                with c_dl1:
                    try:
                        tmp_ply = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
                        tmp_ply.close()
                        export_point_cloud_ply(tmp_ply.name, dem_raster, gsd_m=dem_qc['gsd_m'], texture=dem_tex, subsample_step=1)
                        with open(tmp_ply.name, "rb") as f:
                            ply_data = f.read()
                        st.download_button(
                            "📥 Download 3D Mesh (.PLY)",
                            data=ply_data,
                            file_name=f"lunar_3d_dem_{int(time.time())}.ply",
                            mime="application/octet-stream",
                            icon=":material/download:",
                            width="stretch"
                        )
                    except Exception as e_ply:
                        st.caption(f"PLY export: {e_ply}")

                with c_dl2:
                    try:
                        tmp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                        tmp_tif.close()
                        export_dem_geotiff(dem_raster, tmp_tif.name, gsd_m=dem_qc['gsd_m'])
                        with open(tmp_tif.name, "rb") as f:
                            tif_data = f.read()
                        st.download_button(
                            "🗺️ Download Elevation GeoTIFF",
                            data=tif_data,
                            file_name=f"lunar_elevation_{int(time.time())}.tif",
                            mime="image/tiff",
                            icon=":material/download:",
                            width="stretch"
                        )
                    except Exception as e_tif:
                        st.caption(f"GeoTIFF export: {e_tif}")

                with c_dl3:
                    try:
                        tmp_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
                        tmp_obj.close()
                        export_mesh_obj(tmp_obj.name, dem_raster, gsd_m=dem_qc['gsd_m'], subsample_step=2)
                        with open(tmp_obj.name, "rb") as f:
                            obj_data = f.read()
                        st.download_button(
                            "📦 Download Wavefront 3D (.OBJ)",
                            data=obj_data,
                            file_name=f"lunar_cad_mesh_{int(time.time())}.obj",
                            mime="text/plain",
                            icon=":material/download:",
                            width="stretch"
                        )
                    except Exception as e_obj:
                        st.caption(f"OBJ export: {e_obj}")

    # ── Step 5: Production Product Export Hub ──
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">05</span>
            <div>
                <div class="step-title">Production Product Export Hub</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Export NASA ASP tie-points, GIS GeoTIFF rasters, and certified scientific evaluation reports.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if results.get("backend_job_id"):
            b_jid = results["backend_job_id"]
            st.markdown(f"""
            <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.85rem; color: #E2E8F0;">
                    📡 <strong>FastAPI Backend Certified Job:</strong> <code style="color:#7DD3FC;">{b_jid}</code> (Runtime: {float(results.get('backend_elapsed') or 0.0):.2f}s)
                </span>
                <span style="font-size: 0.75rem; color: #34D399; font-weight: 600;">DIRECT CLOUD ARTIFACTS READY</span>
            </div>
            """, unsafe_allow_html=True)
            c_ba1, c_ba2, c_ba3, c_ba4 = st.columns(4)
            with c_ba1:
                b_gtiff = _fetch_backend_artifact(b_jid, "geotiff")
                if b_gtiff:
                    st.download_button("🗺️ Cloud Warped GeoTIFF", data=b_gtiff, file_name=f"cloud_geotiff_{b_jid[:8]}.tif", mime="image/tiff", width="stretch")
            with c_ba2:
                b_csv = _fetch_backend_artifact(b_jid, "tiepoints")
                if b_csv:
                    st.download_button("📍 Cloud ASP Tie-Points", data=b_csv, file_name=f"cloud_tiepoints_{b_jid[:8]}.csv", mime="text/csv", width="stretch")
            with c_ba3:
                b_pdf = _fetch_backend_artifact(b_jid, "pdf")
                if b_pdf:
                    st.download_button("📑 Cloud PDF Report", data=b_pdf, file_name=f"cloud_certification_{b_jid[:8]}.pdf", mime="application/pdf", width="stretch")
            with c_ba4:
                b_tele = _fetch_backend_artifact(b_jid, "telemetry")
                if b_tele:
                    st.download_button("📊 Cloud Telemetry JSON", data=b_tele, file_name=f"cloud_telemetry_{b_jid[:8]}.json", mime="application/json", width="stretch")
            st.write("")

        exp1, exp2, exp3, exp4, exp5 = st.columns(5)
        with exp1:
            inliers_src_exp = np.asarray(results.get('inliers_src', []))
            inliers_ref_exp = np.asarray(results.get('inliers_ref', []))
            confidences_exp = np.asarray(results.get('confidences', []))
            if len(inliers_src_exp) > 0 and len(inliers_ref_exp) > 0:
                tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                tmp_csv.close()
                export_asp_compatible_tiepoints(inliers_src_exp, inliers_ref_exp, confidences_exp, tmp_csv.name)
                with open(tmp_csv.name, "rb") as f:
                    csv_bytes = f.read()
                st.download_button(
                    "ASP CSV Points",
                    data=csv_bytes,
                    file_name="asp_tiepoints.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    width="stretch"
                )
            else:
                st.button("ASP CSV (No Inliers)", disabled=True, width="stretch")

        with exp2:
            if len(inliers_src_exp) > 0 and len(inliers_ref_exp) > 0:
                tmp_match = tempfile.NamedTemporaryFile(delete=False, suffix=".match")
                tmp_match.close()
                export_asp_match_file(inliers_src_exp, inliers_ref_exp, tmp_match.name)
                with open(tmp_match.name, "rb") as f:
                    match_bytes = f.read()
                st.download_button(
                    "ASP .match File",
                    data=match_bytes,
                    file_name="asp_tiepoints.match",
                    mime="text/plain",
                    icon=":material/download:",
                    width="stretch"
                )
            else:
                st.button("ASP Match (No Inliers)", disabled=True, width="stretch")

        with exp3:
            if img_warped is not None:
                tmp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                tmp_tif.close()
                export_geotiff(img_warped, tmp_tif.name)
                with open(tmp_tif.name, "rb") as f:
                    tif_bytes = f.read()
                st.download_button(
                    "Warped GeoTIFF",
                    data=tif_bytes,
                    file_name="warped_registered.tif",
                    mime="image/tiff",
                    icon=":material/download:",
                    width="stretch"
                )

        with exp4:
            eff_s = src_path if (src_path and os.path.exists(src_path)) else "source_lunar.png"
            eff_r = ref_path if (ref_path and os.path.exists(ref_path)) else "reference_lunar.png"
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp_pdf.close()
            if generate_pdf_report(results, eff_s, eff_r, tmp_pdf.name) and os.path.exists(tmp_pdf.name):
                with open(tmp_pdf.name, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "Scientific PDF",
                    data=pdf_bytes,
                    file_name=f"isro_scientific_report_{int(time.time())}.pdf",
                    mime="application/pdf",
                    icon=":material/picture_as_pdf:",
                    width="stretch"
                )

        with exp5:
            metrics_json = json.dumps({
                'method': results.get('method_used', 'Unknown'),
                'total_matches': results.get('total_matches', 0),
                'inlier_count': results.get('inlier_count', 0),
                'inlier_ratio': float(results.get('inlier_ratio', 0)),
                'rmse_pixels': float(results['rmse_pixels']) if results.get('rmse_pixels') is not None else None,
                'weighted_rmse': float(results['weighted_rmse']) if results.get('weighted_rmse') is not None else None,
                'spatial_coverage': float(results.get('coverage', 0)),
                'scale_src': float(results.get('scale_src', 1.0)),
                'scale_ref': float(results.get('scale_ref', 1.0)),
                'error_stats': results.get('error_stats', {})
            }, indent=2)
            st.download_button(
                "Metrics JSON",
                data=metrics_json,
                file_name="registration_metrics.json",
                mime="application/json",
                icon=":material/download:",
                width="stretch"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SINGLE PAIR REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def render_single_registration_tab(available_images: list, history_mgr: Any, cfg: dict, pipeline_config: dict = None) -> None:
    """Render the Single Scene Co-Registration workflow and inspector tab."""
    p_cfg = pipeline_config or {}
    use_tps = p_cfg.get('use_tps', True)
    max_dim = p_cfg.get('max_dim', 2048)
    enable_subpixel = p_cfg.get('enable_subpixel', True)
    enable_pyramid = p_cfg.get('enable_pyramid', True)
    enable_photo = p_cfg.get('enable_photo', False)

    # ── Step 1: Imagery Ingestion ──
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">01</span>
            <div>
                <div class="step-title">Mission Imagery Ingestion</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Select heterogeneous lunar orbital scenes or upload custom datasets for sub-pixel co-registration.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        input_mode = st.segmented_control(
            "Input Mode",
            options=["Mission Archive Library", "Custom Orbital Upload"],
            default="Mission Archive Library",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns(2, gap="medium")
        src_path_final, ref_path_final = None, None
        src_is_upload, ref_is_upload = False, False
        metadata_kwargs = {}

        if input_mode == "Mission Archive Library":
            if len(available_images) == 0:
                st.warning("⚠️ No browse images found in `data/raw/`.")
            else:
                labels = [img['label'] for img in available_images]

                # Intelligent Curated Stereo Preset Selector
                pair_preset_choice = st.selectbox(
                    "🛰️ Curated Lunar Orbital Stereo Pair (Expected Overlap):",
                    options=[
                        "🌟 Chandrayaan-2 TMC Stereo (2019-12-08 Nadir & Aft — 5m/px) [Recommended]",
                        "🌙 Chandrayaan-2 TMC Stereo (2020-01-08 Nadir & Aft — 5m/px)",
                        "🛰️ Chandrayaan-1 TMC Stereo (2009-05-29 Fore & Aft — 5m/px)",
                        "📂 Manual Pair Selection (Custom Archive Search)"
                    ],
                    index=0,
                    help="Pre-configured orbital passes from the same orbit are expected to overlap; PDS4 geographic validation must confirm the overlap."
                )

                idx_2019_nrn = next((i for i, img in enumerate(available_images) if '20191208' in img['path'] and 'nrn' in img['path']), 13)
                idx_2019_nra = next((i for i, img in enumerate(available_images) if '20191208' in img['path'] and 'nra' in img['path']), 10)
                idx_2020_nrn = next((i for i, img in enumerate(available_images) if '20200108' in img['path'] and 'nrn' in img['path']), 15)
                idx_2020_nra = next((i for i, img in enumerate(available_images) if '20200108' in img['path'] and 'nra' in img['path']), 11)
                idx_2009_nca = next((i for i, img in enumerate(available_images) if '20090529T0853239926' in img['path'] and 'nca' in img['path']), 0)
                idx_2009_ncf = next((i for i, img in enumerate(available_images) if '20090529T0853239926' in img['path'] and 'ncf' in img['path']), 3)

                if "2019-12-08" in pair_preset_choice:
                    preset_src, preset_ref = idx_2019_nrn, idx_2019_nra
                elif "2020-01-08" in pair_preset_choice:
                    preset_src, preset_ref = idx_2020_nrn, idx_2020_nra
                elif "2009-05-29" in pair_preset_choice:
                    preset_src, preset_ref = idx_2009_nca, idx_2009_ncf
                else:
                    preset_src, preset_ref = 0, 1

                with col1:
                    st.markdown("##### 🎯 Source (Target) Scene")
                    src_idx = st.selectbox(
                        "Source Image Selection",
                        range(len(labels)),
                        format_func=lambda i: labels[i],
                        index=preset_src,
                        key="lib_src",
                        label_visibility="collapsed"
                    )
                    src_item = available_images[src_idx]
                    src_path_final = src_item['path']
                    st.markdown(f"""
                    <div class="scene-meta-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span class="telemetry-pill">{src_item['mission']} • {src_item['sensor']}</span>
                            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#38BDF8;">{src_item['res']}</span>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#CBD5E1; word-break:break-all;">{src_item['basename']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if src_path_final and os.path.exists(src_path_final):
                        prev_s = get_safe_preview(src_path_final)
                        if prev_s is not None:
                            st.image(prev_s, caption="Target Lunar Scene Preview", width="stretch")

                with col2:
                    st.markdown("##### ⚓ Reference (Anchor) Scene")
                    ref_idx = st.selectbox(
                        "Reference Image Selection",
                        range(len(labels)),
                        format_func=lambda i: labels[i],
                        index=preset_ref,
                        key="lib_ref",
                        label_visibility="collapsed"
                    )
                    ref_item = available_images[ref_idx]
                    ref_path_final = ref_item['path']
                    metadata_kwargs = _metadata_for_archive_pair(src_path_final, ref_path_final)
                    st.markdown(f"""
                    <div class="scene-meta-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                            <span class="telemetry-pill">{ref_item['mission']} • {ref_item['sensor']}</span>
                            <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#38BDF8;">{ref_item['res']}</span>
                        </div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#CBD5E1; word-break:break-all;">{ref_item['basename']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if ref_path_final and os.path.exists(ref_path_final):
                        prev_r = get_safe_preview(ref_path_final)
                        if prev_r is not None:
                            st.image(prev_r, caption="Anchor Lunar Scene Preview", width="stretch")
        else:
            with col1:
                st.markdown("##### 🎯 Upload Source (Target) Scene")
                src_file = st.file_uploader(
                    "Upload Source Image",
                    type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'img'],
                    key="src_upload",
                    label_visibility="collapsed",
                    help="Accepts PNG, JPG, TIFF, and PDS4/planetary raw .IMG raster files."
                )
                if src_file:
                    src_path_final = _save_uploaded_orbital_image(src_file)
                    src_is_upload = True
                    st.success(f"Loaded: `{src_file.name}` ({len(src_file.getvalue()) // 1024:,} KB)")
                    prev_src = get_safe_preview(src_path_final)
                    if prev_src is not None:
                        st.image(prev_src, caption=f"Uploaded Target: {src_file.name}", width="stretch")

            with col2:
                st.markdown("##### ⚓ Upload Reference (Anchor) Scene")
                ref_file = st.file_uploader(
                    "Upload Reference Image",
                    type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'img'],
                    key="ref_upload",
                    label_visibility="collapsed",
                    help="Accepts PNG, JPG, TIFF, and PDS4/planetary raw .IMG raster files."
                )
                if ref_file:
                    ref_path_final = _save_uploaded_orbital_image(ref_file)
                    ref_is_upload = True
                    st.success(f"Loaded: `{ref_file.name}` ({len(ref_file.getvalue()) // 1024:,} KB)")
                    prev_ref = get_safe_preview(ref_path_final)
                    if prev_ref is not None:
                        st.image(prev_ref, caption=f"Uploaded Anchor: {ref_file.name}", width="stretch")

        st.markdown("##### Independent geographic validation (optional)")
        st.caption("Upload the matching PDS4 XML label and *_g_grd_*.csv grid for each image. These files validate geographic overlap and GT-RMSE without influencing feature matching.")
        meta_src_col, meta_ref_col = st.columns(2)
        with meta_src_col:
            src_label_upload = st.file_uploader("Source PDS4 XML label", type=["xml", "lbl"], key="src_pds4_upload")
            src_grid_upload = st.file_uploader("Source geometry grid CSV", type=["csv"], key="src_grid_upload")
        with meta_ref_col:
            ref_label_upload = st.file_uploader("Reference PDS4 XML label", type=["xml", "lbl"], key="ref_pds4_upload")
            ref_grid_upload = st.file_uploader("Reference geometry grid CSV", type=["csv"], key="ref_grid_upload")

        if src_label_upload and ref_label_upload and src_grid_upload and ref_grid_upload:
            metadata_kwargs = {
                "src_pds4_label": _save_uploaded_metadata(src_label_upload, ".xml"),
                "ref_pds4_label": _save_uploaded_metadata(ref_label_upload, ".xml"),
                "src_geometry_csv": _save_uploaded_metadata(src_grid_upload, ".csv"),
                "ref_geometry_csv": _save_uploaded_metadata(ref_grid_upload, ".csv"),
            }
            st.success("Independent PDS4/geographic validation is configured for this run.")
        elif any((src_label_upload, ref_label_upload, src_grid_upload, ref_grid_upload)):
            st.warning("Upload all four metadata files to enable mission-grade certification.")
        elif src_path_final and ref_path_final:
            auto_meta = _metadata_for_archive_pair(src_path_final, ref_path_final)
            if auto_meta:
                metadata_kwargs = auto_meta
                st.info("🛰️ Auto-detected matching PDS4 XML and geometry grid CSV from project library.")

    # ── Step 2: Registration Command Center ──
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">02</span>
            <div>
                <div class="step-title">Registration Command Center</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Select algorithmic engine architecture and initiate sub-pixel feature extraction.</div>
            </div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0 14px 0;">
            <span style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.3); border-radius:16px; padding:3px 10px; font-size:0.73rem; color:#38BDF8; font-weight:600;">
                ⚡ Simultaneous ThreadPool
            </span>
            <span style="background:rgba(56,189,248,0.10); border:1px solid rgba(56,189,248,0.25); border-radius:16px; padding:3px 10px; font-size:0.73rem; color:#7DD3FC; font-weight:600;">
                🌔 YOLO11 Crater Landmarks
            </span>
            <span style="background:rgba(52,211,153,0.12); border:1px solid rgba(52,211,153,0.3); border-radius:16px; padding:3px 10px; font-size:0.73rem; color:#34D399; font-weight:600;">
                🟢 RootSIFT Two-Tier Baseline
            </span>
            <span style="background:rgba(167,139,250,0.12); border:1px solid rgba(167,139,250,0.3); border-radius:16px; padding:3px 10px; font-size:0.73rem; color:#C084FC; font-weight:600;">
                🔺 Multi-Scale Pyramid (0.5x, 1.0x, 1.5x)
            </span>
            <span style="background:rgba(244,114,182,0.12); border:1px solid rgba(244,114,182,0.3); border-radius:16px; padding:3px 10px; font-size:0.73rem; color:#F472B6; font-weight:600;">
                🎯 Sub-Pixel Parabolic Peak Refinement
            </span>
        </div>
        """, unsafe_allow_html=True)

        engine_mode = st.segmented_control(
            "Engine Architecture Mode",
            options=[
                "⚡ OmniRoute Multi-AI Gateway",
                "🌟 ISRO All-Engine Consensus",
                "🌙 LunarFusion Adaptive AI",
                "🤖 Autonomous Leaderboard (All Engines)",
                "🔬 Single Diagnostic Engine"
            ],
            default="⚡ OmniRoute Multi-AI Gateway",
            label_visibility="collapsed"
        ) or "⚡ OmniRoute Multi-AI Gateway"

        speed_mode = "balanced"
        single_matcher_key = "roma_v2"
        omni_strategy_key = "consensus"
        omni_workers = 4
        omni_selected_models = ["sift", "loftr", "roma_v2", "lightglue"]

        if engine_mode == "⚡ OmniRoute Multi-AI Gateway":
            c_omni1, c_omni2 = st.columns([1.2, 2.8], vertical_alignment="center")
            with c_omni1:
                st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#38BDF8;'>Routing Strategy:</span>", unsafe_allow_html=True)
            with c_omni2:
                selected_strat = st.pills(
                    "OmniRoute Strategy",
                    options=[
                        "🚀 Consensus Boost (Parallel ThreadPool)",
                        "⚡ Auto-Fallback Cascade (Zero Downtime)",
                        "🏁 Fastest-Wins Race (Real-Time)",
                        "🧭 Adaptive Terrain Diagnosis (<50ms)"
                    ],
                    default="🚀 Consensus Boost (Parallel ThreadPool)",
                    label_visibility="collapsed"
                ) or "🚀 Consensus Boost (Parallel ThreadPool)"
                if selected_strat and "Auto-Fallback" in selected_strat:
                    omni_strategy_key = "auto_fallback"
                elif selected_strat and "Fastest-Wins" in selected_strat:
                    omni_strategy_key = "fastest_wins"
                elif selected_strat and "Adaptive" in selected_strat:
                    omni_strategy_key = "adaptive_profile"
                else:
                    omni_strategy_key = "consensus"

            with st.expander("⚙️ OmniRoute Concurrency & Model Pool Tuning", expanded=False):
                c_pool1, c_pool2 = st.columns([2.5, 1.5])
                with c_pool1:
                    omni_model_options = [
                        ("RootSIFT (Deterministic Two-Tier)", "sift"),
                        ("LoFTR (Detector-Free)", "loftr"),
                        ("RoMa v2 (Dense Transformer SOTA)", "roma_v2"),
                        ("YOLO11 (Crater Landmarks)", "yolo11"),
                        ("DISK + LightGlue", "lightglue"),
                        ("NI-LFM (NASA-IBM)", "ni_lfm"),
                        ("ORB (Real-Time)", "orb")
                    ]
                    sel_labels = st.multiselect(
                        "Participating AI Models:",
                        options=[lbl for lbl, _ in omni_model_options],
                        default=["RootSIFT (Deterministic Two-Tier)", "LoFTR (Detector-Free)", "RoMa v2 (Dense Transformer SOTA)"]
                    )
                    label_to_key = dict(omni_model_options)
                    omni_selected_models = [label_to_key[lbl] for lbl in sel_labels] if sel_labels else ["sift", "loftr", "roma_v2"]
                with c_pool2:
                    omni_workers = st.slider("Worker Threads", min_value=1, max_value=8, value=4, help="Concurrent threads in isolated ThreadPool")

            st.caption("⚡ **OmniRoute AI Gateway:** Orchestrates heterogeneous models concurrently in an isolated thread pool with zero-downtime automated fallback and spatial consensus boosting.")

        elif engine_mode == "🌟 ISRO All-Engine Consensus":
            speed_mode = "accuracy"
            st.caption("🚀 **ISRO All-Engine Consensus Mode:** Concurrently executes RoMa v2 Dense Transformer, LoFTR, YOLO11 Crater Landmarks, RootSIFT Two-Tier, DISK+LightGlue, and NI-LFM in parallel. Cross-analyzes putative matches, filters spatial inconsistencies, and synthesizes a single, unified high-precision tie-point field with optimal sub-pixel RMSE.")

        elif engine_mode == "🌙 LunarFusion Adaptive AI":
            c_mode1, c_mode2 = st.columns([1.2, 2], vertical_alignment="center")
            with c_mode1:
                st.markdown("<span style='font-size:0.85rem; font-weight:600; color:#E2E8F0;'>Performance Profile:</span>", unsafe_allow_html=True)
            with c_mode2:
                selected_speed = st.pills(
                    "Speed Profile",
                    options=["Fast (LoFTR + RootSIFT)", "Balanced (Adaptive Multi-AI)", "Maximum Precision (All 6 Engines)"],
                    default="Balanced (Adaptive Multi-AI)",
                    label_visibility="collapsed"
                ) or "Balanced (Adaptive Multi-AI)"
                if selected_speed == "Fast (LoFTR + RootSIFT)":
                    speed_mode = "fast"
                elif selected_speed == "Maximum Precision (All 6 Engines)":
                    speed_mode = "accuracy"
                else:
                    speed_mode = "balanced"
            st.caption("✨ **LunarFusion Engine:** Diagnoses crater density, shadows, and low-texture regolith to optimize RoMa v2, LoFTR, **YOLO11 Crater Landmarks**, **RootSIFT (Deterministic Two-Tier)**, DISK+LightGlue, and NI-LFM. Executes CPU deterministic math simultaneously in parallel with deep models, with hardened 10K MAGSAC++ and multi-scale sub-pixel refinement.")

        elif "Autonomous Leaderboard" in engine_mode:
            st.caption("🤖 Evaluates all engines (LunarFusion, RoMa v2, LoFTR, YOLO11 Crater, RootSIFT Two-Tier, DISK+LightGlue, NI-LFM) sequentially, renders an ISRO photogrammetry leaderboard, and auto-selects the winning homography/TPS model.")

        elif engine_mode == "🔬 Single Diagnostic Engine":
            matcher_choices = [
                ("RoMa v2 (Dense Transformer SOTA)", "roma_v2"),
                ("LoFTR (Detector-Free Transformer)", "loftr"),
                ("YOLO11 (Crater Landmark Matcher)", "yolo11"),
                ("RootSIFT (Deterministic Two-Tier)", "sift"),
                ("DISK + LightGlue (Deep Feature Matcher)", "lightglue"),
                ("NI-LFM (NASA-IBM Lunar Foundation)", "ni_lfm"),
                ("SuperPoint + SuperGlue", "superpoint"),
                ("ORB (Classical Real-Time)", "orb")
            ]
            choice_label = st.selectbox(
                "Single Matcher Selection",
                [c[0] for c in matcher_choices],
                index=0,
                label_visibility="collapsed"
            )
            single_matcher_key = dict(matcher_choices)[choice_label]
            st.caption(f"🎯 Direct execution with single `{single_matcher_key}` pipeline and pushbroom TPS rectification.")

        st.write("")
        use_backend = st.toggle(
            "Use FastAPI backend",
            value=bool(_is_backend_online()),
            help="Send uploads to the backend at the exact contract expected by /api/v1/register and poll /api/v1/jobs/{id}.",
            key="use_backend_api"
        )

        if use_backend:
            with st.expander("🔄 Reconnect / Retrieve Backend Job", expanded=False):
                backend_jobs = _list_backend_jobs() if _is_backend_online() else []
                recent_jid = st.session_state.get('last_backend_job_id', '')
                if backend_jobs:
                    sorted_bjobs = sorted(backend_jobs, key=lambda x: x.get("submitted_at", 0), reverse=True)
                    job_labels = [
                        f"{'🟢' if j.get('status') == 'completed' else '⚡' if j.get('status') == 'processing' else '⏳' if j.get('status') == 'queued' else '🔴'} "
                        f"{j.get('job_id', '')[:8]}... | {(j.get('matcher') or 'unknown').upper()} | {j.get('status', '').upper()} | {float(j.get('elapsed_seconds') or 0.0):.1f}s"
                        for j in sorted_bjobs
                    ]
                    sel_j_idx = st.selectbox(
                        "Recent FastAPI Backend Jobs",
                        range(len(sorted_bjobs)),
                        format_func=lambda i: job_labels[i],
                        key="select_recent_backend_job_tab1"
                    )
                    chosen_jid = sorted_bjobs[sel_j_idx].get("job_id") if sel_j_idx is not None else recent_jid
                else:
                    chosen_jid = recent_jid

                c_bj1, c_bj2, c_bj3 = st.columns([3, 1, 1], vertical_alignment="bottom")
                with c_bj1:
                    lookup_jid = st.text_input("Backend Job ID", value=chosen_jid, placeholder="e.g. 242be56d-1ee6-4a91-a30e-f6c0c9741404", key="input_lookup_jid")
                with c_bj2:
                    if st.button("Fetch Results", width="stretch", icon=":material/sync:"):
                        if lookup_jid.strip():
                            try:
                                j_res = _fetch_backend_job(lookup_jid.strip())
                                st.session_state['active_results'] = j_res
                                if src_path_final and ref_path_final:
                                    st.session_state['active_src_path'] = src_path_final
                                    st.session_state['active_ref_path'] = ref_path_final
                                    from src.pipeline import robust_read, smart_downscale
                                    _s_raw = robust_read(src_path_final, max_dim=max_dim)
                                    _r_raw = robust_read(ref_path_final, max_dim=max_dim)
                                    _s_scaled, _ = smart_downscale(_s_raw, max_dim=max_dim)
                                    _r_scaled, _ = smart_downscale(_r_raw, max_dim=max_dim)
                                    st.session_state['active_img_src'] = _s_scaled
                                    st.session_state['active_img_ref'] = _r_scaled
                                st.success(f"✅ Job {lookup_jid.strip()[:8]} retrieved successfully!")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"❌ Failed to fetch job: {ex}")
                with c_bj3:
                    if st.button("Cancel Job", width="stretch", icon=":material/stop_circle:"):
                        if lookup_jid.strip():
                            if _cancel_backend_job(lookup_jid.strip()):
                                st.warning(f"🛑 Job {lookup_jid.strip()[:8]} cancelled.")
                                st.rerun()
                            else:
                                st.error("Could not cancel job.")

        run_pipeline = st.button(
            "Start Co-Registration Pipeline",
            type="primary",
            icon=":material/rocket_launch:",
            width="stretch"
        )

    # ── Pipeline Execution Routine ──
    if run_pipeline:
        if not (src_path_final and ref_path_final):
            st.warning("⚠️ Please provide both a Source and Reference image.")
            st.stop()

        if os.path.abspath(src_path_final) == os.path.abspath(ref_path_final) or _sha256_file(src_path_final) == _sha256_file(ref_path_final):
            st.error("Source and Reference images are identical. Select two different scenes before starting registration.")
            st.stop()

        if "OmniRoute" in engine_mode:
            active_matcher = "omniroute"
        elif "All-Engine Consensus" in engine_mode or "LunarFusion" in engine_mode:
            active_matcher = "fusion"
        elif "Autonomous" in engine_mode:
            active_matcher = "fusion"
        else:
            active_matcher = single_matcher_key

        if use_backend:
            if not _is_backend_online():
                st.warning("⚠️ FastAPI backend is not reachable at the configured URL. Reverting to local execution.")
            else:
                backend_matcher = active_matcher
                status_placeholder = st.empty()
                p_bar = st.progress(0.0, text=f"⚡ Dispatching {backend_matcher.upper()} to FastAPI backend...")
                try:
                    backend_results = _submit_registration_to_backend(
                        src_path_final, ref_path_final, backend_matcher,
                        max_dim=max_dim, use_tps=use_tps, speed_mode=speed_mode,
                        status_placeholder=status_placeholder, progress_bar=p_bar, timeout_seconds=900
                    )
                    st.session_state['active_results'] = backend_results
                    st.session_state['active_src_path'] = src_path_final
                    st.session_state['active_ref_path'] = ref_path_final
                    from src.pipeline import robust_read, smart_downscale
                    _s_raw = robust_read(src_path_final, max_dim=max_dim)
                    _r_raw = robust_read(ref_path_final, max_dim=max_dim)
                    _s_scaled, _ = smart_downscale(_s_raw, max_dim=max_dim)
                    _r_scaled, _ = smart_downscale(_r_raw, max_dim=max_dim)
                    st.session_state['active_img_src'] = _s_scaled
                    st.session_state['active_img_ref'] = _r_scaled
                    status_placeholder.success("✅ Backend job completed successfully.")
                    p_bar.progress(1.0, text="✅ Backend job completed successfully.")
                    history_mgr.add_entry(
                        src_path_final, ref_path_final,
                        backend_results.get('method_used', backend_matcher), backend_results,
                        {'max_dim': max_dim, 'use_tps': use_tps, 'matcher': backend_matcher, 'backend_job_id': backend_results.get('backend_job_id')}
                    )
                    st.rerun()
                except Exception as exc:
                    status_placeholder.empty()
                    p_bar.empty()
                    st.error(f"❌ Backend submission failed: {exc}")
                    st.stop()

        geom_info = {'src_inc': 45.0, 'src_ema': 0.0, 'src_pha': 45.0,
                     'ref_inc': 30.0, 'ref_ema': 10.0, 'ref_pha': 20.0} if enable_photo else None
        os.environ['APP_USE_PYRAMID'] = str(enable_pyramid)
        cfg.setdefault('pipeline', {})['enable_subpixel'] = enable_subpixel

        img_src, img_ref, results = None, None, None

        if "Autonomous Leaderboard" in engine_mode or "Autonomous Benchmark" in engine_mode:
            ALL_MODELS = [
                ("🌙", "LunarFusion (Adaptive AI)", "fusion"),
                ("🔴", "RoMa v2",          "roma_v2"),
                ("🔵", "LoFTR",            "loftr"),
                ("🟠", "YOLO11 (Crater)",   "yolo11"),
                ("🟢", "RootSIFT (Two-Tier)", "sift"),
                ("🟡", "DISK+LightGlue",   "lightglue"),
                ("🟣", "NI-LFM (NASA-IBM)","ni_lfm"),
            ]

            status_placeholder = st.empty()
            progress_bar = st.progress(0, text=f"🤖 Autonomous Benchmark — Initializing {len(ALL_MODELS)} engines...")

            model_results   = {}
            model_status    = {m[2]: "pending" for m in ALL_MODELS}
            model_inliers   = {m[2]: 0         for m in ALL_MODELS}
            model_rmse      = {m[2]: 999.0     for m in ALL_MODELS}

            def _render_status(current_msg=""):
                dot_map = {
                    "pending": '<span class="model-dot-pending"></span>',
                    "running": '<span class="model-dot-running"></span>',
                    "done":    '<span class="model-dot-done"></span>',
                    "failed":  '<span class="model-dot-failed"></span>',
                }
                status_labels = {
                    "pending": "<span style='color:#CBD5E1'>Waiting…</span>",
                    "running": "<span style='color:#F59E0B'>⚡ Running</span>",
                    "done":    "<span style='color:#10B981'>✅ Done</span>",
                    "failed":  "<span style='color:#F43F5E'>❌ Failed</span>",
                }
                rows_html = ""
                for emoji, label, key in ALL_MODELS:
                    st_tag = model_status[key]
                    inliers = model_inliers[key]
                    rmse    = model_rmse[key]
                    inlier_str = f"{inliers:,} inliers" if st_tag == "done" else ""
                    rmse_str   = f" · {rmse:.3f} px RMSE" if st_tag == "done" and rmse < 900 else ""
                    rows_html += f"""
                    <div class="model-status-row">
                        {dot_map[st_tag]}
                        <span style="color:#E2E8F0; min-width:180px">{emoji} {label}</span>
                        <span style="flex:1">{status_labels[st_tag]}</span>
                        <span style="color:#CBD5E1; font-size:0.78rem">{inlier_str}{rmse_str}</span>
                    </div>"""
                status_placeholder.markdown(f"""
                <div class="autonomous-status-card">
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700;
                                color:#C084FC; margin-bottom:10px; font-size:0.9rem;">
                        🤖 Autonomous Benchmark — Live Engine Status
                    </div>
                    {rows_html}
                    <div style="margin-top:10px; font-size:0.75rem; color:#CBD5E1">{current_msg}</div>
                </div>""", unsafe_allow_html=True)

            best_result = None
            best_img_src = best_img_ref = None
            best_inliers = -1
            best_model_key = None

            for step_i, (emoji, label, m_key) in enumerate(ALL_MODELS):
                model_status[m_key] = "running"
                _render_status(f"▶ Running {emoji} {label}…")
                progress_bar.progress(
                    (step_i) / len(ALL_MODELS),
                    text=f"🤖 Autonomous Benchmark: {emoji} {label} ({step_i+1}/{len(ALL_MODELS)})…"
                )

                try:
                    _img_src, _img_ref, _res = run_registration_pipeline(
                        src_path_final, ref_path_final,
                        use_tps=use_tps, max_dim=max_dim,
                        matcher_type=m_key,
                        geom_info=geom_info,
                        **metadata_kwargs,
                    )
                    inliers = _res.get("inlier_count", 0)
                    rmse    = _res.get("rmse_pixels", 999.0)
                    model_status[m_key]  = "done"
                    model_inliers[m_key] = inliers
                    model_rmse[m_key]    = rmse
                    model_results[m_key] = (_img_src, _img_ref, _res)

                    candidate_passed = bool(_res.get("quality", {}).get("passed", False))
                    candidate_certified = bool(_res.get("mission_grade_certified", False))
                    best_passed = bool(best_result and best_result.get("quality", {}).get("passed", False))
                    best_certified = bool(best_result and best_result.get("mission_grade_certified", False))
                    candidate_rank = (candidate_certified, candidate_passed, -float(rmse if rmse is not None else 999.0), inliers)
                    best_rank = (best_certified, best_passed, -float(model_rmse.get(best_model_key, 999.0)), best_inliers)
                    if candidate_rank > best_rank:
                        best_inliers   = inliers
                        best_result    = _res
                        best_img_src   = _img_src
                        best_img_ref   = _img_ref
                        best_model_key = m_key

                except Exception as _e:
                    model_status[m_key] = "failed"
                    _render_status(f"⚠️ {label} failed: {str(_e)[:80]}")

            _render_status(f"✅ All engines evaluated. Best result: {best_model_key} ({best_inliers:,} inliers)")
            progress_bar.progress(1.0, text="🤖 Autonomous Benchmark complete — best result auto-selected!")

            if best_result is None:
                st.error("❌ All matchers failed. Please check your images and try again.")
                st.stop()

            img_src  = best_img_src
            img_ref  = best_img_ref
            results  = best_result
            results["method_used"] = f"Autonomous [{best_model_key}]"

            # Leaderboard
            st.markdown("#### 🏆 Benchmark Evaluation Leaderboard")
            leaderboard_rows = sorted(
                [(k, model_inliers[k], model_rmse[k]) for k in model_inliers if model_status[k] == "done"],
                key=lambda x: (-x[1], x[2])
            )
            lb_html = ""
            medals = ["🥇", "🥈", "🥉", "4th", "5th", "6th", "7th"]
            label_map = {
                "fusion": "🌙 LunarFusion (Adaptive AI)",
                "roma_v2": "RoMa v2", "roma_omega": "RoMa v2", "roma": "RoMa v2", "loftr": "LoFTR",
                "yolo11": "🟠 YOLO11 (Crater)",
                "sift": "🟢 RootSIFT (Two-Tier)", "rootsift": "🟢 RootSIFT (Two-Tier)",
                "lightglue": "DISK+LightGlue", "ni_lfm": "NI-LFM (NASA-IBM)"
            }
            for rank, (k, inl, rmse) in enumerate(leaderboard_rows):
                highlight = "background: rgba(168,85,247,0.12); border-color: rgba(168,85,247,0.4);" if k == best_model_key else ""
                lb_html += f"""
                <div style="display:flex; align-items:center; gap:12px; padding:8px 14px;
                            border:1px solid rgba(255,255,255,0.07); border-radius:8px;
                            margin-bottom:6px; {highlight}">
                    <span style="font-size:1.1rem; min-width:28px">{medals[rank] if rank < len(medals) else '—'}</span>
                    <span style="color:#E2E8F0; font-weight:600; min-width:180px">{label_map.get(k, k)}</span>
                    <span style="color:#34D399; font-family:'JetBrains Mono',monospace; font-size:0.85rem">{inl:,} inliers</span>
                    <span style="color:#CBD5E1; font-family:'JetBrains Mono',monospace; font-size:0.85rem; margin-left:auto">{rmse:.3f} px RMSE</span>
                    {'<span style="background:#7C3AED; color:white; font-size:0.7rem; padding:2px 8px; border-radius:20px; margin-left:8px; font-weight:600;">AUTO-SELECTED</span>' if k == best_model_key else ''}
                </div>"""
            st.markdown(lb_html, unsafe_allow_html=True)

            history_mgr.add_entry(
                src_path_final, ref_path_final,
                results['method_used'], results,
                {'max_dim': max_dim, 'use_tps': use_tps, 'matcher': 'autonomous'}
            )

        else:
            if "OmniRoute" in engine_mode:
                active_matcher = "omniroute"
                mode_label = f"OmniRoute [{omni_strategy_key.upper()}]"
            elif "All-Engine Consensus" in engine_mode:
                active_matcher = "fusion"
                speed_mode = "accuracy"
                mode_label = "ISRO All-Engine Consensus"
            elif "LunarFusion" in engine_mode:
                active_matcher = "fusion"
                mode_label = f"LunarFusion ({speed_mode.title()})"
            else:
                active_matcher = single_matcher_key
                mode_label = f"Single ({active_matcher.upper()})"

            progress_bar = st.progress(0, text=f"⚡ Initializing registration orchestrator ({mode_label})...")

            try:
                pipeline_kwargs = {
                    "use_tps": use_tps,
                    "max_dim": max_dim,
                    "matcher_type": active_matcher,
                    "progress_callback": lambda s, t, m: progress_bar.progress(s / t, text=m),
                    "geom_info": geom_info,
                }
                pipeline_kwargs.update(metadata_kwargs)
                if active_matcher == "omniroute":
                    pipeline_kwargs["omni_strategy"] = omni_strategy_key
                    pipeline_kwargs["omni_workers"] = omni_workers
                    pipeline_kwargs["preferred_models"] = omni_selected_models
                elif active_matcher == "fusion":
                    pipeline_kwargs["speed_mode"] = speed_mode

                img_src, img_ref, results = run_registration_pipeline(
                    src_path_final, ref_path_final,
                    **pipeline_kwargs
                )

                progress_bar.progress(1.0, text=f"✅ {mode_label} completed successfully!")

                history_mgr.add_entry(
                    src_path_final, ref_path_final,
                    results['method_used'], results,
                    {'max_dim': max_dim, 'use_tps': use_tps, 'matcher': active_matcher, 'strategy': omni_strategy_key if active_matcher == 'omniroute' else ''}
                )

            except Exception as exc:
                st.error(f"❌ Registration failed: {exc}")
                with st.expander("🔍 Full traceback"):
                    st.code(traceback.format_exc())
                st.stop()

        # Persist active run into session_state so user interaction doesn't clear the results
        st.session_state['active_results'] = results
        st.session_state['active_img_src'] = img_src
        st.session_state['active_img_ref'] = img_ref
        st.session_state['active_src_path'] = src_path_final
        st.session_state['active_ref_path'] = ref_path_final

    # ── Step 3, 4, 5: Interactive Telemetry, Inspection, and Export Hub ──
    if st.session_state.get('active_results') is not None:
        c_rst1, c_rst2 = st.columns([5, 1], vertical_alignment="center")
        with c_rst2:
            if st.button("🔄 Clear Results", width="stretch"):
                st.session_state['active_results'] = None
                st.session_state['active_img_src'] = None
                st.session_state['active_img_ref'] = None
                st.session_state['active_src_path'] = None
                st.session_state['active_ref_path'] = None
                st.rerun()

        render_registration_results(
            st.session_state.get('active_results'),
            st.session_state.get('active_img_src'),
            st.session_state.get('active_img_ref'),
            st.session_state.get('active_src_path', src_path_final),
            st.session_state.get('active_ref_path', ref_path_final)
        )

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(6, 214, 160, 0.12) 0%, rgba(14, 165, 233, 0.08) 100%); 
                    border: 1px solid rgba(6, 214, 160, 0.45); 
                    border-radius: 12px; 
                    padding: 14px 18px; 
                    margin: 18px 0 10px 0;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-weight: 700; color: #06D6A0; font-size: 0.92rem; display: flex; align-items: center; gap: 6px;">
                        ⚡ Zero-Friction Registration-to-3D Auto-Binding Ready
                    </span>
                    <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px;">
                        These registered stereo frames and verified tie-points are locked in session memory. 
                        Switch to <strong>Tab 5: 3D Topography &amp; DEM</strong> to generate 3D point clouds and Blender renders immediately without downloading or re-uploading files.
                    </div>
                </div>
                <span style="background: rgba(6, 214, 160, 0.2); border: 1px solid #06D6A0; color: #06D6A0; font-size: 0.72rem; font-weight: 700; padding: 4px 10px; border-radius: 16px; white-space: nowrap;">
                    SESSION LOCKED
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
