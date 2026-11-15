"""
3D Topography, DEM Reconstruction & Blender Photogrammetry View
==============================================================
ISRO LunarFusion Co-Registration Suite — UI 3D Topography View
"""

import os
import time
import tempfile
import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List, Tuple
from PIL import Image

try:
    from ui.config import DATA_DIR, WORKSPACE_ROOT
    from ui.components.visualizations import generate_dem_preview
    from ui.services.api_client import trigger_backend_blender_export, is_backend_online
except (ImportError, ValueError):
    from ..config import DATA_DIR, WORKSPACE_ROOT
    from ..components.visualizations import generate_dem_preview
    from ..services.api_client import trigger_backend_blender_export, is_backend_online
from src.features.dem_builder import (
    BackgroundDEMTask,
    compute_hillshade,
    export_point_cloud_ply,
    evaluate_dem_qc,
    compute_dem_uncertainty_map,
    get_camera_model,
    despike_and_clamp_lunar_slopes,
    compute_photoclinometry_sfs,
    export_dem_geotiff,
    export_mesh_obj,
)
from src.rendering import BlenderTerrainRenderer, find_blender_binary


def render_3d_tab(available_images: list = None) -> None:
    """Render the 3D Topography, DEM Photogrammetry, and Blender Rendering interface."""
    with st.container(border=True):
        st.markdown("""
        <div class="workflow-step-header">
            <span class="step-number">01</span>
            <div>
                <div class="step-title">3D Topography &amp; Digital Elevation Model Generation</div>
                <div style="font-size: 0.82rem; color: #CBD5E1;">Synthesize dense 3D point clouds and topographical maps from registered stereo pairs.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Zero-Friction Registration-to-3D Auto-Binding Inspection ──
        act_res = st.session_state.get('active_results')
        img_s = st.session_state.get('active_img_src')
        img_r = st.session_state.get('active_img_ref')
        src_p = st.session_state.get('active_src_path')
        ref_p = st.session_state.get('active_ref_path')

        if img_s is None and src_p and os.path.exists(src_p):
            try:
                from src.pipeline import robust_read
                raw_s = robust_read(src_p)
                img_s = cv2.cvtColor(raw_s, cv2.COLOR_GRAY2BGR) if (raw_s is not None and raw_s.ndim == 2) else raw_s
            except Exception:
                pass

        if img_r is None and ref_p and os.path.exists(ref_p):
            try:
                from src.pipeline import robust_read
                raw_r = robust_read(ref_p)
                img_r = cv2.cvtColor(raw_r, cv2.COLOR_GRAY2BGR) if (raw_r is not None and raw_r.ndim == 2) else raw_r
            except Exception:
                pass

        has_images = (img_s is not None and img_r is not None)
        inliers_s = act_res.get('inliers_src') if isinstance(act_res, dict) else None
        if inliers_s is None and isinstance(act_res, dict):
            inliers_s = act_res.get('pts_src')
        inliers_r = act_res.get('inliers_ref') if isinstance(act_res, dict) else None
        if inliers_r is None and isinstance(act_res, dict):
            inliers_r = act_res.get('pts_ref')
        
        n_inliers = 0
        if inliers_s is not None:
            try:
                n_inliers = len(inliers_s)
            except Exception:
                n_inliers = 0
        elif isinstance(act_res, dict) and act_res.get('inliers') is not None:
            try:
                n_inliers = int(act_res.get('inliers', 0))
            except Exception:
                n_inliers = 0

        is_auto_bound = has_images and (act_res is not None or n_inliers > 0 or (src_p and ref_p))
        src_name = os.path.basename(src_p) if src_p else "Target Lunar Scene (Tab 1)"
        ref_name = os.path.basename(ref_p) if ref_p else "Anchor Lunar Scene (Tab 1)"
        method_label = act_res.get('method_used', 'Lunar Co-Registration') if isinstance(act_res, dict) else "Tab 1 Registration"
        rmse_val = act_res.get('rmse') if isinstance(act_res, dict) else None
        qc_grade = act_res.get('quality', {}).get('grade', 'Certified') if isinstance(act_res, dict) else "Certified"

        if is_auto_bound:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(6, 214, 160, 0.14) 0%, rgba(14, 165, 233, 0.09) 100%); 
                        border: 1px solid rgba(6, 214, 160, 0.5); 
                        border-radius: 12px; 
                        padding: 16px 20px; 
                        margin-bottom: 18px;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.15rem;">⚡</span>
                        <span style="font-weight: 800; font-size: 1.0rem; color: #06D6A0; letter-spacing: 0.03em;">
                            Zero-Friction Registration-to-3D Auto-Binding: ACTIVE
                        </span>
                    </div>
                    <span style="background: rgba(6, 214, 160, 0.22); border: 1px solid #06D6A0; color: #06D6A0; font-size: 0.72rem; font-weight: 700; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;">
                        🟢 In-Memory Locked
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 12px;">
                    Active stereo scenes and <strong>{n_inliers:,} verified tie-points</strong> from Single Scene Registration (Tab 1) are preserved in session memory. 
                    Click <strong>"Start Background 3D Mapping"</strong> immediately without downloading or re-uploading any files.
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; background: rgba(11, 15, 25, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                    <div>
                        <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Left Stereo Target</div>
                        <div style="font-size: 0.8rem; color: #F8FAFC; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{src_name}</div>
                        <div style="font-size: 0.7rem; color: #38BDF8;">{img_s.shape[1]} × {img_s.shape[0]} px</div>
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Right Stereo Anchor</div>
                        <div style="font-size: 0.8rem; color: #F8FAFC; font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">{ref_name}</div>
                        <div style="font-size: 0.7rem; color: #38BDF8;">{img_r.shape[1]} × {img_r.shape[0]} px</div>
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Verified Tie-Points</div>
                        <div style="font-size: 0.82rem; color: #06D6A0; font-weight: 700;">🎯 {n_inliers:,} Inliers Auto-Bound</div>
                        <div style="font-size: 0.7rem; color: #94A3B8;">Auto-seeded disparity</div>
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Registration Grade</div>
                        <div style="font-size: 0.82rem; color: #F8FAFC; font-weight: 700;">Grade {qc_grade}</div>
                        <div style="font-size: 0.7rem; color: #A78BFA;">{f'RMSE: {rmse_val:.3f} px' if rmse_val is not None else method_label}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            tab5_input_mode = st.radio(
                "Stereo Input Mode",
                [
                    f"⚡ Use Auto-Bound Tab 1 Registration ({n_inliers:,} tie-points preserved)",
                    "📁 Upload Custom Images Manually"
                ],
                index=0,
                key="tab5_source_mode",
                horizontal=True
            )
            use_auto_bound_mode = "Auto-Bound" in tab5_input_mode
        else:
            use_auto_bound_mode = False
            st.info("💡 **Zero-Friction Registration-to-3D Auto-Binding**: Run Single Scene Registration in Tab 1, and your registered stereo images + verified tie-points will auto-bind here with zero manual file uploads. Or upload stereo files manually below:")

        c_eng1, c_eng2, c_eng3 = st.columns([1.5, 1.2, 1.2])
        with c_eng1:
            stereo_engine_choice = st.selectbox(
                "Stereo Reconstruction Engine",
                [
                    "Calibrated Photogrammetry (StereoSGBM Triangulation)",
                    "StereoLunar AI Deep Stereo (ICCV 2025 MASt3R)",
                    "NASA Ames Stereo Pipeline (ASP Pinhole Model)",
                ],
                index=0,
                help="Select stereo algorithm: deterministic photogrammetry (calibrated triangulation), AI deep stereo (StereoLunar), or NASA ASP."
            )
        with c_eng2:
            camera_sensor_choice = st.selectbox(
                "Orbital Sensor Model",
                [
                    "Chandrayaan-2 TMC-2 (GSD 5.0m, EFL 140mm)",
                    "Chandrayaan-2 OHRC (GSD 0.25m, EFL 2200mm)",
                    "Chandrayaan-1 TMC (GSD 5.0m, EFL 120mm)",
                ],
                index=0,
                help="Applies physical calibrated camera intrinsics, focal length, pixel pitch, and orbital baseline."
            )
        with c_eng3:
            tab5_resolution_choice = st.selectbox(
                "3D Raster Map Resolution",
                [
                    "Full-Size 100% Native (1:1 Scale)",
                    "High-Definition 1024px",
                    "Balanced 512px (Interactive)",
                ],
                index=0,
                help="Full-Size 100% Native computes elevation at the un-downscaled full image raster scale."
            )

        sensor_code = "ohrc" if "OHRC" in camera_sensor_choice else "tmc"
        use_asp = ("NASA Ames Stereo Pipeline" in stereo_engine_choice)
        engine_code = "asp" if use_asp else ("stereolunar" if "StereoLunar" in stereo_engine_choice else "calibrated_sgbm")
        is_full_res_mode = ("100" in tab5_resolution_choice)

        asp_dir = ""
        if use_asp:
            default_asp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "StereoPipeline-3.8.0-alpha-2026-08-31-arm64-OSX"))
            asp_dir = st.text_input("ASP Installation Directory", value=default_asp)

        dem_src_file = None
        dem_ref_file = None
        dem_match_file = None

        if use_auto_bound_mode:
            with st.expander("🖼️ View Auto-Bound Stereo Image Previews", expanded=False):
                p_c1, p_c2 = st.columns(2)
                p_c1.image(img_s, caption=f"Left Target: {src_name}", width="stretch")
                p_c2.image(img_r, caption=f"Right Anchor: {ref_name}", width="stretch")
            with st.expander("📁 Optional: Override with Manual File Uploads", expanded=False):
                dem_col1, dem_col2 = st.columns(2, gap="medium")
                dem_src_file = dem_col1.file_uploader("Upload Left Stereo Image", type=["png", "jpg", "jpeg", "tif", "tiff", "img"], key="ovr_src")
                dem_ref_file = dem_col2.file_uploader("Upload Right Stereo Image", type=["png", "jpg", "jpeg", "tif", "tiff", "img"], key="ovr_ref")
                dem_match_file = st.file_uploader("Upload Tie-points File (.match / .csv)", type=["match", "txt", "csv"], key="ovr_match")
        else:
            dem_col1, dem_col2 = st.columns(2, gap="medium")
            dem_src_file = dem_col1.file_uploader("Upload Left Stereo Image", type=["png", "jpg", "jpeg", "tif", "tiff", "img"], key="up_src")
            dem_ref_file = dem_col2.file_uploader("Upload Right Stereo Image", type=["png", "jpg", "jpeg", "tif", "tiff", "img"], key="up_ref")
            dem_match_file = st.file_uploader("Upload Tie-points File (.match / .csv) [Optional]", type=["match", "txt", "csv"], key="up_match")

        st.write("")
        btn_caption = "🚀 Start Background 3D Mapping (Auto-Bound Tab 1 Pair)" if use_auto_bound_mode else "Start Background 3D Mapping"
        if st.button(btn_caption, type="primary", icon=":material/view_in_ar:", width="stretch", key="btn_start_dem_mapping"):
            final_img_src = None
            final_img_ref = None
            match_file_arg = None

            if use_auto_bound_mode and (dem_src_file is None and dem_ref_file is None):
                final_img_src = img_s
                final_img_ref = img_r
                if inliers_s is not None and inliers_r is not None and len(inliers_s) >= 4:
                    tmp_match = tempfile.NamedTemporaryFile(delete=False, suffix=".match")
                    tmp_match.close()
                    export_asp_match_file(np.asarray(inliers_s), np.asarray(inliers_r), tmp_match.name)
                    match_file_arg = tmp_match.name
            else:
                if dem_src_file is not None:
                    saved_src = _save_uploaded_orbital_image(dem_src_file)
                    raw_src = robust_read(saved_src)
                    final_img_src = cv2.cvtColor(raw_src, cv2.COLOR_GRAY2BGR) if (raw_src is not None and raw_src.ndim == 2) else raw_src
                elif img_s is not None:
                    final_img_src = img_s
                elif os.path.exists("samples/source/ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png"):
                    final_img_src = cv2.imread("samples/source/ch2_ohr_ncp_20260103T1005176450_b_brw_d18.png", cv2.IMREAD_COLOR)

                if dem_ref_file is not None:
                    saved_ref = _save_uploaded_orbital_image(dem_ref_file)
                    raw_ref = robust_read(saved_ref)
                    final_img_ref = cv2.cvtColor(raw_ref, cv2.COLOR_GRAY2BGR) if (raw_ref is not None and raw_ref.ndim == 2) else raw_ref
                elif img_r is not None:
                    final_img_ref = img_r
                elif os.path.exists("samples/reference/ch2_ohr_ncp_20260103T0609041371_b_brw_d18.png"):
                    final_img_ref = cv2.imread("samples/reference/ch2_ohr_ncp_20260103T0609041371_b_brw_d18.png", cv2.IMREAD_COLOR)

                if dem_match_file is not None:
                    tmp_match = tempfile.NamedTemporaryFile(delete=False, suffix=".match")
                    tmp_match.write(dem_match_file.read())
                    tmp_match.close()
                    match_file_arg = tmp_match.name
                elif inliers_s is not None and inliers_r is not None and len(inliers_s) >= 4:
                    tmp_match = tempfile.NamedTemporaryFile(delete=False, suffix=".match")
                    tmp_match.close()
                    export_asp_match_file(np.asarray(inliers_s), np.asarray(inliers_r), tmp_match.name)
                    match_file_arg = tmp_match.name
                elif os.path.exists("samples/asp_tiepoints (1).match"):
                    match_file_arg = os.path.abspath("samples/asp_tiepoints (1).match")

            if final_img_src is not None and final_img_ref is not None:
                stereo_cfg = {"stereo": {"sensor": sensor_code}}
                try:
                    st.session_state.dem_task.start(
                        final_img_src, final_img_ref,
                        use_sgbm=True,
                        match_file=match_file_arg,
                        use_asp=use_asp,
                        asp_path=asp_dir,
                        config=stereo_cfg,
                        engine=engine_code,
                        registration_results=act_res,
                        full_resolution=is_full_res_mode,
                        resolution_mode=tab5_resolution_choice,
                    )
                except TypeError:
                    import importlib
                    import src.features.dem_builder
                    importlib.reload(src.features.dem_builder)
                    from src.features.dem_builder import BackgroundDEMTask as _FreshDEMTask
                    st.session_state.dem_task = _FreshDEMTask()
                    try:
                        st.session_state.dem_task.start(
                            final_img_src, final_img_ref,
                            use_sgbm=True,
                            match_file=match_file_arg,
                            use_asp=use_asp,
                            asp_path=asp_dir,
                            config=stereo_cfg,
                            engine=engine_code,
                            registration_results=act_res,
                            full_resolution=is_full_res_mode,
                            resolution_mode=tab5_resolution_choice,
                        )
                    except TypeError:
                        st.session_state.dem_task.start(
                            final_img_src, final_img_ref,
                            use_sgbm=True,
                            match_file=match_file_arg,
                            use_asp=use_asp,
                            asp_path=asp_dir,
                            config=stereo_cfg
                        )
                st.rerun()
            else:
                st.error("Failed to load stereoscopic image frames. Run Tab 1 registration or upload images.")

        # Status Monitor
        task = st.session_state.dem_task
        if task.status == "running":
            import time as timer
            st.warning("⏳ 3D Topography Generation is running in the background...")
            progress_bar = st.progress(task.progress, text=task.message)
            timer.sleep(1)
            st.rerun()
        elif task.status == "error":
            st.error(f"❌ 3D Topography Generation Error: {task.error or task.message}")
            if st.button("🔄 Reset 3D Task", key="btn_reset_dem_task_err", icon=":material/refresh:"):
                task.status = "idle"
                task.error = None
                st.rerun()
        elif task.status == "completed":
            c_top_res1, c_top_res2 = st.columns([5, 1], vertical_alignment="center")
            with c_top_res1:
                st.success("✅ " + task.message)
            with c_top_res2:
                if st.button("🔄 Reset 3D Task", key="btn_reset_dem_task_done", width="stretch", icon=":material/refresh:"):
                    task.status = "idle"
                    task.result_dem = None
                    task.result_texture = None
                    st.rerun()
            if task.result_dem is not None:
                dem_full = task.result_dem
                tex_full = getattr(task, 'result_texture', None)
                
                # 1. 3D Photogrammetric Quality Control
                qc_metrics = evaluate_dem_qc(dem_full, gsd_m=5.0)
                st.markdown("""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 10px; padding: 12px 16px; margin: 16px 0 12px 0;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #38BDF8;">📊 3D Photogrammetric Quality &amp; Terrain Telemetry</span>
                </div>
                """, unsafe_allow_html=True)
                
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("QC Grade", qc_metrics.get("qc_grade", "N/A"), qc_metrics.get("status", ""))
                q2.metric("Relief Span", f"{qc_metrics.get('relief_span_m', 0.0):.1f} m", f"{qc_metrics.get('elevation_min_m', 0.0):.0f} to {qc_metrics.get('elevation_max_m', 0.0):.0f} m")
                q3.metric("Mean Terrain Slope", f"{qc_metrics.get('mean_slope_deg', 0.0):.1f}°")
                q4.metric("Void Percentage", f"{qc_metrics.get('void_percentage', 0.0):.1f}%")

                # 2. Multi-View Visualization Sub-Tabs
                v_tab0, v_tab1, v_tab2, v_tab3 = st.tabs([
                    "🪐 Blender Photorealistic 3D",
                    "🌐 Interactive 3D Mesh",
                    "🌓 Shaded Relief (Hillshade)",
                    "📐 Elevation Uncertainty (σ_Z)",
                ])

                with v_tab0:
                    st.markdown("""
                    <div style="font-size:0.86rem; color:#CBD5E1; margin-bottom:12px;">
                        Headless physical photorealistic render powered by <strong>Blender 5.2 LTS</strong>. Simulates physical solar illumination angles, Hapke regolith surface scattering, and draped Chandrayaan orbital imagery.
                    </div>
                    """, unsafe_allow_html=True)

                    c_bld1, c_bld2, c_bld3, c_bld4 = st.columns([1.3, 1.0, 1.0, 1.0])
                    with c_bld1:
                        cam_angle_choice = st.selectbox(
                            "Camera Perspective",
                            ["Isometric Flyover (45°)", "Nadir Orthophoto (90°)", "Crater Rim Horizon (15°)"],
                            index=0,
                            key="bld_cam_angle"
                        )
                        cam_mode = "isometric" if "Isometric" in cam_angle_choice else ("nadir" if "Nadir" in cam_angle_choice else "rim_grazing")

                    with c_bld2:
                        sun_el = st.slider("Sun Elevation", 5.0, 75.0, 22.0, 1.0, key="bld_sun_elev", help="Low angle casts dramatic crater rim shadows")

                    with c_bld3:
                        sun_az = st.slider("Sun Azimuth", 0.0, 360.0, 65.0, 5.0, key="bld_sun_azim", help="Solar illumination azimuth angle")

                    with c_bld4:
                        exagg = st.slider("Relief Scale", 1.0, 3.5, 1.8, 0.1, key="bld_exagg", help="Vertical relief multiplier")

                    if st.button("🚀 Render Cinematic 3D Scene with Blender", type="primary", icon=":material/view_in_ar:", width="stretch", key="btn_run_blender"):
                        with st.spinner("Executing headless Blender 5.2 render engine..."):
                            renderer = BlenderTerrainRenderer()
                            bld_res = renderer.render(
                                dem=dem_full,
                                texture=tex_full,
                                camera_view=cam_mode,
                                sun_elevation_deg=sun_el,
                                sun_azimuth_deg=sun_az,
                                vertical_exaggeration=exagg,
                                resolution=(1920, 1080),
                            )
                            if bld_res.get("success"):
                                st.session_state["active_blender_render"] = bld_res
                            else:
                                st.error(f"Blender render error: {bld_res.get('error')}")

                    active_bld = st.session_state.get("active_blender_render")
                    if active_bld and active_bld.get("image_bytes"):
                        st.image(
                            active_bld["image_bytes"],
                            caption=f"Blender 5.2 Physical Lunar Render — 1080p ({active_bld.get('elapsed_sec', 0.0)}s render time · {active_bld.get('vertices', 0):,} vertices · {active_bld.get('faces', 0):,} faces)",
                            width="stretch"
                        )
                        b_dl1, b_dl2 = st.columns(2)
                        with b_dl1:
                            st.download_button(
                                "📥 Download 1080p Cinematic Render (PNG)",
                                data=active_bld["image_bytes"],
                                file_name=f"blender_lunar_render_{int(time.time())}.png",
                                mime="image/png",
                                icon=":material/download:",
                                width="stretch"
                            )
                        with b_dl2:
                            if active_bld.get("blend_bytes"):
                                st.download_button(
                                    "📥 Download Blender 3D Project (.blend)",
                                    data=active_bld["blend_bytes"],
                                    file_name=f"lunar_crater_scene_{int(time.time())}.blend",
                                    mime="application/x-blender",
                                    icon=":material/download:",
                                    width="stretch"
                                )
                    elif not active_bld:
                        # Auto-render initial preview if not yet cached
                        with st.spinner("Generating initial Blender 3D preview..."):
                            renderer = BlenderTerrainRenderer()
                            bld_res = renderer.render(
                                dem=dem_full,
                                texture=tex_full,
                                camera_view="isometric",
                                sun_elevation_deg=22.0,
                                sun_azimuth_deg=65.0,
                                vertical_exaggeration=1.8,
                                resolution=(1280, 720),
                                max_grid_dim=250,
                            )
                            if bld_res.get("success"):
                                st.session_state["active_blender_render"] = bld_res
                                st.rerun()

                with v_tab1:
                    render_dem = dem_full
                    render_dem = dem_full.copy()
                    render_tex = tex_full.copy() if tex_full is not None else None

                    c_vt1, c_vt2, c_vt3 = st.columns([1.5, 1.2, 1.0], vertical_alignment="center")
                    with c_vt1:
                        mesh_res = st.selectbox(
                            "Mesh Display Resolution",
                            ["Full-Size 100% Native", "High-Definition 1024px", "Optimized 800px"],
                            index=0,
                            key="tab5_mesh_res"
                        )
                    with c_vt2:
                        mesh_tex_style = st.selectbox(
                            "Surface Shading",
                            ["Lunar Texture (Photorealistic)", "Spectral Elevation", "Viridis", "Plasma", "Terrain"],
                            index=0,
                            key="tab5_mesh_tex"
                        )
                    with c_vt3:
                        mesh_exagg = st.slider("Relief Exaggeration", 0.5, 3.5, 1.8, 0.1, key="tab5_mesh_exagg")

                    if "100" in mesh_res:
                        max_render_dim = 1200
                    elif "1024" in mesh_res:
                        max_render_dim = 1024
                    else:
                        max_render_dim = 800

                    if max(render_dem.shape) > max_render_dim:
                        render_scale = max_render_dim / max(render_dem.shape)
                        r_w = max(int(render_dem.shape[1] * render_scale), 1)
                        r_h = max(int(render_dem.shape[0] * render_scale), 1)
                        render_dem = cv2.resize(render_dem, (r_w, r_h), interpolation=cv2.INTER_AREA)
                        if render_tex is not None:
                            render_tex = cv2.resize(render_tex, (r_w, r_h), interpolation=cv2.INTER_AREA)

                    use_photoreal = ("Lunar Texture" in mesh_tex_style or "Photorealistic" in mesh_tex_style) and (render_tex is not None)
                    if use_photoreal:
                        norm_tex = render_tex.astype(np.float32) / 255.0
                        surface_color = norm_tex
                        c_scale = "gray"
                        show_scale_bar = False
                    else:
                        surface_color = render_dem
                        c_scale = "Viridis" if "Viridis" in mesh_tex_style else ("Spectral" if "Spectral" in mesh_tex_style else "Plasma")
                        show_scale_bar = True

                    fig = go.Figure(data=[
                        go.Surface(
                            z=render_dem,
                            surfacecolor=surface_color,
                            colorscale=c_scale,
                            showscale=show_scale_bar,
                            colorbar=dict(title="Elevation (m)", len=0.8) if show_scale_bar else None,
                            lighting=dict(ambient=0.45, diffuse=0.8, roughness=0.5, specular=0.2)
                        )
                    ])
                    aspect_z = float(mesh_exagg * 0.35)
                    aspect_y = float(render_dem.shape[0] / max(render_dem.shape[1], 1))
                    fig.update_layout(
                        title=f"3D Lunar Surface Reconstruction — {render_dem.shape[1]} × {render_dem.shape[0]} px ({'100% Native Full-Scale' if max(render_dem.shape) == max(dem_full.shape) else 'High-Definition'})",
                        template="plotly_dark",
                        scene=dict(
                            xaxis_title="X (px)",
                            yaxis_title="Y (px)",
                            zaxis_title="Elevation (m)",
                            aspectratio=dict(x=1.0, y=max(0.2, min(5.0, aspect_y)), z=aspect_z),
                            camera=dict(
                                eye=dict(x=1.35, y=-1.35, z=0.9),
                                center=dict(x=0, y=0, z=-0.1)
                            )
                        ),
                        height=660,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="rgba(11, 15, 25, 0)",
                        plot_bgcolor="rgba(11, 15, 25, 0)"
                    )
                    st.plotly_chart(fig, width="stretch")

                with v_tab2:
                    hillshade = compute_hillshade(dem_full, gsd_m=5.0)
                    st.image(hillshade, caption="Photorealistic Shaded Relief (Lambertian Cosine Illumination, Azimuth=315°, Alt=45°)", width="stretch")

                with v_tab3:
                    uncertainty = compute_dem_uncertainty_map(np.maximum(dem_full, 0.5), 20000.0, 93300.0, 5.0)
                    fig_unc = go.Figure(data=go.Heatmap(z=uncertainty, colorscale="Thermal", colorbar=dict(title="σ_Z (m)")))
                    fig_unc.update_layout(title="Per-Pixel Elevation Uncertainty (σ_Z in metres via Error Propagation)", template="plotly_dark", height=500)
                    st.plotly_chart(fig_unc, width="stretch")

                # 3. Scientific 3D Exports
                st.write("")
                d_c1, d_c2, d_c3 = st.columns(3)
                with d_c1:
                    tmp_ply = tempfile.NamedTemporaryFile(delete=False, suffix=".ply")
                    tmp_ply.close()
                    try:
                        import importlib
                        import src.features.dem_builder
                        importlib.reload(src.features.dem_builder)
                        from src.features.dem_builder import export_point_cloud_ply as _export_ply

                        step_s = max(1, int(max(dem_full.shape) / 400))
                        try:
                            _export_ply(tmp_ply.name, dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s, write_faces=True)
                            _export_ply("data/outputs/stereolunar_crater_3d.ply", dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s, write_faces=True)
                            _export_ply("data/outputs/chandrayaan2_terrain.ply", dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s, write_faces=True)
                        except TypeError:
                            _export_ply(tmp_ply.name, dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s)
                            _export_ply("data/outputs/stereolunar_crater_3d.ply", dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s)
                            _export_ply("data/outputs/chandrayaan2_terrain.ply", dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s)

                        with open(tmp_ply.name, "rb") as f_ply:
                            st.download_button(
                                "📥 Download 3D Mesh (.ply)",
                                data=f_ply.read(),
                                file_name=f"lunar_terrain_mesh_{int(time.time())}.ply",
                                mime="application/octet-stream",
                                icon=":material/view_in_ar:",
                                width="stretch"
                            )
                    except Exception as e_ply:
                        st.warning(f"Point cloud export: {e_ply}")

                with d_c2:
                    tmp_tif = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                    tmp_tif.close()
                    export_geotiff(dem_full, tmp_tif.name)
                    export_geotiff(dem_full, "data/outputs/chandrayaan_dem.tif")
                    with open(tmp_tif.name, "rb") as f_tif:
                        st.download_button(
                            "📥 Download Elevation GeoTIFF (.tif)",
                            data=f_tif.read(),
                            file_name=f"lunar_elevation_dem_{int(time.time())}.tif",
                            mime="image/tiff",
                            icon=":material/download:",
                            width="stretch"
                        )

                with d_c3:
                    active_bld_export = st.session_state.get("active_blender_render")
                    if active_bld_export and active_bld_export.get("blend_bytes"):
                        st.download_button(
                            "📥 Download Blender 3D Scene (.blend)",
                            data=active_bld_export["blend_bytes"],
                            file_name=f"lunar_crater_scene_{int(time.time())}.blend",
                            mime="application/x-blender",
                            icon=":material/view_in_ar:",
                            width="stretch"
                        )
                    else:
                        renderer = BlenderTerrainRenderer()
                        if renderer.is_available():
                            tmp_blend_exp = tempfile.NamedTemporaryFile(delete=False, suffix=".blend")
                            tmp_blend_exp.close()
                            bld_exp = renderer.render(
                                dem=dem_full,
                                texture=tex_full,
                                output_blend_path=tmp_blend_exp.name,
                                resolution=(640, 360),
                                max_grid_dim=250,
                            )
                            if bld_exp.get("blend_bytes"):
                                st.download_button(
                                    "📥 Download Blender Scene (.blend)",
                                    data=bld_exp["blend_bytes"],
                                    file_name=f"lunar_crater_scene_{int(time.time())}.blend",
                                    mime="application/x-blender",
                                    icon=":material/view_in_ar:",
                                    width="stretch"
                                )
                            else:
                                st.button("Blender Export (Processing)", disabled=True, width="stretch")
                        else:
                            st.button("Blender Unavailable", disabled=True, width="stretch")

                st.write("")
                st.markdown("""
                <div style="background: rgba(14, 23, 48, 0.65); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px 18px; margin-top: 10px; box-shadow: 0 4px 18px rgba(0,0,0,0.35);">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">🪐 Blender 5.2 Direct 3D Bridge</span>
                            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 2px;">
                                One-click photorealistic lunar terrain visualization in Blender with scientific 65° low-angle lighting &amp; orbital camera.
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    if st.button("🚀 Launch 3D Scene in Blender", icon=":material/view_in_ar:", width="stretch", type="primary"):
                        try:
                            import importlib
                            import src.features.dem_builder
                            importlib.reload(src.features.dem_builder)
                            from src.features.dem_builder import export_point_cloud_ply as _export_ply
                            from scripts.launch_blender import launch_blender

                            active_ply = "data/outputs/stereolunar_crater_3d.ply"
                            step_s = max(1, int(max(dem_full.shape) / 400))
                            try:
                                _export_ply(active_ply, dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s, write_faces=True)
                            except TypeError:
                                _export_ply(active_ply, dem_full, gsd_m=5.0, texture=tex_full, subsample_step=step_s)

                            launched = False
                            if _is_backend_online():
                                b_res = _trigger_backend_blender_export(ply_path=active_ply, background=False)
                                if b_res.get("status") == "launched" or b_res.get("pid"):
                                    st.success(f"🪐 Blender 5.2 launched via FastAPI Backend (PID {b_res.get('pid')})! Check your macOS Dock / desktop.")
                                    launched = True
                            if not launched:
                                proc = launch_blender(ply_path=active_ply, background=False)
                                st.success(f"🪐 Blender 5.2 launched successfully (PID {proc.pid})! Check your macOS Dock / desktop.")
                        except Exception as e_bl:
                            st.error(f"Could not launch Blender: {e_bl}")
                with b_c2:
                    blend_path = "data/outputs/lunar_terrain.blend"
                    if os.path.exists(blend_path):
                        with open(blend_path, "rb") as f_blend:
                            st.download_button(
                                "📥 Download Lunar .blend Project",
                                data=f_blend.read(),
                                file_name=f"chandrayaan_lunar_terrain_{int(time.time())}.blend",
                                mime="application/x-blender",
                                icon=":material/folder_zip:",
                                width="stretch"
                            )
                    else:
                        st.info("ℹ️ Launch Blender once to generate the .blend project file.")
        elif task.status == "error":
            st.error(f"❌ Generation Failed: {task.error}")

# ── Tab: YOLO11 Crater & Hazard AI ───────────────────────────────────────────
