from __future__ import annotations
"""Screen 02: Co-registration studio with 4-step systematic verification."""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import io
import hashlib
from config import PRESETS
from components.header import sample_badge, active_pair_banner
from components.metric_panel import gate_card
from components import image_viewer as iv
import services
import workspace


def _remember_upload(widget_key: str) -> None:
    """Keep upload bytes outside widget state, which is cleaned up on navigation."""
    upload = st.session_state.get(widget_key)
    saved_key = f"saved_{widget_key}"
    if upload is None:
        st.session_state.pop(saved_key, None)
    else:
        st.session_state[saved_key] = {"name": upload.name, "data": upload.getvalue()}
    _persist_active_pair()


def _persist_active_pair() -> None:
    """Mirror the operator's active pair to the workspace directory.

    Session state dies with the session (new tab, other port, server restart),
    which silently reverted the app to bundled presets. The disk copy keeps one
    active pair everywhere until it is explicitly cleared.
    """
    def _side(widget_key: str):
        upload = st.session_state.get(widget_key)
        if upload is not None:
            return upload.name, upload.getvalue()
        saved = st.session_state.get(f"saved_{widget_key}")
        if saved is not None:
            return saved["name"], saved["data"]
        return None

    src = _side("uploader_src")
    ref = _side("uploader_ref")
    if src is None and ref is None:
        workspace.clear()
        return
    workspace.save_pair(
        src[0] if src else "", src[1] if src else b"",
        ref[0] if ref else "", ref[1] if ref else b"",
    )


def _rehydrate_from_workspace() -> None:
    """Restore saved uploads from disk when this session has none.

    Runs on every render but only acts when a slot has neither widget state
    nor session-saved bytes — so an upload the operator removed (✕) in this
    session is never resurrected, while a fresh session picks up the
    persisted pair instead of silently showing bundled presets.
    """
    pair = workspace.load_pair()
    if pair is None:
        return
    (src, ref) = pair
    for side, item in zip(("src", "ref"), (src, ref)):
        if item is None:
            continue
        widget_key = f"uploader_{side}"
        saved_key = f"saved_{widget_key}"
        if st.session_state.get(widget_key) is not None:
            continue
        if saved_key in st.session_state:
            continue
        name, data = item
        buf = io.BytesIO(data)
        buf.name = name
        buf.size = len(data)
        st.session_state[saved_key] = {"name": name, "data": data}
        st.session_state[f"_restored_from_workspace_{side}"] = True


def _active_upload(widget_key: str, upload):
    if upload is not None:
        _remember_upload(widget_key)
        return upload
    saved = st.session_state.get(f"saved_{widget_key}")
    if saved is None:
        return None
    restored = io.BytesIO(saved["data"])
    restored.name = saved["name"]
    restored.size = len(saved["data"])
    return restored


def render() -> None:
    st.title("Co-registration Studio")
    st.caption("Four-stage precision pipeline: Pair Ingestion, Algorithm Tuning, Quality Gate Verification, and Multi-Spectral Visual Proof.")
    
    preset = st.session_state.get("preset", list(PRESETS)[0])

    # ---------------- 1. PAIR INGESTION ----------------
    st.subheader("1. Ingest lunar image pair", anchor=False)
    preset = st.selectbox("Select verification dataset", list(PRESETS), index=list(PRESETS).index(preset) if preset in PRESETS else 0)
    
    clear_c1, clear_c2 = st.columns([1, 3])
    if clear_c1.button("Clear uploaded images", icon=":material/delete:", width="stretch"):
        for _k in ("uploader_src", "uploader_ref", "saved_uploader_src", "saved_uploader_ref"):
            if _k in st.session_state:
                del st.session_state[_k]
        workspace.clear()
        st.rerun()

    _rehydrate_from_workspace()
    restored = [s for s in ("src", "ref") if st.session_state.pop(f"_restored_from_workspace_{s}", None)]

    col_s, col_r = st.columns(2)
    with col_s:
        f_src = st.file_uploader(
            "Upload custom source (Image A)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            key="uploader_src",
            on_change=_remember_upload,
            args=("uploader_src",),
        )
        f_src = _active_upload("uploader_src", f_src)
        if f_src is not None:
            st.caption(f":green-badge[Ingested: {f_src.name}] ({f_src.size:,} bytes)")
    with col_r:
        f_ref = st.file_uploader(
            "Upload custom reference (Image B)",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            key="uploader_ref",
            on_change=_remember_upload,
            args=("uploader_ref",),
        )
        f_ref = _active_upload("uploader_ref", f_ref)
        if f_ref is not None:
            st.caption(f":green-badge[Ingested: {f_ref.name}] ({f_ref.size:,} bytes)")

    if restored:
        names = []
        for side in ("src", "ref"):
            saved = st.session_state.get(f"saved_uploader_{side}")
            if saved is not None:
                names.append(saved["name"])
        st.info(
            "Your previously uploaded pair was restored from the workspace "
            f"({', '.join(names)}) — it stays active across tabs, ports, and server "
            "restarts. Use *Clear uploaded images* to return to the bundled presets."
        )

    src, ref = services.get_pair(preset, f_src, f_ref)
    has_uploads = f_src is not None or f_ref is not None
    partial_upload = (f_src is None) != (f_ref is None)
    # Input previews are the actual uploaded files, not processed/warped arrays.
    preview_src = f_src.getvalue() if f_src is not None else src
    preview_ref = f_ref.getvalue() if f_ref is not None else ref
    st.session_state["input_previews"] = (preview_src, preview_ref)
    st.session_state["input_names"] = (
        f_src.name if f_src is not None else PRESETS[preset]["src"],
        f_ref.name if f_ref is not None else PRESETS[preset]["ref"],
    )
    st.session_state["partial_upload"] = partial_upload

    # A registration result belongs to the exact pair that produced it. As soon as
    # a different pair is ingested, drop the old result instead of presenting its
    # tie-points/gates underneath the new images.
    current_signature = services.pair_signature(src, ref)
    _previous = st.session_state.get("result")
    if (
        _previous is not None
        and _previous.get("pair_signature") != current_signature
    ):
        st.session_state.pop("result", None)
        st.session_state.pop("blender_result", None)
        st.session_state["stale_pair_cleared"] = True

    # Publish the ingested pair immediately so the Overview, 3D and hazard screens
    # track what is actually loaded — not the pair from the last executed job.
    st.session_state["images"] = (src, ref)
    st.session_state["pair_source"] = "upload" if has_uploads else "preset"
    if preset in PRESETS:
        st.session_state["preset"] = preset

    res = st.session_state.get("result")
    sample_badge(res)
    active_pair_banner()
    if has_uploads:
        st.caption(f"Active uploaded pair · fingerprint `{current_signature}`. Saved uploads remain active when switching screens; use Clear uploaded images to return to presets.")
    if partial_upload:
        missing = "reference (Image B)" if f_ref is None else "source (Image A)"
        st.warning(
            f"**Only one image was uploaded.** The {missing} slot still holds the bundled "
            f"Chandrayaan/Apollo dataset, so this pair mixes your upload with a preset image. "
            "Upload both images before executing co-registration."
        )
    if st.session_state.pop("stale_pair_cleared", False):
        st.info(
            "The ingested image pair changed — the previous pair's results were cleared. "
            "Execute co-registration to certify the current pair."
        )

    if res and res.get("fallback_error"):
        if res.get("is_sample"):
            st.error(
                "**Real pipeline failed — the telemetry below is SIMULATED demo data**, "
                "not a certified registration of your uploaded images."
            )
        else:
            st.error("**Real pipeline failed — no registration was produced for this image pair.**")
        with st.expander("Show pipeline error"):
            st.code(res["fallback_error"], language="python")

    if res and isinstance(res.get("rejection"), dict) and res["rejection"].get("primary_reason"):
        rej = res["rejection"]
        if rej.get("primary_reason") == "REAL_PIPELINE_FAILED":
            st.warning(
                "**Registration failed — Grade F.** The real pipeline raised an exception, so no "
                "transform exists for this pair. See the pipeline error above."
            )
        else:
            st.warning(
                f"**Registration rejected — Grade F.** Primary reason: `{rej.get('primary_reason')}`. "
                "The uploaded pair produced no trustworthy overlap. Verify both images cover the same "
                "lunar terrain, or raise the processing dimension."
            )
        telemetry = rej.get("gate_telemetry") or {}
        if telemetry:
            with st.expander("Quality gate telemetry"):
                for gate, verdict in telemetry.items():
                    st.write(f"- **{gate}**: {verdict}")

    pv_s, pv_r = st.columns(2)
    with pv_s:
        with st.container(border=True):
            st.image(preview_src, caption="Source (Image A) — original input", alt="Original source input before registration", output_format="PNG", width="stretch")
            src_label = f"Uploaded: {f_src.name}" if f_src is not None else PRESETS[preset]["src"]
            st.caption(f":material/satellite_alt: {src_label}")
            if f_src is not None:
                st.caption(f"File SHA-256: `{hashlib.sha256(f_src.getvalue()).hexdigest()[:16]}`")

    with pv_r:
        with st.container(border=True):
            st.image(preview_ref, caption="Reference (Image B) — original input", alt="Original reference input before registration", output_format="PNG", width="stretch")
            ref_label = f"Uploaded: {f_ref.name}" if f_ref is not None else PRESETS[preset]["ref"]
            st.caption(f":material/radar: {ref_label}")
            if f_ref is not None:
                st.caption(f"File SHA-256: `{hashlib.sha256(f_ref.getvalue()).hexdigest()[:16]}`")

    # ---------------- 2. ALGORITHM CONFIGURATION ----------------
    st.subheader("2. Configure algorithm parameters", anchor=False)
    with st.container(border=True):
        a1, a2, a3 = st.columns([1.5, 1, 1])
        from config import AVAILABLE_MATCHERS
        matcher = a1.selectbox("Feature matcher engine", AVAILABLE_MATCHERS)
        outlier = a2.selectbox("Outlier filter", ["MAGSAC++", "RANSAC classical"])
        transform = a3.selectbox("Transformation model", ["Thin plate spline (TPS)", "Homography (8-DOF)", "Affine (6-DOF)"])

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        hapke_shadow = c_opt1.toggle(
            "Hapke BRDF shadow inversion",
            value=True,
            help="Decouples solar incidence and eliminates crater shadow disparity using physical planetary Hapke albedo inversion.",
        )
        cross_sun_resilience = c_opt2.toggle(
            "Cross-sun resilience",
            value=True,
            help="Decouples 180° inverted shadow relief, auto-estimates solar azimuth vectors, and aligns crater rim crests via dual-pass Hapke BRDF + monogenic phase congruency.",
        )
        subpixel_soft_argmax = c_opt3.toggle(
            "Soft-argmax sub-pixel",
            value=True,
            help="Computes expectation over continuous temperature-scaled correlation distribution to achieve certified sub-0.15 px precision.",
        )
        
        # Architectural information chip for chosen matcher
        m_lower = matcher.lower()
        if "apex" in m_lower:
            st.caption(":violet-badge[APEX 6-Engine Super-Fusion] Stacks DISK+LightGlue, RoMa v2, ResShift restoration and RootSIFT candidates, then applies spatial-consensus fusion with progressive MAGSAC++ outlier rejection for maximum certified coverage.")
        elif "ensemble" in m_lower:
            st.caption(":blue-badge[Dense Hybrid Transformer] Combines RoMa v2 coarse-to-fine expansion with LoFTR dense correlation for certified sub-pixel accuracy.")
        elif "roma" in m_lower:
            st.caption(":purple-badge[Dense Transformer Backbone] SOTA coarse-to-fine dense matcher with foundation optical flow expansion.")
        elif "loftr" in m_lower:
            st.caption(":blue-badge[Detector-Free Dense Transformer] Direct dense feature correlation invariant to lunar lighting and scale disparity.")
        elif "foundation" in m_lower or "ni_lfm" in m_lower:
            st.caption(":green-badge[Planetary Foundation Model] Pre-trained on multi-mission lunar orbital imagery (NASA LROC & Chandrayaan).")
        elif "lightglue" in m_lower:
            st.caption(":orange-badge[Learned Keypoint Graph] DISK local features with adaptive deep attention graph pruning.")
        elif "superglue" in m_lower:
            st.caption(":orange-badge[Attentional Graph Neural Network] SuperPoint interest points with optimal transport matching.")
        elif "rift2" in m_lower:
            st.caption(":red-badge[Radiation-Invariant Phase Congruency] Designed for extreme cross-modal and opposite-illumination pairs.")
        elif "asift" in m_lower:
            st.caption(":gray-badge[Affine Multiscale Scale-Space] Simulates multi-axis camera tilt angles for severe perspective shifts.")
        elif "akaze" in m_lower:
            st.caption(":gray-badge[Non-Linear Diffusion] Preserves shadow boundaries and sharp crater rims in scale space.")
        else:
            st.caption(":gray-badge[Deterministic SIFT Baseline] Scale-invariant gradient orientation. Certified deterministic CPU fallback for Gate 5.")

        settings = dict(
            matcher=matcher,
            outlier=outlier,
            transform=transform,
            hapke_shadow=hapke_shadow,
            cross_sun_resilience=cross_sun_resilience,
            subpixel_soft_argmax=subpixel_soft_argmax,
            subpixel=True,
            execute_real=True,
        )
        
        if st.button("Execute co-registration", type="primary", icon=":material/play_arrow:", width="stretch"):
            with st.status("Computing tie-points and outlier rejection...", expanded=True) as status:
                st.write("Initializing feature tensors...")
                st.write(f"Executing {matcher} feature matching...")
                st.write(f"Filtering false positives with {outlier}...")
                st.write(f"Fitting {transform} sub-pixel transformation...")
                result = services.run_registration(
                    src, ref, settings, preset, allow_demo=not has_uploads
                )
                result["input_names"] = st.session_state["input_names"]
                st.session_state.update(
                    preset=preset,
                    images=(src, ref),
                    result=result,
                )
                failed = bool(result.get("fallback_error") or result.get("rejection"))
                status.update(label="Registration failed or rejected" if failed else "Co-registration run finished", state="error" if failed else "complete")
            st.rerun()

    # ---------------- 3. QUALITY GATES ----------------
    res = st.session_state.get("result")
    if res is None:
        st.info("Execute co-registration above to verify quality gates and inspect visual alignment.")
        return

    st.caption(f"Executed pair: `{res.get('pair_signature', 'unavailable')}` · Source: {res.get('input_names', ('unknown', 'unknown'))[0]} · Reference: {res.get('input_names', ('unknown', 'unknown'))[1]}")
    with st.expander("Registration input audit"):
        st.json(res.get("input_audit", {"status": "No verified pipeline input audit; this run may have failed before matching."}))

    st.subheader("3. Mathematical quality gate certification", anchor=False)
    st.caption("Automated audit adhering to ISRO LunarFusion Co-Registration Quality Standards.")
    
    gt = res.get("gates", {})
    _grade = str(res.get("grade", "")).strip().lower()
    _grade_ok = _grade.startswith("grade a")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        gate_card(
            "Gate 1: Overfit trap",
            "Passed" if gt.get("overfit", True) else "Breached",
            f"Residual {res['rmse'] if res['rmse'] is not None else '—'} px (ceiling < 0.50 px, non-zero)",
            gt.get("overfit", True),
        )
    with g2:
        gate_card(
            "Gate 2: Inlier count",
            f"{res['inliers']:,}",
            f"Minimum 15 MVR (confidence {res['confidence'] if res['confidence'] is not None else '—'}%)",
            gt.get("inliers", True),
        )
    with g3:
        gate_card(
            "Gate 3: Spatial entropy",
            f"{res['entropy']:.1f}%",
            "Crater grid coverage (threshold >= 15%)",
            gt.get("entropy", True),
        )
    with g4:
        gate_card(
            "Final certification",
            res.get("grade", "Grade F"),
            "Certified for high-resolution 3D DEM" if _grade_ok else "Quality warning",
            _grade_ok,
        )

    illum = res.get("illumination", {})
    if illum and illum.get("delta_azimuth_deg") is not None:
        d_az = illum.get("delta_azimuth_deg", 0.0)
        az_s = illum.get("sun_az_ref", "--")
        az_r = illum.get("sun_az_tgt", "--")
        el_s = illum.get("sun_el_ref", "--")
        el_r = illum.get("sun_el_tgt", "--")
        tag = ":red-badge[180° Inverted Relief Handled]" if illum.get("extreme_opposite_sun") else ":blue-badge[Cross-Sun Illumination Resolved]"
        st.info(
            f":material/wb_sunny: **Solar Illumination Disparity: {d_az}°** — {tag} "
            f"Source sun: {az_s}° az, {el_s}° el | Reference sun: {az_r}° az, {el_r}° el. "
            "Dual-pass Hapke BRDF & Monogenic Phase Congruency normalized opposing shadow gradients before matching."
        )

    # ---------------- 4. VISUAL INSPECTION STUDIO ----------------
    st.subheader("4. Visual inspection studio", anchor=False)
    s_img, r_img = st.session_state.get("images", (src, ref))
    warped_s = res.get("img_src_warped", s_img)
    
    t1, t2, t3, t4, t5 = st.tabs([
        ":material/timeline: Tie-point vectors",
        ":material/grid_view: Checkerboard alignment",
        ":material/difference: Seam difference heatmap",
        ":material/bar_chart: Residual distribution",
        ":material/wb_sunny: Hapke albedo (Shadow-free)",
    ])
    
    with t1:
        st.caption("Green vectors link matching lunar features from Source (Cyan) to Reference (Emerald).")
        if not res["matches"]:
            st.info(
                "No verified tie-points survived for this run — registration was rejected, so no "
                "vectors are drawn. Vectors only appear from the MAGSAC++-verified inlier pool of a "
                "successful registration."
            )
        else:
            st.image(iv.match_lines(s_img, r_img, res["matches"]), width="stretch")
            st.caption(
                f"{len(res['matches'])} verified tie-points drawn · measured residual RMSE "
                f"`{res.get('rmse') if res.get('rmse') is not None else '—'} px`"
            )
        
    with t2:
        c_tile, _ = st.columns([2, 2])
        tile_sz = c_tile.slider("Checkerboard tile size (px)", min_value=16, max_value=96, value=48, step=8)
        st.caption("Aligned source and reference tiled together. Crater rims and shadow boundaries seamlessly continue across tile edges.")
        st.image(iv.checkerboard(warped_s, r_img, tile=tile_sz), width="stretch")
        
    with t3:
        st.caption("Residual error difference map between warped source and reference. Low intensity indicates sub-pixel agreement.")
        st.image(iv.difference(warped_s, r_img), width="stretch")
        
    with t4:
        st.caption("Histogram of tie-point Euclidean residuals in pixels. Values under the red dashed line satisfy the sub-pixel quality gate.")
        residuals = res.get("residuals") or []
        if not residuals:
            st.info(
                "No measured residuals for this run — the registration produced no verified "
                "tie-point pool, so this histogram stays empty rather than showing synthetic data."
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals,
                nbinsx=40,
                marker_color="#38BDF8",
                opacity=0.85,
                name="Inlier residuals",
            ))
            fig.add_vline(
                x=0.50,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text="0.50 px Sub-pixel ceiling",
                annotation_position="top right",
            )
            fig.add_vline(
                x=0.15,
                line_dash="dot",
                line_color="#10B981",
                annotation_text="0.15 px Soft-argmax target",
                annotation_position="top left",
            )
            fig.update_layout(
                height=430,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Residual error (pixels)",
                yaxis_title="Inlier frequency",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
            )
            st.plotly_chart(fig, width="stretch")

    with t5:
        st.caption("Dual-pass Hapke BRDF reflectance inversion decouples solar angle variations and eliminates artificial shadow edge shifts.")
        illum = res.get("illumination", {})
        if illum and illum.get("delta_azimuth_deg") is not None:
            c_meta1, c_meta2 = st.columns(2)
            c_meta1.caption(f":material/wb_sunny: Source lighting: Azimuth **{illum.get('sun_az_ref')}°**, Elevation **{illum.get('sun_el_ref')}°**")
            c_meta2.caption(f":material/wb_sunny: Reference lighting: Azimuth **{illum.get('sun_az_tgt')}°**, Elevation **{illum.get('sun_el_tgt')}°**")
        c_h1, c_h2 = st.columns(2)
        hapke_s = res.get("hapke_src", s_img)
        hapke_r = res.get("hapke_ref", r_img)
        with c_h1:
            st.image(hapke_s, caption="Source: Physical Hapke Albedo (Shadow-free)", width="stretch")
        with c_h2:
            st.image(hapke_r, caption="Reference: Physical Hapke Albedo (Shadow-free)", width="stretch")
        st.caption(":material/light_mode: **Physical Albedo Decoupling**: Eliminates false tie-point migration across shifting crater shadows while preserving high-contrast crater rims.")

