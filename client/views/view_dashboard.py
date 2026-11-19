from __future__ import annotations
"""Screen 01: Academic project overview, architecture, and instant viva demonstration."""
import streamlit as st
from config import PRESETS
from components.header import sample_badge, active_pair_banner
from components.metric_panel import kpis
import services

def render() -> None:
    st.title("Autonomous Sub-pixel Lunar Co-registration & 3D Mapping")
    st.caption("Precision alignment of multimodal Chandrayaan-1 and Chandrayaan-2 imagery using deep feature transformers, MAGSAC++ outlier rejection, and 3D elevation reconstruction.")

    res = st.session_state.get("result")
    sample_badge(res)
    active_pair_banner()

    # ---- Current image pair (uploads win over preset) ----
    pair = st.session_state.get("images")
    pair_sig = None
    if pair is not None:
        try:
            pair_sig = services.pair_signature(pair[0], pair[1])
        except Exception:
            pair_sig = None
    # Only surface a registration that belongs to the pair currently loaded.
    res_matches_pair = bool(
        res is not None and pair_sig is not None and res.get("pair_signature") == pair_sig
    )

    with st.container(border=True):
        st.subheader("Current image pair", anchor=False)
        st.caption("Live view of the pair currently ingested in the studio (Screen 02). Upload new images there and they appear here instantly.")
        if pair is None:
            st.info("No image pair ingested yet. Run the instant demo below or upload a pair on Screen 02 (Co-registration studio).")
        else:
            src, ref = pair
            preview_src, preview_ref = st.session_state.get("input_previews", pair)
            source_name, reference_name = st.session_state.get("input_names", ("Source", "Reference"))
            d1, d2 = st.columns(2)
            with d1:
                st.image(preview_src, caption="Source (Image A) — original input", alt="Original source input before registration", output_format="PNG", width="stretch")
                st.caption(source_name)
            with d2:
                st.image(preview_ref, caption="Reference (Image B) — original input", alt="Original reference input before registration", output_format="PNG", width="stretch")
                st.caption(reference_name)
            if res_matches_pair and res.get("img_src_warped") is not None:
                st.image(res["img_src_warped"], caption="Registered (warped) source after co-registration", width="stretch")
                st.caption(f"Registered from fingerprint `{res.get('pair_signature')}`")
            elif res is not None:
                st.caption(":orange-badge[Pair changed — execute co-registration to certify this pair]")
                st.caption(
                    f"The displayed result was produced from fingerprint "
                    f"`{res.get('pair_signature')}`, but the active pair is `{pair_sig}`."
                )

    # Core KPI cards
    if res_matches_pair:
        inlier_val = f"{res.get('inliers', 0):,}"
        rmse_raw = res.get("rmse")
    else:
        inlier_val = "—"
        rmse_raw = None
    rmse_val = f"{rmse_raw:.2f} px" if isinstance(rmse_raw, (int, float)) else "—"
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
        uploaded_pair = (
            st.session_state.get("pair_source") == "upload"
            and st.session_state.get("images") is not None
        )
        preset = c1.selectbox("Verification dataset", list(PRESETS), disabled=uploaded_pair)
        if uploaded_pair:
            c1.caption(
                ":green-badge[Using the pair uploaded on Screen 02] — this run registers your own "
                "images with the real pipeline. Clear the uploads on Screen 02 to use a bundled dataset."
            )

        if c2.button("Run instant live demo", type="primary", icon=":material/rocket_launch:", width="stretch"):
            with st.status("Executing co-registration suite...", expanded=True) as status:
                st.write("Ingesting multimodal lunar sensor pair...")
                if uploaded_pair:
                    src, ref = st.session_state["images"]
                    st.write("Registering the uploaded pair with the real pipeline...")
                    st.write("Extracting deep transformer tie-points (LoFTR + RoMa v2)...")
                    st.write("Evaluating MAGSAC++ robust inlier pool against Quality Gates 1-5...")
                    result = services.run_registration(
                        src,
                        ref,
                        {
                            "matcher": "LoFTR + RoMa v2 ensemble (ISRO Certified SOTA)",
                            "outlier": "MAGSAC++",
                            "execute_real": True,
                            "hapke_shadow": True,
                            "cross_sun_resilience": True,
                            "subpixel_soft_argmax": True,
                        },
                        preset,
                        allow_demo=False,
                    )
                else:
                    st.write("Extracting deep transformer tie-points (LoFTR + RoMa v2)...")
                    st.write("Evaluating MAGSAC++ robust inlier pool against Quality Gates 1-5...")
                    src, ref = services.get_pair(preset)
                    st.session_state.pop("input_previews", None)
                    st.session_state.pop("input_names", None)
                    st.session_state["pair_source"] = "preset"
                    st.session_state["partial_upload"] = False
                    result = services.run_registration(src, ref, {"matcher": "ensemble", "outlier": "MAGSAC++"}, preset)
                st.session_state.update(
                    preset=preset,
                    images=(src, ref),
                    result=result,
                )
                status.update(label=f"{result.get('grade', 'Result')}: {result['inliers']} inliers with {result['rmse']} px RMSE", state="complete")
            st.toast("Registration complete. Explore screens 02 through 05 via the sidebar.", icon=":material/check_circle:")
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
