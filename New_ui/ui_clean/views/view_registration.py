import streamlit as st
from config import PRESETS
from components.header import sample_badge
from components.metric_panel import gate_card
from components import image_viewer as iv
import services

def render() -> None:
    st.title("Co-registration studio")
    preset = st.session_state.get("preset", list(PRESETS)[0])

    st.subheader("1. Choose the image pair")
    preset = st.selectbox("Preset pair", list(PRESETS), index=list(PRESETS).index(preset))
    up = st.columns(2)
    f_src = up[0].file_uploader("Source (image A)", type=["png", "jpg", "tif", "tiff"])
    f_ref = up[1].file_uploader("Reference (image B)", type=["png", "jpg", "tif", "tiff"])
    src, ref = services.get_pair(preset, f_src, f_ref)
    for col, img, key in zip(st.columns(2), (src, ref), ("src", "ref")):
        with col.container(border=True):
            st.image(img, width="stretch")
            st.caption(PRESETS[preset][key])

    st.subheader("2. Set the algorithm")
    a = st.columns(4)
    settings = dict(
        matcher=a[0].selectbox("Feature matcher", ["LoFTR + RoMa v2 ensemble", "LoFTR", "RoMa v2"]),
        outlier=a[1].selectbox("Outlier filter", ["MAGSAC++", "RANSAC"]),
        transform=a[2].selectbox("Transformation", ["Thin plate spline", "Homography", "Affine"]),
        subpixel=a[3].toggle("Sub-pixel refinement", value=True))
    if st.button("Execute co-registration", type="primary", icon=":material/play_arrow:"):
        with st.status("Matching features and filtering outliers", expanded=False) as s:
            st.session_state.update(preset=preset, images=(src, ref),
                result=services.run_registration(src, ref, settings, preset))
            s.update(label="Registration complete", state="complete")

    res = st.session_state.get("result")
    if res is None:
        st.info("Run the pipeline to see the quality gates and the visual inspection.")
        return
    sample_badge(res)
    st.subheader("3. Read the quality gates")
    g = st.columns(4); gt = res["gates"]
    with g[0]: gate_card("Gate 1: overfit", "Passed" if gt["overfit"] else "Failed", f"{res['points']:,} points, RMSE {res['rmse']} px", gt["overfit"])
    with g[1]: gate_card("Gate 2: inliers", f"{res['inliers']:,}", f"minimum 15, confidence {res['confidence']}%", gt["inliers"])
    with g[2]: gate_card("Gate 3: entropy", f"{res['entropy']}%", "grid coverage", gt["entropy"])
    with g[3]: gate_card("Result", f"Grade {res['grade']}", "certified for 3D DEM" if res["grade"] == "A" else "not certified")

    st.subheader("4. Inspect the alignment")
    s_img, r_img = st.session_state["images"]
    t1, t2, t3 = st.tabs(["Tie-point vectors", "Checkerboard", "Seam difference"])
    t1.image(iv.match_lines(s_img, r_img, res["matches"]), width="stretch")
    t2.image(iv.checkerboard(s_img, r_img), width="stretch")
    t3.image(iv.difference(s_img, r_img), width="stretch")
