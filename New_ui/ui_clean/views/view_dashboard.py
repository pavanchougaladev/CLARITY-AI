import streamlit as st
from config import PRESETS
from components.header import sample_badge
from components.metric_panel import kpis
import services

def render() -> None:
    st.title("Aligning lunar images from different sensors, to under half a pixel")
    st.write("CLARITY\\* co-registers Chandrayaan-1 and Chandrayaan-2 imagery using deep feature "
             "transformers (LoFTR, RoMa v2) with MAGSAC++ outlier rejection, then builds a 3D "
             "elevation model from the aligned pair.")
    res = st.session_state.get("result")
    sample_badge(res)
    kpis([("Mean inliers", f"{(res or {}).get('inliers', 1420):,}", "verified tie points"),
          ("Mean RMSE", f"{(res or {}).get('rmse', 0.38)} px", "certification limit 0.5 px"),
          ("GSD precision", "0.25 m/px", "OHRC reference")])
    st.subheader("Run the viva demo")
    st.caption("Loads a pair, aligns it and prepares every screen, with no manual upload.")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    preset = c1.selectbox("Preset", list(PRESETS))
    if c2.button("Run demo", type="primary", icon=":material/play_arrow:", width="stretch"):
        with st.status("Running pipeline", expanded=False) as status:
            src, ref = services.get_pair(preset)
            st.session_state.update(preset=preset, images=(src, ref),
                result=services.run_registration(src, ref, {}, preset))
            status.update(label="Done. Open the other screens from the sidebar.", state="complete")
        st.rerun()
