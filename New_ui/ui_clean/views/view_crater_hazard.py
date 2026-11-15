import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import PRESETS, GSD_REF
from components.header import sample_badge
from components import image_viewer as iv
import services

def render() -> None:
    st.title("Crater & hazard AI")
    res = st.session_state.get("result"); sample_badge(res)
    if res is None:
        st.info("Run the viva demo or the co-registration first.")
        return
    _, ref = st.session_state["images"]
    z = services.make_dem(seed=PRESETS[st.session_state["preset"]]["seed"]); slope = services.slope_deg(z)
    a, b = st.columns(2)
    with a:
        st.subheader("YOLO11 crater detection")
        st.image(iv.boxes(ref, res["craters"], GSD_REF), width="stretch")
        st.caption("Boxes show estimated crater diameter in metres. Confidence comes from your detector.")
    with b:
        st.subheader("Hazard density")
        fig = go.Figure(go.Heatmap(z=slope, colorscale="YlOrRd", colorbar=dict(title="Slope (°)")))
        fig.add_contour(z=slope, contours=dict(start=15, end=15, size=1, coloring="none"),
                        line=dict(color="#38BDF8", width=2), showscale=False)
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue line marks 15° slope.")

    st.subheader("Safest landing candidates")
    n, blk = z.shape[0], 20; rows = []
    for i in range(0, n, blk):
        for j in range(0, n, blk):
            blkv = slope[i:i + blk, j:j + blk]
            rows.append({"Site": 0, "Position": f"({j + blk // 2}, {i + blk // 2}) px",
                         "Mean slope (°)": round(float(blkv.mean()), 2), "Max slope (°)": round(float(blkv.max()), 2)})
    df = pd.DataFrame(rows).sort_values("Mean slope (°)").head(3).reset_index(drop=True)
    df["Site"] = df.index + 1
    df["Hazard"] = np.where(df["Max slope (°)"] > 15, "Moderate", "Low")
    st.dataframe(df, hide_index=True, width="stretch")
