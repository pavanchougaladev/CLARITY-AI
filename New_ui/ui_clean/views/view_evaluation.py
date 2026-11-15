import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import BENCH

def render() -> None:
    st.title("Why deep matching wins")
    st.write("Same image pair, four methods. The ensemble keeps far more verified points and the "
             "lowest error, at a cost of a few seconds.")
    df = pd.DataFrame(BENCH, columns=["Method", "Inliers", "RMSE (px)", "Reprojection (px)", "Time (s)", "Grade"])
    st.dataframe(df.style.apply(lambda r: ["background-color: rgba(56,189,248,0.12)"] * len(r) if r.name == 0 else [""] * len(r), axis=1),
                 hide_index=True, width="stretch")
    st.subheader("Residual spread")
    rng = np.random.default_rng(1)
    fig = go.Figure()
    for m, rmse in zip(df["Method"], df["RMSE (px)"]):
        fig.add_trace(go.Box(y=np.abs(rng.normal(0, rmse, 200)), name=m, boxpoints=False))
    fig.update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="Residual (px)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Illustrative spread generated from each method's RMSE. Replace with real residuals from your pipeline.")
