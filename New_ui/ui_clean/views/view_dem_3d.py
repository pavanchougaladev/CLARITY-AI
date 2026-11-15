import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config import DEM_SPACING, PRESETS
from components.header import sample_badge
import services

def _layout(fig, h=480):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(aspectmode="data"), font=dict(color="#F8FAFC"))
    return fig

def render() -> None:
    st.title("3D topography & DEM")
    sample_badge(st.session_state.get("result"))
    seed = PRESETS[st.session_state.get("preset", list(PRESETS)[0])]["seed"]
    z = services.make_dem(seed=seed)
    n = z.shape[0]; ax = np.arange(n) * DEM_SPACING
    left, right = st.columns([2.2, 1])
    mode = left.segmented_control("View", ["Mesh", "Point cloud", "Hillshade"], default="Mesh") or "Mesh"
    if mode == "Point cloud":
        X, Y = np.meshgrid(ax, ax)
        fig = go.Figure(go.Scatter3d(x=X.ravel(), y=Y.ravel(), z=z.ravel(), mode="markers",
              marker=dict(size=1.5, color=z.ravel(), colorscale="Viridis")))
    elif mode == "Hillshade":
        gy, gx = np.gradient(z)
        fig = go.Figure(go.Heatmap(z=np.clip(0.5 - 0.5 * (gx + gy), 0, 1), colorscale="Gray", showscale=False))
    else:
        fig = go.Figure(go.Surface(x=ax, y=ax, z=z, colorscale="Viridis", showscale=False))
    left.plotly_chart(_layout(fig), width="stretch")
    with right:
        st.metric("Maximum relief", f"{z.max() - z.min():.1f} m", border=True)
        st.metric("Mean slope", f"{services.slope_deg(z).mean():.1f}°", border=True)
        st.metric("DEM resolution", f"{DEM_SPACING:.1f} m/px", border=True)

    st.subheader("Elevation along a transect")
    row = st.slider("Transect row", 0, n - 1, n // 2)
    prof = go.Figure(go.Scatter(x=ax, y=z[row], mode="lines", line=dict(color="#38BDF8")))
    prof.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Distance (m)",
                       yaxis_title="Elevation (m)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
    st.plotly_chart(prof, width="stretch")

    st.subheader("Export")
    d = st.columns(3)
    tif = services.dem_tiff(z)
    d[0].download_button("TIFF DEM", tif or b"", "dem.tif", disabled=tif is None, width="stretch",
                         help="Plain TIFF. Add georeferencing from your pipeline output.")
    d[1].download_button("PLY point cloud", services.dem_ply(z), "dem.ply", width="stretch")
    d[2].download_button("OBJ 3D mesh", services.dem_obj(z), "dem.obj", width="stretch")
