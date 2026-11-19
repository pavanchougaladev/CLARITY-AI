from __future__ import annotations
"""Screen 04: YOLO11 Crater Detection & Autonomous Landing Site Hazard Assessment."""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import PRESETS, GSD_REF, DEM_SPACING
from components.header import sample_badge
from components import image_viewer as iv
import services

def render() -> None:
    st.title("Crater Detection & Landing Hazard AI")
    st.caption("Fine-tuned YOLO11 deep learning crater identification fused with 3D topographic slope and multi-factor obstacle avoidance for certified lunar touchdown.")

    preset = st.session_state.get("preset", list(PRESETS)[0])
    res = st.session_state.get("result")
    sample_badge(res)

    # Ingest reference image from active session or preset
    images = st.session_state.get("images")
    if images is not None and len(images) > 1:
        ref = images[1]
    else:
        _, ref = services.get_pair(preset)

    # Topographic DEM reconstruction for physical 3D crater depth profiling
    z, _ = services.make_dem_from_image(ref, n=96, is_hybrid=True)

    # ---------------- 1. DETECTION & HAZARD CONTROLS ----------------
    with st.expander("Crater detection & hazard envelope controls", expanded=False, icon=":material/tune:"):
        c_conf, c_sam, c_micro, c_slope, c_foot = st.columns([1.1, 1.2, 1.0, 1.0, 1.0])
        conf = c_conf.slider(
            "YOLO11 neural confidence",
            min_value=0.10,
            max_value=0.60,
            value=0.18,
            step=0.02,
            help="Minimum neural detection confidence threshold for YOLO11 macro-crater detector.",
        )
        enable_sam2 = c_sam.toggle(
            "SAM 2 rim polygons",
            value=True,
            help="Segment Anything Model 2 extracts organic, pixel-accurate crater rim contours and polygonal hazard masks.",
        )
        include_micro = c_micro.toggle(
            "Micro-craters",
            value=True,
            help="Enables circular radial relief transform to detect sub-resolution micro-craters down to 2m.",
        )
        max_safe_slope = c_slope.slider(
            "Max safe slope (°)",
            min_value=8.0,
            max_value=18.0,
            value=12.0,
            step=1.0,
            help="Chandrayaan-3 lander design envelope: <= 12° nominal safe slope, 15° hard abort ceiling.",
        )
        lander_footprint_m = c_foot.slider(
            "Footprint dia (m)",
            min_value=20.0,
            max_value=80.0,
            value=40.0,
            step=5.0,
            help="Touchdown footprint zone requiring continuous obstacle-free terrain.",
        )

    # Execute Hybrid Crater Detection with 3D Morphometric Depth Profiling & SAM 2 Rim Polygons
    craters = services.detect_craters_hybrid(
        ref,
        conf=conf,
        gsd=GSD_REF,
        z_dem=z,
        include_micro=include_micro,
        enable_sam2=enable_sam2,
    )

    # Execute Autonomous Landing Hazard Evaluation
    safety_map, slope, hazard_mask, ranked_sites = services.evaluate_landing_hazards(
        z_dem=z,
        craters=craters,
        gsd_m=DEM_SPACING,
        max_safe_slope=max_safe_slope,
        lander_footprint_m=lander_footprint_m,
        img_size=ref.shape[:2],
    )

    # ---------------- 2. SUMMARY KPI METRICS ROW ----------------
    safe_area_pct = float((slope < max_safe_slope).mean() * 100.0)
    max_d_m = max([c["diameter_m"] for c in craters]) if craters else 0.0
    max_depth_m = max([c["depth_m"] for c in craters]) if craters else 0.0
    primary_safe = len(ranked_sites) > 0 and "SAFE" in ranked_sites[0]["status"].upper()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Detected craters",
        f"{len(craters):,}",
        help="Macro & micro craters identified by YOLO11 and radial relief morphometry",
        border=True,
    )
    k2.metric(
        "Safe surface area",
        f"{safe_area_pct:.1f}%",
        f"Slope < {max_safe_slope:.0f}° ceiling",
        border=True,
    )
    k3.metric(
        "Largest hazard crater",
        f"{max_d_m:.0f} m",
        f"Depth: {max_depth_m:.1f} m",
        border=True,
    )
    k4.metric(
        "Autonomous landing rating",
        "GO FOR DESCENT" if primary_safe else "CAUTION / CONTINGENCY",
        f"Top Site Safety: {ranked_sites[0]['safety_score']:.1f}%" if ranked_sites else "No safe site",
        border=True,
    )

    # ---------------- 3. VISUAL INSPECTION STUDIO TABS ----------------
    st.subheader("Visual inspection & landing site selection studio", anchor=False)

    t1, t2, t3, t4 = st.tabs([
        ":material/target: YOLO11 + SAM 2 Crater Segmentation",
        ":material/flight_land: Autonomous Landing Site Selection",
        ":material/warning: Multi-Factor Hazard Heatmap",
        ":material/analytics: Crater Morphometry & Depth Analysis",
    ])

    with t1:
        st.caption(
            "YOLO11 deep learning detection fused with SAM 2 pixel-exact crater rim contours and radial relief micro-crater morphometry. "
            "Telemetry badges report physical diameter (D), calibrated depth (d), and isoperimetric circularity (C)."
        )

        view_col, opacity_col = st.columns([1.8, 1.2])
        with view_col:
            view_mode = st.radio(
                "Delineation style",
                ["SAM 2 Pixel-Exact Rim Polygons", "YOLO11 Bounding Boxes", "Fused Overlay"],
                horizontal=True,
            )
        with opacity_col:
            poly_alpha = st.slider(
                "Polygon fill opacity",
                min_value=0.10,
                max_value=0.80,
                value=0.35,
                step=0.05,
                disabled="Boxes" in view_mode,
            )

        col_img, col_stat = st.columns([2.2, 1])
        with col_img:
            if "Boxes" in view_mode:
                annotated_img = iv.boxes(ref, craters, GSD_REF, show_depth=True)
            elif "Fused" in view_mode:
                annotated_img = iv.polygons(
                    ref,
                    craters,
                    GSD_REF,
                    show_depth=True,
                    fill_alpha=poly_alpha,
                    show_boxes=True,
                )
            else:
                annotated_img = iv.polygons(
                    ref,
                    craters,
                    GSD_REF,
                    show_depth=True,
                    fill_alpha=poly_alpha,
                    show_boxes=False,
                )
            st.image(annotated_img, width="stretch")
        with col_stat:
            with st.container(border=True):
                st.markdown("##### Detection & Segmentation Breakdown")
                n_macro = sum(1 for c in craters if c["diameter_m"] >= 40.0)
                n_micro = sum(1 for c in craters if c["diameter_m"] < 40.0)
                n_crit = sum(1 for c in craters if c["hazard_level"] == "CRITICAL HAZARD")
                avg_circ = float(np.mean([c.get("circularity", 0.95) for c in craters])) if craters else 0.95
                sam_engine = craters[0].get("segmentation_engine", "SAM 2") if craters else "SAM 2"

                st.write(f"• **Macroscopic craters (D ≥ 40m):** {n_macro}")
                st.write(f"• **Micro-crater pits (D < 40m):** {n_micro}")
                st.write(f"• **Critical hazard craters:** {n_crit}")
                st.write(f"• **Avg Circularity Index ($C$):** {avg_circ:.2f}")
                st.write(f"• **Rim Delineation:** :green[{sam_engine}]")
                st.divider()
                st.markdown("##### Crater Hazard Legend")
                st.caption(":red[■] **Critical Hazard**: D ≥ 80m or Depth ≥ 15m")
                st.caption(":orange[■] **Moderate Hazard**: D ≥ 35m or Rim Slope ≥ 12°")
                st.caption(":blue[■] **Micro Hazard**: Small pits / shallow basins")

    with t2:
        st.caption(
            "Ranked safe landing footprint circles overlaid directly on the reference lunar image. "
            "Neon green targets mark certified candidate landing sites filtered for minimal slope and 1.35x crater rim clearance."
        )
        col_site_img, col_site_meta = st.columns([2.2, 1])
        with col_site_img:
            site_overlay = iv.draw_landing_sites_overlay(
                ref,
                ranked_sites,
                craters,
                gsd=GSD_REF,
                footprint_radius_m=lander_footprint_m / 2.0,
            )
            st.image(site_overlay, width="stretch")
        with col_site_meta:
            if ranked_sites:
                best = ranked_sites[0]
                with st.container(border=True):
                    st.markdown(f"##### Recommended Primary Site: :green[{best['site_id']}]")
                    st.write(f"• **Safety Score:** {best['safety_score']:.1f}%")
                    st.write(f"• **Touchdown Center:** ({best['x_m']}m, {best['y_m']}m)")
                    st.write(f"• **Mean Slope:** {best['mean_slope']:.1f}°")
                    st.write(f"• **Peak Slope:** {best['max_slope']:.1f}°")
                    st.write(f"• **Crater Rim Clearance:** {best['crater_clearance_m']:.1f} m")
                    st.write(f"• **Certification:** :green[{best['status']}]")
            else:
                st.warning("No terrain patch meets the certified safety criteria under current settings.")

    with t3:
        st.caption(
            "Comprehensive lunar surface hazard evaluation combining slope gradient, crater exclusion buffer, and micro-roughness."
        )
        col_hz1, col_hz2 = st.columns(2)
        n_dem = z.shape[0]
        ax_m = np.arange(n_dem) * DEM_SPACING

        with col_hz1:
            st.markdown("###### Topographic Slope Distribution (°)")
            fig_slope = go.Figure(go.Heatmap(
                x=ax_m,
                y=ax_m,
                z=slope,
                colorscale="YlOrRd",
                colorbar=dict(title="Slope (°)"),
            ))
            # Contour at nominal threshold and abort threshold
            fig_slope.add_contour(
                x=ax_m,
                y=ax_m,
                z=slope,
                contours=dict(start=max_safe_slope, end=max_safe_slope, size=1, coloring="none"),
                line=dict(color="#38BDF8", width=2),
                name=f"Safe Limit ({max_safe_slope}°)",
            )
            fig_slope.add_contour(
                x=ax_m,
                y=ax_m,
                z=slope,
                contours=dict(start=15.0, end=15.0, size=1, coloring="none"),
                line=dict(color="#EF4444", width=2.5, dash="dash"),
                name="Abort Ceiling (15°)",
            )
            fig_slope.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
            )
            st.plotly_chart(fig_slope, width="stretch")

        with col_hz2:
            st.markdown("###### Composite Landing Safety Index (LSI %)")
            fig_safety = go.Figure(go.Heatmap(
                x=ax_m,
                y=ax_m,
                z=safety_map,
                colorscale="Viridis",
                colorbar=dict(title="Safety (%)"),
            ))
            # Overlay ranked landing site dots
            if ranked_sites:
                fig_safety.add_trace(go.Scatter(
                    x=[s["x_m"] for s in ranked_sites],
                    y=[s["y_m"] for s in ranked_sites],
                    mode="markers+text",
                    text=[s["site_id"] for s in ranked_sites],
                    textposition="top center",
                    textfont=dict(color="#FFFFFF", size=11),
                    marker=dict(size=12, color="#10B981", symbol="circle-open", line=dict(width=2, color="#10B981")),
                    name="Candidate Sites",
                ))
            fig_safety.update_layout(
                height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
            )
            st.plotly_chart(fig_safety, width="stretch")

    with t4:
        st.caption(
            "Empirical crater morphometric distribution comparing measured physical depth vs. diameter against the Pike (1977) lunar scaling law."
        )
        if craters:
            df_cr = pd.DataFrame(craters)
            col_sc, col_hist = st.columns([1.5, 1])

            with col_sc:
                # Diameter vs. Depth scatter plot
                d_vals = np.linspace(min(df_cr["diameter_m"]), max(df_cr["diameter_m"]), 50)
                # Pike 1977 simple crater scaling law: d = 0.196 * D^1.01
                pike_depth = 0.196 * (d_vals ** 1.01)

                fig_morph = go.Figure()
                fig_morph.add_trace(go.Scatter(
                    x=df_cr["diameter_m"],
                    y=df_cr["depth_m"],
                    mode="markers",
                    text=[f"Crater #{c['id']} ({c['morphology']})" for c in craters],
                    marker=dict(
                        size=8,
                        color=df_cr["depth_diam_ratio"],
                        colorscale="Tealgrn",
                        colorbar=dict(title="d/D Ratio"),
                        showscale=True,
                    ),
                    name="Detected Craters",
                ))
                fig_morph.add_trace(go.Scatter(
                    x=d_vals,
                    y=pike_depth,
                    mode="lines",
                    line=dict(color="#EF4444", dash="dash", width=2),
                    name="Pike (1977) Lunar Simple Crater Model",
                ))
                fig_morph.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Crater Diameter D (metres)",
                    yaxis_title="Crater Depth d (metres)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                )
                st.plotly_chart(fig_morph, width="stretch")

            with col_hist:
                fig_circ = go.Figure()
                fig_circ.add_trace(go.Scatter(
                    x=df_cr["diameter_m"],
                    y=df_cr.get("circularity", pd.Series([0.95] * len(df_cr))),
                    mode="markers",
                    text=[f"Crater #{c['id']} (Circularity: {c.get('circularity', 0.95):.3f})" for c in craters],
                    marker=dict(
                        size=8,
                        color=df_cr.get("circularity", pd.Series([0.95] * len(df_cr))),
                        colorscale="Viridis",
                        colorbar=dict(title="Circularity (C)"),
                        showscale=True,
                    ),
                    name="Crater Circularity",
                ))
                # Add theoretical circle baseline C = 1.0
                fig_circ.add_hline(y=1.0, line_dash="dot", line_color="#94A3B8", annotation_text="Ideal Circle (C=1.0)")
                fig_circ.update_layout(
                    height=320,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Crater Diameter D (metres)",
                    yaxis_title="Circularity Index C = 4πA/P²",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                )
                st.plotly_chart(fig_circ, width="stretch")

            # Crater Telemetry Table with SAM 2 Polygon Geometry
            st.markdown("##### Detailed Crater Morphometry & SAM 2 Geometry")
            display_cols = ["id", "diameter_m", "depth_m", "depth_diam_ratio", "circularity", "ellipticity", "area_m2", "perimeter_m", "rim_slope", "hazard_level", "segmentation_engine"]
            available_cols = [c for c in display_cols if c in df_cr.columns]
            df_table = df_cr[available_cols].copy()
            col_rename = {
                "id": "ID",
                "diameter_m": "Diameter (m)",
                "depth_m": "Depth (m)",
                "depth_diam_ratio": "d/D Ratio",
                "circularity": "Circularity (C)",
                "ellipticity": "Ellipticity (b/a)",
                "area_m2": "Area (m²)",
                "perimeter_m": "Perimeter (m)",
                "rim_slope": "Rim Slope (°)",
                "hazard_level": "Hazard Rating",
                "segmentation_engine": "Delineation Engine",
            }
            df_table.rename(columns=col_rename, inplace=True)
            st.dataframe(df_table, hide_index=True, width="stretch")

            # GeoJSON & CSV Downloads for QGIS & GIS Tools
            c_g1, c_g2, _ = st.columns([1.2, 1.2, 1.6])
            geojson_str = services.export_craters_geojson(craters, gsd_m=GSD_REF)
            c_g1.download_button(
                "Export crater polygons (GeoJSON)",
                data=geojson_str.encode("utf-8"),
                file_name="chandrayaan_sam2_crater_polygons.geojson",
                mime="application/geo+json",
                icon=":material/download:",
                help="Standard GeoJSON polygon layer ready for drag-and-drop into QGIS or ArcGIS.",
            )
            c_g2.download_button(
                "Export crater morphometry (CSV)",
                data=df_table.to_csv(index=False).encode("utf-8"),
                file_name="lunar_crater_morphometry.csv",
                mime="text/csv",
                icon=":material/data_object:",
                help="Complete morphometry table including SAM 2 perimeters, areas, and Pike scaling.",
            )

    # ---------------- 4. RANKED CANDIDATE SITES TABLE ----------------
    with st.container(border=True):
        st.subheader("Autonomous landing candidate sites ranking", anchor=False)
        st.caption(
            "Ranked optimal candidate zones filtered for continuous slope stability, obstacle absence, and crater clearance."
        )

        if ranked_sites:
            df_sites = pd.DataFrame(ranked_sites)[[
                "site_id",
                "x_m",
                "y_m",
                "mean_slope",
                "max_slope",
                "crater_clearance_m",
                "safety_score",
                "status",
            ]].copy()

            df_sites.columns = [
                "Site ID",
                "Coord X (m)",
                "Coord Y (m)",
                "Mean Slope (°)",
                "Max Slope (°)",
                "Crater Clearance (m)",
                "Safety Rating (%)",
                "Mission Certification",
            ]

            st.dataframe(
                df_sites,
                hide_index=True,
                width="stretch",
                column_config={
                    "Safety Rating (%)": st.column_config.ProgressColumn(
                        "Safety Rating",
                        help="Autonomous landing safety rating (0-100%)",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                    "Mean Slope (°)": st.column_config.NumberColumn(format="%.1f°"),
                    "Max Slope (°)": st.column_config.NumberColumn(format="%.1f°"),
                    "Crater Clearance (m)": st.column_config.NumberColumn(format="%.1f m"),
                },
            )

            # Export Landing Telemetry
            col_d1, col_d2, _ = st.columns([1, 1, 2])
            csv_data = df_sites.to_csv(index=False).encode("utf-8")
            col_d1.download_button(
                "Export landing sites (CSV)",
                data=csv_data,
                file_name="lunar_safe_landing_sites.csv",
                mime="text/csv",
                icon=":material/download:",
            )

            json_telemetry = {
                "mission": "CLARITY* Lunar Landing Hazard AI",
                "preset": preset,
                "detected_craters": len(craters),
                "safe_area_fraction": round(safe_area_pct, 1),
                "ranked_sites": ranked_sites,
            }
            col_d2.download_button(
                "Export mission telemetry (JSON)",
                data=json.dumps(json_telemetry, indent=2).encode("utf-8"),
                file_name="landing_hazard_telemetry.json",
                mime="application/json",
                icon=":material/data_object:",
            )
        else:
            st.info("No landing sites meet the safety criteria under current slope and footprint settings.")
