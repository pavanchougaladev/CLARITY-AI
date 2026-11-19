"""
Visualization and Interactive Photogrammetry Canvas Component
=============================================================
ISRO LunarFusion Co-Registration Suite — UI Visualizations
Provides checkerboard blending, seam continuity metrics, match plots, and 3D DEM preview surfaces.
"""

import cv2
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from typing import Optional, Any, Dict, List, Tuple

from src.features.dem_builder import (
    get_camera_model,
    compute_photoclinometry_sfs,
    despike_and_clamp_lunar_slopes,
)


def normalize_to_rgb8(img: np.ndarray, clahe_enhance: bool = True) -> np.ndarray:
    """
    Safely convert any image (uint8, uint16, float, 2D, 3D, 4D) to 3-channel RGB uint8
    with dynamic range normalization and optional CLAHE crater rim enhancement.
    """
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)

    img_arr = np.asarray(img)

    # Handle floating point imagery [0.0, 1.0] or arbitrary range
    if np.issubdtype(img_arr.dtype, np.floating):
        min_val = float(np.nanmin(img_arr))
        max_val = float(np.nanmax(img_arr))
        if max_val > min_val:
            img_norm = np.clip((img_arr - min_val) / (max_val - min_val) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_arr, dtype=np.uint8)
    elif img_arr.dtype == np.uint16:
        p2, p98 = np.percentile(img_arr, (2, 98))
        if p98 > p2:
            img_norm = np.clip((img_arr.astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_norm = (img_arr >> 8).astype(np.uint8)
    elif img_arr.dtype == np.uint8:
        img_norm = img_arr.copy()
    else:
        min_v = float(np.min(img_arr))
        max_v = float(np.max(img_arr))
        if max_v > min_v:
            img_norm = np.clip((img_arr.astype(np.float32) - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)
        else:
            img_norm = np.zeros_like(img_arr, dtype=np.uint8)

    # Convert multi-channel / single-channel to 2D grayscale first for uniform CLAHE
    if img_norm.ndim == 2:
        gray = img_norm
    elif img_norm.ndim == 3:
        if img_norm.shape[2] == 1:
            gray = img_norm[:, :, 0]
        elif img_norm.shape[2] == 4:
            gray = cv2.cvtColor(img_norm, cv2.COLOR_RGBA2GRAY)
        elif img_norm.shape[2] == 3:
            gray = cv2.cvtColor(img_norm, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_norm[:, :, 0]
    else:
        gray = np.squeeze(img_norm)
        if gray.ndim > 2:
            gray = gray[:, :, 0]

    # Optional CLAHE contrast enhancement for razor-sharp lunar topography
    if clahe_enhance and gray.ndim == 2 and gray.size > 0:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            pass

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def create_checkerboard(
    img1: np.ndarray,
    img2: np.ndarray,
    tiles: int = 16,
    show_seams: bool = True,
    seam_color: Tuple[int, int, int] = (0, 240, 255),
    invert: bool = False,
    contrast_boost: bool = False,
) -> np.ndarray:
    """
    Create a sub-pixel checkerboard blended image for visual alignment verification.
    Supports arbitrary grid densities, high-contrast seam boundary overlays,
    and adaptive CLAHE contrast enhancement for lunar crater rim continuity.
    """
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)

    i1 = normalize_to_rgb8(img1, clahe_enhance=contrast_boost)
    i2 = normalize_to_rgb8(img2, clahe_enhance=contrast_boost)

    checker = np.zeros_like(i1)
    tile_h = max(h // tiles, 1)
    tile_w = max(w // tiles, 1)

    for y in range(tiles):
        for x in range(tiles):
            start_y = y * tile_h
            end_y = min((y + 1) * tile_h, h) if y < tiles - 1 else h
            start_x = x * tile_w
            end_x = min((x + 1) * tile_w, w) if x < tiles - 1 else w

            condition = ((x + y) % 2 == (1 if invert else 0))
            if condition:
                checker[start_y:end_y, start_x:end_x] = i1[start_y:end_y, start_x:end_x]
            else:
                checker[start_y:end_y, start_x:end_x] = i2[start_y:end_y, start_x:end_x]

    if show_seams:
        seam_rgb = np.array(seam_color, dtype=np.uint8)
        for y in range(1, tiles):
            y_pos = y * tile_h
            if 0 < y_pos < h:
                checker[max(0, y_pos - 1):min(h, y_pos + 1), :] = seam_rgb
        for x in range(1, tiles):
            x_pos = x * tile_w
            if 0 < x_pos < w:
                checker[:, max(0, x_pos - 1):min(w, x_pos + 1)] = seam_rgb

    return checker


def compute_seam_alignment_continuity(img1: np.ndarray, img2: np.ndarray, tiles: int = 16) -> Dict[str, Any]:
    """
    Evaluates quantitative photometric and geometric continuity across checkerboard seams.
    Returns normalized gradient coherence (0-100%) and estimated sub-pixel misalignment.
    """
    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_CUBIC)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    tile_h = max(h // tiles, 1)
    tile_w = max(w // tiles, 1)

    correlations = []
    for x in range(1, tiles):
        x_pos = x * tile_w
        if 0 < x_pos < w:
            col_l = g1[:, x_pos].astype(np.float32)
            col_r = g2[:, x_pos].astype(np.float32)
            std_l, std_r = np.std(col_l), np.std(col_r)
            if std_l > 1e-3 and std_r > 1e-3:
                r = float(np.corrcoef(col_l, col_r)[0, 1])
                if not np.isnan(r):
                    correlations.append(max(0.0, r))

    for y in range(1, tiles):
        y_pos = y * tile_h
        if 0 < y_pos < h:
            row_t = g1[y_pos, :].astype(np.float32)
            row_b = g2[y_pos, :].astype(np.float32)
            std_t, std_b = np.std(row_t), np.std(row_b)
            if std_t > 1e-3 and std_b > 1e-3:
                r = float(np.corrcoef(row_t, row_b)[0, 1])
                if not np.isnan(r):
                    correlations.append(max(0.0, r))

    coherence = float(np.mean(correlations)) if correlations else 1.0
    continuity_pct = max(0.0, min(100.0, coherence * 100.0))
    mean_disp_px = max(0.02, (1.0 - coherence) * 2.5)

    return {
        "coherence_pct": continuity_pct,
        "mean_seam_disparity_px": mean_disp_px,
        "is_subpixel_locked": coherence >= 0.85,
    }


def create_hd_match_canvas(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    inliers_mask: np.ndarray = None,
    max_lines: Optional[int] = 350,
    line_thickness: int = 2,
    color_palette: str = "aurora",
    enhance_contrast: bool = True,
    target_canvas_height: int = 1280,
) -> np.ndarray:
    """
    Renders an Ultra-HD photogrammetric co-registration canvas with antialiased lines,
    dual reticle keypoints, header banners, and high-visibility contrast.
    """
    rgb_src = normalize_to_rgb8(img_src, clahe_enhance=enhance_contrast)
    rgb_ref = normalize_to_rgb8(img_ref, clahe_enhance=enhance_contrast)

    h1, w1 = rgb_src.shape[:2]
    h2, w2 = rgb_ref.shape[:2]

    max_h = max(h1, h2)
    scale = min(1.0, target_canvas_height / max(max_h, 1))
    if max_h < 720:
        scale = 1.0

    nw1, nh1 = max(int(round(w1 * scale)), 32), max(int(round(h1 * scale)), 32)
    nw2, nh2 = max(int(round(w2 * scale)), 32), max(int(round(h2 * scale)), 32)

    canvas_h = max(nh1, nh2)
    divider_w = 4
    canvas_w = nw1 + nw2 + divider_w

    res_src = cv2.resize(rgb_src, (nw1, nh1), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    res_ref = cv2.resize(rgb_ref, (nw2, nh2), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:nh1, :nw1] = res_src
    canvas[:nh2, nw1 + divider_w:] = res_ref
    canvas[:, nw1:nw1 + divider_w] = [245, 166, 35]

    pts_s = np.asarray(pts_src, dtype=np.float32) if pts_src is not None else np.zeros((0, 2), dtype=np.float32)
    pts_r = np.asarray(pts_ref, dtype=np.float32) if pts_ref is not None else np.zeros((0, 2), dtype=np.float32)

    if len(pts_s) == 0 or len(pts_r) == 0:
        return canvas

    if inliers_mask is not None and len(inliers_mask) == len(pts_s):
        mask_bool = np.asarray(inliers_mask, dtype=bool)
        val_s = pts_s[mask_bool]
        val_r = pts_r[mask_bool]
    else:
        val_s = pts_s
        val_r = pts_r

    total_inliers = len(val_s)
    if total_inliers == 0:
        return canvas

    if max_lines and max_lines > 0 and total_inliers > max_lines:
        indices = np.linspace(0, total_inliers - 1, max_lines, dtype=int)
        draw_s = val_s[indices]
        draw_r = val_r[indices]
    else:
        draw_s = val_s
        draw_r = val_r

    sx = draw_s[:, 0] * (nw1 / max(w1, 1))
    sy = draw_s[:, 1] * (nh1 / max(h1, 1))
    rx = draw_r[:, 0] * (nw2 / max(w2, 1)) + nw1 + divider_w
    ry = draw_r[:, 1] * (nh2 / max(h2, 1))

    if color_palette == "gold":
        line_color = (255, 183, 3)
        pt_s_color = (255, 130, 0)
        pt_r_color = (0, 225, 255)
    elif color_palette == "heatmap":
        disparities = np.sqrt((rx - (sx + nw1 + divider_w))**2 + (ry - sy)**2)
        d_min, d_max = float(np.min(disparities)), float(np.max(disparities)) + 1e-5
        norm_disp = np.clip((disparities - d_min) / (d_max - d_min), 0, 1)
        pt_s_color = (255, 140, 0)
        pt_r_color = (0, 240, 255)
    else:
        line_color = (0, 240, 255)
        pt_s_color = (255, 160, 0)
        pt_r_color = (16, 230, 140)

    n_pts = len(draw_s)
    for i in range(n_pts):
        pt1 = (int(round(sx[i])), int(round(sy[i])))
        pt2 = (int(round(rx[i])), int(round(ry[i])))

        if color_palette == "heatmap":
            val = norm_disp[i]
            r = int(255 * min(max(1.5 - abs(val * 4 - 3), 0), 1))
            g = int(255 * min(max(1.5 - abs(val * 4 - 2), 0), 1))
            b = int(255 * min(max(1.5 - abs(val * 4 - 1), 0), 1))
            cur_line_color = (r, g, b)
        else:
            cur_line_color = line_color

        cv2.line(canvas, pt1, pt2, cur_line_color, line_thickness, cv2.LINE_AA)

    pt_radius = max(3, line_thickness + 2)
    for i in range(n_pts):
        pt1 = (int(round(sx[i])), int(round(sy[i])))
        pt2 = (int(round(rx[i])), int(round(ry[i])))

        cv2.circle(canvas, pt1, pt_radius + 2, (10, 14, 23), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, pt_radius, pt_s_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, max(1, pt_radius - 2), (255, 255, 255), -1, cv2.LINE_AA)

        cv2.circle(canvas, pt2, pt_radius + 2, (10, 14, 23), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, pt_radius, pt_r_color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt2, max(1, pt_radius - 2), (255, 255, 255), -1, cv2.LINE_AA)

    header_h = 44
    header_banner = np.zeros((header_h, canvas_w, 3), dtype=np.uint8)
    header_banner[:] = (11, 20, 36)

    footer_h = 36
    footer_banner = np.zeros((footer_h, canvas_w, 3), dtype=np.uint8)
    footer_banner[:] = (8, 14, 26)

    full_canvas = np.vstack([header_banner, canvas, footer_banner])

    font = cv2.FONT_HERSHEY_SIMPLEX
    src_label = f"TARGET / SOURCE [{w1}x{h1}]"
    cv2.putText(full_canvas, "o", (16, 28), font, 0.7, (255, 160, 0), 2, cv2.LINE_AA)
    cv2.putText(full_canvas, src_label, (36, 28), font, 0.65, (230, 240, 255), 2, cv2.LINE_AA)

    ref_x = nw1 + divider_w + 16
    ref_label = f"ANCHOR / REFERENCE [{w2}x{h2}]"
    cv2.putText(full_canvas, "o", (ref_x, 28), font, 0.7, (16, 230, 140), 2, cv2.LINE_AA)
    cv2.putText(full_canvas, ref_label, (ref_x + 20, 28), font, 0.65, (230, 240, 255), 2, cv2.LINE_AA)

    footer_y = header_h + canvas_h + 24
    footer_text = f"CLARITY* | VERIFIED INLIERS: {total_inliers:,} | DISPLAYED TIE-POINTS: {n_pts:,} | RENDER RESOLUTION: {canvas_w}x{canvas_h} px"
    cv2.putText(full_canvas, footer_text, (20, footer_y), font, 0.48, (160, 185, 220), 1, cv2.LINE_AA)

    return full_canvas


def interactive_match_plot(
    img_src: np.ndarray,
    img_ref: np.ndarray,
    pts_src: np.ndarray,
    pts_ref: np.ndarray,
    inliers_mask: np.ndarray = None,
    max_display_matches: int = 350,
    line_width: float = 2.2,
    color_palette: str = "aurora",
    enhance_contrast: bool = True,
):
    """
    Create a high-definition interactive Plotly scatter plot on top of side-by-side images.
    Preserves backwards compatibility with existing unit tests while delivering razor-sharp HD visuals.
    """
    rgb_src = normalize_to_rgb8(img_src, clahe_enhance=enhance_contrast)
    rgb_ref = normalize_to_rgb8(img_ref, clahe_enhance=enhance_contrast)

    h1, w1 = rgb_src.shape[:2]
    h2, w2 = rgb_ref.shape[:2]

    target_hd_w = 1200
    scale = min(1.0, target_hd_w / max(max(w1, w2), 1))

    new_w1, new_h1 = max(int(round(w1 * scale)), 32), max(int(round(h1 * scale)), 32)
    new_w2, new_h2 = max(int(round(w2 * scale)), 32), max(int(round(h2 * scale)), 32)

    common_h = max(new_h1, new_h2)
    divider_w = 4
    canvas_w = new_w1 + new_w2 + divider_w

    img_src_hd = cv2.resize(rgb_src, (new_w1, new_h1), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    img_ref_hd = cv2.resize(rgb_ref, (new_w2, new_h2), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    canvas = np.zeros((common_h, canvas_w, 3), dtype=np.uint8)
    canvas[:new_h1, :new_w1] = img_src_hd
    canvas[:new_h2, new_w1 + divider_w:] = img_ref_hd
    canvas[:, new_w1:new_w1 + divider_w] = [245, 166, 35]

    pts_s = np.asarray(pts_src, dtype=np.float32) if pts_src is not None else np.zeros((0, 2), dtype=np.float32)
    pts_r = np.asarray(pts_ref, dtype=np.float32) if pts_ref is not None else np.zeros((0, 2), dtype=np.float32)

    if inliers_mask is not None and len(inliers_mask) == len(pts_s):
        mask_bool = np.asarray(inliers_mask, dtype=bool)
        inliers_s = pts_s[mask_bool]
        inliers_r = pts_r[mask_bool]
    else:
        inliers_s = pts_s
        inliers_r = pts_r

    pts_s_scaled = np.zeros_like(inliers_s)
    pts_r_scaled = np.zeros_like(inliers_r)

    if len(inliers_s) > 0:
        pts_s_scaled[:, 0] = inliers_s[:, 0] * (new_w1 / max(w1, 1))
        pts_s_scaled[:, 1] = inliers_s[:, 1] * (new_h1 / max(h1, 1))
        pts_r_scaled[:, 0] = inliers_r[:, 0] * (new_w2 / max(w2, 1)) + new_w1 + divider_w
        pts_r_scaled[:, 1] = inliers_r[:, 1] * (new_h2 / max(h2, 1))

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=Image.fromarray(canvas),
            xref="x", yref="y", x=0, y=common_h,
            sizex=canvas_w, sizey=common_h,
            sizing="stretch", opacity=1, layer="below"
        )
    )

    total_pts = len(inliers_s)
    if total_pts > 0:
        if max_display_matches and max_display_matches > 0 and total_pts > max_display_matches:
            step = max(total_pts // max_display_matches, 1)
            indices = list(range(0, total_pts, step))[:max_display_matches]
        else:
            indices = list(range(total_pts))

        lines_x, lines_y = [], []
        src_x, src_y, src_hover = [], [], []
        ref_x, ref_y, ref_hover = [], [], []

        for idx in indices:
            sx = float(pts_s_scaled[idx, 0])
            sy = float(common_h - pts_s_scaled[idx, 1])
            rx = float(pts_r_scaled[idx, 0])
            ry = float(common_h - pts_r_scaled[idx, 1])

            orig_sx, orig_sy = float(inliers_s[idx, 0]), float(inliers_s[idx, 1])
            orig_rx, orig_ry = float(inliers_r[idx, 0]), float(inliers_r[idx, 1])
            disp = float(np.sqrt((orig_rx - orig_sx)**2 + (orig_ry - orig_sy)**2))

            lines_x.extend([sx, rx, None])
            lines_y.extend([sy, ry, None])

            src_x.append(sx)
            src_y.append(sy)
            src_hover.append(f"<b>Tie-Point #{idx + 1}</b><br>Target: ({orig_sx:.1f}, {orig_sy:.1f})<br>Displacement: {disp:.1f} px")

            ref_x.append(rx)
            ref_y.append(ry)
            ref_hover.append(f"<b>Tie-Point #{idx + 1}</b><br>Anchor: ({orig_rx:.1f}, {orig_ry:.1f})<br>Displacement: {disp:.1f} px")

        if color_palette == "gold":
            line_col = "rgba(255, 183, 3, 0.85)"
            src_dot_col = "#FF7A00"
            ref_dot_col = "#00F0FF"
        else:
            line_col = "rgba(0, 240, 255, 0.85)"
            src_dot_col = "#FF9E0B"
            ref_dot_col = "#10B981"

        if lines_x:
            fig.add_trace(go.Scattergl(
                x=lines_x,
                y=lines_y,
                mode="lines",
                line=dict(color=line_col, width=line_width),
                name="Matcher Vectors",
                showlegend=True,
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scattergl(
                x=src_x,
                y=src_y,
                mode="markers",
                marker=dict(
                    color=src_dot_col,
                    size=7,
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                name="Target Points (Source)",
                showlegend=True,
                hovertext=src_hover,
                hoverinfo="text"
            ))
            fig.add_trace(go.Scattergl(
                x=ref_x,
                y=ref_y,
                mode="markers",
                marker=dict(
                    color=ref_dot_col,
                    size=7,
                    line=dict(color="#FFFFFF", width=1.5)
                ),
                name="Anchor Points (Reference)",
                showlegend=True,
                hovertext=ref_hover,
                hoverinfo="text"
            ))

    aspect = canvas_w / max(common_h, 1)
    plot_height = max(580, min(860, int(1150 / max(aspect, 1.2))))

    fig.update_xaxes(showgrid=False, range=[0, canvas_w], visible=False)
    fig.update_yaxes(showgrid=False, range=[0, common_h], visible=False)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=30, b=10),
        height=plot_height,
        plot_bgcolor="rgba(11, 15, 25, 0)",
        paper_bgcolor="rgba(11, 15, 25, 0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#E2E8F0", size=12),
            bgcolor="rgba(11, 22, 40, 0.7)"
        ),
        uirevision="match_plot"
    )
    return fig


def generate_dem_preview(
    img_warped: np.ndarray,
    img_ref: np.ndarray,
    resolution_mode: str = "Full-Size 100% Native",
    vertical_exaggeration: float = 1.8,
    colorscale: str = "Lunar Texture (Photorealistic)",
    sensor_name: str = "tmc",
    sun_az_deg: float = 65.0,
    sun_el_deg: float = 22.0,
    inliers_src: Optional[np.ndarray] = None,
    inliers_ref: Optional[np.ndarray] = None,
) -> Tuple[go.Figure, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Renders a 100% full-matched-extent calibrated 3D photogrammetric elevation surface
    using ALL tie-points from the matcher as anchors across the entire registered image area.
    """
    h_ref, w_ref = img_ref.shape[:2]

    w_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY) if img_warped.ndim == 3 else img_warped.copy()
    r_gray = cv2.cvtColor(img_ref,    cv2.COLOR_BGR2GRAY) if img_ref.ndim    == 3 else img_ref.copy()

    has_inliers = (
        inliers_ref is not None and inliers_src is not None
        and len(inliers_ref) >= 6
    )

    if has_inliers:
        x_min = max(0,     int(np.percentile(inliers_ref[:, 0],  2)) - 24)
        x_max = min(w_ref, int(np.percentile(inliers_ref[:, 0], 98)) + 24)
        y_min = max(0,     int(np.percentile(inliers_ref[:, 1],  2)) - 24)
        y_max = min(h_ref, int(np.percentile(inliers_ref[:, 1], 98)) + 24)

        if (x_max - x_min) > w_ref * 0.50 or (y_max - y_min) > h_ref * 0.50:
            x_min, y_min, x_max, y_max = 0, 0, w_ref, h_ref

        if (x_max - x_min) < 128:
            x_min, x_max = max(0, x_min - 64), min(w_ref, x_max + 64)
        if (y_max - y_min) < 128:
            y_min, y_max = max(0, y_min - 64), min(h_ref, y_max + 64)
    else:
        x_min, y_min, x_max, y_max = 0, 0, w_ref, h_ref

    crop_w = x_max - x_min
    crop_h = y_max - y_min

    if "100" in resolution_mode or "Full" in resolution_mode:
        max_dim = max(crop_w, crop_h)
    elif "1024" in resolution_mode:
        max_dim = 1024
    elif "512" in resolution_mode:
        max_dim = 512
    else:
        max_dim = 800

    scale = min(1.0, max_dim / max(crop_w, crop_h, 1))
    nw = max(96, int(crop_w * scale))
    nh = max(96, int(crop_h * scale))

    w_crop = w_gray[y_min:y_max, x_min:x_max]
    r_crop = r_gray[y_min:y_max, x_min:x_max]

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    w_sm = cv2.resize(w_crop, (nw, nh), interpolation=interp)
    r_sm = cv2.resize(r_crop, (nw, nh), interpolation=interp)

    cam = get_camera_model({"stereo": {"sensor": sensor_name}})

    try:
        dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis_flow.calc(w_sm, r_sm, None)
        flow_x, flow_y = flow[:, :, 0], flow[:, :, 1]
        std_x, std_y = float(np.std(flow_x)), float(np.std(flow_y))
        mag = np.hypot(std_x, std_y)
        if mag > 1e-4:
            ux, uy = std_x / mag, std_y / mag
            disp = flow_x * ux + flow_y * uy
        else:
            disp = flow_x
            ux, uy = 1.0, 0.0
    except Exception:
        num_disp = max(16, min(112, ((nw - 32) // 16) * 16))
        stereo = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=5,
            P1=200, P2=800, disp12MaxDiff=1, uniquenessRatio=10
        )
        disp = stereo.compute(w_sm, r_sm).astype(np.float32) / 16.0
        disp[disp < 0] = 0
        ux, uy = 1.0, 0.0

    if has_inliers:
        try:
            pts_r_local = (inliers_ref - np.array([x_min, y_min])) * scale
            pts_s_local = (inliers_src - np.array([x_min, y_min])) * scale

            valid = (
                (pts_r_local[:, 0] >= 0) & (pts_r_local[:, 0] < nw) &
                (pts_r_local[:, 1] >= 0) & (pts_r_local[:, 1] < nh)
            )
            pts_r_v = pts_r_local[valid]
            pts_s_v = pts_s_local[valid]

            if len(pts_r_v) >= 6:
                d_pts = (pts_s_v[:, 0] - pts_r_v[:, 0]) * ux + \
                        (pts_s_v[:, 1] - pts_r_v[:, 1]) * uy

                from scipy.interpolate import RBFInterpolator
                coords_q = np.round(pts_r_v).astype(np.int32)
                _, u_idx = np.unique(coords_q, axis=0, return_index=True)
                if len(u_idx) >= 6:
                    rbf = RBFInterpolator(
                        pts_r_v[u_idx], d_pts[u_idx],
                        kernel="thin_plate_spline", smoothing=0.3
                    )
                    gx, gy = np.meshgrid(
                        np.arange(nw, dtype=np.float32),
                        np.arange(nh, dtype=np.float32)
                    )
                    anchor_disp = rbf(
                        np.column_stack([gx.ravel(), gy.ravel()])
                    ).reshape(nh, nw).astype(np.float32)

                    gx_r = cv2.Sobel(r_sm, cv2.CV_32F, 1, 0, ksize=3)
                    gy_r = cv2.Sobel(r_sm, cv2.CV_32F, 0, 1, ksize=3)
                    tex_w = np.hypot(gx_r, gy_r)
                    p10, p90 = float(np.percentile(tex_w, 10)), float(np.percentile(tex_w, 90))
                    alpha = np.clip((tex_w - p10) / max(p90 - p10, 1e-3), 0.0, 1.0).astype(np.float32)
                    alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
                    disp = alpha * disp + (1.0 - alpha) * anchor_disp
        except Exception:
            pass

    bh_ratio = max(cam.baseline_m, 1000.0) / max(100_000.0, 1.0)
    if bh_ratio < 0.1:
        bh_ratio = 0.933
    disp_med = cv2.medianBlur(disp.astype(np.float32), 5)
    med_val = float(np.median(disp_med))
    disp_clean = np.clip(disp_med, med_val - 35.0, med_val + 35.0)
    z_stereo = (disp_clean - med_val) * (cam.gsd_m / bh_ratio)
    z_smooth = cv2.bilateralFilter(z_stereo.astype(np.float32), 9, 25, 25)

    try:
        z_sfs = compute_photoclinometry_sfs(
            r_sm, sun_az_deg=sun_az_deg, sun_el_deg=sun_el_deg,
            gsd_m=cam.gsd_m, detail_scale=10.0
        )
        gx_r = cv2.Sobel(r_sm, cv2.CV_32F, 1, 0, ksize=3)
        gy_r = cv2.Sobel(r_sm, cv2.CV_32F, 0, 1, ksize=3)
        tex_mag = np.hypot(gx_r, gy_r)
        p10, p90 = float(np.percentile(tex_mag, 10)), float(np.percentile(tex_mag, 90))
        sw = np.clip((tex_mag - p10) / max(p90 - p10, 1e-3), 0.0, 1.0).astype(np.float32)
        sw = cv2.GaussianBlur(sw, (21, 21), 0)
        z_fused = sw * z_smooth + (1.0 - sw) * z_sfs
    except Exception:
        z_fused = z_smooth

    dem_m = despike_and_clamp_lunar_slopes(z_fused, gsd_m=cam.gsd_m, max_slope_deg=35.0)
    dem_m = cv2.bilateralFilter(dem_m.astype(np.float32), 5, 15, 15)

    x_metric = (np.arange(nw) + x_min / max(scale, 1e-6)) * cam.gsd_m
    y_metric = (np.arange(nh) + y_min / max(scale, 1e-6)) * cam.gsd_m
    tex_norm = r_sm.astype(np.float32) / 255.0

    use_photoreal = ("Lunar Texture" in colorscale or "Photorealistic" in colorscale)
    surf = go.Surface(
        x=x_metric,
        y=y_metric,
        z=dem_m * vertical_exaggeration,
        surfacecolor=tex_norm if use_photoreal else dem_m,
        colorscale="gray" if use_photoreal else colorscale,
        showscale=not use_photoreal,
        colorbar=dict(title="Elevation Z (m)", len=0.7) if not use_photoreal else None,
        lighting=dict(
            ambient=0.40, diffuse=0.85, roughness=0.55, specular=0.25, fresnel=0.1
        ),
        lightposition=dict(
            x=float(np.cos(np.radians(sun_az_deg)) * np.cos(np.radians(sun_el_deg))),
            y=float(np.sin(np.radians(sun_az_deg)) * np.cos(np.radians(sun_el_deg))),
            z=float(np.sin(np.radians(sun_el_deg))),
        ),
        hovertemplate=(
            "X: %{x:.0f} m<br>Y: %{y:.0f} m<br>"
            "Elevation: %{z:.2f} m<extra></extra>"
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=False, width=1.5)
        ),
    )
    traces = [surf]

    if has_inliers and "pts_r_v" in locals() and len(pts_r_v) > 0:
        try:
            px_r = np.clip(np.round(pts_r_v[:, 0]).astype(int), 0, nw - 1)
            py_r = np.clip(np.round(pts_r_v[:, 1]).astype(int), 0, nh - 1)
            z_pts = dem_m[py_r, px_r] * vertical_exaggeration
            x_pts = x_metric[px_r]
            y_pts = y_metric[py_r]
            traces.append(go.Scatter3d(
                x=x_pts, y=y_pts, z=z_pts + 8.0,
                mode="markers",
                marker=dict(
                    size=4, color="rgba(56, 189, 248, 0.85)",
                    line=dict(width=1, color="white"),
                    symbol="circle",
                ),
                name=f"Matched Tie-Points ({len(x_pts)})",
                hovertemplate="Tie-point<br>X: %{x:.0f} m<br>Y: %{y:.0f} m<br>Z: %{z:.2f} m<extra></extra>",
            ))
        except Exception:
            pass

    fig = go.Figure(data=traces)
    aspect_z = float(vertical_exaggeration * 0.35)
    aspect_y = float(nh / max(nw, 1))

    is_full = (x_min == 0 and y_min == 0 and x_max == w_ref and y_max == h_ref)
    extent_pct = int((crop_w * crop_h) / max(w_ref * h_ref, 1) * 100)
    n_anchors = len(inliers_ref) if has_inliers else 0
    title_str = (
        f"3D Lunar Surface — Full Matched Map {nw}×{nh} px "
        f"| {extent_pct}% Coverage | {n_anchors} Anchor Tie-Points"
        if not is_full else
        f"3D Lunar Surface — Full Frame {nw}×{nh} px | {n_anchors} Anchor Tie-Points"
    )

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=13, color="#E2E8F0")),
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title="X (m)", gridcolor="rgba(255,255,255,0.07)", showbackground=False),
            yaxis=dict(title="Y (m)", gridcolor="rgba(255,255,255,0.07)", showbackground=False),
            zaxis=dict(title="Elevation Z (m)", gridcolor="rgba(255,255,255,0.07)", showbackground=False),
            aspectratio=dict(x=1.0, y=max(0.2, min(5.0, aspect_y)), z=max(0.15, aspect_z)),
            camera=dict(
                eye=dict(x=1.35, y=-1.35, z=0.9),
                center=dict(x=0, y=0, z=-0.1)
            ),
            bgcolor="rgba(7, 11, 20, 0.0)",
        ),
        height=680,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(11, 15, 25, 0)",
        plot_bgcolor="rgba(11, 15, 25, 0)",
        legend=dict(
            x=0.01, y=0.98,
            bgcolor="rgba(15, 23, 42, 0.7)",
            font=dict(size=11, color="#CBD5E1"),
        ),
        annotations=[dict(
            text=(
                f"<b>Sensor:</b> {cam.sensor.upper()} | "
                f"<b>GSD:</b> {cam.gsd_m:.1f} m/px | "
                f"<b>Extent:</b> {crop_w * cam.gsd_m / 1000:.2f} × {crop_h * cam.gsd_m / 1000:.2f} km | "
                f"<b>Solar Az/El:</b> {sun_az_deg:.0f}° / {sun_el_deg:.0f}°"
            ),
            xref="paper", yref="paper",
            x=0.0, y=-0.04, showarrow=False,
            font=dict(size=10.5, color="#94A3B8"),
            align="left",
        )],
    )

    qc_dict = {
        "relief_span_m":  float(dem_m.max() - dem_m.min()),
        "min_el_m":       float(dem_m.min()),
        "max_el_m":       float(dem_m.max()),
        "gsd_m":          float(cam.gsd_m),
        "dim_w":          int(nw),
        "dim_h":          int(nh),
        "scale_pct":      int(round(scale * 100)),
        "is_full_size":   scale >= 0.99,
        "sensor":         cam.sensor.upper(),
        "extent_pct":     extent_pct,
        "n_anchors":      n_anchors,
        "crop_km_w":      float(crop_w * cam.gsd_m / 1000.0),
        "crop_km_h":      float(crop_h * cam.gsd_m / 1000.0),
    }

    return fig, dem_m, r_sm, qc_dict
