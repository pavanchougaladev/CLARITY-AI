"""Side-by-side, checkerboard, difference, homography warp, and match-line views."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

def align_image(src: np.ndarray, ref: np.ndarray, matches: list) -> np.ndarray:
    """Warps source image onto reference coordinate frame using homography from tie-points."""
    try:
        import cv2
        if len(matches) >= 4:
            pts_s = np.float32([m[0] for m in matches])
            pts_r = np.float32([m[1] for m in matches])
            h_mat, _ = cv2.findHomography(pts_s, pts_r, cv2.RANSAC, 3.0)
            if h_mat is not None:
                warped = cv2.warpPerspective(src, h_mat, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR)
                return warped
    except Exception:
        pass
    return src

def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 48) -> Image.Image:
    """Interleaves alternating tiles of images a and b to evaluate boundary seam continuity."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy // tile) + (xx // tile)) % 2 == 0
    composite = np.where(mask, a[:h, :w], b[:h, :w]).astype("uint8")
    img = Image.fromarray(composite).convert("RGB")
    
    # Overlay subtle grid lines along tile borders for visual clarity
    draw = ImageDraw.Draw(img)
    for y in range(0, h, tile):
        draw.line([(0, y), (w, y)], fill=(56, 189, 248, 80), width=1)
    for x in range(0, w, tile):
        draw.line([(x, 0), (x, h)], fill=(56, 189, 248, 80), width=1)
    return img

def difference(a: np.ndarray, b: np.ndarray, amplify: int = 4) -> Image.Image:
    """Computes amplified absolute difference map between images."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    diff = np.abs(a[:h, :w].astype(np.int32) - b[:h, :w].astype(np.int32))
    diff_amplified = np.clip(diff * amplify, 0, 255).astype("uint8")
    
    # Colormap: grayscale base with cyan highlight on residual errors
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(diff_amplified * 0.4, 0, 255)
    rgb[:, :, 1] = np.clip(diff_amplified * 0.8, 0, 255)
    rgb[:, :, 2] = diff_amplified
    return Image.fromarray(rgb)

def match_lines(a: np.ndarray, b: np.ndarray, matches: list, max_lines: int = 140) -> Image.Image:
    """Renders side-by-side tie-point vector field connecting source and reference."""
    h = min(a.shape[0], b.shape[0])
    w1, w2 = a.shape[1], b.shape[1]
    canvas = Image.fromarray(np.hstack([a[:h, :w1], b[:h, :w2]])).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    
    step = max(1, len(matches) // max_lines)
    sub_matches = matches[::step][:max_lines]
    
    for (x1, y1), (x2, y2) in sub_matches:
        if 0 <= y1 < h and 0 <= x1 < w1 and 0 <= y2 < h and 0 <= x2 < w2:
            # Tie line
            draw.line([(x1, y1), (x2 + w1, y2)], fill=(16, 185, 129, 140), width=1)
            # Source point (Cyan)
            draw.ellipse([x1 - 2.5, y1 - 2.5, x1 + 2.5, y1 + 2.5], fill=(56, 189, 248), outline=(255, 255, 255))
            # Reference point (Emerald)
            draw.ellipse([x2 + w1 - 2.5, y2 - 2.5, x2 + w1 + 2.5, y2 + 2.5], fill=(16, 185, 129), outline=(255, 255, 255))
            
    return canvas

def boxes(img: np.ndarray, craters: list, gsd: float = 0.25, show_depth: bool = True) -> Image.Image:
    """Draws YOLO11 and morphological crater bounding annotations with diameter and depth telemetry."""
    out = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(out)

    for idx, c in enumerate(craters):
        if isinstance(c, dict):
            cx = float(c.get("cx", 0))
            cy = float(c.get("cy", 0))
            r = float(c.get("r", 15))
            d_m = float(c.get("diameter_m", 2 * r * gsd))
            depth_m = float(c.get("depth_m", 0.0))
            hazard = c.get("hazard_level", "MODERATE HAZARD")
            conf = float(c.get("confidence", 0.90))
            cid = c.get("id", idx + 1)
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
            d_m = 2 * r * gsd
            depth_m = d_m * 0.14
            hazard = "CRITICAL HAZARD" if d_m >= 80 else ("MODERATE HAZARD" if d_m >= 35 else "MICRO HAZARD")
            conf = float(c[3]) if len(c) > 3 else 0.85
            cid = idx + 1
        else:
            continue

        # Color based on hazard rating
        if hazard == "CRITICAL HAZARD":
            color = (239, 68, 68)      # Coral Red
        elif hazard == "MODERATE HAZARD":
            color = (245, 158, 11)     # Amber Orange
        else:
            color = (56, 189, 248)      # Sky Cyan

        # Bounding circle & box
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Centroid crosshair
        draw.line([(cx - 3, cy), (cx + 3, cy)], fill=(255, 255, 255), width=1)
        draw.line([(cx, cy - 3), (cx, cy + 3)], fill=(255, 255, 255), width=1)

        # Telemetry badge
        if show_depth and depth_m > 0:
            label = f"C-{cid:02d} D:{d_m:.0f}m d:{depth_m:.1f}m"
        else:
            label = f"C-{cid:02d} D:{d_m:.0f}m ({int(conf*100)}%)"

        bw = max(60, len(label) * 7 + 6)
        by1 = max(0.0, y1 - 16)
        by2 = max(14.0, y1)
        draw.rectangle([x1, by1, x1 + bw, by2], fill=(10, 10, 13))
        draw.text((x1 + 3, by1 + 1), label, fill=color)

    return out


def polygons(
    img: np.ndarray,
    craters: list,
    gsd: float = 0.25,
    show_depth: bool = True,
    fill_alpha: float = 0.35,
    show_boxes: bool = False,
) -> Image.Image:
    """Renders SAM 2 pixel-exact crater rim polygons with translucent fills, 2px crest contours, and circularity telemetry."""
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)

    for idx, c in enumerate(craters):
        if not isinstance(c, dict):
            continue

        cx = float(c.get("cx", 0))
        cy = float(c.get("cy", 0))
        r = float(c.get("r", 15))
        d_m = float(c.get("diameter_m", 2 * r * gsd))
        depth_m = float(c.get("depth_m", 0.0))
        hazard = c.get("hazard_level", "MODERATE HAZARD")
        circ = float(c.get("circularity", 0.95))
        cid = c.get("id", idx + 1)
        poly = c.get("polygon_pts", [])

        if hazard == "CRITICAL HAZARD":
            rgb = (239, 68, 68)      # Coral Red
        elif hazard == "MODERATE HAZARD":
            rgb = (245, 158, 11)     # Amber Orange
        else:
            rgb = (56, 189, 248)      # Sky Cyan

        rgba_fill = (rgb[0], rgb[1], rgb[2], int(fill_alpha * 255))
        rgba_line = (rgb[0], rgb[1], rgb[2], 255)

        if poly and len(poly) >= 3:
            poly_tuples = [(float(p[0]), float(p[1])) for p in poly]
            draw_ov.polygon(poly_tuples, fill=rgba_fill)
            for i in range(len(poly_tuples)):
                p1 = poly_tuples[i]
                p2 = poly_tuples[(i + 1) % len(poly_tuples)]
                draw_ov.line([p1, p2], fill=rgba_line, width=2)
        else:
            x1, y1 = cx - r, cy - r
            x2, y2 = cx + r, cy + r
            draw_ov.ellipse([x1, y1, x2, y2], fill=rgba_fill, outline=rgba_line, width=2)

        if show_boxes:
            x1, y1 = cx - r, cy - r
            x2, y2 = cx + r, cy + r
            draw_ov.rectangle([x1, y1, x2, y2], outline=(rgb[0], rgb[1], rgb[2], 120), width=1)

        # Centroid marker
        draw_ov.line([(cx - 3.5, cy), (cx + 3.5, cy)], fill=(255, 255, 255, 255), width=1)
        draw_ov.line([(cx, cy - 3.5), (cx, cy + 3.5)], fill=(255, 255, 255, 255), width=1)

        # Telemetry Badge
        if show_depth and depth_m > 0:
            label = f"C-{cid:02d} D:{d_m:.0f}m d:{depth_m:.1f}m (C:{circ:.2f})"
        else:
            label = f"C-{cid:02d} D:{d_m:.0f}m (C:{circ:.2f})"

        bw = max(68, len(label) * 7 + 8)
        by1 = max(0.0, cy - r - 16)
        by2 = max(14.0, cy - r)
        bx1 = max(0.0, cx - r)
        draw_ov.rectangle([bx1, by1, bx1 + bw, by2], fill=(10, 10, 13, 220))
        draw_ov.rectangle([bx1, by1, bx1 + bw, by2], outline=(rgb[0], rgb[1], rgb[2], 180), width=1)
        draw_ov.text((bx1 + 4, by1 + 1), label, fill=rgba_line)

    composited = Image.alpha_composite(base, overlay)
    return composited.convert("RGB")


def draw_landing_sites_overlay(
    img: np.ndarray,
    sites: list,
    craters: list,
    gsd: float = 0.25,
    footprint_radius_m: float = 20.0,
) -> Image.Image:
    """Renders autonomous candidate landing footprints with safety ratings and crater exclusion buffers."""
    out = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(out)
    h_img, w_img = img.shape[:2]

    # 1. Hazard exclusion boundary around known craters (True Polygon or Circle)
    for c in craters:
        if isinstance(c, dict):
            cx, cy, r = float(c.get("cx", 0)), float(c.get("cy", 0)), float(c.get("r", 15))
            poly = c.get("polygon_pts", [])
            if poly and len(poly) >= 3:
                buf_poly = []
                for px, py in poly:
                    bx = (px - cx) * 1.35 + cx
                    by = (py - cy) * 1.35 + cy
                    buf_poly.append((bx, by))
                for i in range(len(buf_poly)):
                    p1 = buf_poly[i]
                    p2 = buf_poly[(i + 1) % len(buf_poly)]
                    draw.line([p1, p2], fill=(239, 68, 68, 140), width=1)
                continue
        elif isinstance(c, (list, tuple)) and len(c) >= 3:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])
        else:
            continue
        # Exclusion radius = 1.35x crater rim fallback
        r_ex = r * 1.35
        draw.ellipse([cx - r_ex, cy - r_ex, cx + r_ex, cy + r_ex], outline=(239, 68, 68, 120), width=1)

    # 2. Render ranked landing sites
    r_px = max(8.0, footprint_radius_m / max(gsd, 0.05))

    for idx, s in enumerate(sites):
        cx = float(s.get("cx_px", w_img // 2))
        cy = float(s.get("cy_px", h_img // 2))
        score = float(s.get("safety_score", 95.0))
        status = s.get("status", "CERTIFIED SAFE")
        site_id = s.get("site_id", f"SITE-{idx+1:02d}")
        mean_s = float(s.get("mean_slope", 4.5))

        if "SAFE" in status.upper():
            c_ring = (34, 197, 94)    # Neon Green
            c_fill = (16, 185, 129)
        elif "MARGINAL" in status.upper() or "CAUTION" in status.upper():
            c_ring = (245, 158, 11)   # Amber
            c_fill = (245, 158, 11)
        else:
            c_ring = (239, 68, 68)    # Red
            c_fill = (239, 68, 68)

        # Concentric landing footprint circles
        draw.ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px], outline=c_ring, width=2)
        draw.ellipse([cx - r_px * 0.45, cy - r_px * 0.45, cx + r_px * 0.45, cy + r_px * 0.45], outline=c_ring, width=1)
        # Center target crosshair
        draw.line([(cx - r_px * 0.8, cy), (cx + r_px * 0.8, cy)], fill=(255, 255, 255), width=1)
        draw.line([(cx, cy - r_px * 0.8), (cx, cy + r_px * 0.8)], fill=(255, 255, 255), width=1)

        # Site Info Badge
        badge_txt = f"★ {site_id} [{score:.1f}%] Slope:{mean_s:.1f}°" if idx == 0 else f"{site_id} [{score:.1f}%]"
        bw = max(70, len(badge_txt) * 7 + 8)
        bx1 = max(0.0, cx - bw // 2)
        by1 = max(0.0, cy - r_px - 18)
        bx2 = bx1 + bw
        by2 = by1 + 16

        draw.rectangle([bx1, by1, bx2, by2], fill=(10, 10, 13))
        draw.rectangle([bx1, by1, bx2, by2], outline=c_ring, width=1)
        draw.text((bx1 + 4, by1 + 1), badge_txt, fill=c_fill)

    return out
