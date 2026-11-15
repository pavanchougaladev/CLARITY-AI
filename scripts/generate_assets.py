"""
Generate high-resolution native macOS App Icon (.icns) and DMG installer background.
"""

import os
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont

def create_lunar_icon(size=1024):
    """Draw a rich, modern vector-style lunar registration app icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Rounded squircle background (macOS Big Sur+ style)
    # macOS icon grid: radius ~ 22.5% of size = 230px
    radius = int(size * 0.225)
    margin = int(size * 0.04)
    bbox = [margin, margin, size - margin, size - margin]
    
    # Background gradient fill (Deep space indigo / navy)
    # We create a base mask
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(bbox, radius=radius, fill=255)

    # Base background: deep space blue
    bg = Image.new("RGBA", (size, size), (15, 23, 42, 255))
    bg_draw = ImageDraw.Draw(bg)

    # Radial gradient or concentric space glow
    cx, cy = size // 2, size // 2
    for r in range(int(size * 0.45), 0, -10):
        alpha = int(25 * (1 - r / (size * 0.45)))
        bg_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(30, 58, 138, alpha))

    # Draw Moon Sphere
    moon_radius = int(size * 0.30)
    moon_bbox = [cx - moon_radius, cy - moon_radius, cx + moon_radius, cy + moon_radius]
    bg_draw.ellipse(moon_bbox, fill=(226, 232, 240, 255), outline=(203, 213, 225, 255), width=4)

    # Moon Craters
    craters = [
        (cx - int(moon_radius * 0.35), cy - int(moon_radius * 0.2), int(moon_radius * 0.22)),
        (cx + int(moon_radius * 0.25), cy - int(moon_radius * 0.35), int(moon_radius * 0.16)),
        (cx - int(moon_radius * 0.1), cy + int(moon_radius * 0.3), int(moon_radius * 0.25)),
        (cx + int(moon_radius * 0.35), cy + int(moon_radius * 0.2), int(moon_radius * 0.18)),
        (cx + int(moon_radius * 0.05), cy - int(moon_radius * 0.05), int(moon_radius * 0.12)),
    ]
    for ccx, ccy, cr in craters:
        bg_draw.ellipse([ccx - cr, ccy - cr, ccx + cr, ccy + cr], fill=(203, 213, 225, 255), outline=(148, 163, 184, 255), width=3)
        # Crater shadow
        bg_draw.ellipse([ccx - int(cr*0.7), ccy - int(cr*0.7), ccx + int(cr*0.7), ccy + int(cr*0.7)], fill=(148, 163, 184, 180))

    # Crosshairs / Reticle for Registration Alignment (Cyan / Saffron / Neon Gold)
    # Reticle rings
    bg_draw.ellipse([cx - int(moon_radius * 1.15), cy - int(moon_radius * 1.15),
                     cx + int(moon_radius * 1.15), cy + int(moon_radius * 1.15)],
                    outline=(56, 189, 248, 200), width=6)
    
    # Tick marks / crosshair ticks
    tick_len = int(size * 0.06)
    # North, South, East, West ticks
    bg_draw.line([(cx, cy - int(moon_radius * 1.25)), (cx, cy - int(moon_radius * 1.25) + tick_len)], fill=(56, 189, 248, 240), width=6)
    bg_draw.line([(cx, cy + int(moon_radius * 1.25)), (cx, cy + int(moon_radius * 1.25) - tick_len)], fill=(56, 189, 248, 240), width=6)
    bg_draw.line([(cx - int(moon_radius * 1.25), cy), (cx - int(moon_radius * 1.25) + tick_len, cy)], fill=(56, 189, 248, 240), width=6)
    bg_draw.line([(cx + int(moon_radius * 1.25), cy), (cx + int(moon_radius * 1.25) - tick_len, cy)], fill=(56, 189, 248, 240), width=6)

    # Spacecraft orbit trajectory arc (Golden / ISRO Orange #FF9933)
    orbit_bbox = [cx - int(size * 0.42), cy - int(size * 0.18), cx + int(size * 0.42), cy + int(size * 0.38)]
    bg_draw.arc(orbit_bbox, start=200, end=350, fill=(249, 115, 22, 255), width=8)

    # Satellite diamond symbol on the orbit
    sat_x = cx + int(size * 0.36)
    sat_y = cy - int(size * 0.05)
    s_rad = int(size * 0.035)
    bg_draw.polygon([
        (sat_x, sat_y - s_rad),
        (sat_x + s_rad, sat_y),
        (sat_x, sat_y + s_rad),
        (sat_x - s_rad, sat_y)
    ], fill=(255, 255, 255, 255), outline=(249, 115, 22, 255))

    # Apply rounded squircle mask
    img.paste(bg, (0, 0), mask)
    return img

def build_icns(icon_img, output_icns_path):
    """Build .icns file using macOS iconutil tool."""
    iconset_dir = "/tmp/AppIcon.iconset"
    os.makedirs(iconset_dir, exist_ok=True)

    sizes = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]

    for s, name in sizes:
        resized = icon_img.resize((s, s), Image.Resampling.LANCZOS)
        resized.save(os.path.join(iconset_dir, name))

    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", output_icns_path], check=True)
    print(f"[✓] Generated native macOS AppIcon: {output_icns_path}")

def create_dmg_background(output_path, width=800, height=450):
    """Create a sleek, retina-ready DMG installer background with installation guide."""
    # Scale 2x for crisp Retina rendering
    scale = 2
    w, h = width * scale, height * scale
    img = Image.new("RGB", (w, h), (13, 17, 23)) # GitHub dark style
    draw = ImageDraw.Draw(img)

    # Subtle gradient bands
    for y in range(h):
        ratio = y / h
        r = int(13 * (1 - ratio) + 22 * ratio)
        g = int(17 * (1 - ratio) + 27 * ratio)
        b = int(23 * (1 - ratio) + 34 * ratio)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Top banner bar (ISRO lunar theme)
    draw.rectangle([0, 0, w, int(70 * scale)], fill=(19, 24, 32))
    draw.line([0, int(70 * scale), w, int(70 * scale)], fill=(48, 54, 61), width=scale)

    # Title & Subtitle text
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/SFPro-Bold.ttf", 20 * scale)
        font_sub = ImageFont.truetype("/System/Library/Fonts/SFPro-Regular.ttf", 13 * scale)
        font_inst = ImageFont.truetype("/System/Library/Fonts/SFPro-Medium.ttf", 14 * scale)
        font_tag = ImageFont.truetype("/System/Library/Fonts/SFPro-Semibold.ttf", 11 * scale)
    except Exception:
        font_title = font_sub = font_inst = font_tag = ImageFont.load_default()

    draw.text((35 * scale, 18 * scale), "🌔 ISRO Lunar Image Registration Pipeline", fill=(240, 246, 252), font=font_title)
    draw.text((35 * scale, 42 * scale), "Chandrayaan-1 / Chandrayaan-2 Automated Co-Registration Suite", fill=(139, 148, 158), font=font_sub)

    # Installation Drag-and-Drop Arrow in the center
    # Icons will be at x=200, y=220 (App) and x=600, y=220 (Applications)
    # Center arrow between x=330 and x=470
    arrow_y = int(220 * scale)
    arrow_start_x = int(330 * scale)
    arrow_end_x = int(470 * scale)

    # Draw stylish dashed / gradient arrow
    draw.line([(arrow_start_x, arrow_y), (arrow_end_x, arrow_y)], fill=(88, 166, 255), width=4 * scale)
    
    # Arrow head
    head_size = int(16 * scale)
    draw.polygon([
        (arrow_end_x, arrow_y),
        (arrow_end_x - head_size, arrow_y - head_size // 2),
        (arrow_end_x - head_size, arrow_y + head_size // 2),
    ], fill=(88, 166, 255))

    # Installation Instruction below arrow
    inst_text = "Drag to Applications to Install"
    draw.text((int(320 * scale), arrow_y + int(25 * scale)), inst_text, fill=(139, 148, 158), font=font_inst)

    # Bottom footer with ARM64 badge and ISRO SAC note
    draw.line([0, h - int(45 * scale), w, h - int(45 * scale)], fill=(48, 54, 61), width=scale)
    draw.text((35 * scale, h - int(32 * scale)), "Apple Silicon ARM64 Native  •  FastAPI REST API  •  Streamlit UI  •  Deep Learning LoFTR/LightGlue", fill=(110, 118, 129), font=font_tag)

    img.save(output_path, "PNG")
    print(f"[✓] Generated DMG Installer Background: {output_path}")

def main():
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
    os.makedirs(assets_dir, exist_ok=True)

    icon_img = create_lunar_icon(1024)
    png_icon_path = os.path.join(assets_dir, "app_icon.png")
    icon_img.save(png_icon_path)
    print(f"[✓] Saved raw app icon: {png_icon_path}")

    icns_path = os.path.join(assets_dir, "AppIcon.icns")
    build_icns(icon_img, icns_path)

    bg_path = os.path.join(assets_dir, "dmg_background.png")
    create_dmg_background(bg_path, width=800, height=450)

if __name__ == "__main__":
    main()
