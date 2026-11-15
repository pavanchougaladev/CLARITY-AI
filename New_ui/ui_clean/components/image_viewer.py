"""Side-by-side, checkerboard, difference and match-line views."""
import numpy as np
from PIL import Image, ImageDraw

def checkerboard(a: np.ndarray, b: np.ndarray, tile: int = 64) -> Image.Image:
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    mask = ((yy // tile) + (xx // tile)) % 2 == 0
    return Image.fromarray(np.where(mask, a, b).astype("uint8"))

def difference(a: np.ndarray, b: np.ndarray) -> Image.Image:
    d = np.abs(a.astype(int) - b.astype(int))
    return Image.fromarray(np.clip(d * 4, 0, 255).astype("uint8"))

def match_lines(a: np.ndarray, b: np.ndarray, matches) -> Image.Image:
    w = a.shape[1]
    canvas = Image.fromarray(np.hstack([a, b])).convert("RGB")
    d = ImageDraw.Draw(canvas)
    for (x1, y1), (x2, y2) in matches:
        d.line([(x1, y1), (x2 + w, y2)], fill=(16, 185, 129), width=1)
    return canvas

def boxes(img: np.ndarray, craters, gsd: float) -> Image.Image:
    out = Image.fromarray(img).convert("RGB"); d = ImageDraw.Draw(out)
    for x, y, r in craters:
        d.rectangle([x - r, y - r, x + r, y + r], outline=(56, 189, 248), width=2)
        d.text((x - r, y - r - 12), f"{2 * r * gsd:.0f} m", fill=(56, 189, 248))
    return out
