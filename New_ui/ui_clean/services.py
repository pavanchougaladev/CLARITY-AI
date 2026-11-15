"""All data access lives here so views stay free of model code.

DEMO mode generates synthetic lunar-like data so the UI runs on its own.
To use your real pipeline, set USE_REAL_PIPELINE = True and fill in
_real_registration() and (optionally) the other TODO hooks.
"""
import io, time
import numpy as np
import streamlit as st
from PIL import Image
from config import PRESETS, GSD_REF, DEM_SPACING

USE_REAL_PIPELINE = False

# ---------- synthetic data (demo mode) ----------
@st.cache_data
def synth_patch(seed: int, noise_seed: int, shift=(0, 0), size=512, n=18):
    """Return a uint8 lunar-like patch and its crater list (x, y, r)."""
    rng = np.random.default_rng(seed)
    craters = [(rng.uniform(50, size - 50), rng.uniform(50, size - 50), rng.uniform(12, 48)) for _ in range(n)]
    noise = np.random.default_rng(noise_seed).standard_normal((size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    img = 0.35 + 0.03 * noise
    for cx, cy, r in craters:
        cx, cy = cx + shift[0], cy + shift[1]
        d = np.hypot(xx - cx, yy - cy)
        img -= 0.18 * np.exp(-(d / r) ** 4)
        img += 0.12 * np.exp(-((d - r) / 3) ** 2) * (1 + np.cos(np.arctan2(yy - cy, xx - cx) + 0.8)) / 2
    return (np.clip(img, 0, 1) * 255).astype("uint8"), craters

def to_gray512(file) -> np.ndarray:
    return np.array(Image.open(file).convert("L").resize((512, 512)))

def get_pair(preset: str, up_src=None, up_ref=None):
    """Source and reference images (uploads win over the preset)."""
    seed = PRESETS[preset]["seed"]
    src = to_gray512(up_src) if up_src else synth_patch(seed, 1)[0]
    ref = to_gray512(up_ref) if up_ref else synth_patch(seed, 2, shift=(6, -4))[0]
    return src, ref

@st.cache_data
def make_dem(n=80, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    z = np.zeros((n, n))
    for _ in range(5):
        cx, cy, r = rng.uniform(10, n - 10), rng.uniform(10, n - 10), rng.uniform(5, 14)
        d = np.hypot(xx - cx, yy - cy)
        z += -r * 0.9 * np.exp(-(d / r) ** 2) + r * 0.3 * np.exp(-((d - r) / (0.25 * r)) ** 2)
    return z + 0.3 * rng.standard_normal((n, n))

def slope_deg(z: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(z, DEM_SPACING)
    return np.degrees(np.arctan(np.hypot(gx, gy)))

# ---------- registration ----------
def _real_registration(src, ref, settings) -> dict:
    """TODO: call your pipeline here and return the same keys as the demo result."""
    raise NotImplementedError("Connect src/pipeline.py here")

def run_registration(src, ref, settings: dict, preset: str) -> dict:
    if USE_REAL_PIPELINE:
        return _real_registration(src, ref, settings)
    time.sleep(0.5)
    seed = PRESETS[preset]["seed"]
    rng = np.random.default_rng(seed)
    _, cs = synth_patch(seed, 1)
    matches = [((x, y), (x + 6 + rng.normal(0, .3), y - 4 + rng.normal(0, .3))) for x, y, _ in cs]
    res = dict(is_sample=True, points=1284, inliers=1420, rmse=0.38, confidence=99.4,
               entropy=84.2, matches=matches, craters=cs, settings=settings)
    res["gates"] = {"overfit": res["rmse"] < 0.5, "inliers": res["inliers"] >= 15, "entropy": res["entropy"] >= 60}
    res["grade"] = "A" if all(res["gates"].values()) else "C"
    return res

# ---------- exports ----------
def dem_obj(z: np.ndarray) -> str:
    n = z.shape[0]
    lines = [f"v {x*DEM_SPACING} {y*DEM_SPACING} {z[y, x]:.3f}" for y in range(n) for x in range(n)]
    for y in range(n - 1):
        for x in range(n - 1):
            a = y * n + x + 1
            lines += [f"f {a} {a+1} {a+n}", f"f {a+1} {a+n+1} {a+n}"]
    return "\n".join(lines)

def dem_ply(z: np.ndarray) -> str:
    n = z.shape[0]
    head = f"ply\nformat ascii 1.0\nelement vertex {n*n}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    return head + "\n".join(f"{x*DEM_SPACING} {y*DEM_SPACING} {z[y, x]:.3f}" for y in range(n) for x in range(n))

def dem_tiff(z: np.ndarray):
    try:
        import tifffile
    except ImportError:
        return None
    buf = io.BytesIO(); tifffile.imwrite(buf, z.astype("float32")); return buf.getvalue()
