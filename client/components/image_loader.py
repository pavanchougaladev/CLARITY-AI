"""
Image and Metadata Ingestion Service
===================================
ISRO LunarFusion Co-Registration Suite — UI Image Loader
Handles safe previews, PDS4 archive metadata resolution, and orbital image ingestion.
"""

import os
import glob
import tempfile
import hashlib
import shutil
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional, Dict, List, Any
import logging

# Workspace paths resolved relative to this module (client/components/ → project root)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.abspath(os.path.join(WORKSPACE_ROOT, "data"))

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def save_uploaded_metadata(uploaded_file, suffix: str) -> str:
    """Persist a metadata upload for the current Streamlit run."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    return temp_file.name


def save_uploaded_orbital_image(uploaded_file) -> str:
    """
    Persist an uploaded image (PNG, JPG, TIFF, or PDS .img), preserving its
    original filename so sibling PDS4 metadata and rasterio loaders can bind properly.
    """
    upload_dir = os.path.join(tempfile.gettempdir(), "lunar_orbital_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    clean_name = os.path.basename(uploaded_file.name)
    target_path = os.path.join(upload_dir, clean_name)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer() if hasattr(uploaded_file, "getbuffer") else uploaded_file.read())

    # For .img files, check if a matching .xml exists in project data/ and copy alongside if available
    ext = os.path.splitext(clean_name)[1].lower()
    if ext == ".img":
        base = os.path.splitext(clean_name)[0]
        xml_name = f"{base}.xml"
        target_xml = os.path.join(upload_dir, xml_name)
        if not os.path.exists(target_xml):
            search_root = DATA_DIR
            for root, dirs, files in os.walk(search_root):
                if xml_name in files:
                    try:
                        shutil.copyfile(os.path.join(root, xml_name), target_xml)
                    except Exception:
                        pass
                    break

    return target_path


def sha256_file(path: str) -> str:
    """Calculate cryptographic SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_archive_metadata(image_path: str) -> dict:
    """Resolve the image's sibling PDS4 label and geographic grid products."""
    image_name = os.path.basename(image_path)
    stem = image_name.rsplit(".", 1)[0]
    if "_b_brw_" in stem:
        product_stem = stem.replace("_b_brw_", "_d_img_")
    elif "_d_img_" in stem:
        product_stem = stem
    else:
        product_stem = stem
    grid_stem = product_stem.replace("_d_img_", "_g_grd_")
    search_root = WORKSPACE_ROOT
    img_dir = os.path.dirname(os.path.abspath(image_path))

    # Priority 1: Check immediate sibling directory
    sibling_labels = [p for p in [os.path.join(img_dir, product_stem + ".xml"), os.path.join(img_dir, grid_stem + ".xml")] if os.path.exists(p)]
    sibling_grids = [p for p in [os.path.join(img_dir, grid_stem + ".csv")] if os.path.exists(p)]

    labels = sibling_labels if sibling_labels else [p for p in glob.glob(os.path.join(search_root, "**", product_stem + ".xml"), recursive=True) if os.path.exists(p)]
    if not labels:
        labels = [p for p in glob.glob(os.path.join(search_root, "**", grid_stem + ".xml"), recursive=True) if os.path.exists(p)]

    grids = sibling_grids if sibling_grids else [p for p in glob.glob(os.path.join(search_root, "**", grid_stem + ".csv"), recursive=True) if os.path.exists(p)]

    if not labels or not grids:
        return {}
    return {"label": labels[0], "grid": grids[0]}


def metadata_for_archive_pair(source_path: str, reference_path: str) -> dict:
    """Resolve archive metadata pairs for source and reference images."""
    source = find_archive_metadata(source_path)
    reference = find_archive_metadata(reference_path)
    if not source or not reference:
        return {}
    return {
        "src_pds4_label": source["label"],
        "ref_pds4_label": reference["label"],
        "src_geometry_csv": source["grid"],
        "ref_geometry_csv": reference["grid"],
    }


@st.cache_data(max_entries=64)
def get_safe_preview(image_path: str, max_dim: int = 800) -> Optional[np.ndarray]:
    """
    Safely load a lightweight display thumbnail for any image format (including
    gigantic 1-billion-pixel PDS4/GeoTIFF images like OHRC) without triggering
    DecompressionBombError or running out of memory.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".img":
            try:
                from src.ingest.pds4_parser import load_img_raster
                return load_img_raster(image_path, max_dim=max_dim)
            except Exception:
                pass
        elif ext in [".tif", ".tiff"]:
            try:
                import rasterio
                with rasterio.open(image_path) as ds:
                    h, w = ds.height, ds.width
                    scale = max_dim / max(h, w)
                    nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
                    arr = ds.read(1, out_shape=(nh, nw))
                    if arr.dtype != np.uint8:
                        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    return arr
            except Exception:
                import tifffile
                with tifffile.TiffFile(image_path) as tif:
                    page = tif.pages[0]
                    h, w = page.shape[:2]
                    step = max(1, int(max(h, w) / max_dim))
                    arr = page.asarray()[::step, ::step]
                    if arr.ndim == 3:
                        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                    if arr.dtype != np.uint8:
                        arr = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    return arr
        else:
            arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                h, w = arr.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    arr = cv2.resize(arr, (max(16, int(w * scale)), max(16, int(h * scale))), interpolation=cv2.INTER_AREA)
                return arr
    except Exception:
        pass

    try:
        with Image.open(image_path) as im:
            im.thumbnail((max_dim, max_dim))
            return np.array(im.convert("L"))
    except Exception:
        return None


@st.cache_data
def discover_browse_images(data_dir: str) -> list:
    """Discover available browse images from the dataset directory."""
    patterns = [
        os.path.join(data_dir, "**", "*_brw_*.png"),
        os.path.join(data_dir, "**", "*_b_brw_*.png"),
    ]
    found_paths = set()
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if "lunar_crater_yolo" not in p and "outputs" not in p and "archive" not in p:
                found_paths.add(p)
    found = sorted(list(found_paths))
    images = []
    for path in found:
        basename = os.path.basename(path)
        parts = basename.split("_")
        mission = parts[0].upper() if len(parts) > 0 else "CH2"
        sensor_code = parts[1].upper() if len(parts) > 1 else "TMC"
        sub_code = parts[2] if len(parts) > 2 else ""

        if sensor_code == "OHR":
            sensor_name = "OHRC (High-Res Pan 25cm)"
            resolution_label = "0.25 m/px"
        elif sensor_code == "TMC":
            if "nra" in sub_code:
                sensor_name = "TMC-2 Aft (5m)"
                resolution_label = "5.0 m/px"
            elif "nrf" in sub_code:
                sensor_name = "TMC-2 Fore (5m)"
                resolution_label = "5.0 m/px"
            elif "nrn" in sub_code:
                sensor_name = "TMC-2 Nadir (5m)"
                resolution_label = "5.0 m/px"
            elif "ncf" in sub_code:
                sensor_name = "TMC Fore (Ch-1)"
                resolution_label = "5.0 m/px"
            else:
                sensor_name = "TMC (Terrain Camera)"
                resolution_label = "5.0 m/px"
        elif "IIR" in sensor_code or "APOLLO" in path.upper():
            sensor_name = "Apollo / IIRS Multi-Mission"
            resolution_label = "Orbital"
        else:
            sensor_name = sensor_code
            resolution_label = "Orbital"

        date_str = "2023-08"
        path_parts = path.replace("\\", "/").split("/")
        for p in path_parts:
            if len(p) == 8 and p.isdigit():
                date_str = f"{p[:4]}-{p[4:6]}-{p[6:]}"
                break

        label = f"📷 {mission} | {sensor_name} | {date_str} ({basename[:25]}...)"
        images.append({
            "path": path, "label": label, "basename": basename,
            "mission": mission, "sensor": sensor_name, "date": date_str,
            "res": resolution_label
        })
    return images


@st.cache_resource
def get_cached_crater_detector(model_path: str = "models/yolo11_lunar_crater.pt"):
    """Cached loader for YOLO11 Lunar Crater detector model."""
    try:
        from src.features.crater_detector import LunarCraterDetector
        detector = LunarCraterDetector(model_path)
        if getattr(detector, "model", None) is not None:
            return detector
    except Exception as e:
        logger.warning(f"Primary Lunar YOLO11 model load failed ({e}), attempting fallback.")
    try:
        from src.features.crater_detector import LunarCraterDetector
        detector = LunarCraterDetector("yolo11n.pt")
        if getattr(detector, "model", None) is not None:
            return detector
    except Exception as e:
        logger.error(f"Fallback YOLO model load failed: {e}")
    return None


# Backward compatibility aliases
_save_uploaded_metadata = save_uploaded_metadata
_save_uploaded_orbital_image = save_uploaded_orbital_image
_sha256_file = sha256_file
_find_archive_metadata = find_archive_metadata
_metadata_for_archive_pair = metadata_for_archive_pair
