"""PDS4 metadata and geographic-grid validation for lunar registration."""

import csv
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


_FIELD_ALIASES = {
    "pixel_resolution": "gsd_m",
    "pixel_resolution_m": "gsd_m",
    "focal_length": "focal_length_mm",
    "start_date_time": "start_time",
    "stop_date_time": "stop_time",
    "solar_incidence": "incidence",
    "solar_elevation": "sun_elevation",
    "sensor": "sensor",
    "instrument": "sensor",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_pds4_metadata(label_path: str) -> Dict[str, object]:
    """Read useful scalar and corner metadata from a Chandrayaan PDS4 label."""
    if not label_path or not os.path.exists(label_path):
        raise FileNotFoundError(f"PDS4 label not found: {label_path}")

    root = ET.parse(label_path).getroot()
    values: Dict[str, object] = {"label_path": label_path, "corners": {}}
    for element in root.iter():
        if element.text is None or not element.text.strip():
            continue
        name = _local_name(element.tag)
        text = element.text.strip()
        key = _FIELD_ALIASES.get(name, name)
        unit = str(element.attrib.get("unit", "")).lower()
        value: object = text
        try:
            value = float(text.split()[0])
        except ValueError:
            pass

        if name.endswith("latitude") or name.endswith("longitude"):
            corner_name = name.replace("_latitude", "").replace("_longitude", "")
            values["corners"].setdefault(corner_name, {})[name.rsplit("_", 1)[-1]] = value
        elif key in {"gsd_m", "focal_length_mm", "roll", "pitch", "yaw", "spacecraft_altitude", "incidence", "emission", "phase", "solar_incidence", "sun_elevation", "sun_azimuth"}:
            values[key] = value
        elif key in {"start_time", "stop_time", "sensor"}:
            values[key] = value
        elif unit in {"m/pixel", "m/px"} and isinstance(value, float):
            values["gsd_m"] = value
        elif unit == "mm" and isinstance(value, float):
            values["focal_length_mm"] = value

    return values


def load_geometry_grid(csv_path: str) -> np.ndarray:
    """Load Longitude, Latitude, Pixel, Scan rows from a PDS4 geometry CSV."""
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Geometry CSV not found: {csv_path}")

    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Longitude", "Latitude", "Pixel", "Scan"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Geometry CSV must contain {sorted(required)}: {csv_path}")
        for row in reader:
            try:
                rows.append((float(row["Longitude"]), float(row["Latitude"]), float(row["Pixel"]), float(row["Scan"])))
            except (TypeError, ValueError):
                continue

    if not rows:
        raise ValueError(f"Geometry CSV contains no valid rows: {csv_path}")
    return np.asarray(rows, dtype=np.float64)


def build_geographic_gcp_pairs(
    source_grid: np.ndarray,
    reference_grid: np.ndarray,
    max_distance_m: float = 250.0,
    max_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Build independent pixel/scan GCP pairs from common lunar coordinates."""
    if len(source_grid) == 0 or len(reference_grid) == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), {"status": "empty_grid"}

    # Longitude distance is scaled by latitude so polar grids use local metric distance.
    mean_lat = np.deg2rad(np.mean(np.concatenate((source_grid[:, 1], reference_grid[:, 1]))))
    scale_lon = max(abs(math.cos(mean_lat)), 1e-3)
    source_xy = np.column_stack((source_grid[:, 0] * scale_lon, source_grid[:, 1]))
    reference_xy = np.column_stack((reference_grid[:, 0] * scale_lon, reference_grid[:, 1]))
    tree = cKDTree(reference_xy)
    distances_deg, indices = tree.query(source_xy, k=1)
    distances_m = distances_deg * 111_320.0
    valid = distances_m <= max_distance_m

    matched_source = source_grid[valid]
    matched_reference = reference_grid[indices[valid]]
    if len(matched_source) > max_points:
        keep = np.linspace(0, len(matched_source) - 1, max_points).astype(int)
        matched_source = matched_source[keep]
        matched_reference = matched_reference[keep]

    source_points = matched_source[:, [2, 3]].astype(np.float32)
    reference_points = matched_reference[:, [2, 3]].astype(np.float32)
    diagnostics = {
        "status": "valid" if len(source_points) >= 4 else "insufficient_overlap",
        "candidate_count": int(len(source_grid)),
        "matched_count": int(len(source_points)),
        "max_distance_m": float(max_distance_m),
    }
    return source_points, reference_points, diagnostics


def validate_geographic_overlap(source_grid_path: str, reference_grid_path: str, **kwargs):
    """Load two PDS4 grids and return independent source/reference GCPs."""
    source_grid = load_geometry_grid(source_grid_path)
    reference_grid = load_geometry_grid(reference_grid_path)
    return build_geographic_gcp_pairs(source_grid, reference_grid, **kwargs)
