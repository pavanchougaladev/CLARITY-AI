"""
ImageDescriptor — Typed Image Metadata Container
=================================================
Implements the typed image descriptor required by §3 of the CLARITY3D brief.

Fields
------
Every image processed by the pipeline is described by an ``ImageDescriptor``
that records:

* Identity — image_id, path, sha256 checksum
* Sensor — sensor name, product_level, band descriptions
* Radiometry — dtype, scale, offset, nodata, saturation flag
* Geometry — CRS, affine transform, pixel_size, footprint
* Illumination — sun_azimuth, sun_elevation, acquisition time
* Provenance — PDS label path, calibration metadata reference

This descriptor is serialised to ``provenance.json`` alongside every
registration job so that every artifact can be traced back to its origin.

Notes
-----
- For JPEG/PNG inputs without trustworthy metadata the geographic fields are
  left as ``None`` rather than being invented.
- For planetary PDS4 products the CRS is always declared in a lunar frame
  (LROC, MOON 2000, or IAU-derived).  Earth CRS is never silently assigned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate-convention sentinel so callers know what (0,0) means
# ---------------------------------------------------------------------------
PIXEL_ORIGIN_CENTRE = "pixel_centre"   # PDS4 convention
PIXEL_ORIGIN_CORNER = "pixel_corner"   # GeoTIFF / rasterio convention


@dataclass
class ImageDescriptor:
    """
    Typed metadata container for one image entering the registration pipeline.

    All fields are optional except *path*; missing values are explicitly None.
    Never invent values that are not present in labels or metadata.
    """

    # --- Identity ---
    image_id: str = ""
    path: str = ""
    sha256: Optional[str] = None        # populated lazily via compute_checksum()

    # --- Sensor ---
    sensor: str = ""                     # e.g. "TMC-2", "OHRC", "IIRS", "LROC-WAC"
    product_level: str = ""              # e.g. "EDR", "CDR", "RDR"
    dims: Tuple[int, int] = (0, 0)      # (height, width) in pixels
    bands: int = 1
    band_descriptions: List[str] = field(default_factory=list)

    # --- Radiometry ---
    dtype: str = ""                      # numpy dtype string, e.g. "uint16"
    radiometric_scale: Optional[float] = None
    radiometric_offset: Optional[float] = None
    nodata: Optional[float] = None
    saturation_dn: Optional[float] = None
    quality_mask_available: bool = False

    # --- Geospatial ---
    crs_wkt: Optional[str] = None        # WKT2 string; None for ungeoreferenced
    affine_transform: Optional[List[float]] = None  # 6-element affine [a,b,c,d,e,f]
    pixel_size_m: Optional[float] = None  # GSD in metres/pixel (square assumed)
    footprint_wkt: Optional[str] = None   # WKT polygon in declared CRS
    pixel_origin: str = PIXEL_ORIGIN_CORNER
    crs_uncertainty: str = ""            # "trusted", "estimated", "unknown"

    # --- Illumination / Acquisition ---
    sun_azimuth_deg: Optional[float] = None
    sun_elevation_deg: Optional[float] = None
    incidence_angle_deg: Optional[float] = None
    emission_angle_deg: Optional[float] = None
    phase_angle_deg: Optional[float] = None
    acquisition_start_utc: Optional[str] = None
    acquisition_stop_utc: Optional[str] = None

    # --- Provenance ---
    pds_label_path: Optional[str] = None
    calibration_reference: Optional[str] = None

    # --- Internal ---
    _ingest_timestamp: float = field(default_factory=time.time, repr=False)

    # ------------------------------------------------------------------
    def compute_checksum(self, force: bool = False) -> Optional[str]:
        """Compute SHA-256 of the file at *path* and cache it."""
        if self.sha256 and not force:
            return self.sha256
        if not self.path or not os.path.isfile(self.path):
            return None
        digest = hashlib.sha256()
        try:
            with open(self.path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            self.sha256 = digest.hexdigest()
        except OSError as exc:
            logger.warning("Cannot compute checksum for %s: %s", self.path, exc)
        return self.sha256

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe dictionary (no numpy, no private fields)."""
        d = asdict(self)
        d.pop("_ingest_timestamp", None)
        # Convert tuples to lists for JSON
        if isinstance(d.get("dims"), tuple):
            d["dims"] = list(d["dims"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageDescriptor":
        """Reconstruct an ImageDescriptor from a plain dict."""
        # Normalise dims tuple
        if "dims" in data and isinstance(data["dims"], list):
            data["dims"] = tuple(data["dims"])
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_path(
        cls,
        path: str,
        image_id: Optional[str] = None,
        compute_hash: bool = False,
        pds_metadata: Optional[Dict[str, Any]] = None,
    ) -> "ImageDescriptor":
        """
        Build a descriptor for *path* with whatever information is available.

        Parameters
        ----------
        path : str
            Absolute path to the image file.
        image_id : str, optional
            Explicit ID; defaults to the file basename without extension.
        compute_hash : bool
            If True, compute SHA-256 immediately (slow for large files).
        pds_metadata : dict, optional
            Pre-parsed PDS4 metadata dict (from ``parse_pds4_metadata``).
        """
        desc = cls(
            image_id=image_id or os.path.splitext(os.path.basename(path))[0],
            path=path,
        )

        # Try to read basic raster metadata via rasterio
        try:
            import rasterio
            with rasterio.open(path) as ds:
                desc.dims = (ds.height, ds.width)
                desc.bands = ds.count
                desc.dtype = str(ds.dtypes[0])
                desc.nodata = ds.nodata
                if ds.crs is not None:
                    desc.crs_wkt = ds.crs.to_wkt()
                    desc.crs_uncertainty = "trusted"
                t = ds.transform
                desc.affine_transform = [t.a, t.b, t.c, t.d, t.e, t.f]
                if t.a != 0:
                    desc.pixel_size_m = abs(t.a)
                desc.pixel_origin = PIXEL_ORIGIN_CORNER
        except Exception:
            # Fall back: just record dimensions via cv2 / numpy
            try:
                import cv2
                tmp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tmp is not None:
                    desc.dims = (tmp.shape[0], tmp.shape[1])
                    desc.dtype = str(tmp.dtype)
            except Exception:
                pass

        # Overlay PDS4 metadata when supplied
        if pds_metadata:
            desc.sun_azimuth_deg = pds_metadata.get("sun_azimuth") or pds_metadata.get("sun_azimuth_deg")
            desc.sun_elevation_deg = pds_metadata.get("sun_elevation") or pds_metadata.get("sun_elevation_deg")
            desc.incidence_angle_deg = pds_metadata.get("incidence")
            desc.emission_angle_deg = pds_metadata.get("emission")
            desc.phase_angle_deg = pds_metadata.get("phase")
            desc.acquisition_start_utc = pds_metadata.get("start_time")
            desc.acquisition_stop_utc = pds_metadata.get("stop_time")
            desc.sensor = pds_metadata.get("instrument_id", desc.sensor)
            desc.product_level = pds_metadata.get("product_type", desc.product_level)
            gsd = pds_metadata.get("gsd_m")
            if gsd:
                desc.pixel_size_m = float(gsd)

        if compute_hash:
            desc.compute_checksum()

        return desc


# ---------------------------------------------------------------------------
# Pair descriptor — two images that form one registration job
# ---------------------------------------------------------------------------
@dataclass
class ImagePairDescriptor:
    """
    Descriptor for the fixed/moving image pair entering registration.

    Convention: *fixed* = reference, *moving* = source  (§2 of brief).
    """
    fixed: ImageDescriptor = field(default_factory=ImageDescriptor)
    moving: ImageDescriptor = field(default_factory=ImageDescriptor)
    overlap_estimate: Optional[float] = None   # 0–1 score
    gsd_ratio: Optional[float] = None          # moving_gsd / fixed_gsd
    sensor_pair: str = ""                       # e.g. "ohrc_tmc", "iirs_optical"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fixed": self.fixed.to_dict(),
            "moving": self.moving.to_dict(),
            "overlap_estimate": self.overlap_estimate,
            "gsd_ratio": self.gsd_ratio,
            "sensor_pair": self.sensor_pair,
            "notes": self.notes,
        }
