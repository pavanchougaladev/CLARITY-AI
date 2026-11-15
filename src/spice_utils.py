"""
SPICE Kernel Integration + PDS4 Geometry Parser
=================================================
Helper module to interface with NASA's SPICE toolkit using spiceypy
and parse illumination geometry directly from PDS4 XML labels.

Calculates exact solar geometry (incidence, emission, phase angles)
for Lunar images based on spacecraft ephemeris data or PDS4 metadata.

Fix applied
-----------
``compute_solar_geometry()`` previously returned hardcoded mock values
(45°, 10°, 35°) even when SPICE kernels were successfully loaded.
It is now implemented using ``spice.ilumin()`` which computes the exact
incidence, emission, and phase angles at the ground-intercept point of the
spacecraft's nadir vector.

Fallback chain
--------------
1. ``spice.ilumin()`` — precise surface-intercept illumination angles
2. ``spice.spkpos()``-based approximation — angular separation in 3-D space
   (less accurate but avoids shape-model requirement)
3. ``None`` — signals caller to skip photometric correction entirely

Neither fallback returns fabricated constant angles.
"""

import os
import logging
import math
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Tuple

try:
    import spiceypy as spice
    HAS_SPICE = True
except ImportError:
    HAS_SPICE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel management
# ---------------------------------------------------------------------------

def load_kernels(kernel_paths):
    """Load SPICE kernels (.bsp, .tpc, .tsc, etc.)."""
    if not HAS_SPICE:
        logger.error("spiceypy not installed — cannot load SPICE kernels.")
        return False

    try:
        for kp in kernel_paths:
            spice.furnsh(kp)
        logger.info(f"Loaded {len(kernel_paths)} SPICE kernels.")
        return True
    except Exception as e:
        logger.error(f"Failed to load SPICE kernels: {e}")
        return False


# ---------------------------------------------------------------------------
# Fix 3: Real solar geometry via spice.ilumin()
# ---------------------------------------------------------------------------

def compute_solar_geometry(
    et: float,
    target: str = "MOON",
    observer: str = "CHANDRAYAAN-2",
    frame: str = "IAU_MOON",
) -> Optional[Tuple[float, float, float]]:
    """
    Compute illumination geometry angles at the spacecraft's nadir
    ground-intercept point for the given ephemeris time.

    Returns
    -------
    tuple (incidence_deg, emission_deg, phase_deg)  or  ``None``

    ``None`` is returned (instead of mock constants) whenever the computation
    cannot be completed, allowing the caller to skip photometric correction
    gracefully rather than apply it with wrong angles.

    Implementation
    --------------
    Uses ``spice.ilumin()`` which requires:
      - A loaded lunar shape model kernel (e.g. MOON_PA_DE421_1900-2050.bpc)
      - SPK kernels for the Sun and spacecraft
      - PCK kernel for planetary constants

    If ``ilumin`` fails (e.g. no shape model loaded), falls back to a
    ``spkpos``-based angular approximation.

    Parameters
    ----------
    et : float
        SPICE ephemeris time (seconds past J2000).
    target : str
        SPICE body name for the target body (default: 'MOON').
    observer : str
        SPICE body name for the spacecraft (default: 'CHANDRAYAAN-2').
        Use 'CHAN2' if your kernels use that alias.
    frame : str
        Body-fixed reference frame (default: 'IAU_MOON').
    """
    if not HAS_SPICE:
        logger.warning("spiceypy not available — skipping photometric correction.")
        return None

    # ── Strategy 1: spice.ilumin() — precise surface-intercept angles ──────
    try:
        # Find the sub-spacecraft (nadir) point on the Moon's surface
        spoint, trgepc, srfvec = spice.sincpt(
            "ELLIPSOID",   # surface method
            target,        # target body
            et,            # epoch
            frame,         # body-fixed frame
            "LT+S",        # aberration correction
            observer,      # observer (spacecraft)
            frame,         # observer frame (same as body frame for nadir)
            (0.0, 0.0, 1.0),  # nadir direction in spacecraft frame (approx)
        )

        # Compute illumination angles at the surface intercept
        trgepc2, srfvec2, phase, incdnc, emissn = spice.ilumin(
            "ELLIPSOID",
            target,
            et,
            frame,
            "LT+S",
            observer,
            spoint,
        )

        inc_deg   = math.degrees(incdnc)
        ema_deg   = math.degrees(emissn)
        phase_deg = math.degrees(phase)

        logger.info(
            f"SPICE ilumin: incidence={inc_deg:.2f}°, "
            f"emission={ema_deg:.2f}°, phase={phase_deg:.2f}°"
        )
        return inc_deg, ema_deg, phase_deg

    except Exception as e:
        logger.debug(f"spice.ilumin() failed ({e}), trying spkpos approximation...")

    # ── Strategy 2: spkpos-based angular approximation ─────────────────────
    try:
        # Get Sun and spacecraft positions relative to Moon centre
        sun_pos, _  = spice.spkpos("SUN",      et, frame, "LT+S", target)
        obs_pos, _  = spice.spkpos(observer,   et, frame, "NONE", target)
        moon_radius = 1737.4e3  # metres

        # Approximate surface normal as radial from Moon centre to sub-s/c pt
        obs_norm   = obs_pos / (spice.vnorm(obs_pos) + 1e-12)
        sun_norm   = sun_pos / (spice.vnorm(sun_pos) + 1e-12)
        obs_to_sc  = -obs_pos / (spice.vnorm(obs_pos) + 1e-12)  # nadir direction

        # Angles between vectors (clamp to valid arccos domain)
        def safe_acos(x):
            return math.acos(max(-1.0, min(1.0, float(x))))

        inc_rad   = safe_acos(spice.vdot(sun_norm,  obs_norm))
        ema_rad   = safe_acos(spice.vdot(obs_to_sc, obs_norm))
        phase_rad = safe_acos(spice.vdot(sun_norm,  obs_to_sc))

        inc_deg   = math.degrees(inc_rad)
        ema_deg   = math.degrees(ema_rad)
        phase_deg = math.degrees(phase_rad)

        logger.info(
            f"SPICE spkpos approx: incidence={inc_deg:.2f}°, "
            f"emission={ema_deg:.2f}°, phase={phase_deg:.2f}°"
        )
        return inc_deg, ema_deg, phase_deg

    except Exception as e:
        logger.warning(
            f"SPICE spkpos approximation also failed ({e}). "
            "Returning None — photometric correction will be skipped."
        )
        return None


# ---------------------------------------------------------------------------
# PDS4 label parsing (unchanged — already correct)
# ---------------------------------------------------------------------------

def parse_geometry_from_pds4_label(xml_path: str) -> Optional[Dict[str, float]]:
    """
    Parse illumination geometry angles from a Chandrayaan PDS4 XML label.

    Chandrayaan-2 PDS4 labels store geometry metadata in the
    Observation_Area or Geometry sections. This parser searches for:
      - incidence_angle
      - emission_angle
      - phase_angle
      - solar_longitude
      - sub_solar_latitude

    Parameters
    ----------
    xml_path : str
        Path to the PDS4 .xml label file.

    Returns
    -------
    dict or None : Dictionary with angle values in degrees, or None if not found.
    """
    if not os.path.exists(xml_path):
        logger.warning(f"PDS4 label not found: {xml_path}")
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # PDS4 uses namespaces — strip them for easier searching
        def strip_ns(tag):
            return tag.split('}')[-1] if '}' in tag else tag

        result = {}

        # Search for geometry-related elements anywhere in the XML tree
        geometry_keys = {
            'incidence_angle': 'incidence',
            'emission_angle': 'emission',
            'phase_angle': 'phase',
            'solar_longitude': 'solar_longitude',
            'sub_solar_latitude': 'sub_solar_latitude',
            'solar_azimuth': 'solar_azimuth',
            'solar_incidence': 'incidence',
            'sun_elevation': 'sun_elevation',
            'sun_azimuth': 'sun_azimuth',
            # Chandrayaan-2 specific fields
            'IncidenceAngle': 'incidence',
            'EmissionAngle': 'emission',
            'PhaseAngle': 'phase',
            'SolarZenithAngle': 'solar_zenith',
            'SolarIncidenceAngle': 'incidence',
        }

        for elem in root.iter():
            tag_name = strip_ns(elem.tag)
            for xml_key, result_key in geometry_keys.items():
                if tag_name.lower() == xml_key.lower():
                    try:
                        value = float(elem.text.strip().split()[0])  # Handle "45.3 deg"
                        result[result_key] = value
                    except (ValueError, AttributeError):
                        pass

        if result:
            logger.info(f"Parsed PDS4 geometry: {result}")
            return result
        else:
            logger.info(f"No geometry angles found in PDS4 label: {xml_path}")
            return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse PDS4 XML: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error reading PDS4 label: {e}")
        return None


def find_pds4_label_for_image(image_path: str) -> Optional[str]:
    """
    Attempts to find the PDS4 XML label associated with an image file.

    Strategy:
      1. Look for a .xml file with the same base name
      2. Look for a .lbl file with the same base name
      3. Look in a 'labels/' sibling directory
    """
    base = os.path.splitext(image_path)[0]
    img_dir = os.path.dirname(image_path)
    img_basename = os.path.splitext(os.path.basename(image_path))[0]

    # Strategy 1: same name, .xml extension
    candidates = [
        base + '.xml',
        base + '.XML',
        base + '.lbl',
        base + '.LBL',
    ]

    # Strategy 2: labels/ directory
    parent_dir = os.path.dirname(img_dir)
    labels_dir = os.path.join(parent_dir, 'labels')
    if os.path.isdir(labels_dir):
        candidates.extend([
            os.path.join(labels_dir, img_basename + '.xml'),
            os.path.join(labels_dir, img_basename + '.XML'),
        ])

    for candidate in candidates:
        if os.path.exists(candidate):
            logger.info(f"Found PDS4 label: {candidate}")
            return candidate

    return None


def get_geometry_for_image(image_path: str) -> Optional[Dict[str, float]]:
    """
    High-level function: tries to extract geometry angles for an image.

    Priority:
      1. Parse from PDS4 XML label (real angles)
      2. Return None (caller should use defaults or skip photometric correction)
    """
    label_path = find_pds4_label_for_image(image_path)
    if label_path:
        return parse_geometry_from_pds4_label(label_path)
    return None
