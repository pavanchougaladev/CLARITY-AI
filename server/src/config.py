"""
Configuration Loader
=====================
Loads YAML configuration files and provides a global config object
with sensible defaults for the entire pipeline.
"""

import logging
import os
import yaml
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)



_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'configs', 'default.yaml'
)

# In-memory cache
_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str or None
        Path to the YAML config file.  Falls back to configs/default.yaml.

    Returns
    -------
    dict : Parsed configuration dictionary.
    """
    global _config

    if config_path is None:
        config_path = os.path.abspath(_DEFAULT_CONFIG_PATH)

    if not os.path.exists(config_path):
        logger.warning(f"[config] Config file not found: {config_path}, using built-in defaults.")
        _config = _builtin_defaults()
        return _config

    with open(config_path, 'r') as f:
        _config = yaml.safe_load(f)

    if _config is None:
        _config = _builtin_defaults()

    return _config


def get_config() -> Dict[str, Any]:
    """Return the currently loaded config, loading defaults if needed."""
    global _config
    if _config is None:
        load_config()
    return _config


def get(key_path: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.

    Example: get('pipeline.max_dim', 2048)
    """
    cfg = get_config()
    keys = key_path.split('.')
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def _builtin_defaults() -> Dict[str, Any]:
    """Hard-coded fallback defaults when no YAML is available."""
    return {
        'pipeline': {
            'max_dim': 2048,
            'use_tps': True,
            'matcher': 'loftr',
            'outlier_threshold': 1.5,
            'magsac_confidence': 0.999,
            'magsac_max_iters': 10000,
            'pyramid_levels': 3,
            'enable_subpixel': True,
            'subpixel_patch_size': 128,
            'subpixel_response_threshold': 0.05,
            'epipolar_filter': False,
            'epipolar_threshold_px': 2.0,
            'triplet_validation': False,
        },
        'preprocessing': {
            'clahe_clip_limit': 2.0,
            'clahe_tile_size': [8, 8],
            'homomorphic_radius': 30,
            'homomorphic_low_gain': 0.5,
            'use_phase_congruency': True,    # Item 9: illumination-invariant crater detection
            'wallis_target_mean': 110.0,     # Item 10: tuned for Chandrayaan TMC DN range
            'wallis_target_std': 35.0,       # Item 10: avoids over-stretching low-texture maria
        },
        'photometric': {
            'enabled': False,
            'model': 'hapke',
            'target_incidence': 30.0,
            'target_emission': 0.0,
            'target_phase': 30.0,
        },
        'sift_fallback': {
            'nfeatures': 8000,
            'ratio_threshold': 0.75,
            'ratio_threshold_secondary': 0.82,
        },
        'yolo11_crater': {
            'model_path': 'models/yolo11_lunar_crater.pt',
            'conf': 0.20,
            'min_similarity': 0.35,
            'include_rim_points': True,
        },
        'export': {
            'asp_format': True,
            'geotiff': True,
        },
        'stereo': {
            'sensor': 'tmc',
            'focal_length_mm': 140.0,
            'pixel_size_um': 7.0,
            'baseline_m': 93300.0,
            'gsd_m': 5.0,
        },
        'visualization': {
            'max_draw_matches': 150,
            'density_grid': [8, 8],
            'overlay_opacity': 0.5,
            'checkerboard_tiles': 8,
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': True,
            'log_dir': 'logs/',
        },
        'history': {
            'enabled': True,
            'storage_file': 'registration_history.json',
            'max_entries': 500,
        },
        'provenance': {
            'enabled': True,
            'hash_algorithm': 'sha256',
        },
        'auth': {
            'enabled': False,
            'secret_key_env': 'JWT_SECRET_KEY',
            'token_expire_minutes': 60,
            'users_file': 'users.json',
        },
    }
