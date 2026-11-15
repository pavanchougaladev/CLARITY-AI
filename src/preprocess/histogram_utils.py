"""
Histogram Utilities for Cross-Sensor Radiometric Alignment
==========================================================
Paper Pipeline B (IIRS ↔ WAC Hyperspectral Registration):
Matches the cumulative distribution function (CDF) of the source image
to the reference image to bridge extreme radiometric discrepancy.
"""

import numpy as np


def apply_histogram_matching(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Match the cumulative histogram of `source` to `reference`.

    Parameters
    ----------
    source : np.ndarray
        Source image (uint8 or float) whose histogram is to be adjusted.
    reference : np.ndarray
        Reference template image (uint8 or float).

    Returns
    -------
    np.ndarray : Histogram-matched image with dtype matching source.
    """
    try:
        from skimage.exposure import match_histograms
        matched = match_histograms(source, reference)
        return matched.astype(source.dtype)
    except Exception:
        pass

    # Pure NumPy implementation if skimage is unavailable or errors
    src_flat = source.ravel()
    ref_flat = reference.ravel()

    # Get unique values and counts
    s_values, bin_idx, s_counts = np.unique(src_flat, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(ref_flat, return_counts=True)

    # Compute normalized empirical CDFs
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    # Interpolate source values into reference CDF
    interp_values = np.interp(s_quantiles, r_quantiles, r_values)

    matched = interp_values[bin_idx].reshape(source.shape)
    if np.issubdtype(source.dtype, np.integer):
        return np.clip(np.round(matched), 0, 255).astype(source.dtype)
    return matched.astype(source.dtype)
