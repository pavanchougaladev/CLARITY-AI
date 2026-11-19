"""
Enhanced Shadow Normalization — Multi-Scale CLAHE + Adaptive Wallis Filter
===========================================================================
Improved illumination-invariant preprocessing for lunar terrain.

- Multi-scale CLAHE: applies at multiple tile sizes and blends for
  better adaptation across both small craters and large terrain features.
- Wallis filter: local contrast normalization proven more effective
  than CLAHE alone for planetary imagery with extreme shadow gradients.
  Kernel size now ADAPTS to image resolution so statistics are computed
  over a spatially meaningful neighbourhood at any scale.
- Homomorphic filtering: attenuates low-frequency illumination component.
"""

import cv2
import numpy as np


def normalize_illumination(
    img: np.ndarray,
    use_wallis: bool = True,
    wallis_target_mean: float = 110.0,
    wallis_target_std: float = 35.0,
) -> np.ndarray:
    """
    Full illumination normalization pipeline:
      1. Homomorphic filtering (reduce low-frequency illumination bias)
      2. Multi-scale CLAHE (local contrast at multiple spatial scales)
      3. Adaptive Wallis filter (force local mean/variance to target values;
         kernel auto-scales with image resolution)

    Parameters
    ----------
    img               : 2D uint8 grayscale image.
    use_wallis        : Whether to apply Wallis filter (recommended for lunar).
    wallis_target_mean: Target local mean for Wallis (default 110, tuned for TMC).
    wallis_target_std : Target local std for Wallis (default 35, avoids over-stretch
                        on low-texture lunar maria; was 50 which was too aggressive).

    Returns
    -------
    np.ndarray : Normalized uint8 image.
    """
    # Step 1: Homomorphic filtering
    homo = _homomorphic_filter(img)

    # Step 2: Multi-scale CLAHE
    ms_clahe = _multiscale_clahe(homo)

    # Step 3: Adaptive Wallis filter for local mean/variance normalization
    if use_wallis:
        result = _wallis_filter(ms_clahe,
                                target_mean=wallis_target_mean,
                                target_std=wallis_target_std)
    else:
        result = ms_clahe

    return result


def _homomorphic_filter(
    img: np.ndarray,
    radius: int = 30,
    low_gain: float = 0.5,
    high_gain: float = 1.5,
) -> np.ndarray:
    """
    Homomorphic filtering with configurable low/high frequency gains.
    Separates illumination (low-freq) from reflectance (high-freq).
    """
    img_f = img.astype(np.float32) + 1.0
    log_img = np.log(img_f)
    rows, cols = log_img.shape
    crow, ccol = rows // 2, cols // 2

    dft = np.fft.fftshift(np.fft.fft2(log_img))

    # Gaussian-based high/low frequency separation
    y, x = np.ogrid[:rows, :cols]
    dist_sq = (x - ccol) ** 2 + (y - crow) ** 2
    # Smooth Gaussian transition instead of hard cutoff
    H = (high_gain - low_gain) * (1 - np.exp(-dist_sq / (2.0 * radius ** 2))) + low_gain

    filtered = np.fft.ifft2(np.fft.ifftshift(dft * H))
    result = np.abs(np.exp(np.real(filtered))) - 1.0
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return result


def _multiscale_clahe(img: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE at three different tile scales and blends the results.
    - Small tiles (4x4): enhances fine crater detail
    - Medium tiles (8x8): standard local contrast
    - Large tiles (16x16): broad terrain normalization
    """
    clahe_small  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe_medium = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_large  = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16))

    result_small  = clahe_small.apply(img).astype(np.float32)
    result_medium = clahe_medium.apply(img).astype(np.float32)
    result_large  = clahe_large.apply(img).astype(np.float32)

    # Weighted blend: medium gets most weight
    blended = result_small * 0.25 + result_medium * 0.50 + result_large * 0.25
    return np.clip(blended, 0, 255).astype(np.uint8)


def _wallis_filter(
    img: np.ndarray,
    target_mean: float = 127.0,
    target_std: float = 50.0,
    contrast_gain: float = 0.8,
    brightness_gain: float = 0.9,
    kernel_size: int = None,          # None → auto-scale with image size
) -> np.ndarray:
    """
    Wallis adaptive contrast filter.

    Forces local image statistics (mean, std) toward target values,
    providing consistent local contrast regardless of illumination.
    Widely used in photogrammetry for aerial and satellite imagery.

    FIX: kernel_size now auto-scales with image resolution.
    A 31×31 kernel on a 2048px image covers only 1.5% of the image width —
    too small to capture the broad illumination gradients of lunar terrain.
    The auto-scaled kernel covers ~1/64 of the image width, which corresponds
    to a few crater diameters at typical TMC resolution.

    Parameters
    ----------
    img             : uint8 input image.
    target_mean     : Desired local mean (typically 127).
    target_std      : Desired local standard deviation (typically 50).
    contrast_gain   : Contrast control [0, 1]. Higher = more aggressive.
    brightness_gain : Brightness control [0, 1]. Higher = more aggressive.
    kernel_size     : Window size for local statistics computation.
                      None = auto: max(31, image_width // 64), always odd.
    """
    img_f = img.astype(np.float64)
    h, w = img.shape[:2]

    # Auto-scale kernel to image resolution
    if kernel_size is None:
        kernel_size = max(31, w // 64)
        # Ensure odd for symmetric filtering
        if kernel_size % 2 == 0:
            kernel_size += 1

    # Compute local mean and std using box filter
    local_mean    = cv2.blur(img_f, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(img_f ** 2, (kernel_size, kernel_size))
    local_var     = np.maximum(local_sq_mean - local_mean ** 2, 1e-6)
    local_std     = np.sqrt(local_var)

    # Wallis formula
    denominator = contrast_gain * local_std + (1 - contrast_gain) * target_std + 1e-10
    r1 = contrast_gain * target_std / denominator
    r0 = brightness_gain * target_mean + (1 - brightness_gain - r1) * local_mean

    result = r1 * img_f + r0

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_image_inversion(img: np.ndarray) -> np.ndarray:
    """
    Research Paper Pipeline A (ISRO / Chandrayaan-2):
    Invert image intensities (255 - I or 1.0 - I) to convert deeply shadowed
    crater floors into high-contrast highlights, aiding feature detectors.
    """
    if img.dtype == np.uint8:
        return 255 - img
    elif np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            return 1.0 - img
        else:
            return 255.0 - img
    else:
        max_val = np.iinfo(img.dtype).max if np.issubdtype(img.dtype, np.integer) else 255
        return (max_val - img).astype(img.dtype)


def apply_morphological_dilation(
    img: np.ndarray,
    kernel_size: int = 3,
    kernel_shape: str = "ellipse",
    iterations: int = 1,
) -> np.ndarray:
    """
    Research Paper Pipeline A (ISRO / Chandrayaan-2):
    Apply morphological dilation to bridge broken crater rim features,
    strengthen thin ridges, and expand local illumination gradients.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image (uint8 or float).
    kernel_size : int
        Size of structuring element (must be odd, default: 3).
    kernel_shape : str
        Structuring element shape: 'ellipse', 'rect', or 'cross'.
    iterations : int
        Number of dilation iterations (default: 1).
    """
    k_size = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    shapes = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
        "cross": cv2.MORPH_CROSS,
    }
    shape_flag = shapes.get(kernel_shape.lower(), cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(shape_flag, (k_size, k_size))

    is_float = np.issubdtype(img.dtype, np.floating)
    if is_float:
        orig_max = img.max()
        u8 = (img * 255.0).clip(0, 255).astype(np.uint8) if orig_max <= 1.0 else img.clip(0, 255).astype(np.uint8)
        dilated = cv2.dilate(u8, kernel, iterations=iterations)
        return dilated.astype(img.dtype) / (255.0 if orig_max <= 1.0 else 1.0)
    else:
        return cv2.dilate(img, kernel, iterations=iterations)

