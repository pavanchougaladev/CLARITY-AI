import cv2
import numpy as np

def build_gaussian_pyramid(img: np.ndarray, levels: int = 3) -> list:
    """Builds a Gaussian pyramid."""
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def build_laplacian_pyramid(img: np.ndarray, levels: int = 3) -> list:
    """Builds a Laplacian pyramid."""
    gaussian_pyr = build_gaussian_pyramid(img, levels)
    laplacian_pyr = []
    
    for i in range(levels - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i+1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)
    
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def resize_to_common_resolution(img: np.ndarray, target_gsd: float, current_gsd: float) -> np.ndarray:
    """
    Resizes an image so its Ground Sample Distance (GSD) matches the target.
    Scale factor = current_gsd / target_gsd.
    E.g., 5m/px image to match 25cm/px needs a 20x upscale.
    """
    scale_factor = current_gsd / target_gsd
    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
