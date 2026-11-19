import numpy as np
from skimage.filters import gabor_kernel
import cv2

def phase_congruency_map(img: np.ndarray, scales: int = 4, orientations: int = 6) -> np.ndarray:
    """
    Simplified multi-scale, multi-orientation phase congruency response.
    Production implementation should use Kovesi's PC algorithm (ported to Python).
    This function acts as a robust descriptor resilient to illumination changes.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    img_fft = np.fft.fft2(img)
    responses = []
    
    for s in range(scales):
        sigma = 2 ** s
        for o in range(orientations):
            theta = o * np.pi / orientations
            kernel = np.real(gabor_kernel(frequency=0.2 / sigma, theta=theta))
            kernel_fft = np.fft.fft2(kernel, s=img.shape)
            filtered = np.abs(np.fft.ifft2(img_fft * kernel_fft))
            responses.append(filtered)
            
    pc_map = np.sum(responses, axis=0) / (np.mean(responses, axis=0) + 1e-6)
    max_val = pc_map.max()
    if max_val > 0:
        pc_map = pc_map / max_val
    else:
        pc_map = np.zeros_like(pc_map)
    return pc_map
