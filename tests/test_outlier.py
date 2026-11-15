import numpy as np
from src.matching.outlier_rejection import estimate_transform
import cv2

def test_estimate_transform(synthetic_points):
    src, ref, _ = synthetic_points
    # Add more valid points to ensure MAGSAC has enough to work with
    extra_src = np.array([[50, 50], [60, 10], [10, 60], [80, 80]], dtype=np.float32)
    extra_ref = np.array([[55, 55], [65, 15], [15, 65], [85, 85]], dtype=np.float32)
    
    src_ext = np.vstack([src, extra_src])
    ref_ext = np.vstack([ref, extra_ref])
    
    # Add an outlier
    src_with_outlier = np.vstack([src_ext, [100, 100]])
    ref_with_outlier = np.vstack([ref_ext, [10, 10]]) # Bad match
    
    # Needs at least 4 points, we have 9
    H, inliers = estimate_transform(src_with_outlier, ref_with_outlier, use_tps=False)
    
    assert H is not None
    assert H.shape == (3, 3)
    
    # First 8 should be inliers, 9th should be outlier
    assert inliers[0] == True
    assert inliers[8] == False
