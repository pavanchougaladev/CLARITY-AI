import pytest
try:
    import torch
except (ImportError, Exception):
    pytest.skip("PyTorch missing or dylib not available", allow_module_level=True)
import numpy as np
from src.features.deep_matcher import LunarMatcher

def test_lunar_matcher_initialization():
    """Verify all matchers initialize without error."""
    for m_type in ["sift", "loftr", "tiny_roma", "roma", "roma_v2", "roma_omega"]:
        matcher = LunarMatcher(matcher_type=m_type)
        assert matcher._available, f"Matcher {m_type} should be available"

def test_roma_v2_matching():
    """Verify RoMa v2 produces valid matches with proper shapes and zero NaNs on synthetic tensors."""
    matcher = LunarMatcher(matcher_type="roma_v2", confidence_threshold=0.1)
    # Synthetic 256x256 image with crater pattern
    img = np.zeros((256, 256), dtype=np.uint8)
    import cv2
    cv2.circle(img, (128, 128), 30, 200, -1)
    cv2.circle(img, (128, 128), 20, 50, -1)
    cv2.GaussianBlur(img, (9, 9), 3, dst=img)

    pts_src, pts_ref, confs = matcher.match(img, img)
    assert isinstance(pts_src, np.ndarray)
    assert isinstance(pts_ref, np.ndarray)
    assert isinstance(confs, np.ndarray)
    assert len(pts_src) == len(pts_ref) == len(confs)
    if len(pts_src) > 0:
        assert not np.isnan(pts_src).any(), "pts_src must not contain NaNs"
        assert not np.isnan(pts_ref).any(), "pts_ref must not contain NaNs"
        assert not np.isnan(confs).any(), "confs must not contain NaNs"

