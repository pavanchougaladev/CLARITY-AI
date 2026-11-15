import numpy as np
import cv2
from src.features.tile_matcher import tile_match, create_sift_tile_matcher, _deduplicate_matches

def test_tile_match():
    # Create two textured images with a known shift
    tex1 = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    tex1 = cv2.GaussianBlur(tex1, (5, 5), 1.0)
    for x, y in [(50, 50), (150, 50), (250, 50), (100, 200), (200, 200)]:
        cv2.circle(tex1, (x, y), 15, 220, -1)
        cv2.rectangle(tex1, (x-8, y-8), (x+8, y+8), 100, 2)

    M = np.float32([[1, 0, 4], [0, 1, 4]])
    tex2 = cv2.warpAffine(tex1, M, (300, 300))

    sift_fn = create_sift_tile_matcher()
    pts_s, pts_r, confs = tile_match(tex1, tex2, sift_fn, grid=(2, 2), overlap=0.2)

    assert len(pts_s) > 0
    assert len(pts_s) == len(pts_r) == len(confs)


def test_deduplicate_matches():
    pts_s = np.array([[10, 10], [10.5, 10.5], [50, 50]], dtype=np.float32)
    pts_r = np.array([[15, 15], [15.5, 15.5], [55, 55]], dtype=np.float32)
    confs = np.array([0.9, 0.5, 0.8], dtype=np.float32)

    dedup_s, dedup_r, dedup_c = _deduplicate_matches(pts_s, pts_r, confs, radius=2.0)
    assert len(dedup_s) == 2
    assert np.isclose(float(dedup_c[0]), 0.9, rtol=1e-4), f"Expected ~0.9, got {dedup_c[0]}"  # Higher confidence retained
