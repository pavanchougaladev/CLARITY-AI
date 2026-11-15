import os
import numpy as np
from src.pipeline import robust_read, smart_downscale
from src.registration.export import export_asp_compatible_tiepoints
import cv2

def test_smart_downscale(random_image):
    large_image = np.zeros((3000, 2000), dtype=np.uint8)
    scaled, factor = smart_downscale(large_image, max_dim=1500)
    
    assert factor == 0.5
    assert scaled.shape == (1500, 1000)
    
    # Test skipping downscale
    small_image = np.zeros((1000, 1000), dtype=np.uint8)
    scaled, factor = smart_downscale(small_image, max_dim=2048)
    assert factor == 1.0
    assert scaled.shape == (1000, 1000)

def test_robust_read(tmp_path):
    img = np.zeros((10, 10), dtype=np.uint8)
    path = tmp_path / "test.png"
    cv2.imwrite(str(path), img)
    
    read_img = robust_read(str(path))
    assert read_img.shape == (10, 10)
    assert read_img.dtype == np.uint8

def test_export_asp_compatible_tiepoints(tmp_path, synthetic_points):
    src, ref, conf = synthetic_points
    out_path = tmp_path / "tiepoints.csv"
    export_asp_compatible_tiepoints(src, ref, conf, str(out_path))
    
    assert os.path.exists(str(out_path))
    with open(str(out_path), 'r') as f:
        lines = f.readlines()
        assert len(lines) == 5 # header + 4 points
        assert lines[0].strip() == "x1,y1,x2,y2,weight"
