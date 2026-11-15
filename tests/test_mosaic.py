import numpy as np
import cv2
from src.mosaic import MosaicEngine

def test_mosaic_engine():
    engine = MosaicEngine()
    
    # Create two overlapping synthetic images
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img1, (50, 50), 20, 255, -1)
    cv2.rectangle(img1, (20, 20), (40, 40), 180, -1)
    
    img2 = np.roll(img1, 5, axis=1) # Shift by 5px
    
    engine.add_image(img1, "strip_1")
    engine.add_image(img2, "strip_2")
    
    mosaic_out, meta = engine.compute_mosaic(base_idx=0)
    
    assert mosaic_out is not None
    assert isinstance(mosaic_out, np.ndarray)
    assert meta["success"] is True
    assert meta["count"] >= 1
