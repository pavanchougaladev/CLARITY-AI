import numpy as np


def test_iirs_camera_profile_is_available():
    from src.features.dem_builder import get_camera_model

    model = get_camera_model({"stereo": {"sensor": "iirs"}})
    assert model.sensor == "iirs"
    assert model.gsd_m > 0
from src.preprocess.shadow_normalization import normalize_illumination
from src.preprocess.photometric_correction import apply_hapke_correction, apply_lunar_lambert

def test_normalize_illumination(random_image):
    norm = normalize_illumination(random_image)
    assert norm.shape == random_image.shape
    assert norm.dtype == np.uint8

def test_hapke_correction(random_image):
    corrected = apply_hapke_correction(random_image, 45.0, 10.0, 35.0)
    assert corrected.shape == random_image.shape
    assert corrected.dtype == np.uint8

def test_lunar_lambert(random_image):
    corrected = apply_lunar_lambert(random_image, 45.0, 10.0, 35.0)
    assert corrected.shape == random_image.shape
    assert corrected.dtype == np.uint8
