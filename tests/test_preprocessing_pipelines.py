import numpy as np
import pytest
from src.preprocess import (
    apply_image_inversion,
    apply_morphological_dilation,
    apply_pca_enhancement,
    apply_log_transform,
    apply_histogram_matching,
    lee_speckle_filter,
    sar_log_ratio_normalization,
    SensorModalityRouter,
)


def test_image_inversion():
    img_u8 = np.array([[0, 50], [200, 255]], dtype=np.uint8)
    inv = apply_image_inversion(img_u8)
    np.testing.assert_array_equal(inv, [[255, 205], [55, 0]])

    img_f = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
    inv_f = apply_image_inversion(img_f)
    np.testing.assert_allclose(inv_f, [[1.0, 0.8], [0.2, 0.0]], atol=1e-5)


def test_morphological_dilation():
    img = np.zeros((15, 15), dtype=np.uint8)
    img[7, 7] = 255
    dilated = apply_morphological_dilation(img, kernel_size=3)
    assert dilated[7, 7] == 255
    # Neighbors dilated
    assert dilated[6, 7] == 255
    assert dilated[7, 6] == 255
    assert dilated[8, 7] == 255
    assert dilated[7, 8] == 255
    assert dilated[0, 0] == 0


def test_pca_enhancement():
    # Synthetic lunar crater texture
    np.random.seed(42)
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    pca_img = apply_pca_enhancement(img)
    assert pca_img.shape == (64, 64)
    assert pca_img.dtype == np.uint8
    assert pca_img.max() > pca_img.min()


def test_log_transform():
    img = np.array([[0, 10], [50, 255]], dtype=np.uint8)
    log_img = apply_log_transform(img)
    assert log_img.shape == (2, 2)
    assert log_img.dtype == np.uint8
    assert log_img[0, 0] == 0
    assert log_img[1, 1] == 255
    assert log_img[0, 1] > 10  # Shadow expansion


def test_histogram_matching():
    src = np.full((32, 32), 50, dtype=np.uint8)
    ref = np.full((32, 32), 180, dtype=np.uint8)
    matched = apply_histogram_matching(src, ref)
    assert matched.shape == (32, 32)
    assert matched.mean() > 100


def test_lee_speckle_filter():
    sar = np.random.exponential(scale=50.0, size=(64, 64)).clip(0, 255).astype(np.uint8)
    filtered = lee_speckle_filter(sar, kernel_size=5, enl=4.0)
    assert filtered.shape == (64, 64)
    assert filtered.std() <= sar.std()  # Speckle variance reduced


def test_sensor_modality_router():
    src = np.random.randint(20, 220, (100, 100), dtype=np.uint8)
    ref = np.random.randint(20, 220, (100, 100), dtype=np.uint8)

    # Pipeline A
    out_s, out_r, tele = SensorModalityRouter.route_and_preprocess(
        src, ref, modality_or_pair="pipeline_a"
    )
    assert "Pipeline A" in tele["pipeline_selected"]
    assert out_s.shape == (100, 100)

    # Pipeline B
    out_s, out_r, tele = SensorModalityRouter.route_and_preprocess(
        src, ref, sensor_type_src="iirs", sensor_type_ref="wac"
    )
    assert "Pipeline B" in tele["pipeline_selected"]
    assert out_s.shape == (100, 100)

    # Pipeline C
    out_s, out_r, tele = SensorModalityRouter.route_and_preprocess(
        src, ref, sensor_type_src="dfsar", sensor_type_ref="selene"
    )
    assert "Pipeline C" in tele["pipeline_selected"]
    assert out_s.shape == (100, 100)
