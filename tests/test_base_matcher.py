import pytest
import numpy as np
import cv2

from src.features.base_matcher import (
    BaseLunarMatcher,
    MatcherOutput,
    sanitize_lunar_input,
    apply_illumination_weighting,
    spatially_balance_matches,
    RootSIFTAdapter,
    get_matcher_adapter,
)


class DummyLunarMatcher(BaseLunarMatcher):
    def __init__(self, fail=False, return_pts=None, **kwargs):
        super().__init__(matcher_name="dummy", **kwargs)
        self.fail = fail
        self.return_pts = return_pts

    def _match_implementation(self, img_src, img_ref, **kwargs):
        if self.fail:
            raise RuntimeError("Synthetic failure in matcher")
        if self.return_pts is not None:
            return self.return_pts
        # Return simple matching points
        pts_s = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        pts_r = np.array([[12, 12], [22, 22], [32, 32]], dtype=np.float32)
        conf = np.array([0.9, 0.85, 0.4], dtype=np.float32)
        return pts_s, pts_r, conf, {"dummy_meta": True}


def test_sanitize_lunar_input_uint8():
    img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    sanitized = sanitize_lunar_input(img)
    assert sanitized.dtype == np.uint8
    assert sanitized.shape == (128, 128)


def test_sanitize_lunar_input_bgr():
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    sanitized = sanitize_lunar_input(img)
    assert sanitized.dtype == np.uint8
    assert sanitized.shape == (128, 128)


def test_sanitize_lunar_input_float_normalized():
    img = np.random.rand(100, 100).astype(np.float32)
    sanitized = sanitize_lunar_input(img)
    assert sanitized.dtype == np.uint8
    assert sanitized.shape == (100, 100)


def test_matcher_output_dataclass():
    pts_s = np.zeros((5, 2), dtype=np.float32)
    pts_r = np.zeros((5, 2), dtype=np.float32)
    confs = np.ones(5, dtype=np.float32)
    out = MatcherOutput(pts_s, pts_r, confs, matcher_name="test")
    assert len(out) == 5
    s, r, c = out.as_tuple()
    assert len(s) == 5 and len(r) == 5 and len(c) == 5


def test_base_matcher_filtering_and_safety():
    matcher = DummyLunarMatcher(confidence_threshold=0.5)
    img_s = np.full((100, 100), 128, dtype=np.uint8)
    img_r = np.full((100, 100), 128, dtype=np.uint8)

    out = matcher.match(img_s, img_r)
    assert isinstance(out, MatcherOutput)
    # The dummy has confs [0.9, 0.85, 0.4]. With threshold 0.5, only 2 points survive
    assert len(out) == 2
    assert out.matcher_name == "dummy"
    assert out.metadata.get("dummy_meta") is True


def test_base_matcher_error_isolation():
    failing_matcher = DummyLunarMatcher(fail=True)
    img_s = np.full((100, 100), 128, dtype=np.uint8)
    img_r = np.full((100, 100), 128, dtype=np.uint8)

    out = failing_matcher.match(img_s, img_r)
    assert len(out) == 0
    assert "error" in out.metadata
    assert "Synthetic failure" in out.metadata["error"]


def test_base_matcher_rejects_undersized_image():
    matcher = DummyLunarMatcher()
    tiny_img = np.zeros((8, 8), dtype=np.uint8)
    normal_img = np.zeros((100, 100), dtype=np.uint8)

    out = matcher.match(tiny_img, normal_img)
    assert len(out) == 0
    assert out.metadata.get("error") == "image_too_small"


def test_illumination_weighting():
    img_shadow = np.zeros((100, 100), dtype=np.uint8)
    img_lit = np.full((100, 100), 200, dtype=np.uint8)

    pts = np.array([[50, 50]], dtype=np.float32)
    conf = np.array([0.9], dtype=np.float32)

    weighted = apply_illumination_weighting(pts, pts, conf, img_shadow, img_lit, shadow_threshold_dn=25.0)
    # Shadow region should have penalized confidence
    assert weighted[0] < conf[0]


def test_spatial_balancing():
    # 50 points clustered in top-left cell
    pts_src = np.random.uniform(0, 10, size=(50, 2)).astype(np.float32)
    pts_ref = pts_src + 1.0
    confs = np.linspace(0.5, 0.99, 50).astype(np.float32)

    b_src, b_ref, b_conf = spatially_balance_matches(
        pts_src, pts_ref, confs, (100, 100), max_per_cell=5, grid=(4, 4)
    )
    assert len(b_src) <= 5
    assert len(b_src) == len(b_ref) == len(b_conf)


def test_root_sift_adapter_synthetic():
    adapter = RootSIFTAdapter(nfeatures=500)
    # Textured synthetic images with circles
    img_s = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img_s, (60, 60), 20, 255, -1)
    cv2.circle(img_s, (140, 140), 25, 180, -1)
    cv2.circle(img_s, (60, 140), 15, 220, -1)

    # Shifted image
    M = np.float32([[1, 0, 5], [0, 1, 5]])
    img_r = cv2.warpAffine(img_s, M, (200, 200))

    out = adapter.match(img_s, img_r)
    assert isinstance(out, MatcherOutput)
    assert out.matcher_name == "root_sift"


def test_get_matcher_adapter_factory():
    sift_adapter = get_matcher_adapter("root_sift")
    assert isinstance(sift_adapter, RootSIFTAdapter)


try:
    from src.features.crater_detector import _ULTRALYTICS_AVAILABLE as _HAS_ULTRALYTICS
except ImportError:
    _HAS_ULTRALYTICS = False


@pytest.mark.skipif(
    not _HAS_ULTRALYTICS,
    reason="ultralytics not installed; install 'ultralytics>=8.3.0' to enable crater adapter tests",
)
def test_get_crater_adapter_factory():
    crater_adapter = get_matcher_adapter("crater")
    assert crater_adapter.matcher_name == "yolo11_crater"
