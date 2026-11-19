"""The bytes that reach the real pipeline must be the uploaded files."""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

CLIENT = Path(__file__).resolve().parents[1] / "client"
ROOT = Path(__file__).resolve().parents[1]
for path in (str(CLIENT), str(ROOT), str(ROOT / "server")):
    if path not in sys.path:
        sys.path.insert(0, path)

import services
from config import NAV, WORKING_MAX_SIDE


def png_bytes(value, size=64):
    stream = io.BytesIO()
    Image.fromarray(np.full((size, size), value, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def capture_pipeline_call():
    """Patch the pipeline and record the pixels it actually receives."""
    seen = {}

    def fake_pipeline(src_path, ref_path, **kwargs):
        with Image.open(src_path) as img_s:
            seen["src"] = np.array(img_s.convert("L"))
        with Image.open(ref_path) as img_r:
            seen["ref"] = np.array(img_r.convert("L"))
        seen["kwargs"] = kwargs
        empty = np.empty((0, 2), dtype=np.float32)
        return None, None, {
            "inliers_src": empty,
            "inliers_ref": empty,
            "inlier_count": 0,
            "rmse_pixels": None,
            "coverage": 0.0,
            "total_matches": 0,
            "quality": {"primary_reason": "TEST_NO_OVERLAP"},
            "quality_grade": "F",
            "img_src_warped": seen["src"],
        }

    return seen, fake_pipeline


def test_pipeline_receives_the_uploaded_pixels():
    app = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value(NAV[1]).run()
    app.file_uploader[0].upload("src.png", png_bytes(37)).run()
    app.file_uploader[1].upload("ref.png", png_bytes(213)).run()
    assert not app.exception

    seen, fake_pipeline = capture_pipeline_call()
    with patch("src.pipeline.run_registration_pipeline", side_effect=fake_pipeline):
        next(b for b in app.button if b.label == "Execute co-registration").click().run()

    assert not app.exception
    assert seen, "the real pipeline was never invoked"
    # Grayscale working copy of the uploaded 64x64 uniform images
    # (aspect preserved, longest side = WORKING_MAX_SIDE).
    expected_src = np.full((WORKING_MAX_SIDE, WORKING_MAX_SIDE), 37, dtype=np.uint8)
    expected_ref = np.full((WORKING_MAX_SIDE, WORKING_MAX_SIDE), 213, dtype=np.uint8)
    np.testing.assert_array_equal(seen["src"], expected_src)
    np.testing.assert_array_equal(seen["ref"], expected_ref)

    result = app.session_state["result"]
    assert result["pair_signature"] == services.pair_signature(expected_src, expected_ref)
    assert result["input_audit"]["persisted_pixels_verified"] is True
    assert result["input_names"] == ("src.png", "ref.png")


def test_replacing_an_upload_registers_the_new_pixels_not_the_old_ones():
    app = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value(NAV[1]).run()
    app.file_uploader[0].upload("src.png", png_bytes(37)).run()
    app.file_uploader[1].upload("ref.png", png_bytes(213)).run()
    assert not app.exception

    seen, fake_pipeline = capture_pipeline_call()
    with patch("src.pipeline.run_registration_pipeline", side_effect=fake_pipeline):
        next(b for b in app.button if b.label == "Execute co-registration").click().run()
    np.testing.assert_array_equal(seen["src"], np.full((WORKING_MAX_SIDE, WORKING_MAX_SIDE), 37, dtype=np.uint8))

    # Replace only the source with a different image.
    app.file_uploader[0].upload("src.png", png_bytes(155)).run()
    assert not app.exception
    seen2, fake_pipeline2 = capture_pipeline_call()
    with patch("src.pipeline.run_registration_pipeline", side_effect=fake_pipeline2):
        next(b for b in app.button if b.label == "Execute co-registration").click().run()

    np.testing.assert_array_equal(seen2["src"], np.full((WORKING_MAX_SIDE, WORKING_MAX_SIDE), 155, dtype=np.uint8))
    np.testing.assert_array_equal(seen2["ref"], np.full((WORKING_MAX_SIDE, WORKING_MAX_SIDE), 213, dtype=np.uint8))


def test_real_run_residuals_are_measured_not_random():
    """Residuals must be the measured ||H·src - ref|| distances, never random noise."""
    import services

    src = np.full((64, 64), 37, dtype=np.uint8)
    ref = np.full((64, 64), 90, dtype=np.uint8)
    pts_s = np.array([[10, 10], [20, 15], [30, 30], [40, 25], [50, 40]], dtype=np.float32)
    pts_r = pts_s + np.array([2.0, 1.0], dtype=np.float32)
    # The fitted transform maps source onto (+2, 0), while the true offset is
    # (+2, +1): every tie-point therefore carries a measured residual of 1.0 px.
    shift_h = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    results = dict(
        inliers_src=pts_s,
        inliers_ref=pts_r,
        inliers_src_proj=None,
        H_fallback=shift_h,
        inlier_count=5,
        rmse_pixels=None,
        coverage=0.5,
        total_matches=5,
        quality={},
        quality_grade="F",
        img_src_warped=src,
        norm_src=src,
        norm_ref=ref,
        illumination_disparity={},
        provenance={},
    )
    with patch("src.pipeline.run_registration_pipeline", return_value=(None, None, results)):
        res = services._real_registration(src, ref, {"matcher": "sift", "transform": "homography"})

    expected = 1.0
    assert len(res["residuals"]) == 5
    assert all(abs(r - expected) < 1e-6 for r in res["residuals"])

    # A rejected run with no fitted transform must yield NO fabricated residuals.
    results_no_transform = dict(results, H_fallback=None, inliers_src=np.empty((0, 2), np.float32),
                                inliers_ref=np.empty((0, 2), np.float32), inlier_count=0, total_matches=0)
    with patch("src.pipeline.run_registration_pipeline", return_value=(None, None, results_no_transform)):
        res_empty = services._real_registration(src, ref, {"matcher": "sift", "transform": "homography"})
    assert res_empty["residuals"] == []
