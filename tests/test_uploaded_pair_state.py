"""Uploads must remain the active pair across navigation and replacement."""
import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

CLIENT = Path(__file__).resolve().parents[1] / "client"
sys.path.insert(0, str(CLIENT))

import services
from config import NAV, PRESETS, WORKING_MAX_SIDE


def png_bytes(value):
    stream = io.BytesIO()
    Image.fromarray(np.full((32, 32), value, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def test_uploaded_pair_survives_navigation_and_replacement():
    app = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=20).run()
    app.radio[0].set_value(NAV[1]).run()
    app.file_uploader[0].upload("source.png", png_bytes(40)).run()
    app.file_uploader[1].upload("reference.png", png_bytes(90)).run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state["images"][0], 40)
    assert app.session_state["input_previews"] == (png_bytes(40), png_bytes(90))
    assert app.session_state["input_names"] == ("source.png", "reference.png")
    original_sig = services.pair_signature(*app.session_state["images"])

    app.radio[0].set_value(NAV[0]).run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state["images"][0], 40)
    app.radio[0].set_value(NAV[1]).run()
    assert not app.exception
    assert services.pair_signature(*app.session_state["images"]) == original_sig
    assert app.session_state["pair_source"] == "upload"

    # A stale legacy result without a fingerprint must also be invalidated.
    app.session_state["result"] = {"inliers": 100}
    app.file_uploader[0].upload("source.png", png_bytes(180)).run()
    assert not app.exception
    np.testing.assert_array_equal(app.session_state["images"][0], 180)
    assert app.session_state["input_previews"][0] == png_bytes(180)
    np.testing.assert_array_equal(app.session_state["images"][1], 90)
    assert app.session_state.get("result") is None
    assert services.pair_signature(*app.session_state["images"]) != original_sig

    next(b for b in app.button if b.label == "Clear uploaded images").click().run()
    assert not app.exception
    assert app.session_state["pair_source"] == "preset"
    assert app.session_state.get("saved_uploader_src") is None
    assert app.session_state.get("saved_uploader_ref") is None


def test_persisted_pair_is_restored_in_a_fresh_session():
    """A new session (new tab / other port / server restart) must restore the
    persisted uploaded pair instead of silently falling back to presets."""
    import workspace

    first = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=20).run()
    first.radio[0].set_value(NAV[1]).run()
    first.file_uploader[0].upload("persist_src.png", png_bytes(64)).run()
    first.file_uploader[1].upload("persist_ref.png", png_bytes(128)).run()
    assert not first.exception

    # A brand-new session: no widget state, no session-saved bytes.
    fresh = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=20).run()
    fresh.radio[0].set_value(NAV[1]).run()
    assert not fresh.exception
    np.testing.assert_array_equal(fresh.session_state["images"][0], 64)
    np.testing.assert_array_equal(fresh.session_state["images"][1], 128)
    assert fresh.session_state["input_names"] == ("persist_src.png", "persist_ref.png")
    assert fresh.session_state["pair_source"] == "upload"
    restored_note = " ".join(i.value for i in fresh.info)
    assert "restored from the workspace" in restored_note

    # Clearing in the new session must wipe both session state and disk.
    next(b for b in fresh.button if b.label == "Clear uploaded images").click().run()
    assert not fresh.exception
    assert workspace.load_pair() is None
    assert fresh.session_state["pair_source"] == "preset"


def test_partial_upload_is_flagged_and_mixes_in_a_preset():
    """Uploading only one image must be visible, not silently paired with a preset."""
    app = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value(NAV[1]).run()
    app.file_uploader[0].upload("only_src.png", png_bytes(77)).run()
    assert not app.exception

    assert app.session_state["partial_upload"] is True
    warnings = " ".join(w.value for w in app.warning)
    assert "Only one image was uploaded" in warnings
    # The reference slot still holds the bundled preset, which is what the warning is about.
    # Aspect ratio is preserved (longest side = WORKING_MAX_SIDE), so the
    # strip-shaped TMC-2 preset is not squashed.
    np.testing.assert_array_equal(app.session_state["images"][0], 77)
    assert max(app.session_state["images"][1].shape) == WORKING_MAX_SIDE

    # Completing the pair clears the warning.
    app.file_uploader[1].upload("only_ref.png", png_bytes(88)).run()
    assert not app.exception
    assert app.session_state["partial_upload"] is False
    assert "Only one image was uploaded" not in " ".join(w.value for w in app.warning)


def test_active_pair_banner_is_shown_on_every_screen():
    app = AppTest.from_file(str(CLIENT / "app.py"), default_timeout=30).run()
    nav_radio = next(r for r in app.radio if r.options == NAV)
    nav_radio.set_value(NAV[1]).run()
    app.file_uploader[0].upload("b_src.png", png_bytes(37)).run()
    app.file_uploader[1].upload("b_ref.png", png_bytes(90)).run()
    assert not app.exception

    expected = services.pair_signature(*app.session_state["images"])
    for screen in NAV:
        nav_radio = next(r for r in app.radio if r.options == NAV)
        nav_radio.set_value(screen).run()
        assert not app.exception
        captions = " ".join(c.value for c in app.caption)
        assert expected in captions, f"active-pair fingerprint missing on {screen}"


def test_disallowing_demo_always_executes_real_pipeline():
    src = np.full((32, 32), 40, dtype=np.uint8)
    ref = np.full((32, 32), 90, dtype=np.uint8)
    with patch.object(services, "_real_registration", return_value={"is_sample": False}) as real:
        result = services.run_registration(src, ref, {}, next(iter(PRESETS)), allow_demo=False)
    real.assert_called_once_with(src, ref, {})
    assert not result["is_sample"]
    assert result["pair_signature"] == services.pair_signature(src, ref)


def test_upload_decode_is_independent_of_stream_cursor():
    upload = io.BytesIO(png_bytes(125))
    upload.read()
    np.testing.assert_array_equal(services.to_gray512(upload), 125)
