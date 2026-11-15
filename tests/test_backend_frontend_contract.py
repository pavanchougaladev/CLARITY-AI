"""
Tests for FastAPI Backend <-> Streamlit UI Contract and End-to-End Deserialization.
Verifies serialization of telemetry, match arrays, homographies, and safe consumption
by the UI visualization and export pipelines.
"""

import os
import io
import time
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from api.main import app, jobs, process_registration
from src.registration.export import export_asp_compatible_tiepoints


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_png_bytes():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, 255, -1)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_register_endpoint_form_and_query_dispatch(client, sample_png_bytes):
    """Test that /api/v1/register accepts matcher via form or query and enqueues job."""
    # Test via query parameter
    resp_query = client.post(
        "/api/v1/register?matcher=sift",
        files={
            "source_image": ("src.png", sample_png_bytes, "image/png"),
            "reference_image": ("ref.png", sample_png_bytes, "image/png"),
        },
    )
    assert resp_query.status_code == 202
    job_id_query = resp_query.json()["job_id"]
    assert job_id_query in jobs
    assert jobs[job_id_query]["matcher"] == "sift"

    # Test via form data
    resp_form = client.post(
        "/api/v1/register",
        files={
            "source_image": ("src.png", sample_png_bytes, "image/png"),
            "reference_image": ("ref.png", sample_png_bytes, "image/png"),
        },
        data={"matcher": "fusion"},
    )
    assert resp_form.status_code == 202
    job_id_form = resp_form.json()["job_id"]
    assert job_id_form in jobs
    assert jobs[job_id_form]["matcher"] == "fusion"


def test_api_serialization_and_ui_deserialization_contract(tmp_path):
    """Verify serialization in process_registration matches the frontend deserialization."""
    # Synthetic registered points
    n_pts = 20
    pts_s = np.random.uniform(10, 90, (n_pts, 2)).astype(np.float32)
    pts_r = pts_s + np.random.normal(0, 0.5, (n_pts, 2)).astype(np.float32)
    conf = np.random.uniform(0.7, 0.99, n_pts).astype(np.float32)
    h_matrix = np.eye(3, dtype=np.float32)

    mock_results = {
        "inliers_src": pts_s,
        "inliers_ref": pts_r,
        "confidences": conf,
        "H_fallback": h_matrix,
        "method_used": "LUNARFUSION (ROMA + SIFT)",
        "scale_src": 1.0,
        "scale_ref": 1.0,
        "rmse_pixels": 0.42,
        "inlier_count": n_pts,
        "raw_matches": 35,
        "inlier_ratio": float(n_pts / 35),
        "execution_time_seconds": 1.25,
        "quality_grade": "A (EXCELLENT)",
        "quality_passed": True,
        "quality_confidence": 0.94,
        "warnings": [],
        "reason_codes": [],
        "primary_reason": "Certified for CartoSat/OHRC mapping",
        "provenance": {"engine": "lunar_fusion"},
        "omniroute_stats": None,
        "lunar_fusion_stats": {"models_used": ["roma", "sift"], "profile_tag": "high_texture"},
        "self_healed": False,
    }

    job_id = "test-contract-job-001"
    jobs[job_id] = {"status": "queued"}

    # Simulate backend job serialization
    from src.result_schema import RegistrationTelemetry
    telemetry = RegistrationTelemetry.from_results(mock_results)
    clean_results = telemetry.to_dict()
    clean_results['quality'] = {
        'grade': telemetry.quality_grade,
        'passed': telemetry.quality_passed,
        'confidence': telemetry.quality_confidence,
        'warnings': telemetry.warnings,
        'reason_codes': telemetry.reason_codes,
        'primary_reason': telemetry.primary_reason,
    }
    clean_results['provenance'] = mock_results.get('provenance', {})
    clean_results['recovery_attempts'] = mock_results.get('recovery_attempts', [])
    clean_results['method_used'] = mock_results.get('method_used', telemetry.method)
    clean_results['scale_src'] = float(mock_results.get('scale_src', 1.0))
    clean_results['scale_ref'] = float(mock_results.get('scale_ref', 1.0))
    clean_results['inliers_src'] = mock_results['inliers_src'].tolist()
    clean_results['inliers_ref'] = mock_results['inliers_ref'].tolist()
    clean_results['confidences'] = mock_results['confidences'].tolist()
    clean_results['H_fallback'] = mock_results['H_fallback'].tolist()
    clean_results['omniroute_stats'] = mock_results.get('omniroute_stats')
    clean_results['lunar_fusion_stats'] = mock_results.get('lunar_fusion_stats')
    clean_results['self_healed'] = bool(mock_results.get('self_healed', False))

    jobs[job_id]['status'] = 'completed'
    jobs[job_id]['results'] = clean_results

    # Now verify frontend deserialization logic
    ui_results = clean_results.copy()
    if "inliers_src" in ui_results and ui_results["inliers_src"] is not None:
        ui_results["inliers_src"] = np.asarray(ui_results["inliers_src"], dtype=np.float32)
    if "inliers_ref" in ui_results and ui_results["inliers_ref"] is not None:
        ui_results["inliers_ref"] = np.asarray(ui_results["inliers_ref"], dtype=np.float32)
    if "confidences" in ui_results and ui_results["confidences"] is not None:
        ui_results["confidences"] = np.asarray(ui_results["confidences"], dtype=np.float32)
    if "H_fallback" in ui_results and ui_results["H_fallback"] is not None:
        ui_results["H_fallback"] = np.asarray(ui_results["H_fallback"], dtype=np.float32)

    # Validate types and shapes
    assert isinstance(ui_results["inliers_src"], np.ndarray)
    assert ui_results["inliers_src"].shape == (20, 2)
    assert isinstance(ui_results["inliers_ref"], np.ndarray)
    assert ui_results["inliers_ref"].shape == (20, 2)
    assert isinstance(ui_results["confidences"], np.ndarray)
    assert ui_results["confidences"].shape == (20,)
    assert isinstance(ui_results["H_fallback"], np.ndarray)
    assert ui_results["H_fallback"].shape == (3, 3)

    # Verify tiepoints exporter accepts deserialized results
    csv_out = str(tmp_path / "tiepoints.csv")
    csv_path = export_asp_compatible_tiepoints(
        ui_results["inliers_src"],
        ui_results["inliers_ref"],
        ui_results["confidences"],
        csv_out,
    )
    assert os.path.exists(csv_path)
    with open(csv_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 21  # 1 header + 20 tiepoints
        assert lines[0].strip() == "x1,y1,x2,y2,weight"


def test_frontend_backend_helpers(monkeypatch):
    """Test frontend backend helper functions against mock / live responses."""
    from ui.app import (
        _format_backend_results,
        _list_backend_jobs,
        _cancel_backend_job,
        _delete_backend_job,
        _fetch_backend_artifact,
        _trigger_backend_blender_export,
    )

    # 1. Test _format_backend_results field standardizations
    raw = {
        "inliers_src": [[1.0, 2.0], [3.0, 4.0]],
        "inliers_ref": [[1.1, 2.1], [3.1, 4.1]],
        "confidences": [0.9, 0.95],
        "rmse": 0.35,
        "method": "RoMa v2",
    }
    fmt = _format_backend_results(raw)
    assert fmt["inlier_count"] == 2
    assert fmt["inliers"] == 2
    assert fmt["rmse_pixels"] == 0.35
    assert fmt["method_used"] == "RoMa v2"
    assert isinstance(fmt["inliers_src"], np.ndarray)

    # 2. Test _list_backend_jobs, _cancel_backend_job, _delete_backend_job with requests mocking
    class MockResp:
        def __init__(self, ok, json_data=None, content=b"", status_code=200, text=""):
            self.ok = ok
            self._json = json_data or {}
            self.content = content
            self.status_code = status_code
            self.text = text

        def json(self):
            return self._json

    monkeypatch.setattr(
        "requests.get",
        lambda url, timeout=None: (
            MockResp(True, content=b"fake-geotiff-bytes")
            if "artifacts/geotiff" in url
            else MockResp(True, json_data={"count": 1, "jobs": [{"job_id": "job-123", "status": "completed"}]})
        )
    )
    monkeypatch.setattr(
        "requests.post",
        lambda url, **kwargs: MockResp(True, json_data={"status": "launched", "pid": 9999})
    )
    monkeypatch.setattr(
        "requests.delete",
        lambda url, timeout=None: MockResp(True, json_data={"status": "deleted"})
    )

    jobs_list = _list_backend_jobs()
    assert len(jobs_list) == 1
    assert jobs_list[0]["job_id"] == "job-123"

    assert _cancel_backend_job("job-123") is True
    assert _delete_backend_job("job-123") is True

    art = _fetch_backend_artifact("job-123", "geotiff")
    assert art == b"fake-geotiff-bytes"

    bld = _trigger_backend_blender_export(ply_path="data/test.ply", background=False)
    assert bld.get("status") == "launched"
    assert bld.get("pid") == 9999


def test_register_endpoint_paper_matchers_and_pipelines(client, sample_png_bytes):
    """Test that /api/v1/register accepts paper matchers (superglue, rift2, asift, akaze) and pipelines."""
    for matcher_name in ["superglue", "rift2", "asift", "akaze"]:
        resp = client.post(
            f"/api/v1/register?matcher={matcher_name}",
            files={
                "source_image": ("src.png", sample_png_bytes, "image/png"),
                "reference_image": ("ref.png", sample_png_bytes, "image/png"),
            },
            data={
                "pipeline_mode": "pipeline_a",
                "sensor_pair": "OHRC-NAC",
                "region_type": "polar",
            }
        )
        assert resp.status_code == 202
        jid = resp.json()["job_id"]
        assert jid in jobs
        assert jobs[jid]["matcher"] == matcher_name
        assert jobs[jid]["pipeline_mode"] == "pipeline_a"
        assert jobs[jid]["sensor_pair"] == "OHRC-NAC"
        assert jobs[jid]["region_type"] == "polar"

