from fastapi.testclient import TestClient
import os
import tempfile
import threading
from api.main import app, jobs, cancellation_events, _sanitize_filename

def test_api_health():
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
    assert "max_concurrent_jobs" in resp.json()

def test_api_root():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ISRO" in resp.json()["message"]


def test_api_rejects_unsupported_matcher():
    client = TestClient(app)
    response = client.post(
        "/api/v1/register?matcher=not-a-matcher",
        files={
            "source_image": ("source.png", b"not-an-image", "image/png"),
            "reference_image": ("reference.png", b"not-an-image", "image/png"),
        },
    )
    assert response.status_code == 422


def test_api_rejects_unsupported_file_type():
    client = TestClient(app)
    response = client.post(
        "/api/v1/register",
        files={
            "source_image": ("source.txt", b"not-an-image", "text/plain"),
            "reference_image": ("reference.png", b"not-an-image", "image/png"),
        },
    )
    assert response.status_code == 415


def test_api_cancel_job():
    client = TestClient(app)
    job_id = "test-cancel-id-123"
    token = threading.Event()
    jobs[job_id] = {
        "status": "processing",
        "job_id": job_id,
        "job_dir": None,
    }
    cancellation_events[job_id] = token

    resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelling"
    assert token.is_set()


def test_api_delete_job():
    client = TestClient(app)
    job_id = "test-delete-id-456"
    tmp_dir = tempfile.mkdtemp(prefix="test_del_")
    jobs[job_id] = {
        "status": "completed",
        "job_id": job_id,
        "job_dir": tmp_dir,
    }

    resp = client.delete(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == job_id
    assert job_id not in jobs
    assert not os.path.exists(tmp_dir)


def test_api_artifact_download():
    client = TestClient(app)
    job_id = "test-art-id-789"
    tmp_dir = tempfile.mkdtemp(prefix="test_art_")
    telemetry_file = os.path.join(tmp_dir, "telemetry.json")
    with open(telemetry_file, "w") as f:
        f.write('{"status": "verified"}')

    jobs[job_id] = {
        "status": "completed",
        "job_id": job_id,
        "job_dir": tmp_dir,
        "artifact_paths": {
            "telemetry": telemetry_file,
        },
    }

    resp = client.get(f"/api/v1/jobs/{job_id}/artifacts/telemetry")
    assert resp.status_code == 200
    assert resp.json()["status"] == "verified"

    # Incomplete job gives 400
    jobs["pending-job"] = {"status": "processing"}
    resp_pending = client.get("/api/v1/jobs/pending-job/artifacts/telemetry")
    assert resp_pending.status_code == 400


def test_sanitize_filename_traversal():
    assert _sanitize_filename("../../etc/passwd") == "passwd"
    assert _sanitize_filename("valid_image_01.tif") == "valid_image_01.tif"
