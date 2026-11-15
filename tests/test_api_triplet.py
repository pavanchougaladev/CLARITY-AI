"""
Unit Tests for Triplet Validation REST API Endpoint (Feature 1)
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from api.main import app, jobs


def test_api_triplet_validate_explicit_matrices():
    client = TestClient(app)

    # 3 shift matrices that sum to zero: (20, 10), (10, 15), (-30, -25)
    H_AB = [[1.0, 0.0, 20.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]
    H_BC = [[1.0, 0.0, 10.0], [0.0, 1.0, 15.0], [0.0, 0.0, 1.0]]
    H_CA = [[1.0, 0.0, -30.0], [0.0, 1.0, -25.0], [0.0, 0.0, 1.0]]

    req_body = {
        "H_AB": H_AB,
        "H_BC": H_BC,
        "H_CA": H_CA,
        "image_shape": [1024, 1024],
        "tolerance_px": 2.0,
    }

    resp = client.post("/api/v1/triplet-validate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "PASS"
    assert data["is_consistent"] is True
    assert data["mean_residual_px"] < 1e-3


def test_api_triplet_validate_with_job_ids():
    client = TestClient(app)

    # Seed 3 mock completed jobs into jobs registry
    H_AB = [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    H_BC = [[1.0, 0.0, 5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    H_CA = [[1.0, 0.0, -10.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    jobs["job-ab"] = {"status": "completed", "results": {"H_fallback": H_AB}}
    jobs["job-bc"] = {"status": "completed", "results": {"H_fallback": H_BC}}
    jobs["job-ca"] = {"status": "completed", "results": {"H_fallback": H_CA}}

    req_body = {
        "job_id_ab": "job-ab",
        "job_id_bc": "job-bc",
        "job_id_ca": "job-ca",
        "tolerance_px": 2.0,
    }

    resp = client.post("/api/v1/triplet-validate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "PASS"
    assert data["is_consistent"] is True


def test_api_triplet_validate_missing_inputs():
    client = TestClient(app)
    # Post with incomplete data
    resp = client.post("/api/v1/triplet-validate", json={"tolerance_px": 3.0})
    assert resp.status_code == 400
