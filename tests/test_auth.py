"""
Unit Tests for Authentication & Role-Based Access Control (Feature 3)
"""

import os
import pytest
from fastapi.testclient import TestClient
from src.auth.jwt_handler import (
    create_access_token,
    decode_access_token,
    authenticate_user,
    hash_password,
    verify_password,
)
from api.main import app, jobs


def test_password_hashing():
    plain = "chandrayaan_mission_2026"
    h = hash_password(plain)
    assert len(h) == 64
    assert verify_password(plain, h) is True
    assert verify_password("wrong_password", h) is False


def test_jwt_lifecycle_and_expiry():
    token = create_access_token({"sub": "astronaut", "role": "pilot"}, expires_minutes=1)
    payload = decode_access_token(token)
    assert payload["sub"] == "astronaut"
    assert payload["role"] == "pilot"

    # Expired token test
    expired_token = create_access_token({"sub": "old_user"}, expires_minutes=-1)
    with pytest.raises(ValueError, match="expired"):
        decode_access_token(expired_token)

    # Tampered token test
    parts = token.split(".")
    tampered_token = f"{parts[0]}.{parts[1]}.badsignature"
    with pytest.raises(ValueError, match="Invalid token signature"):
        decode_access_token(tampered_token)


def test_authenticate_default_users():
    admin = authenticate_user("admin", "isro_admin_2026")
    assert admin is not None
    assert admin["role"] == "admin"

    analyst = authenticate_user("analyst", "isro_analyst_2026")
    assert analyst is not None
    assert analyst["role"] == "analyst"

    invalid = authenticate_user("admin", "wrong_pass")
    assert invalid is None


def test_api_auth_token_endpoint():
    client = TestClient(app)
    # Valid login
    resp = client.post("/api/v1/auth/token", json={"username": "admin", "password": "isro_admin_2026"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["role"] == "admin"

    # Invalid login
    resp_bad = client.post("/api/v1/auth/token", json={"username": "admin", "password": "invalid_password"})
    assert resp_bad.status_code == 401


def test_api_auth_enforcement_when_enabled(monkeypatch):
    client = TestClient(app)
    import api.main

    # Simulate AUTH_ENABLED = True
    monkeypatch.setattr(api.main, "AUTH_ENABLED", True)

    # Calling a protected endpoint without token should fail
    resp_unauth = client.delete("/api/v1/jobs/dummy-job")
    assert resp_unauth.status_code == 401

    # Calling with valid admin token
    admin_token = create_access_token({"sub": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {admin_token}"}

    # Job not found should be 404, not 401
    resp_auth = client.delete("/api/v1/jobs/dummy-job", headers=headers)
    assert resp_auth.status_code == 404

    # Calling with analyst role should be 403 Forbidden on admin endpoint
    analyst_token = create_access_token({"sub": "analyst", "role": "analyst"})
    headers_analyst = {"Authorization": f"Bearer {analyst_token}"}
    resp_forbidden = client.delete("/api/v1/jobs/dummy-job", headers=headers_analyst)
    assert resp_forbidden.status_code == 403
