"""
Lightweight RFC 7519 HS256 JWT & Role-Based Access Control (RBAC) Handler
========================================================================
ISRO LunarFusion Co-Registration Suite — Planetary Data API

Zero-dependency standard library implementation (hmac, hashlib, base64, json).
Provides cryptographically secure authentication, token lifecycle management,
and role verification (admin, analyst, viewer).
"""

import os
import time
import json
import hmac
import hashlib
import base64
from typing import Dict, Any, Optional

DEFAULT_SECRET = "isro-lunar-fusion-secret-key-3d-mapping-2026"
DEFAULT_USERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "users.json"
)


def _b64url_encode(data: bytes) -> str:
    """Base64url encode without trailing padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data_str: str) -> bytes:
    """Base64url decode with restored padding."""
    pad_len = (-len(data_str)) % 4
    return base64.urlsafe_b64decode(data_str + ("=" * pad_len))


def get_jwt_secret() -> str:
    """Fetch active secret key from environment or default."""
    return os.getenv("JWT_SECRET_KEY", DEFAULT_SECRET)


def hash_password(password: str) -> str:
    """Compute deterministic SHA-256 hex digest for password storage."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against stored SHA-256 hash in constant time."""
    candidate_hash = hash_password(plain_password)
    return hmac.compare_digest(candidate_hash, hashed_password)


def create_access_token(
    claims: Dict[str, Any],
    expires_minutes: Optional[int] = None,
    secret_key: Optional[str] = None,
) -> str:
    """
    Generate an RFC 7519 compliant JSON Web Token using HS256 algorithm.

    Parameters
    ----------
    claims : dict
        Subject, roles, and custom claims to encode.
    expires_minutes : int, optional
        Expiration lifespan (defaults to 60 minutes).
    secret_key : str, optional
        Secret key override.
    """
    key = secret_key or get_jwt_secret()
    duration = expires_minutes if expires_minutes is not None else 60
    now = int(time.time())

    header = {"alg": "HS256", "typ": "JWT"}
    payload = claims.copy()
    payload.setdefault("iat", now)
    payload["exp"] = now + int(duration * 60)

    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}"

    signature = hmac.new(key.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{signing_input}.{sig_b64}"


def decode_access_token(token: str, secret_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate signature, timing, and decode payload of an HS256 JWT.

    Raises
    ------
    ValueError
        If the token is malformed, has an invalid signature, or is expired.
    """
    key = secret_key or get_jwt_secret()
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT token: must contain header, payload, and signature.")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}"

    # Verify signature in constant time
    expected_sig = hmac.new(key.encode("utf-8"), signing_input.encode("utf-8"), hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
    except Exception:
        raise ValueError("Invalid signature encoding in token.")

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid token signature: cryptographic verification failed.")

    # Decode and parse payload
    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse token payload: {e}")

    # Expiration check
    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp):
        raise ValueError("Token has expired.")

    return payload


def load_users(users_file: Optional[str] = None) -> Dict[str, Any]:
    """Load users database, seeding defaults if missing."""
    path = users_file or DEFAULT_USERS_FILE
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Seed default user roles
    default_users = {
        "admin": {
            "password_hash": hash_password("isro_admin_2026"),
            "role": "admin",
            "name": "Mission Commander",
        },
        "analyst": {
            "password_hash": hash_password("isro_analyst_2026"),
            "role": "analyst",
            "name": "Photogrammetry Analyst",
        },
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)
    except Exception:
        pass
    return default_users


def authenticate_user(username: str, password: str, users_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Verify username and password against users registry."""
    users = load_users(users_file)
    user_record = users.get(username)
    if not user_record:
        return None
    stored_hash = user_record.get("password_hash", "")
    if verify_password(password, stored_hash):
        return {
            "username": username,
            "role": user_record.get("role", "viewer"),
            "name": user_record.get("name", username),
        }
    return None
