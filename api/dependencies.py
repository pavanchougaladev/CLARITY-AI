"""
FastAPI Route Dependencies & Request Validation Models
======================================================
ISRO LunarFusion Co-Registration Suite — API Security
"""

from typing import Dict, Any, Optional, List, Tuple
from fastapi import HTTPException, Header, Depends
from pydantic import BaseModel

from .config import AUTH_ENABLED
from src.auth.jwt_handler import decode_access_token


class LoginRequest(BaseModel):
    username: str
    password: str


class TripletValidationRequest(BaseModel):
    job_id_ab: Optional[str] = None
    job_id_bc: Optional[str] = None
    job_id_ca: Optional[str] = None
    H_AB: Optional[List[List[float]]] = None
    H_BC: Optional[List[List[float]]] = None
    H_CA: Optional[List[List[float]]] = None
    image_shape: Optional[Tuple[int, int]] = (2048, 2048)
    tolerance_px: Optional[float] = 5.0


def _is_auth_active() -> bool:
    import sys
    if "api.main" in sys.modules:
        main_mod = sys.modules["api.main"]
        if hasattr(main_mod, "AUTH_ENABLED"):
            return bool(getattr(main_mod, "AUTH_ENABLED"))
    return bool(AUTH_ENABLED)


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency: Extract and verify caller identity and role."""
    if not _is_auth_active():
        return {"username": "anonymous", "role": "admin"}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token)
        return {
            "username": payload.get("sub", "unknown"),
            "role": payload.get("role", "viewer"),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication token invalid or expired: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """RBAC Guard: Enforce administrator clearance."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Administrator role required for this action")
    return user
