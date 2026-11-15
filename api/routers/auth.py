"""
Authentication & JWT Issuance Endpoints
=======================================
ISRO LunarFusion Co-Registration Suite — Access Control
"""

from fastapi import APIRouter, HTTPException
from ..dependencies import LoginRequest
from src.auth.jwt_handler import authenticate_user, create_access_token

router = APIRouter(tags=["Authentication"])


@router.post("/api/v1/auth/token")
@router.post("/auth/token")
async def login_for_access_token(credentials: LoginRequest):
    """Authenticate with username/password and receive an HS256 JWT access token."""
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "username": user["username"],
    }
