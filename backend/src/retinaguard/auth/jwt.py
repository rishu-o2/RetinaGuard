"""JWT token handling and password hashing."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import bcrypt
if not hasattr(bcrypt, "__about__"):
    bcrypt.__about__ = type("about", (), {"__version__": getattr(bcrypt, "__version__", "4.0.0")})

from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"
DEFAULT_EXPIRE_MINUTES = 60


def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        # For development / test fallback
        secret = "retinaguard-insecure-dev-secret-key-change-in-prod-123456"
    return secret


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        minutes = int(os.environ.get("JWT_EXPIRE_MINUTES", DEFAULT_EXPIRE_MINUTES))
        expire = now + timedelta(minutes=minutes)

    to_encode.update({"exp": expire, "iat": now})
    secret = get_jwt_secret()
    return jwt.encode(to_encode, secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials or token expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

