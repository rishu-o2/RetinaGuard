"""Authentication API endpoints (login, rate-limiting)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from retinaguard.auth.deps import require_role
from retinaguard.auth.jwt import create_access_token, hash_password, verify_password
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import get_db

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/auth", tags=["auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class CreateUserRequest(BaseModel):
    email: str
    password: str
    role: UserRole = UserRole.CLINICIAN


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_role(UserRole.ADMIN)),
) -> UserResponse:
    """Create a new user account (Admin only)."""
    email_clean = user_data.email.strip().lower()
    stmt = select(User).where(User.email == email_clean)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="User with this email already exists")

    new_user = User(
        email=email_clean,
        hashed_password=hash_password(user_data.password),
        role=user_data.role,
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return UserResponse(
        id=new_user.id,
        email=new_user.email,
        role=new_user.role.value,
        is_active=new_user.is_active,
    )


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Authenticate user with username (email) and password."""
    stmt = select(User).where(User.email == form_data.username.strip().lower())
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.id, "role": user.role.value, "email": user.email})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user.role.value,
        email=user.email,
    )


@router.get("/config")
def get_auth_config() -> dict[str, bool]:
    """Return public authentication configuration."""
    import os
    allow_anon = os.environ.get("ALLOW_ANONYMOUS_SCREENING", "false").lower() in ("true", "1", "yes")
    return {"allow_anonymous_screening": allow_anon}


