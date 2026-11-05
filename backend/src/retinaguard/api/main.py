"""FastAPI main application for RetinaGuard."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import select

from retinaguard import __version__
from retinaguard.api.auth import limiter, router as auth_router
from retinaguard.api.screen import router as screen_router
from retinaguard.api.screenings import router as screenings_router
from retinaguard.api.voice import router as voice_router
from retinaguard.auth.jwt import hash_password
from retinaguard.core.logging import get_logger
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import create_all_tables, get_sessionmaker

logger = get_logger("retinaguard.api")


async def seed_admin_user() -> None:
    """Seed initial admin account from environment variables if configured."""
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")
    if not admin_email or not admin_password:
        return

    admin_email = admin_email.strip().lower()
    session_factory = get_sessionmaker()
    async with session_factory() as db:
        stmt = select(User).where(User.email == admin_email)
        result = await db.execute(stmt)
        existing = result.scalar_one_or_none()
        if not existing:
            admin_user = User(
                email=admin_email,
                hashed_password=hash_password(admin_password),
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(admin_user)
            await db.commit()
            logger.info("Seeded initial admin user: %s", admin_email)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    try:
        await create_all_tables()
        await seed_admin_user()
    except Exception as exc:
        logger.warning("DB table creation or admin seed warning: %s", exc)
    yield


app = FastAPI(
    title="RetinaGuard Decision-Support API",
    description="Clinical decision-support service for Diabetic Retinopathy screening (CDSCO / DPDP compliant)",
    version=__version__,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route registrations
app.include_router(screen_router)
app.include_router(auth_router)
app.include_router(screenings_router)
app.include_router(voice_router)

# v1 prefixed routes
app.include_router(screen_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(screenings_router, prefix="/api/v1")
app.include_router(voice_router, prefix="/api/v1")


@app.get("/")
def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "RetinaGuard API",
        "description": "Diabetic Retinopathy Clinical Decision Support System",
        "version": __version__,
        "mode": "DECISION_SUPPORT_ONLY",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for orchestration, uptime probes, and load balancers."""
    logger.info("Health check ping received.")
    return {
        "status": "healthy",
        "service": "retinaguard-api",
        "version": __version__,
        "country_context": "IN",
    }
