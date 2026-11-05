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


def verify_model_artifacts() -> None:
    """Check whether ONNX model and OOD stats exist and log/print a clear error if missing."""
    from retinaguard.core.config import _find_repo_root
    repo_root = _find_repo_root()
    registry_path = repo_root / "models" / "registry.json"

    onnx_path = repo_root / "ml" / "exports" / "retinaguard_dr_v1.onnx"
    ood_path = repo_root / "models" / "ood_stats_v1.npz"

    if registry_path.exists():
        try:
            import json
            with open(registry_path, encoding="utf-8") as f:
                reg = json.load(f)
            entries = reg.get("models", [])
            if entries:
                entry = entries[0]
                if "onnx_path" in entry:
                    onnx_path = repo_root / entry["onnx_path"]
                if "ood_stats_path" in entry:
                    ood_path = repo_root / entry["ood_stats_path"]
        except Exception as exc:
            logger.warning("Could not read registry.json: %s", exc)

    missing = []
    if not onnx_path.exists():
        missing.append(f"ONNX Model: {onnx_path}")
    if not ood_path.exists():
        missing.append(f"OOD Statistics: {ood_path}")

    if missing:
        msg = (
            "\n" + "=" * 70 + "\n"
            "STARTUP WARNING: Missing AI model artifacts:\n" +
            "\n".join(f"  - {m}" for m in missing) + "\n\n"
            "Inference will fall back to UNCERTAIN until these artifacts are provided.\n"
            "To place them:\n"
            "  Mount external weights in container: docker compose up (with ./models mounted)\n"
            "  Or mount directly: docker run -v /path/to/models:/app/models ...\n"
            "  Or generate locally: python ml/export_onnx.py && python ml/generate_ood_stats.py\n"
            + "=" * 70
        )
        logger.error(msg)
        print(msg)
    else:
        logger.info("AI model artifacts verified: ONNX model and OOD stats present.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle."""
    try:
        await create_all_tables()
        await seed_admin_user()
        verify_model_artifacts()
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
