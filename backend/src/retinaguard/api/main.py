"""FastAPI main application for RetinaGuard."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from retinaguard import __version__
from retinaguard.core.logging import get_logger
from retinaguard.api.screen import router as screen_router

logger = get_logger("retinaguard.api")

app = FastAPI(
    title="RetinaGuard Decision-Support API",
    description="Clinical decision-support service for Diabetic Retinopathy screening (CDSCO / DPDP compliant)",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(screen_router)
app.include_router(screen_router, prefix="/api/v1")


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

