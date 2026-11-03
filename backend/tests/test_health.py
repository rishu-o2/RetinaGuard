"""Tests for FastAPI /health endpoint."""

from fastapi.testclient import TestClient
from retinaguard.api.main import app

client = TestClient(app)


def test_health_check_endpoint() -> None:
    """Verify that /health returns 200 OK with expected status and service identity."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "retinaguard-api"
    assert data["country_context"] == "IN"


def test_root_endpoint() -> None:
    """Verify that root endpoint returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["mode"] == "DECISION_SUPPORT_ONLY"

