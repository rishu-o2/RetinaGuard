"""Tests for authentication, JWT handling, and role-based access control."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from retinaguard.api.main import app
from retinaguard.auth.jwt import create_access_token, hash_password
from retinaguard.db.base import Base
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import get_db


@pytest.fixture
async def test_db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def client(test_db: AsyncSession):
    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_auth_login_success(client: AsyncClient, test_db: AsyncSession):
    # Create test user
    user = User(
        email="clinician@hospital.in",
        hashed_password=hash_password("DoctorPass123!"),
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    test_db.add(user)
    await test_db.commit()

    resp = await client.post(
        "/auth/login",
        data={"username": "clinician@hospital.in", "password": "DoctorPass123!"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "clinician"
    assert data["email"] == "clinician@hospital.in"


@pytest.mark.asyncio
async def test_auth_login_invalid_password(client: AsyncClient, test_db: AsyncSession):
    user = User(
        email="screener@clinic.in",
        hashed_password=hash_password("CorrectPass123!"),
        role=UserRole.SCREENER,
        is_active=True,
    )
    test_db.add(user)
    await test_db.commit()

    resp = await client.post(
        "/auth/login",
        data={"username": "screener@clinic.in", "password": "WrongPassword!"},
    )
    assert resp.status_code == 401
    assert "Incorrect email or password" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_login_nonexistent_user(client: AsyncClient, test_db: AsyncSession):
    resp = await client.post(
        "/auth/login",
        data={"username": "ghost@clinic.in", "password": "SomePassword123!"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_role_guard_blocks_unauthorized_role(client: AsyncClient, test_db: AsyncSession):
    # Screener user should NOT be able to access /screenings (requires CLINICIAN or ADMIN)
    screener = User(
        email="screener2@clinic.in",
        hashed_password=hash_password("Pass123!"),
        role=UserRole.SCREENER,
        is_active=True,
    )
    test_db.add(screener)
    await test_db.commit()

    token = create_access_token(data={"sub": screener.id, "role": "screener"})
    resp = await client.get("/screenings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403
    assert "Operation not permitted" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_role_guard_allows_clinician(client: AsyncClient, test_db: AsyncSession):
    clinician = User(
        email="doctor@hospital.in",
        hashed_password=hash_password("Pass123!"),
        role=UserRole.CLINICIAN,
        is_active=True,
    )
    test_db.add(clinician)
    await test_db.commit()

    token = create_access_token(data={"sub": clinician.id, "role": "clinician"})
    resp = await client.get("/screenings", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
