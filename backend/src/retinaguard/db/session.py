"""Database engine and session management for RetinaGuard.

Supports SQLite (local default via aiosqlite) and PostgreSQL (via asyncpg).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from retinaguard.db.base import Base

# ---------------------------------------------------------------------------
# Database URL resolution
# ---------------------------------------------------------------------------


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./retinaguard.db")
    # Handle standard postgres:// or postgresql:// -> postgresql+asyncpg://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://") and not url.startswith("sqlite+"):
        url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        url = get_database_url()
        connect_args = {}
        if "sqlite" in url:
            connect_args["check_same_thread"] = False
        _engine = create_async_engine(
            url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        engine = get_engine()
        _sessionmaker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _sessionmaker


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for yielding an async database session."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    """Create all tables in the database (useful for SQLite & dev/tests)."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
