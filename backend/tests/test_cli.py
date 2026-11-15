"""Tests for RetinaGuard administration CLI."""

import os
import subprocess
import sys
import pytest
from sqlalchemy import select

from retinaguard.auth.jwt import hash_password, verify_password
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import create_all_tables, get_sessionmaker


@pytest.mark.asyncio
async def test_reset_admin_password_refuses_without_dev_flag(monkeypatch, capsys):
    from retinaguard.cli import reset_admin_password

    monkeypatch.setenv("RETINAGUARD_ENV", "production")
    monkeypatch.delenv("RETINAGUARD_DEV", raising=False)
    monkeypatch.delenv("DEV", raising=False)
    monkeypatch.setenv("ADMIN_EMAIL", "admin@hospital.in")
    monkeypatch.setenv("ADMIN_PASSWORD", "NewSecretPass123!")

    code = await reset_admin_password(load_env=False)
    assert code == 1
    captured = capsys.readouterr()
    assert "DEV flag is not set" in captured.err
    assert "NewSecretPass123!" not in captured.err
    assert "NewSecretPass123!" not in captured.out


@pytest.mark.asyncio
async def test_reset_admin_password_success(monkeypatch, capsys):
    from retinaguard.cli import reset_admin_password

    test_email = "admin-cli-test@hospital.in"
    initial_pass = "OldPassword123!"
    new_pass = "UpdatedAdminPassword123!"

    await create_all_tables()
    session_factory = get_sessionmaker()

    # Pre-seed user with old password
    async with session_factory() as db:
        stmt = select(User).where(User.email == test_email)
        res = await db.execute(stmt)
        user = res.scalar_one_or_none()
        if not user:
            user = User(
                email=test_email,
                hashed_password=hash_password(initial_pass),
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(user)
        else:
            user.hashed_password = hash_password(initial_pass)
        await db.commit()

    # Set DEV flag and run CLI function
    monkeypatch.setenv("RETINAGUARD_DEV", "1")
    monkeypatch.setenv("ADMIN_EMAIL", test_email)
    monkeypatch.setenv("ADMIN_PASSWORD", new_pass)

    code = await reset_admin_password()
    assert code == 0

    captured = capsys.readouterr()
    assert f"Admin password successfully updated for {test_email}" in captured.out
    # Never leak password
    assert new_pass not in captured.out
    assert new_pass not in captured.err

    # Verify new password in DB
    async with session_factory() as db:
        stmt = select(User).where(User.email == test_email)
        res = await db.execute(stmt)
        updated_user = res.scalar_one_or_none()
        assert updated_user is not None
        assert verify_password(new_pass, updated_user.hashed_password)
        assert not verify_password(initial_pass, updated_user.hashed_password)


def test_cli_invocation_via_subprocess(tmp_path):
    """Test full CLI invocation via python -m retinaguard.cli."""
    env = os.environ.copy()
    env["RETINAGUARD_DEV"] = "1"
    env["ADMIN_EMAIL"] = "cli-subproc@hospital.in"
    env["ADMIN_PASSWORD"] = "SubprocPass123!"

    result = subprocess.run(
        [sys.executable, "-m", "retinaguard.cli", "reset-admin-password"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Admin password successfully updated for cli-subproc@hospital.in" in result.stdout
    assert "SubprocPass123!" not in result.stdout
    assert "SubprocPass123!" not in result.stderr
