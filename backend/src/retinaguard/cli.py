"""CLI commands for RetinaGuard administration.

Usage:
    python -m retinaguard.cli reset-admin-password
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from sqlalchemy import select

from retinaguard.auth.jwt import hash_password
from retinaguard.db.models import User, UserRole
from retinaguard.db.session import create_all_tables, get_sessionmaker


def is_dev_mode() -> bool:
    """Check whether a DEV environment flag is set."""
    env = os.environ.get("RETINAGUARD_ENV", "").lower()
    if env in ("production", "prod"):
        return False
    dev_flag = os.environ.get("RETINAGUARD_DEV", "").lower()
    dev_generic = os.environ.get("DEV", "").lower()
    return env in ("development", "dev", "local", "test") or dev_flag in ("1", "true", "yes") or dev_generic in ("1", "true", "yes")


async def reset_admin_password(load_env: bool = True) -> int:
    """Reset or seed the admin password from environment variables."""
    if load_env:
        load_dotenv()
    if not is_dev_mode():
        sys.stderr.write(
            "Error: Refusing to reset admin password. DEV flag is not set.\n"
            "Set RETINAGUARD_DEV=1 or RETINAGUARD_ENV=development to allow local password reset.\n"
        )
        return 1

    admin_email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    admin_password = os.environ.get("ADMIN_PASSWORD", "")

    if not admin_email or not admin_password:
        sys.stderr.write("Error: ADMIN_EMAIL or ADMIN_PASSWORD environment variable is not set.\n")
        return 1

    await create_all_tables()
    session_factory = get_sessionmaker()

    async with session_factory() as db:
        stmt = select(User).where(User.email == admin_email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        new_hash = hash_password(admin_password)

        if user:
            user.hashed_password = new_hash
            user.is_active = True
            user.role = UserRole.ADMIN
        else:
            user = User(
                email=admin_email,
                hashed_password=new_hash,
                role=UserRole.ADMIN,
                is_active=True,
            )
            db.add(user)

        await db.commit()

    # Never print the password
    print(f"Admin password successfully updated for {admin_email}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="retinaguard.cli", description="RetinaGuard Admin CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "reset-admin-password",
        help="Reset admin password from ADMIN_EMAIL and ADMIN_PASSWORD (DEV only)",
    )

    args = parser.parse_args()

    if args.command == "reset-admin-password":
        exit_code = asyncio.run(reset_admin_password())
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
