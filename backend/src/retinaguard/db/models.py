"""Database ORM models for RetinaGuard.

Tables:
- User: system accounts with roles (screener / clinician / admin)
- Screening: one record per submitted fundus image (pseudonymised, no PHI)
- Review: clinician decision linked to a Screening
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from retinaguard.db.base import Base


# ---------------------------------------------------------------------------
# Python enumerations
# ---------------------------------------------------------------------------


class UserRole(str, enum.Enum):
    SCREENER = "screener"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class ScreeningStatus(str, enum.Enum):
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    OVERRIDDEN = "overridden"


class ReviewDecision(str, enum.Enum):
    CONFIRMED = "confirmed"
    OVERRIDDEN = "overridden"


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """System user account."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"), nullable=False, default=UserRole.SCREENER
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # relationships
    reviews: Mapped[list["Review"]] = relationship("Review", back_populates="clinician")


class Screening(Base):
    """One DR screening event (PHI-free: only pseudonym hash stored)."""

    __tablename__ = "screenings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # SHA-256 hash of the patient_pseudonym_id — no real patient data
    pseudonym_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    audit_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    status: Mapped[ScreeningStatus] = mapped_column(
        Enum(ScreeningStatus, name="screening_status"),
        nullable=False,
        default=ScreeningStatus.PENDING_REVIEW,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # relationships
    reviews: Mapped[list["Review"]] = relationship("Review", back_populates="screening")


class Review(Base):
    """Clinician review decision for a Screening."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    screening_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("screenings.id"), nullable=False, index=True
    )
    clinician_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    decision: Mapped[ReviewDecision] = mapped_column(
        Enum(ReviewDecision, name="review_decision"), nullable=False
    )
    final_grade: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # relationships
    screening: Mapped["Screening"] = relationship("Screening", back_populates="reviews")
    clinician: Mapped["User"] = relationship("User", back_populates="reviews")
