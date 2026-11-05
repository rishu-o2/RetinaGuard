"""Screening and clinician review queue API endpoints."""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from retinaguard.auth.deps import get_current_user, require_role
from retinaguard.db.models import Review, ReviewDecision, Screening, ScreeningStatus, User, UserRole
from retinaguard.db.session import get_db
from retinaguard.services.audit import AuditLedger
from retinaguard.services.pdf import PDFReportService

router = APIRouter(prefix="/screenings", tags=["screenings"])
_pdf_service = PDFReportService()
_audit_ledger = AuditLedger()


class ReviewRequest(BaseModel):
    decision: ReviewDecision
    final_grade: Optional[int] = Field(None, ge=0, le=4)
    note: Optional[str] = None


class ScreeningSummary(BaseModel):
    id: str
    pseudonym_id: str
    status: str
    created_at: str
    grade: Optional[int] = None
    grade_label: Optional[str] = None
    urgency: Optional[str] = None
    referral: Optional[str] = None
    is_uncertain: bool = False


@router.get("", response_model=list[ScreeningSummary])
async def list_screenings(
    screening_status: Optional[ScreeningStatus] = Query(ScreeningStatus.PENDING_REVIEW, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN)),
) -> list[ScreeningSummary]:
    """List screenings in the review queue (clinicians and admins only)."""
    stmt = select(Screening)
    if screening_status is not None:
        stmt = stmt.where(Screening.status == screening_status)
    stmt = stmt.order_by(desc(Screening.created_at)).offset(offset).limit(limit)

    result = await db.execute(stmt)
    screenings = result.scalars().all()

    summaries = []
    for s in screenings:
        try:
            rdata = json.loads(s.result_json)
        except Exception:
            rdata = {}

        summaries.append(
            ScreeningSummary(
                id=s.id,
                pseudonym_id=s.pseudonym_id,
                status=s.status.value,
                created_at=s.created_at.isoformat(),
                grade=rdata.get("grade"),
                grade_label=rdata.get("grade_label_en") or rdata.get("grade_label"),
                urgency=rdata.get("urgency"),
                referral=rdata.get("referral"),
                is_uncertain=rdata.get("is_uncertain", False),
            )
        )
    return summaries


@router.get("/{screening_id}")
async def get_screening(
    screening_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get single screening details including full result JSON."""
    stmt = select(Screening).where(Screening.id == screening_id)
    result = await db.execute(stmt)
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    try:
        rdata = json.loads(screening.result_json)
    except Exception:
        rdata = {}

    return {
        "id": screening.id,
        "pseudonym_id": screening.pseudonym_id,
        "status": screening.status.value,
        "model_version": screening.model_version,
        "audit_hash": screening.audit_hash,
        "created_at": screening.created_at.isoformat(),
        "result": rdata,
    }


@router.post("/{screening_id}/review")
async def review_screening(
    screening_id: str,
    review_data: ReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_role(UserRole.CLINICIAN, UserRole.ADMIN)),
) -> dict[str, Any]:
    """Submit a clinician review (Confirm or Override) for a screening."""
    stmt = select(Screening).where(Screening.id == screening_id)
    result = await db.execute(stmt)
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    if review_data.decision == ReviewDecision.OVERRIDDEN and review_data.final_grade is None:
        raise HTTPException(
            status_code=400,
            detail="final_grade is required when decision is overridden",
        )

    # 1. Create Review record
    review = Review(
        screening_id=screening.id,
        clinician_id=current_user.id,
        decision=review_data.decision,
        final_grade=review_data.final_grade,
        note=review_data.note,
    )
    db.add(review)

    # 2. Update Screening status
    if review_data.decision == ReviewDecision.CONFIRMED:
        screening.status = ScreeningStatus.CONFIRMED
    else:
        screening.status = ScreeningStatus.OVERRIDDEN

    # Update result_json with clinician review summary
    try:
        rdata = json.loads(screening.result_json)
        rdata["clinician_confirmation"] = review_data.decision.value
        rdata["final_grade_override"] = review_data.final_grade
        screening.result_json = json.dumps(rdata)
    except Exception:
        pass

    # 3. Append to Audit Ledger
    _audit_ledger.append_review(
        screening_id=screening.id,
        clinician_id=current_user.id,
        decision=review_data.decision.value,
        final_grade=review_data.final_grade,
        note=review_data.note,
    )

    await db.flush()

    return {
        "status": "success",
        "screening_id": screening.id,
        "review_id": review.id,
        "decision": review_data.decision.value,
        "screening_status": screening.status.value,
    }


@router.get("/{screening_id}/report.pdf")
async def download_report_pdf(
    screening_id: str,
    lang: str = Query("en", pattern="^(en|hi)$"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Response:
    """Generate and download server-side PDF referral report."""
    stmt = select(Screening).where(Screening.id == screening_id)
    result = await db.execute(stmt)
    screening = result.scalar_one_or_none()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    try:
        result_json = json.loads(screening.result_json)
    except Exception:
        result_json = {}

    pdf_bytes = _pdf_service.generate(
        screening_id=screening.id,
        result_json=result_json,
        language=lang,
    )

    filename = f"retinaguard_report_{screening.id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        },
    )
