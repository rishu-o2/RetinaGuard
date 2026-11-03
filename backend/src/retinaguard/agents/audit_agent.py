"""Audit Agent for RetinaGuard.

ADR-005 & ADR-006: Cryptographically logs every screening event into a tamper-evident,
SHA-256 hash-chained JSONL ledger.

Strict Privacy Guard:
- Zero PHI: Only pseudonym IDs and image SHA-256 hashes are recorded.
- No image bytes or personal identifiers are stored in the audit record.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from retinaguard.agents.state import ScreeningAgentState
from retinaguard.core.logging import get_logger
from retinaguard.services.audit import AuditLedger

logger = get_logger("retinaguard.agents.audit")


class AuditAgent:
    """Agent that appends tamper-evident audit records for all screening decisions."""

    def __init__(self, ledger: Optional[AuditLedger] = None) -> None:
        self.ledger = ledger or AuditLedger()

    def __call__(self, state: ScreeningAgentState) -> dict:
        """Append the screening event to the hash-chained audit ledger."""
        patient_context = state["patient_context"]
        metadata = state["image_metadata"]
        decision = state.get("referral_decision")
        inference_out = state.get("inference_output")
        is_uncertain = state.get("is_uncertain", False)

        # Build outcome summary
        if is_uncertain or decision is None:
            event_type = "SCREENING_UNCERTAIN"
            outcome_summary = f"UNCERTAIN; Reasons={'; '.join(state.get('uncertainty_reasons', []))}"
            grade_val = None
        else:
            event_type = "SCREENING_COMPLETED"
            grade_val = inference_out.predicted_grade.value if inference_out else None
            outcome_summary = (
                f"Grade={grade_val}; "
                f"Referral={decision.referral.value}; "
                f"Urgency={decision.urgency.value}; "
                f"IntervalMonths={decision.follow_up_interval_months}"
            )

        screening_id = str(uuid4())

        # Zero PHI verification: Ensure pseudonym ONLY, no raw image bytes
        entry = self.ledger.append(
            event_type=event_type,
            screening_id=screening_id,
            patient_pseudonym_id=patient_context.patient_pseudonym_id,
            image_sha256=metadata.image_sha256,
            outcome_summary=outcome_summary,
            extra={
                "camera_type": metadata.camera_type.value,
                "eye_side": metadata.eye_side.value,
                "is_uncertain": is_uncertain,
                "clinician_review_mandatory": True,
            },
        )

        logger.info(
            "AuditAgent: Appended entry %s (hash=%s, prev=%s)",
            entry.entry_id,
            entry.entry_hash[:16],
            entry.previous_hash[:16],
        )

        return {
            "audit_hash": entry.entry_hash,
            "previous_audit_hash": entry.previous_hash,
        }
