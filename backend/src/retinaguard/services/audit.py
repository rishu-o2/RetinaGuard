"""Audit Ledger Service — Hash-Chained Tamper-Evident Audit Log.

ADR-005: Every screening event is appended to an append-only ledger with a
SHA-256 chain hash linking each entry to its predecessor. This makes
retrospective tampering detectable.

Properties:
- Append-only: entries can only be added, never modified.
- Hash-chained: each entry's `entry_hash` covers (previous_hash + entry payload).
- PHI-free storage: the ledger stores pseudonym IDs and screening IDs only,
  never names, Aadhaar, ABHA, or clinical images.
- DPDP Act 2023 compatible: audit trail retained per regulatory requirement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from retinaguard.core.logging import get_logger

logger = get_logger("retinaguard.services.audit")

# Sentinel genesis hash for the very first entry in a ledger
_GENESIS_HASH = "0" * 64


class AuditEntry:
    """Immutable single-line audit ledger record.

    The entry is serialised to a compact JSON line for append to the ledger file.
    ``entry_hash`` is computed over (previous_hash + deterministic JSON payload).
    """

    __slots__ = (
        "entry_id",
        "timestamp",
        "event_type",
        "screening_id",
        "patient_pseudonym_id",
        "image_sha256",
        "outcome_summary",
        "previous_hash",
        "entry_hash",
        "extra",
    )

    def __init__(
        self,
        *,
        event_type: str,
        screening_id: str,
        patient_pseudonym_id: str,
        image_sha256: str,
        outcome_summary: str,
        previous_hash: str,
        extra: Optional[dict[str, Any]] = None,
        entry_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self.entry_id: str = entry_id or str(uuid4())
        self.timestamp: str = timestamp or datetime.now(timezone.utc).isoformat()
        self.event_type: str = event_type
        self.screening_id: str = screening_id
        self.patient_pseudonym_id: str = patient_pseudonym_id
        self.image_sha256: str = image_sha256
        self.outcome_summary: str = outcome_summary
        self.previous_hash: str = previous_hash
        self.extra: dict[str, Any] = extra or {}
        self.entry_hash: str = self._compute_hash()

    def _payload_dict(self) -> dict[str, Any]:
        """Canonical, sorted payload for deterministic hashing."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "screening_id": self.screening_id,
            "patient_pseudonym_id": self.patient_pseudonym_id,
            "image_sha256": self.image_sha256,
            "outcome_summary": self.outcome_summary,
            "previous_hash": self.previous_hash,
            "extra": self.extra,
        }

    def _compute_hash(self) -> str:
        payload_json = json.dumps(self._payload_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(payload_json.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        d = self._payload_dict()
        d["entry_hash"] = self.entry_hash
        return d

    def to_json_line(self) -> str:
        """Serialise to a single-line JSON string (no trailing newline)."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEntry":
        entry = cls(
            event_type=data["event_type"],
            screening_id=data["screening_id"],
            patient_pseudonym_id=data["patient_pseudonym_id"],
            image_sha256=data["image_sha256"],
            outcome_summary=data["outcome_summary"],
            previous_hash=data["previous_hash"],
            extra=data.get("extra", {}),
            entry_id=data["entry_id"],
            timestamp=data["timestamp"],
        )
        return entry

    def verify(self) -> bool:
        """Re-compute hash and confirm it matches the stored entry_hash."""
        return self._compute_hash() == self.entry_hash


class AuditLedger:
    """Append-only SHA-256 hash-chained audit ledger.

    Thread-safety: append() acquires a file-level exclusive lock using
    a .lock sentinel file (compatible with Windows; does not use fcntl).

    Usage::

        ledger = AuditLedger(Path("audit/retinaguard.audit.jsonl"))
        ledger.append(
            event_type="SCREENING_COMPLETED",
            screening_id=str(result.screening_id),
            patient_pseudonym_id=patient.patient_pseudonym_id,
            image_sha256=meta.image_sha256,
            outcome_summary=f"GRADE={grade} REFERRAL={referral}",
        )
    """

    def __init__(self, ledger_path: Optional[Path | str] = None) -> None:
        if ledger_path is None:
            from retinaguard.core.config import _find_repo_root
            ledger_path = _find_repo_root() / "data" / "audit" / "retinaguard.audit.jsonl"
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.ledger_path.with_suffix(".lock")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_last_hash(self) -> str:
        """Return the entry_hash of the most recently appended entry, or genesis."""
        if not self.ledger_path.exists() or self.ledger_path.stat().st_size == 0:
            return _GENESIS_HASH
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            last_line = ""
            for line in f:
                stripped = line.strip()
                if stripped:
                    last_line = stripped
        if not last_line:
            return _GENESIS_HASH
        try:
            data = json.loads(last_line)
            return data.get("entry_hash", _GENESIS_HASH)
        except json.JSONDecodeError:
            logger.error("Corrupt last line in audit ledger — resetting chain hash to genesis.")
            return _GENESIS_HASH

    def _acquire_lock(self) -> None:
        """Busy-wait lock using a sentinel file (Windows-safe)."""
        import time
        timeout = 10.0
        interval = 0.05
        elapsed = 0.0
        while elapsed < timeout:
            try:
                self._lock_path.touch(exist_ok=False)
                return
            except FileExistsError:
                time.sleep(interval)
                elapsed += interval
        raise TimeoutError(f"Could not acquire audit ledger lock after {timeout}s")

    def _release_lock(self) -> None:
        try:
            self._lock_path.unlink(missing_ok=True)
        except Exception:
            pass  # Best-effort unlock

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        event_type: str,
        screening_id: str,
        patient_pseudonym_id: str,
        image_sha256: str,
        outcome_summary: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> AuditEntry:
        """Create and append one hash-chained audit entry.

        Returns the entry so the caller can embed its hash in the ScreeningResult.
        """
        self._acquire_lock()
        try:
            previous_hash = self._read_last_hash()
            entry = AuditEntry(
                event_type=event_type,
                screening_id=screening_id,
                patient_pseudonym_id=patient_pseudonym_id,
                image_sha256=image_sha256,
                outcome_summary=outcome_summary,
                previous_hash=previous_hash,
                extra=extra or {},
            )
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json_line() + "\n")
            logger.info(
                "Audit entry appended: event=%s screening=%s hash=%s...",
                event_type,
                screening_id,
                entry.entry_hash[:12],
            )
            return entry
        finally:
            self._release_lock()

    def verify_chain(self) -> tuple[bool, list[str]]:
        """Verify integrity of the entire chain.

        Returns:
            (is_intact, list_of_error_messages)
            Empty error list means the chain is intact.
        """
        if not self.ledger_path.exists():
            return True, []

        errors: list[str] = []
        previous_hash = _GENESIS_HASH
        line_num = 0

        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line_num += 1
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Line {line_num}: JSON parse error — {exc}")
                    continue

                entry = AuditEntry.from_dict(data)

                if not entry.verify():
                    errors.append(
                        f"Line {line_num}: entry_hash mismatch for entry_id={entry.entry_id}"
                    )

                if entry.previous_hash != previous_hash:
                    errors.append(
                        f"Line {line_num}: chain break — expected previous_hash={previous_hash[:12]}... "
                        f"got {entry.previous_hash[:12]}..."
                    )

                previous_hash = entry.entry_hash

        intact = len(errors) == 0
        if intact:
            logger.info("Audit chain verified: %d entries, all intact.", line_num)
        else:
            logger.error("Audit chain CORRUPTED: %d errors in %d entries.", len(errors), line_num)
        return intact, errors

    def tail(self, n: int = 20) -> list[AuditEntry]:
        """Return the last n audit entries (most recent last)."""
        if not self.ledger_path.exists():
            return []
        lines: list[str] = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for raw in f:
                stripped = raw.strip()
                if stripped:
                    lines.append(stripped)
        entries = []
        for raw in lines[-n:]:
            try:
                entries.append(AuditEntry.from_dict(json.loads(raw)))
            except Exception:
                pass
        return entries

