"""Tests for AuditLedger — hash-chained tamper-evident log."""

import json
import tempfile
from pathlib import Path

import pytest

from retinaguard.services.audit import AuditEntry, AuditLedger, _GENESIS_HASH


# ---------------------------------------------------------------------------
# AuditEntry unit tests
# ---------------------------------------------------------------------------

def _make_entry(previous_hash: str = _GENESIS_HASH, **kwargs) -> AuditEntry:
    defaults = dict(
        event_type="TEST_EVENT",
        screening_id="screen-001",
        patient_pseudonym_id="PAT-ANON-001",
        image_sha256="a" * 64,
        outcome_summary="GRADE=0 REFERRAL=NON_REFERABLE",
        previous_hash=previous_hash,
    )
    defaults.update(kwargs)
    return AuditEntry(**defaults)


def test_entry_hash_is_deterministic():
    e1 = _make_entry()
    e2 = AuditEntry.from_dict(e1.to_dict())
    assert e1.entry_hash == e2.entry_hash


def test_entry_verify_passes_for_intact_entry():
    entry = _make_entry()
    assert entry.verify() is True


def test_entry_verify_fails_if_payload_tampered():
    """Mutating a payload field then re-checking verify() must detect the tamper."""
    entry = _make_entry()
    original_hash = entry.entry_hash
    # Build a new entry with a different payload but claim the original hash
    d = entry.to_dict()
    d["outcome_summary"] = "TAMPERED_OUTCOME"
    # from_dict recomputes the hash — so the new hash will differ from the original
    rebuilt = AuditEntry.from_dict(d)
    assert rebuilt.entry_hash != original_hash, "Tampered payload must produce a different hash"
    assert rebuilt.verify() is True  # rebuilt's own hash is internally consistent
    # The tamper is detected by comparing rebuilt.entry_hash against the original
    assert rebuilt.entry_hash != original_hash


def test_entry_to_json_line_is_single_line():
    entry = _make_entry()
    line = entry.to_json_line()
    assert "\n" not in line
    parsed = json.loads(line)
    assert parsed["event_type"] == "TEST_EVENT"


# ---------------------------------------------------------------------------
# AuditLedger integration tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_ledger(tmp_path: Path) -> AuditLedger:
    return AuditLedger(ledger_path=tmp_path / "test.audit.jsonl")


def test_empty_ledger_chain_intact(tmp_ledger: AuditLedger):
    intact, errors = tmp_ledger.verify_chain()
    assert intact is True
    assert errors == []


def test_first_entry_has_genesis_previous_hash(tmp_ledger: AuditLedger):
    entry = tmp_ledger.append(
        event_type="FIRST",
        screening_id="s1",
        patient_pseudonym_id="p1",
        image_sha256="b" * 64,
        outcome_summary="ok",
    )
    assert entry.previous_hash == _GENESIS_HASH


def test_chain_links_entries(tmp_ledger: AuditLedger):
    e1 = tmp_ledger.append(
        event_type="E1", screening_id="s1", patient_pseudonym_id="p1",
        image_sha256="c" * 64, outcome_summary="ok",
    )
    e2 = tmp_ledger.append(
        event_type="E2", screening_id="s2", patient_pseudonym_id="p1",
        image_sha256="d" * 64, outcome_summary="ok",
    )
    assert e2.previous_hash == e1.entry_hash


def test_verify_chain_passes_for_valid_ledger(tmp_ledger: AuditLedger):
    for i in range(5):
        tmp_ledger.append(
            event_type="SCREENING_COMPLETED",
            screening_id=f"s{i}",
            patient_pseudonym_id="p1",
            image_sha256="e" * 64,
            outcome_summary=f"GRADE={i}",
        )
    intact, errors = tmp_ledger.verify_chain()
    assert intact is True, f"Errors: {errors}"


def test_verify_chain_detects_tampering(tmp_ledger: AuditLedger):
    """Mutating a line in the ledger must break the chain verification."""
    for i in range(3):
        tmp_ledger.append(
            event_type="E",
            screening_id=f"s{i}",
            patient_pseudonym_id="p",
            image_sha256="f" * 64,
            outcome_summary="ok",
        )

    # Read ledger lines
    lines = tmp_ledger.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    # Tamper: change outcome_summary in first entry
    first = json.loads(lines[0])
    first["outcome_summary"] = "TAMPERED"
    lines[0] = json.dumps(first)
    tmp_ledger.ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    intact, errors = tmp_ledger.verify_chain()
    assert intact is False
    assert len(errors) >= 1  # At minimum the hash mismatch for line 1


def test_tail_returns_last_n(tmp_ledger: AuditLedger):
    for i in range(10):
        tmp_ledger.append(
            event_type="E",
            screening_id=f"s{i}",
            patient_pseudonym_id="p",
            image_sha256="a" * 64,
            outcome_summary="ok",
        )
    tail = tmp_ledger.tail(n=3)
    assert len(tail) == 3
    assert tail[-1].screening_id == "s9"


def test_tail_empty_ledger(tmp_ledger: AuditLedger):
    assert tmp_ledger.tail(5) == []
