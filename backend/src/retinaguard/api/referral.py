"""Eye-facility referral directory endpoint.

GET /referral?state=&district=
  Returns up to 3 facilities from configs/facilities.csv matching the given
  state and district (case-insensitive).  Returns [] when the directory is
  empty or no match is found.  No authentication required (public data).

The CSV must only contain verified, real facilities added by authorised
personnel following the instructions in docs/FACILITIES.md.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query

from retinaguard.core.config import _find_repo_root
from retinaguard.core.logging import get_logger

logger = get_logger("retinaguard.api.referral")

router = APIRouter(prefix="/referral", tags=["referral"])

_FACILITIES_CSV = _find_repo_root() / "configs" / "facilities.csv"
_MAX_RESULTS = 3


def _load_facilities() -> list[dict]:
    """Read the facilities CSV and return a list of row dicts."""
    if not _FACILITIES_CSV.exists():
        logger.warning("facilities.csv not found at %s", _FACILITIES_CSV)
        return []
    rows: list[dict] = []
    try:
        with open(_FACILITIES_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
    except Exception as exc:
        logger.error("Could not read facilities.csv: %s", exc)
    return rows


@router.get(
    "",
    summary="Return up to 3 eye-care facilities for a state/district",
)
def get_referral_facilities(
    state: Optional[str] = Query(default=None, description="State name"),
    district: Optional[str] = Query(default=None, description="District name"),
) -> list[dict]:
    """Return up to 3 facilities matching the given state and district.

    Matching is case-insensitive and strips leading/trailing whitespace.
    When state or district is omitted, the corresponding filter is skipped.
    Returns an empty list when no entries match or the CSV is empty.
    """
    all_facilities = _load_facilities()
    if not all_facilities:
        return []

    state_q = (state or "").strip().lower()
    district_q = (district or "").strip().lower()

    matched = []
    for row in all_facilities:
        row_state = (row.get("state") or "").strip().lower()
        row_district = (row.get("district") or "").strip().lower()

        state_match = (not state_q) or (row_state == state_q)
        district_match = (not district_q) or (row_district == district_q)

        if state_match and district_match:
            matched.append(row)
            if len(matched) >= _MAX_RESULTS:
                break

    return matched
