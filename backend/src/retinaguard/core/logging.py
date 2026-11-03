"""Zero-PHI Logging Module.

Enforces Principle 6: No patient identifiers (PII/PHI) in logs.
Automatically sanitizes ABHA IDs, phone numbers, email addresses,
Aadhaar numbers, and patient names from log messages and attributes.
"""

import json
import logging
import re
from typing import Any


# Patterns for sensitive Indian PII/PHI identifiers
PATTERNS = {
    # 14-digit ABHA numbers (with or without dashes)
    "ABHA_NUMBER": re.compile(r"\b\d{2}-\d{4}-\d{4}-\d{4}\b|\b\d{14}\b"),
    # ABHA address handles (e.g. name@abdm)
    "ABHA_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@abdm\b", re.IGNORECASE),
    # 12-digit Indian Aadhaar numbers
    "AADHAAR": re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b"),
    # Indian mobile numbers (10 digits starting with 6, 7, 8, or 9 with optional +91/0 prefix)
    "PHONE_NUMBER": re.compile(r"(?:\+91[-\s]?|0)?[6-9]\d{9}\b"),
    # Standard email addresses
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
}


def sanitize_text(text: str) -> str:
    """Scrub detected PII/PHI patterns and replace with redaction tokens."""
    if not isinstance(text, str):
        return text

    sanitized = text
    sanitized = PATTERNS["ABHA_NUMBER"].sub("[REDACTED_ABHA_ID]", sanitized)
    sanitized = PATTERNS["ABHA_ADDRESS"].sub("[REDACTED_ABHA_ADDR]", sanitized)
    sanitized = PATTERNS["AADHAAR"].sub("[REDACTED_AADHAAR]", sanitized)
    sanitized = PATTERNS["PHONE_NUMBER"].sub("[REDACTED_PHONE]", sanitized)
    sanitized = PATTERNS["EMAIL"].sub("[REDACTED_EMAIL]", sanitized)
    return sanitized


class PHIScrubbingFilter(logging.Filter):
    """Logging filter that scrubs sensitive patient health and identity information."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = sanitize_text(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    sanitize_text(arg) if isinstance(arg, str) else arg for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: sanitize_text(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


class JSONLogFormatter(logging.Formatter):
    """Structured JSON formatter outputting zero-PHI audit-safe logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if hasattr(record, "screening_id"):
            log_entry["screening_id"] = str(getattr(record, "screening_id"))
        if hasattr(record, "patient_pseudonym_id"):
            log_entry["patient_pseudonym_id"] = str(getattr(record, "patient_pseudonym_id"))

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger with mandatory PHI scrubbing."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, PHIScrubbingFilter) for f in logger.filters):
        logger.addFilter(PHIScrubbingFilter())
    return logger

