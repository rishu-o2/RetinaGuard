"""PDF Referral Report Service.

Generates a PHI-free, bilingual PDF clinical decision-support report using
ReportLab Platypus.  Hindi text is rendered via the Noto Sans Devanagari TTF
font; falls back to English gracefully if the font file is not present.

ADR-003: clinician_review_mandatory is ALWAYS stated in the report.
ADR-010: DPDP Act 2023 — only pseudonym_id is included; no real patient data.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retinaguard.core.config import _find_repo_root

logger = logging.getLogger("retinaguard.services.pdf")

# ---------------------------------------------------------------------------
# Font paths
# ---------------------------------------------------------------------------

_REPO_ROOT = _find_repo_root()
_FONTS_DIR = _REPO_ROOT / "backend" / "fonts"
_NOTO_DEVANAGARI = _FONTS_DIR / "NotoSansDevanagari-Regular.ttf"

# ---------------------------------------------------------------------------
# ReportLab imports (guarded so the module can be imported without reportlab
# in test environments that only check for the class's existence)
# ---------------------------------------------------------------------------

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _REPORTLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REPORTLAB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Models registry helper
# ---------------------------------------------------------------------------

_REGISTRY_PATH = _REPO_ROOT / "models" / "registry.json"


def _get_model_version() -> str:
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        return data["models"][0]["version"]
    except Exception:  # noqa: BLE001
        return "v1.0.0"


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_URGENCY_COLOURS: dict[str, Any] = {}

def _urgency_colour(urgency: str) -> "colors.Color":  # type: ignore[name-defined]
    mapping = {
        "ROUTINE": colors.HexColor("#28a745"),
        "SOON": colors.HexColor("#fd7e14"),
        "PRIORITY": colors.HexColor("#dc3545"),
        "URGENT": colors.HexColor("#dc3545"),
        "EMERGENCY": colors.HexColor("#9b0000"),
    }
    return mapping.get(urgency.upper(), colors.black)


# ---------------------------------------------------------------------------
# Font registration
# ---------------------------------------------------------------------------

_HINDI_FONT_NAME = "NotoSansDevanagari"
_HINDI_FONT_REGISTERED = False


def _ensure_hindi_font() -> bool:
    """Register Noto Sans Devanagari TTF and return True on success."""
    global _HINDI_FONT_REGISTERED  # noqa: PLW0603
    if _HINDI_FONT_REGISTERED:
        return True
    if not _NOTO_DEVANAGARI.exists():
        logger.warning(
            "Hindi font not found at %s — falling back to Helvetica (Hindi text may not render).",
            _NOTO_DEVANAGARI,
        )
        return False
    try:
        pdfmetrics.registerFont(TTFont(_HINDI_FONT_NAME, str(_NOTO_DEVANAGARI)))
        _HINDI_FONT_REGISTERED = True
        logger.info("Registered Hindi font from %s", _NOTO_DEVANAGARI)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to register Hindi font: %s — falling back to Helvetica.", exc)
        return False


# ---------------------------------------------------------------------------
# PDF Report Service
# ---------------------------------------------------------------------------


class PDFReportService:
    """Builds a RetinaGuard Clinical Decision Support PDF report.

    Returns raw PDF bytes.  No PHI beyond pseudonym_id is included.
    """

    def generate(
        self,
        screening_id: str,
        result_json: dict[str, Any],
        language: str = "en",
    ) -> bytes:
        """Generate a PDF report.

        Args:
            screening_id: UUID string of the screening record.
            result_json:  Dict as returned by ReportService.build_json_report()
                          plus "report_text" key containing the localised summary.
            language:     "en" or "hi".

        Returns:
            Raw PDF bytes (starts with b"%PDF").
        """
        if not _REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab is not installed; cannot generate PDF.")

        is_hi = language == "hi"
        hindi_ok = _ensure_hindi_font() if is_hi else False
        body_font = _HINDI_FONT_NAME if (is_hi and hindi_ok) else "Helvetica"
        body_font_bold = "Helvetica-Bold"  # Bold variant from Helvetica family always available

        model_version = _get_model_version()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title="RetinaGuard Clinical Decision Support Report",
            author="RetinaGuard",
            subject="Diabetic Retinopathy Screening Report",
            pageCompression=0,
        )

        styles = getSampleStyleSheet()
        story = []

        # ── 1. Header ────────────────────────────────────────────────────────
        header_style = ParagraphStyle(
            "RGHeader",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=6,
        )
        story.append(Paragraph("RetinaGuard Clinical Decision Support Report", header_style))
        story.append(Spacer(1, 0.3 * cm))

        # ── 2. Disclaimer box ────────────────────────────────────────────────
        disclaimer_style = ParagraphStyle(
            "RGDisclaimer",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#b22222"),
            spaceAfter=4,
        )
        disclaimer_text = (
            "⚠ Prototype, not clinically validated. "
            "Must not be used for clinical decisions without regulatory clearance (CDSCO/FDA/CE)."
        )
        disclaimer_para = Paragraph(disclaimer_text, disclaimer_style)
        disclaimer_table = Table([[disclaimer_para]], colWidths=[doc.width])
        disclaimer_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 1.5, colors.red),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff5f5")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(disclaimer_table)
        story.append(Spacer(1, 0.4 * cm))

        # ── 3. Case info table ───────────────────────────────────────────────
        label_style = ParagraphStyle(
            "RGLabel",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=colors.HexColor("#555555"),
        )
        value_style = ParagraphStyle(
            "RGValue",
            fontName=body_font,
            fontSize=9,
        )

        pseudonym_id = result_json.get("patient_pseudonym_id", "UNKNOWN")
        date_str = result_json.get("created_at", timestamp)
        eye_side = result_json.get("eye_side", "")
        camera_type = result_json.get("camera_type", "")

        case_data = [
            [Paragraph("Pseudonymous ID", label_style), Paragraph(str(pseudonym_id), value_style)],
            [Paragraph("Date", label_style), Paragraph(str(date_str), value_style)],
            [Paragraph("Eye Side", label_style), Paragraph(str(eye_side), value_style)],
            [Paragraph("Camera Type", label_style), Paragraph(str(camera_type), value_style)],
            [Paragraph("Model Version", label_style), Paragraph(model_version, value_style)],
        ]
        col_w = doc.width / 2
        case_table = Table(case_data, colWidths=[col_w * 0.4, col_w * 1.6])
        case_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(Paragraph("CASE INFORMATION", ParagraphStyle(
            "RGSectionHead",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=4,
        )))
        story.append(case_table)
        story.append(Spacer(1, 0.4 * cm))

        # ── 4. Result section ────────────────────────────────────────────────
        section_head_style = ParagraphStyle(
            "RGSectionHead2",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=colors.HexColor("#1a1a2e"),
            spaceAfter=4,
        )
        story.append(Paragraph("FINDINGS", section_head_style))

        grade = result_json.get("grade")
        grade_label_key = "grade_label_hi" if is_hi else "grade_label_en"
        grade_label = result_json.get(grade_label_key) or result_json.get("grade_label_en", "Unknown")
        urgency = result_json.get("urgency", "ROUTINE")
        follow_up = result_json.get("follow_up_interval_months", "N/A")
        referral = result_json.get("referral", "")

        urgency_colour = _urgency_colour(str(urgency))

        result_data = [
            [
                Paragraph("Grade", label_style),
                Paragraph(f"{grade} — {grade_label}" if grade is not None else str(grade_label), value_style),
            ],
            [
                Paragraph("Urgency", label_style),
                Paragraph(
                    f'<font color="{urgency_colour.hexval() if hasattr(urgency_colour, "hexval") else "#000000"}">'
                    f"<b>{urgency}</b></font>",
                    value_style,
                ),
            ],
            [
                Paragraph("Follow-up Interval", label_style),
                Paragraph(f"{follow_up} month(s)", value_style),
            ],
            [
                Paragraph("Referral Decision", label_style),
                Paragraph(str(referral), value_style),
            ],
        ]
        result_table = Table(result_data, colWidths=[col_w * 0.4, col_w * 1.6])
        result_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(result_table)
        story.append(Spacer(1, 0.4 * cm))

        # ── 5. Report text ───────────────────────────────────────────────────
        report_text = result_json.get("report_text", "")
        if report_text:
            story.append(Paragraph("REPORT SUMMARY", section_head_style))
            report_style = ParagraphStyle(
                "RGReport",
                fontName=body_font,
                fontSize=9,
                leading=14,
                spaceAfter=4,
            )
            # Escape angle brackets for HTML-ish ReportLab paragraphs
            safe_text = (
                str(report_text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )
            story.append(Paragraph(safe_text, report_style))
            story.append(Spacer(1, 0.4 * cm))

        # ── 6. Clinician review section ──────────────────────────────────────
        story.append(Paragraph("CLINICIAN REVIEW", section_head_style))

        clinician_confirmation = result_json.get("clinician_confirmation", "pending")
        final_grade = result_json.get("final_grade_override")
        note_hash = result_json.get("note_hash", "")

        review_lines = [f"Clinician Confirmation: {clinician_confirmation}"]
        if final_grade is not None:
            review_lines.append(f"Final Grade (Overridden): {final_grade}")
        if note_hash:
            review_lines.append(f"Note Hash: {note_hash}")

        review_style = ParagraphStyle(
            "RGReview",
            fontName="Helvetica",
            fontSize=9,
            leading=14,
        )
        for line in review_lines:
            story.append(Paragraph(line, review_style))
        story.append(Spacer(1, 0.6 * cm))

        # ── 7. Footer ────────────────────────────────────────────────────────
        footer_style = ParagraphStyle(
            "RGFooter",
            fontName="Helvetica",
            fontSize=8,
            textColor=colors.HexColor("#888888"),
            alignment=TA_CENTER,
            borderPad=4,
        )
        footer_text = (
            f"Generated by RetinaGuard {model_version} | {timestamp} | For clinician use only"
        )
        footer_line = Table(
            [[Paragraph(footer_text, footer_style)]],
            colWidths=[doc.width],
        )
        footer_line.setStyle(
            TableStyle(
                [
                    ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.HexColor("#cccccc")),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(footer_line)

        # Build PDF
        doc.build(story)
        return buf.getvalue()
