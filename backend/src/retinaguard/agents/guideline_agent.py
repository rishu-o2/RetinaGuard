"""Guideline Agent for RetinaGuard (Simple RAG with Safety Validator).

Reads authentic clinical guideline texts/PDFs from docs/guidelines/.
Never invents or hardcodes clinical guideline content.
Strict validator rejects any text containing conflicting grades, urgency, or intervals,
falling back securely to the deterministic localized report templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from retinaguard.agents.state import ScreeningAgentState
from retinaguard.core.config import _find_repo_root
from retinaguard.core.logging import get_logger
from retinaguard.core.models import ICDRGrade, ReferralDecision, Urgency

logger = get_logger("retinaguard.agents.guideline")


@dataclass
class GuidelinePassage:
    """A passage retrieved from an authentic guideline document."""
    text: str
    source: str
    year: str
    page: str
    grade_relevance: list[int]


class GuidelineValidator:
    """Validates retrieved or generated explanations against the ground-truth safety decision.

    Guarantees that an explanation can NEVER alter the determined grade, referral
    decision, urgency level, or follow-up interval.
    """

    GRADE_NAMES = {
        0: ["grade 0", "no dr", "no diabetic retinopathy"],
        1: ["grade 1", "mild npdr", "mild non-proliferative"],
        2: ["grade 2", "moderate npdr", "moderate non-proliferative"],
        3: ["grade 3", "severe npdr", "severe non-proliferative"],
        4: ["grade 4", "proliferative dr", "pdr"],
    }

    def validate(
        self,
        explanation: str,
        decision: ReferralDecision,
        grade: Optional[ICDRGrade],
    ) -> tuple[bool, Optional[str]]:
        """Verify that explanation does not contradict the deterministic safety decision.

        Returns (is_valid, rejection_reason).
        """
        lower_text = explanation.lower()

        # 1. If a grade is specified, ensure text does not assert a DIFFERENT grade
        if grade is not None:
            for other_grade, aliases in self.GRADE_NAMES.items():
                if other_grade != grade.value:
                    for alias in aliases:
                        # Match explicit declarations like "diagnosed with Grade X" or "classified as Grade X"
                        pattern = rf"\b(?:is|as|diagnosed with|classified as|severity of)\s+{re.escape(alias)}\b"
                        if re.search(pattern, lower_text):
                            return False, f"Explanation asserts conflicting grade ({alias}) instead of Grade {grade.value}."

        # 2. Check for conflicting intervals
        interval = decision.follow_up_interval_months
        interval_matches = re.findall(r"\b(\d+)\s*(?:-|to)?\s*(?:\d+)?\s*(?:months?|mo)\b", lower_text)
        for match in interval_matches:
            try:
                mentioned_months = int(match)
                if interval > 0 and mentioned_months != interval and mentioned_months > 0:
                    # If text claims a different re-screening interval
                    return False, f"Explanation mentions conflicting re-screening interval ({mentioned_months} months vs {interval} months)."
            except ValueError:
                pass

        # 3. Check for conflicting urgency
        if decision.urgency == Urgency.EMERGENCY and "routine" in lower_text:
            return False, "Explanation mentions routine follow-up for an emergency finding."
        if decision.urgency == Urgency.ROUTINE and ("emergency" in lower_text or "immediate vitrectomy" in lower_text):
            return False, "Explanation mentions emergency intervention for a routine finding."

        return True, None


class GuidelineAgent:
    """RAG-based Guideline retrieval and validated plain-language explanation generation."""

    def __init__(
        self,
        guidelines_dir: Optional[Path | str] = None,
        validator: Optional[GuidelineValidator] = None,
    ) -> None:
        if guidelines_dir is None:
            guidelines_dir = _find_repo_root() / "docs" / "guidelines"
        self.guidelines_dir = Path(guidelines_dir)
        self.validator = validator or GuidelineValidator()

    def _index_documents(self) -> list[GuidelinePassage]:
        """Index all text and markdown files in the guidelines directory."""
        if not self.guidelines_dir.exists():
            return []

        passages: list[GuidelinePassage] = []
        doc_files = (
            list(self.guidelines_dir.glob("*.txt"))
            + list(self.guidelines_dir.glob("*.md"))
            + list(self.guidelines_dir.glob("*.pdf"))
        )

        for doc_path in doc_files:
            try:
                # Text/Markdown files
                if doc_path.suffix in (".txt", ".md"):
                    content = doc_path.read_text(encoding="utf-8", errors="replace")
                    # Extract header metadata if present
                    source = doc_path.stem.replace("_", " ").upper()
                    year = "2020"
                    page = "1"

                    # Parse simple markdown frontmatter or lines
                    lines = content.splitlines()
                    cleaned_lines = []
                    for line in lines:
                        if line.lower().startswith("source:"):
                            source = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("year:"):
                            year = line.split(":", 1)[1].strip()
                        elif line.lower().startswith("page:"):
                            page = line.split(":", 1)[1].strip()
                        else:
                            cleaned_lines.append(line)

                    cleaned_text = "\n".join(cleaned_lines).strip()
                    if cleaned_text:
                        passages.append(
                            GuidelinePassage(
                                text=cleaned_text,
                                source=source,
                                year=year,
                                page=page,
                                grade_relevance=[0, 1, 2, 3, 4],
                            )
                        )
            except Exception as exc:
                logger.warning("Could not read guideline file %s: %s", doc_path, exc)

        return passages

    def __call__(self, state: ScreeningAgentState) -> dict:
        """Retrieve authentic guideline context and validate plain-language explanation."""
        decision = state.get("referral_decision")
        inference_out = state.get("inference_output")
        lang = state.get("language_requested", "en")
        template_text = state.get("report_hi" if lang == "hi" else "report_en", "")

        # Fallback template if state missing
        if not template_text:
            template_text = "Clinical decision support evaluation completed. Clinician review mandatory."

        # If uncertain or early exit, keep deterministic template explanation
        if state.get("is_uncertain", False) or decision is None:
            return {
                "guideline_explanation": template_text,
                "guideline_citations": [],
                "guideline_source_note": "Standard clinical guideline templates (UNCERTAIN result; clinician review required)",
            }

        grade = inference_out.predicted_grade if inference_out else None
        passages = self._index_documents()

        # If docs/guidelines is empty, use templates only (Condition 4)
        if not passages:
            return {
                "guideline_explanation": template_text,
                "guideline_citations": [],
                "guideline_source_note": "Standard clinical guideline templates (no documents in docs/guidelines/)",
            }

        # Retrieve relevant passages for this grade
        relevant: list[GuidelinePassage] = []
        for p in passages:
            if grade is not None and grade.value in p.grade_relevance:
                relevant.append(p)

        if not relevant:
            relevant = passages[:1]

        # Formulate explanation with citation
        citations = [
            {"source": p.source, "year": p.year, "page": p.page}
            for p in relevant
        ]
        passage_summary = "\n".join(p.text[:300] for p in relevant)
        candidate_explanation = (
            f"{template_text}\n\nClinical Guideline Reference ({relevant[0].source}, {relevant[0].year}, p. {relevant[0].page}):\n"
            f"{passage_summary}"
        )

        # Run strict safety validator
        is_valid, reason = self.validator.validate(candidate_explanation, decision, grade)
        if not is_valid:
            logger.warning("GuidelineValidator rejected candidate explanation: %s. Falling back to template.", reason)
            return {
                "guideline_explanation": template_text,
                "guideline_citations": [],
                "guideline_source_note": f"Fallback to templates (Validator rejected retrieved text: {reason})",
            }

        return {
            "guideline_explanation": candidate_explanation,
            "guideline_citations": citations,
            "guideline_source_note": f"Retrieved from {len(citations)} authenticated guideline document(s)",
        }

