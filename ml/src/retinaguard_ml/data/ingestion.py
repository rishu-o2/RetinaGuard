"""Clinical Ingestion Pipeline for Locally Collected Indian Validation Sets.

Handles:
- Multiple ophthalmologist grader IDs and annotations.
- Adjudication logic (consensus or senior vitreoretinal surgeon override).
- Inter-rater reliability / agreement analysis.
- Standardized manifest generation tagged with device and screening site.
"""

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from retinaguard_ml.data.registry import ImageRecord


@dataclass
class MultiGraderAnnotation:
    """Record of individual clinician grading."""
    grader_id: str
    experience_level: str  # e.g., 'RESIDENT', 'FELLOW', 'VITREORETINAL_CONSULTANT'
    grade: int  # ICDR 0-4
    notes: Optional[str] = None


@dataclass
class ClinicalIntakeRecord:
    """Intake record for a local prospective screening study."""
    screening_id: str
    patient_id: str
    eye: str  # 'OD' or 'OS'
    device: str  # e.g., 'Remidio NM-FOP', 'Forus 3nethra', 'Zeiss Clarus'
    site: str  # e.g., 'PHC_Koppal', 'Camp_Madurai', 'AIIMS_Delhi'
    file_path: str
    grader_annotations: List[MultiGraderAnnotation] = field(default_factory=list)
    senior_adjudicator_grade: Optional[int] = None
    adjudicated_grade: Optional[int] = None
    adjudication_method: Optional[str] = None


class LocalClinicalDatasetIngestor:
    """Ingests multi-grader local validation datasets and standardizes them for evaluation."""

    @staticmethod
    def adjudicate(record: ClinicalIntakeRecord) -> int:
        """Determine adjudicated ground-truth DR grade.

        Priority:
        1. Explicit senior vitreoretinal surgeon override.
        2. Majority consensus across graders.
        3. Most conservative (highest grade) in the event of a tie.
        """
        if record.senior_adjudicator_grade is not None:
            record.adjudicated_grade = record.senior_adjudicator_grade
            record.adjudication_method = "SENIOR_SPECIALIST_OVERRIDE"
            return record.adjudicated_grade

        if not record.grader_annotations:
            raise ValueError(f"Screening {record.screening_id} has no grader annotations.")

        grades = [g.grade for g in record.grader_annotations]
        counts = Counter(grades)
        most_common = counts.most_common()

        # Check for tie
        top_count = most_common[0][1]
        tied_grades = [grade for grade, cnt in most_common if cnt == top_count]

        if len(tied_grades) == 1:
            record.adjudicated_grade = tied_grades[0]
            record.adjudication_method = "MAJORITY_VOTE"
        else:
            # Conservative clinical tie-breaker: select the higher grade for patient safety
            record.adjudicated_grade = max(tied_grades)
            record.adjudication_method = "CONSERVATIVE_SAFETY_TIE_BREAKER"

        return record.adjudicated_grade

    @classmethod
    def ingest_records(
        cls, intake_records: List[ClinicalIntakeRecord], target_split: str = "external_test"
    ) -> List[ImageRecord]:
        """Process intake records into standardized pipeline ImageRecord objects."""
        standardized: List[ImageRecord] = []
        for intake in intake_records:
            final_grade = cls.adjudicate(intake)
            standardized.append(
                ImageRecord(
                    image_id=intake.screening_id,
                    patient_id=intake.patient_id,
                    grade=final_grade,
                    eye=intake.eye,
                    device=intake.device,
                    site=intake.site,
                    file_path=intake.file_path,
                    dataset_name="local_clinical_validation",
                    split=target_split,
                    extra_metadata={
                        "adjudication_method": intake.adjudication_method,
                        "num_graders": len(intake.grader_annotations),
                    },
                )
            )
        return standardized

    @staticmethod
    def compute_pairwise_agreement(intake_records: List[ClinicalIntakeRecord]) -> dict[str, float]:
        """Compute inter-grader percent agreement across multiple evaluations."""
        agreements = []
        for r in intake_records:
            if len(r.grader_annotations) >= 2:
                grades = [g.grade for g in r.grader_annotations]
                # Check if all graders agree
                agreements.append(1.0 if len(set(grades)) == 1 else 0.0)

        rate = float(np.mean(agreements)) if agreements else 1.0
        return {
            "num_evaluated_cases": len(agreements),
            "complete_agreement_rate": rate,
        }

