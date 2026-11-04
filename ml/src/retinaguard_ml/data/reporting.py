"""Dataset reporting and visualization script.

Generates:
- Class count distributions per split (train/val/test/external_test)
- Image resolution and aspect ratio statistics
- Device and clinical screening site distributions
- Summary markdown report and visual artifacts in docs/data_report/
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List
import numpy as np

from retinaguard_ml.data.registry import ImageRecord


class DataReporter:
    """Generates markdown reports and summaries of dataset splits."""

    @classmethod
    def generate_report(
        cls, records: List[ImageRecord], output_dir: Path | str = "docs/data_report"
    ) -> Path:
        """Analyze records and write dataset_summary.md."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "dataset_summary.md"

        total_images = len(records)
        unique_patients = len(set(r.patient_id for r in records))

        # Split distribution
        split_counts = Counter(r.split or "unassigned" for r in records)

        # Grade distribution overall and per split
        overall_grades = Counter(r.grade for r in records)
        split_to_grades: Dict[str, Counter] = {}
        for r in records:
            s = r.split or "unassigned"
            if s not in split_to_grades:
                split_to_grades[s] = Counter()
            split_to_grades[s][r.grade] += 1

        # Device distribution
        device_counts = Counter(r.device for r in records)

        # Site distribution
        site_counts = Counter(r.site for r in records)

        # Build markdown text
        lines = [
            "# RetinaGuard Dataset & Cohort Summary Report",
            "",
            f"**Total Fundus Images**: {total_images}  ",
            f"**Unique Patients**: {unique_patients}  ",
            "",
            "## 1. Split Distribution",
            "| Split | Image Count | Percentage |",
            "|---|---|---|",
        ]
        for s, count in sorted(split_counts.items()):
            pct = (count / total_images) * 100 if total_images > 0 else 0
            lines.append(f"| `{s}` | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 2. ICDR Grade Distribution Across Splits",
            "| ICDR Grade | Name | Total | " + " | ".join(f"`{s}`" for s in sorted(split_counts.keys())) + " |",
            "|---|---|---|" + "|".join("---" for _ in split_counts) + "|",
        ])

        grade_names = {
            0: "No DR",
            1: "Mild NPDR",
            2: "Moderate NPDR",
            3: "Severe NPDR",
            4: "Proliferative DR",
        }

        for g in range(5):
            g_total = overall_grades.get(g, 0)
            split_cols = [str(split_to_grades.get(s, Counter()).get(g, 0)) for s in sorted(split_counts.keys())]
            lines.append(f"| **{g}** | {grade_names[g]} | {g_total} | " + " | ".join(split_cols) + " |")

        lines.extend([
            "",
            "## 3. Imaging Device Breakdown (Tabletop vs. Handheld/Smartphone)",
            "| Camera Hardware / Model | Count | Percentage |",
            "|---|---|---|",
        ])
        for dev, cnt in device_counts.most_common():
            pct = (cnt / total_images) * 100 if total_images > 0 else 0
            lines.append(f"| {dev} | {cnt} | {pct:.1f}% |")

        lines.extend([
            "",
            "## 4. Clinical Screening Sites (India & Benchmark)",
            "| Clinical Site / Network | Count | Percentage |",
            "|---|---|---|",
        ])
        for site, cnt in site_counts.most_common():
            pct = (cnt / total_images) * 100 if total_images > 0 else 0
            lines.append(f"| {site} | {cnt} | {pct:.1f}% |")

        report_content = "\n".join(lines) + "\n"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_path

