"""IDRiD (Indian Diabetic Retinopathy Image Dataset) loader.

Enforces ADR-008: IDRiD records are strictly locked as external_test and
kept out of model training to provide an Indian ground-truth evaluation benchmark.
"""

import csv
from pathlib import Path
from typing import List

from retinaguard_ml.data.registry import DATASET_REGISTRY, ImageRecord


class IDRiDLoader:
    """Loader for IDRiD (Dr. Ramanath Eye Hospital, Nanded, Maharashtra, India)."""

    def __init__(self, dataset_key: str = "idrid"):
        self.meta = DATASET_REGISTRY[dataset_key]

    def load_records(
        self,
        root_dir: Path | str,
        csv_relative_path: str = "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
        images_subfolder: str = "1. Original Images/a. Training Set",
    ) -> List[ImageRecord]:
        """Parse IDRiD labels CSV and images directory."""
        root_path = Path(root_dir)
        csv_path = root_path / csv_relative_path
        images_dir = root_path / images_subfolder

        if not csv_path.exists():
            # Check direct fallback
            if (root_path / "labels.csv").exists():
                csv_path = root_path / "labels.csv"
                images_dir = root_path / "images"
            else:
                raise FileNotFoundError(f"IDRiD labels CSV not found at {csv_path}")

        records: List[ImageRecord] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Column name variants in IDRiD
                img_name = row.get("Image name") or row.get("image_id") or row.get("Image_name", "")
                img_name = img_name.strip()

                grade_raw = row.get("Retinopathy grade") or row.get("diagnosis") or row.get("grade", "0")
                grade = int(grade_raw.strip())

                img_path = images_dir / f"{img_name}.jpg"
                if not img_path.exists():
                    img_path = images_dir / f"{img_name}.tif"

                records.append(
                    ImageRecord(
                        image_id=img_name,
                        patient_id=f"IDRID_{img_name}",
                        grade=grade,
                        eye=None,
                        device=self.meta.default_device,
                        site=self.meta.default_site,
                        file_path=str(img_path),
                        dataset_name=self.meta.key,
                        split="external_test",  # STRICTLY LOCKED AS EXTERNAL TEST SET
                    )
                )

        return records

