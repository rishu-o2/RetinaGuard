from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

from retinaguard_ml.data.registry import DATASET_REGISTRY, ImageRecord


class PatientLeakageError(Exception):
    pass


class ExternalHoldoutContaminationError(Exception):
    """Raised when an external-test-only dataset is assigned to training or validation."""
    pass


class PatientStratifiedSplitter:

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self, records: List[ImageRecord]
    ) -> Dict[str, List[ImageRecord]]:
        """Split records into train, val, test, and external_test."""
        rng = np.random.default_rng(self.seed)

        # 1. Separate external holdout records (e.g. IDRiD)
        training_eligible: List[ImageRecord] = []
        external_test: List[ImageRecord] = []

        for r in records:
            d_meta = DATASET_REGISTRY.get(r.dataset_name)
            if d_meta and d_meta.is_external_test_only:
                if r.split in ("train", "val"):
                    raise ExternalHoldoutContaminationError(
                        f"Dataset '{r.dataset_name}' is registered as external_test_only, "
                        f"but record '{r.image_id}' was assigned split='{r.split}'."
                    )
                r.split = "external_test"
                external_test.append(r)
            elif r.split == "external_test":
                external_test.append(r)
            else:
                training_eligible.append(r)

        # 2. Group eligible records by patient_id
        patient_to_records: Dict[str, List[ImageRecord]] = defaultdict(list)
        patient_to_max_grade: Dict[str, int] = {}

        for r in training_eligible:
            patient_to_records[r.patient_id].append(r)
            # Patient stratification key = maximum DR grade across eyes/visits
            current_max = patient_to_max_grade.get(r.patient_id, -1)
            patient_to_max_grade[r.patient_id] = max(current_max, r.grade)

        # 3. Stratify patients by grade
        grade_to_patients: Dict[int, List[str]] = defaultdict(list)
        for pid, grade in patient_to_max_grade.items():
            grade_to_patients[grade].append(pid)

        train_patients: Set[str] = set()
        val_patients: Set[str] = set()
        test_patients: Set[str] = set()

        for grade, pids in sorted(grade_to_patients.items()):
            pids_arr = np.array(pids)
            rng.shuffle(pids_arr)

            n_total = len(pids_arr)
            n_train = int(np.round(n_total * self.train_ratio))
            n_val = int(np.round(n_total * self.val_ratio))

            # Ensure at least 1 sample in train if available
            if n_total > 0 and n_train == 0:
                n_train = 1

            train_p = pids_arr[:n_train]
            val_p = pids_arr[n_train : n_train + n_val]
            test_p = pids_arr[n_train + n_val :]

            train_patients.update(train_p)
            val_patients.update(val_p)
            test_patients.update(test_p)

        # 4. Map patients back to image records and tag split
        splits: Dict[str, List[ImageRecord]] = {
            "train": [],
            "val": [],
            "test": [],
            "external_test": external_test,
        }

        for pid, patient_records in patient_to_records.items():
            if pid in train_patients:
                target_split = "train"
            elif pid in val_patients:
                target_split = "val"
            else:
                target_split = "test"

            for rec in patient_records:
                rec.split = target_split
                splits[target_split].append(rec)

        # 5. Enforce mathematical zero-leakage check
        self.assert_no_leakage(splits)

        return splits

    @staticmethod
    def assert_no_leakage(splits: Dict[str, List[ImageRecord]]) -> None:
        """Verify that patient sets are strictly disjoint across train, val, and test."""
        # 1. External holdout contamination check
        for s in ("train", "val"):
            for r in splits.get(s, []):
                d_meta = DATASET_REGISTRY.get(r.dataset_name)
                if (d_meta and d_meta.is_external_test_only) or r.split == "external_test":
                    raise ExternalHoldoutContaminationError(
                        f"External holdout contamination! Record '{r.image_id}' from dataset '{r.dataset_name}' found in {s} split!"
                    )

        train_p = {r.patient_id for r in splits.get("train", [])}
        val_p = {r.patient_id for r in splits.get("val", [])}
        test_p = {r.patient_id for r in splits.get("test", [])}

        train_val_overlap = train_p & val_p
        if train_val_overlap:
            raise PatientLeakageError(
                f"Data leakage detected! {len(train_val_overlap)} patients found in both train and val: {list(train_val_overlap)[:5]}"
            )

        train_test_overlap = train_p & test_p
        if train_test_overlap:
            raise PatientLeakageError(
                f"Data leakage detected! {len(train_test_overlap)} patients found in both train and test: {list(train_test_overlap)[:5]}"
            )

        val_test_overlap = val_p & test_p
        if val_test_overlap:
            raise PatientLeakageError(
                f"Data leakage detected! {len(val_test_overlap)} patients found in both val and test: {list(val_test_overlap)[:5]}"
            )

    @staticmethod
    def save_manifest(splits: Dict[str, List[ImageRecord]], output_path: Path | str) -> None:
        """Serialize split manifest to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialized: Dict[str, List[dict]] = {}
        for split_name, recs in splits.items():
            serialized[split_name] = [
                {
                    "image_id": r.image_id,
                    "patient_id": r.patient_id,
                    "grade": r.grade,
                    "eye": r.eye,
                    "device": r.device,
                    "site": r.site,
                    "dataset_name": r.dataset_name,
                    "file_path": r.file_path,
                    "split": r.split,
                }
                for r in recs
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)




class ContentGroupSplitter(PatientStratifiedSplitter):

    
    pass
