"""APTOS 2019 Blindness Detection dataset loader."""

import csv
from pathlib import Path
from typing import List

from retinaguard_ml.data.registry import DATASET_REGISTRY, ImageRecord


class APTOS2019Loader:
    """Loader for APTOS 2019 Blindness Detection dataset (Aravind Eye Hospital, India)."""

    def __init__(self, dataset_key: str = "aptos2019"):
        self.meta = DATASET_REGISTRY[dataset_key]

    def load_records(self, root_dir: Path | str, csv_filename: str = "train.csv") -> List[ImageRecord]:
        """Parse APTOS train.csv and images directory."""
        root_path = Path(root_dir)
        csv_path = root_path / csv_filename
        images_dir = root_path / "train_images"

        if not csv_path.exists():
            raise FileNotFoundError(f"APTOS CSV not found at {csv_path}")

        records: List[ImageRecord] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id_code = row["id_code"].strip()
                grade = int(row["diagnosis"].strip())

                img_path = images_dir / f"{id_code}.png"
                if not img_path.exists():
                    # Check jpg fallback
                    img_path = images_dir / f"{id_code}.jpg"

                records.append(
                    ImageRecord(
                        image_id=id_code,
                        patient_id=f"APTOS_{id_code}",  # Patient mapping when not explicitly unblinded
                        grade=grade,
                        eye=None,
                        device=self.meta.default_device,
                        site=self.meta.default_site,
                        file_path=str(img_path),
                        dataset_name=self.meta.key,
                        split=None,  # Will be assigned by PatientStratifiedSplitter
                    )
                )

        return records




import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from retinaguard_ml.data.augmentations import FundusClinicalAugmentor
from retinaguard_ml.data.imbalance import compute_class_weights, create_weighted_sampler
from retinaguard_ml.data.preprocessing import ben_graham_enhancement
from retinaguard_ml.data.split import ExternalHoldoutContaminationError

FUNDUS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
FUNDUS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class APTOSDataset(Dataset):
    """Production PyTorch Dataset reading pre-processed cached fundus images and split manifest."""

    def __init__(
        self,
        manifest_path: Path | str,
        cache_dir: Path | str,
        split: str = "train",
        dataset_name: str = "aptos2019",
        augment: bool | None = None,
        use_ben_graham: bool = False,
        image_size: int | tuple[int, int] | list[int] | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.dataset_name = dataset_name
        self.use_ben_graham = use_ben_graham
        if image_size is not None:
            self.image_size: tuple[int, int] | None = (image_size, image_size) if isinstance(image_size, int) else (image_size[0], image_size[1])
        else:
            self.image_size = None

        # External-test guard verification
        d_meta = DATASET_REGISTRY.get(dataset_name)
        if d_meta and d_meta.is_external_test_only:
            if split in ("train", "val"):
                raise ExternalHoldoutContaminationError(
                    f"Dataset '{dataset_name}' is registered as external_test_only and cannot be used in split='{split}'!"
                )

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Read manifest rows for the selected split
        self.records: list[dict] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == self.split:
                    self.records.append({
                        "id_code": row["id_code"],
                        "diagnosis": int(row["diagnosis"]),
                        "group_id": row.get("group_id", row["id_code"]),
                    })

        if not self.records:
            raise ValueError(f"No records found for split='{self.split}' in {self.manifest_path}")

        # Augmentation setup (active for train split by default)
        self.augment = (self.split == "train") if augment is None else augment
        self.augmentor = FundusClinicalAugmentor() if self.augment else None

    def __len__(self) -> int:
        return len(self.records)

    def get_labels(self) -> list[int]:
        """Return list of all diagnostic labels in this split."""
        return [r["diagnosis"] for r in self.records]

    def get_class_weights(self, method: str = "effective_samples") -> torch.Tensor:
        """Calculate class weights based on the empirical label frequencies in this dataset."""
        return compute_class_weights(self.get_labels(), num_classes=5, method=method)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler balancing all ICDR grades equally."""
        return create_weighted_sampler(self.get_labels(), num_classes=5)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        id_code = rec["id_code"]
        label = rec["diagnosis"]

        img_path = self.cache_dir / f"{id_code}.png"
        if not img_path.exists():
            img_path = self.cache_dir / f"{id_code}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"Cached image not found for id_code='{id_code}' at {img_path}")

        with Image.open(img_path) as img:
            arr = np.array(img.convert("RGB"))

        # Optional resize at load time if specified and dimension differs
        if self.image_size is not None:
            th, tw = self.image_size
            if arr.shape[0] != th or arr.shape[1] != tw:
                arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LINEAR)

        # Optional Ben Graham contrast enhancement
        if self.use_ben_graham:
            arr = ben_graham_enhancement(arr)

        # Clinical augmentations (train only)
        if self.augmentor:
            arr = self.augmentor(arr)

        # Normalize and convert to float32 tensor [3, H, W]
        arr_norm = arr.astype(np.float32) / 255.0
        arr_norm = (arr_norm - FUNDUS_MEAN) / FUNDUS_STD
        tensor = torch.from_numpy(arr_norm.transpose(2, 0, 1)).float()

        return tensor, label
