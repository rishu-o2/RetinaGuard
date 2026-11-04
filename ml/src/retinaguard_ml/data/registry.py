"""Centralized Dataset Registry and Metadata Management.

Enforces licensing compliance, camera hardware tracking, download provenance,
and clinical site documentation.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol


class CameraDeviceCategory(str, Enum):
    """Categorization of fundus imaging hardware."""
    TABLETOP = "TABLETOP"
    PORTABLE_SMARTPHONE = "PORTABLE_SMARTPHONE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DatasetMeta:
    """Metadata describing a retinal fundus dataset."""
    key: str
    name: str
    source_url: str
    download_method: str  # "kagglehub", "manual", "api"
    local_path: str
    license_note: str
    used_for: str  # "training", "external_test_only", "pretraining"
    license: str
    citation: str
    default_device: str
    device_category: CameraDeviceCategory
    default_site: str
    country: str
    is_external_test_only: bool
    description: str


@dataclass
class ImageRecord:
    """Standardized record describing a single retinal fundus image in the pipeline."""
    image_id: str
    patient_id: str
    grade: int  # ICDR 0-4
    eye: str | None = None  # 'OD', 'OS', or None
    device: str = "UNKNOWN"
    site: str = "UNKNOWN"
    file_path: str = ""
    dataset_name: str = ""
    split: str | None = None  # 'train', 'val', 'test', 'external_test'
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0 <= self.grade <= 4):
            raise ValueError(f"Invalid ICDR grade {self.grade}; must be in range 0-4.")


class BaseFundusLoader(Protocol):
    """Protocol for plug-in dataset loaders."""

    def load_records(self, root_dir: Path | str) -> list[ImageRecord]:
        """Parse raw directory and return standardized ImageRecord objects."""
        ...


# Central Registry of Datasets
DATASET_REGISTRY: dict[str, DatasetMeta] = {
    "aptos2019": DatasetMeta(
        key="aptos2019",
        name="APTOS 2019 Blindness Detection",
        source_url="https://www.kaggle.com/c/aptos2019-blindness-detection/data",
        download_method="kagglehub",
        local_path="data/raw/aptos",
        license_note="Kaggle competition data: check the competition rules for commercial use; treat as research/prototype only until verified",
        used_for="training",
        license="Kaggle Competition Rules (Non-Commercial / Research Use Only)",
        citation="Karthik, K. et al. APTOS 2019 Blindness Detection, Kaggle (2019)",
        default_device="Multi-Site Clinical Tabletop & Field Cameras",
        device_category=CameraDeviceCategory.TABLETOP,
        default_site="Aravind Eye Hospital, Tamil Nadu, India",
        country="IN",
        is_external_test_only=False,
        description="Fundus photos collected from patients in rural and urban clinics across Tamil Nadu, India.",
    ),
    "idrid": DatasetMeta(
        key="idrid",
        name="Indian Diabetic Retinopathy Image Dataset (IDRiD)",
        source_url="https://idrid.grand-challenge.org/",
        download_method="manual",
        local_path="data/raw/idrid",
        license_note="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        used_for="external_test_only",
        license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        citation="Porwal, P. et al. Indian Diabetic Retinopathy Image Dataset (IDRiD), IEEE Dataport (2018)",
        default_device="Kowa VX-10α (50° Field of View)",
        device_category=CameraDeviceCategory.TABLETOP,
        default_site="Dr. Ramanath Eye Hospital, Nanded, Maharashtra, India",
        country="IN",
        is_external_test_only=True,
        description="Authoritative Indian clinical benchmark dataset used exclusively as an external test set.",
    ),
    "eyepacs": DatasetMeta(
        key="eyepacs",
        name="EyePACS Diabetic Retinopathy",
        source_url="https://www.kaggle.com/c/diabetic-retinopathy-detection",
        download_method="manual",
        local_path="data/raw/eyepacs",
        license_note="Kaggle Competition License: research / academic evaluation only",
        used_for="pretraining",
        license="Kaggle Competition License",
        citation="Cuadros, J. & Bresnick, G. Eyepacs: an adaptable telemedicine system for diabetic retinopathy (2009)",
        default_device="Diverse Tabletop (Canon CR-DGi, Centervue DRS, Topcon NW)",
        device_category=CameraDeviceCategory.TABLETOP,
        default_site="Multi-clinic Telemedicine Network, USA",
        country="US",
        is_external_test_only=False,
        description="Large-scale heterogeneous fundus dataset with multi-vendor cameras.",
    ),
    "messidor2": DatasetMeta(
        key="messidor2",
        name="Messidor-2",
        source_url="https://www.adcis.net/en/third-party/messidor2/",
        download_method="manual",
        local_path="data/raw/messidor2",
        license_note="Messidor Research Agreement: non-commercial academic research",
        used_for="external_test_only",
        license="Messidor Research Agreement",
        citation="Decencière, E. et al. Feedback on a publicly distributed image database: the Messidor database (2014)",
        default_device="Topcon TRC NW6 (45° Field of View)",
        device_category=CameraDeviceCategory.TABLETOP,
        default_site="Multi-hospital consortium, France",
        country="FR",
        is_external_test_only=True,
        description="Standardized French benchmark dataset with paired eye examinations.",
    ),
    "ddr": DatasetMeta(
        key="ddr",
        name="DDR (Dataset for Diabetic Retinopathy)",
        source_url="https://github.com/nkicsl/DDR-dataset",
        download_method="manual",
        local_path="data/raw/ddr",
        license_note="Academic / Research Use Only",
        used_for="pretraining",
        license="Academic / Research Use Only",
        citation="Li, T. et al. Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening (2019)",
        default_device="Tabletop Fundus Cameras",
        device_category=CameraDeviceCategory.TABLETOP,
        default_site="Multi-hospital network, China",
        country="CN",
        is_external_test_only=False,
        description="Comprehensive dataset with image-level grades and lesion annotations.",
    ),
}


def get_dataset_metadata(key: str) -> DatasetMeta:
    """Retrieve metadata, license, and download info for a registered dataset."""
    key_clean = key.lower().strip()
    if key_clean not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{key}' not in registry. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[key_clean]
