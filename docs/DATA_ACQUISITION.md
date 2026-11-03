# Dataset Acquisition Guide - Diabetic Retinopathy Datasets

This document provides verified instructions for manually acquiring retinal fundus datasets for training, evaluation, and external validation of RetinaGuard.

> [!IMPORTANT]
> **No Automatic Downloads**: In compliance with clinical research governance, ethics board guidelines, and dataset-specific terms of use, datasets must be downloaded manually by authorized researchers who have accepted each respective license.

---

## 1. Primary Datasets Overview

| Dataset | Origin / Site | Images | Primary Camera / Device Modality | License | Usage in RetinaGuard |
|---|---|---|---|---|---|
| **APTOS 2019** | Aravind Eye Hospital, India | 3,662 | Multi-site clinical cameras in rural/urban Tamil Nadu | Kaggle Competition Terms (Research/Non-Commercial) | Primary Training & Internal Validation |
| **IDRiD** | Nanded, Maharashtra, India | 516 | Kowa VX-10α (50° FOV, Tabletop) | Creative Commons Attribution 4.0 (CC BY 4.0) | **Strict External Test Set** (Held out from training) |
| **EyePACS** | Multi-clinic, USA | ~88,000 | Diverse tabletop cameras (Centervue, Canon, Topcon) | Kaggle Competition Terms | Secondary Pretraining / Scaling |
| **Messidor-2** | France (Brest, Paris, Saint-Étienne) | 1,748 | Topcon TRC NW6 (45° FOV) | Messidor Research License | European External Benchmark |
| **DDR** | Multi-hospital, China | 13,673 | Diverse tabletop cameras | Research Use Only | Auxiliary Benchmark |

---

## 2. Step-by-Step Acquisition Instructions

### A. APTOS 2019 Blindness Detection
1. Visit Kaggle: https://www.kaggle.com/c/aptos2019-blindness-detection/data
2. Log in and accept the competition rules.
3. Download `aptos2019-blindness-detection.zip`.
4. Extract into:
   ```text
   data/raw/aptos2019/
   ├── train_images/
   ├── train.csv
   └── test.csv
   ```

### B. IDRiD (Indian Diabetic Retinopathy Image Dataset)
1. Visit IEEE Dataport: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
2. License: CC BY 4.0 (A. Prasanna Porwal et al., 2018).
3. Download Part B: Disease Grading (`1. Original Images` and `2. Groundtruths`).
4. Extract into:
   ```text
   data/raw/idrid/
   ├── a. Disease Grading/
   │   ├── 1. Original Images/
   │   │   ├── a. Training Set/
   │   │   └── b. Testing Set/
   │   └── 2. Groundtruths/
   │       ├── a. IDRiD_Disease Grading_Training Labels.csv
   │       └── b. IDRiD_Disease Grading_Testing Labels.csv
   ```

---

## 3. Directory Layout Expected by Loaders

```text
data/
├── raw/
│   ├── aptos2019/
│   ├── idrid/
│   ├── eyepacs/
│   └── local_clinical/
│
├── splits/
│   └── split_manifest.json    # Created deterministically by PatientStratifiedSplitter
│
└── processed/
    └── 512x512/               # Preprocessed & contrast-enhanced images
```

