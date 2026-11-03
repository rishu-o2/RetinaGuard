# RetinaGuard Dataset Download & Setup Guide

This guide describes how to configure Kaggle credentials, download the APTOS 2019 dataset reproducibly, and manually acquire the IDRiD external validation dataset.

---

## 1. APTOS 2019 Dataset Download (Automated via KaggleHub)

The APTOS 2019 Blindness Detection dataset contains 3,662 retinal fundus images from patients screened in rural and urban clinics across Tamil Nadu, India (Aravind Eye Hospital).

### Step 1: Accept Competition Rules in Browser (Mandatory)
Before the Kaggle API can download the data, your Kaggle account must have accepted the competition terms:
1. Navigate to: **[https://www.kaggle.com/c/aptos2019-blindness-detection/rules](https://www.kaggle.com/c/aptos2019-blindness-detection/rules)**
2. Log in with your Kaggle account.
3. Review the rules and click **"I Understand and Accept"**.

> [!WARNING]
> If you omit this step, the download script will halt with an **HTTP 403 Forbidden** error.

### Step 2: Configure Kaggle API Credentials
You can provide your Kaggle credentials via a credentials file or environment variables. **Never commit credentials to version control.**

#### Option A: Kaggle JSON File (Recommended)
1. Go to your Kaggle profile settings: `https://www.kaggle.com/settings`.
2. Scroll to the **API** section and click **"Create New Token"**.
3. A file named `kaggle.json` will be downloaded to your computer containing:
   ```json
   {"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_API_KEY"}
   ```
4. Move this file to your home directory:
   - **Linux / macOS**: `~/.kaggle/kaggle.json` (ensure permissions: `chmod 600 ~/.kaggle/kaggle.json`)
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

#### Option B: Environment Variables
Alternatively, export the credentials directly in your terminal:
```bash
# Linux / macOS / Git Bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Windows PowerShell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

### Step 3: Run the Download and Verification Pipeline
Execute the automated target:
```bash
make data-aptos
```
Or run the scripts directly:
```bash
python ml/scripts/download_aptos.py
python ml/scripts/verify_aptos.py
```

#### What `download_aptos.py` does:
- Checks for minimum **10 GB** available disk space.
- Sets `KAGGLEHUB_CACHE` to `data/raw/_kagglehub_cache/` so cache files stay inside the repository.
- Extracts `train.csv` and `train_images/` into `data/raw/aptos/`.
- **Excludes `test_images/` and `test.csv`**: Competition test set labels are held private by Kaggle. We retain only labeled data.
- **Idempotent**: Re-running will not re-download or re-copy if the dataset already exists. Use `--force` to overwrite.

---

## 2. IDRiD Dataset (Manual Download - External Test Set)

The **Indian Diabetic Retinopathy Image Dataset (IDRiD)** was acquired at Dr. Ramanath Eye Hospital in Nanded, Maharashtra, India using a 50° FOV Kowa VX-10α fundus camera. In RetinaGuard, it is strictly maintained as an **external test set** kept completely out of training and validation (ADR-008 & ADR-015).

### Download Steps:
1. Visit the IEEE DataPort portal: **[https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)**
   *(Or the challenge portal: [https://idrid.grand-challenge.org/](https://idrid.grand-challenge.org/))*
2. Register and accept the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license terms.
3. Download the archive **`B. Disease Grading.zip`** (Part B: Disease Grading, ~516 high-resolution 4288x2848 images).
4. Extract the ZIP into `data/raw/idrid_download/`. The expected folder layout is:
   ```text
   data/raw/idrid_download/
   └── B. Disease Grading/
       ├── CC-BY-4.0.txt
       ├── LICENSE.txt
       ├── 1. Original Images/
       │   ├── a. Training Set/
       │   │   └── IDRiD_001.jpg ... IDRiD_413.jpg (413 images)
       │   └── b. Testing Set/
       │       └── IDRiD_001.jpg ... IDRiD_103.jpg (103 images)
       └── 2. Groundtruths/
           ├── a. IDRiD_Disease Grading_Training Labels.csv
           └── b. IDRiD_Disease Grading_Testing Labels.csv
   ```
5. Run the ingestion and verification target:
   ```bash
   make data-idrid
   ```
   Or execute directly:
   ```bash
   python ml/scripts/prepare_idrid.py
   python ml/scripts/verify_idrid.py
   ```
   This generates `data/processed/idrid_manifest.csv` with globally unique image identifiers (`IDRiD_train_001` and `IDRiD_test_001`), verified disk paths, and groundtruth DR/DME labels.

---

## 3. Dataset Verification Tool

To check the integrity of your downloaded dataset at any time, run:
```bash
make data-verify
```
Or:
```bash
python ml/scripts/verify_aptos.py --data-dir data/raw/aptos --sample-size 50
```

The verification tool checks:
- Column names (`id_code`, `diagnosis`).
- 1:1 parity between image files on disk and CSV entries.
- Diagnostic labels bounded strictly in $\{0, 1, 2, 3, 4\}$.
- PIL structural decodability across random sample images.
- Empirical class distribution report.

