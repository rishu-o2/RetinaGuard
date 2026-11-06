# Running RetinaGuard Demo

> **⚠️ PROTOTYPE NOTICE: NOT CLINICALLY VALIDATED**
> RetinaGuard is an investigational clinical decision-support tool designed to assist licensed clinicians in diabetic retinopathy triage. It does NOT provide autonomous medical diagnoses. All patient outcomes require licensed ophthalmologist review and sign-off.

---

## 1. Prerequisites

- **Python**: 3.11+
- **Node.js**: 20+ (with `npm`)
- **Git**
- Optional: **Docker & Docker Compose** (for containerized deployment)

---

## 2. Environment Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Generate a secure `JWT_SECRET_KEY` and set a strong `ADMIN_PASSWORD`:
   ```bash
   # Generate random 32-byte secret key:
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. Open `.env` and set:
   ```env
   # Mandatory security credentials (never commit .env to version control)
   JWT_SECRET_KEY=<generated-secure-token>
   ADMIN_EMAIL=admin@hospital.in
   ADMIN_PASSWORD=<your-strong-password>

   # Local database (SQLite default for local development)
   DATABASE_URL=sqlite+aiosqlite:///./retinaguard.db

   # Local screening access: false requires Bearer JWT token; true allows guest screener button
   ALLOW_ANONYMOUS_SCREENING=false

   # Voice synthesis (optional — falls back to browser Web Speech API if empty)
   SARVAM_API_KEY=
   ```

---

## 3. Place Model Artifacts

RetinaGuard uses an ONNX runtime engine and Mahalanobis feature covariance statistics for out-of-distribution (OOD) detection. These files are excluded from Git to prevent large binary tracking.

Verify or place the model files:
```
models/
├── registry.json             # Tracked in Git
├── retinaguard_dr_v1.onnx    # ONNX export (input: 256x256 RGB)
└── ood_stats_v1.npz          # Mean vector & precision matrix for OOD
```

If you trained or exported models locally:
```bash
# Ensure ONNX model is in place:
cp ml/exports/retinaguard_dr_v1.onnx models/retinaguard_dr_v1.onnx
```

---

## 4. Start the Application

### Option A: Local Development (Recommended for quick testing)

**Terminal 1 — Backend (FastAPI):**
```bash
# Install backend dependencies in your active Python environment
pip install -e "./backend[dev]"

# Start backend server on port 8000
uvicorn retinaguard.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Wait until you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```
*(On first startup, database tables are automatically initialized and the initial admin user is seeded).*

**Terminal 2 — Frontend (React + Vite):**
```bash
cd frontend
npm ci
npm run dev
```
Open your browser at **http://localhost:5173**.

---

### Option B: Docker Compose

```bash
# Build and run the entire stack (PostgreSQL + FastAPI + Vite Frontend)
docker compose up --build -d

# Check health
docker compose ps
curl http://localhost:8000/health
```
Access the application at **http://localhost:8000** (or port configured in compose).

---

## 5. Walkthrough Demo Script

Follow these steps to experience the complete clinical workflow:

### Step 1: Sign In
1. Navigate to `http://localhost:5173`.
2. Sign in using the Admin credentials configured in your `.env`:
   - **Email**: `admin@hospital.in`
   - **Password**: *(The password you set in `.env`)*
3. You will be logged in with `ADMIN` privileges and directed to the navigation bar.

### Step 2: Screen a Good Fundus Image (Passes Quality Gate)
1. Click the **New Screening** tab.
2. Select Camera Hardware Modality (e.g., `Tabletop Fundus Camera`).
3. Laterality: `OD (Right Eye)`.
4. Enter Patient Pseudonym ID: `PAT-VAL-001`.
5. Check the **DPDP Act 2023 Consent** checkbox.
6. Drop or upload a valid fundus image (e.g., from `data/raw/aptos/train_images/00cc2b75cddd.png` or `071435a218ec.png`).
7. Click **Run Screening**.
8. **Observe Results**:
   - The quality gate checks pass (Laplacian blur, brightness, FOV, 256px resolution).
   - The ONNX model infers the ICDR grade (e.g., Grade 0 or Grade 2) and confidence.
   - Clinical rules engine assigns referral (`NON_REFERABLE` or `REFERABLE`) and urgency (`ROUTINE`, `PRIORITY`, or `URGENT`).
   - An explanation with guideline citations (AIOS 2020 / RSSDI 2022) is generated.

### Step 3: Screen a Defective Image (UNCERTAIN Routing)
1. While still on the **New Screening** tab, upload a non-fundus image, a solid color square, or an extremely low-resolution / blurry photo.
2. Ensure consent is checked and click **Run Screening**.
3. **Observe UNCERTAIN Card**:
   - Distinct orange warning styling.
   - Shows rejection reason (e.g., failed blur check or FOV circle fit).
   - Prominently advises: **"Please recapture the fundus image — retake the photo if possible. A clinician will review this case. Do not act on this result."**
   - Automatically persisted to the clinician review queue flagged **"needs recapture or review"**.

### Step 4: Clinician Review Queue & Confirmation
1. Click the **Review Queue** tab in the top navigation.
2. View pending cases: both confident diagnoses and UNCERTAIN screenings are present.
3. Click **Review Case →** on any pending case:
   - Review findings, predicted grade, urgency, and clinical notes.
   - Click **✓ Confirm AI Assessment** to approve the screening.
   - Or click **✗ Override Assessment**, select the revised ICDR grade (0–4), provide a clinical note, and submit.
4. The audit ledger appends a tamper-evident hash chain entry recording the clinician's decision.

### Step 5: Bilingual Referral Report (PDF)
1. On the case review screen (or result card), click **Download PDF (EN)** or **Download PDF (HI)**.
2. Open the downloaded PDF:
   - Displays patient pseudonym, camera modality, ICDR grade, and referral timeframe.
   - Embeds the prototype warning banner.
   - In Hindi mode (`lang=hi`), renders native Unicode Devanagari characters via embedded Noto Sans Devanagari.

### Step 6: Audio Speech Output
1. On the screening result card, click **🔊 Play Audio**.
2. If `SARVAM_API_KEY` is configured in `.env`, streaming audio is returned from the provider.
3. If no API key is provided, the browser automatically falls back to client-side Web Speech synthesis (`SpeechSynthesisUtterance`).

---

## 6. Running Test Suites

Run all automated verification tests locally:

```bash
# 1. Backend tests (FastAPI, auth, PDF, audit ledger, clinical rules)
python -m pytest backend/tests/ -v

# 2. End-to-end smoke test (3 real images, PDF generation, audit check)
python -m pytest backend/tests/test_e2e_smoke.py -v

# 3. Machine Learning pipeline tests (export, evaluation, OOD detection)
python -m pytest ml/tests/ -v

# 4. Frontend DOM component tests
cd frontend
npm test
```

---

## 7. Reading GitHub Actions CI Results

RetinaGuard uses GitHub Actions (`.github/workflows/ci.yml`) to enforce clinical and code quality standards on every commit:

1. In GitHub, navigate to the **Actions** tab of the repository.
2. Click on the latest workflow run named **RetinaGuard CI**.
3. You will see 6 dedicated jobs:
   - **`Lint & Style Checks`**: Verifies backend and ML code formatting using `ruff check` and `ruff format`.
   - **`Backend Test Suite`**: Runs all backend unit, integration, and security tests under Python 3.11.
   - **`ML Test Suite`**: Validates PyTorch/ONNX inference, quality metrics, and rules engine under Python 3.11.
   - **`Frontend Test Suite`**: Runs Node.js 20 DOM tests (`npm test`) covering ResultCard, UncertainCard, LoginPage, and ReviewDetail.
   - **`Dependency Security Scan`**: Audits Python packages (`pip-audit`) and npm dependencies (`npm audit`) for known CVEs.
   - **`Docker Build & Health Check`**: On pushes to `main`, builds the container stack, launches services with `docker compose up -d`, and validates HTTP 200 on `GET /health`.
4. Green checkmarks (✔) indicate all quality gates passed. If a job fails, click on the job to inspect the terminal logs for the failing assertion.
