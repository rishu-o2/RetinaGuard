# RetinaGuard 👁️🛡️

> **Diabetic Retinopathy (DR) Clinical Decision-Support System**  
> *Target Market: India (Diabetologist Clinics, Primary Health Centers [PHCs], Community Screening Camps)*

RetinaGuard is an assistive clinical decision-support platform designed to screen for Diabetic Retinopathy from retinal fundus photographs. It evaluates image quality, detects out-of-distribution inputs, classifies DR according to the **ICDR (0–4)** scale, deterministically calculates referral urgency and follow-up intervals, and generates plain-language reports in the patient's local language.

---

## ⚠️ Core Architectural Principles

1. **Decision Support, Not Diagnosis**: RetinaGuard is an assistive tool for a clinician, never an autonomous diagnostic device.
2. **Separation of Concerns**: The ML model strictly predicts ICDR grade (0–4) and calibrated confidence. Referral decisions, urgency, and follow-up intervals are governed 100% by a deterministic, unit-tested Python rules engine.
3. **Guardrailed Explanations**: Generative LLM/RAG may only synthesize plain-language explanations from authenticated clinical guideline texts. It cannot alter grades, urgency, or numbers.
4. **Fail-Safe Gatekeeping**: Blurry, over/underexposed, or out-of-distribution (non-fundus) images fail-safe to `"UNCERTAIN - needs clinician review"`. The model is never permitted to guess.
5. **Tamper-Evident Ledger**: Every screening transaction is cryptographically hash-chained (SHA-256) for audit integrity.
6. **Privacy & India DPDP Act 2023**: Zero PHI in logs (automatic regex scrubbing of ABHA IDs, Aadhaar, phone numbers, and emails), field-level authenticated AES-256-GCM encryption at rest, and strict India data residency.

---

## 🇮🇳 India Context & Guidelines

- **Clinical Consensus**: Referral thresholds and urgency tiers are directly grounded in:
  - **AIOS** (All India Ophthalmological Society) Clinical Practice Guidelines
  - **RSSDI** (Research Society for the Study of Diabetes in India) Clinical Recommendations
  - **ICMR** (Indian Council of Medical Research) Guidelines for Type 2 Diabetes
  - **ICO** (International Council of Ophthalmology) Diabetic Eye Care Guidelines
- **Hardware Agility**: Pre-calibrated quality profiles for both **tabletop** cameras (Zeiss, Topcon) and **portable/smartphone** fundus cameras (Remidio NM-FOP, Forus 3nethra).
- **ABDM/ABHA Ready**: Native data models support 14-digit ABHA numbers and PHR handles.
- **Multilingual**: English and Hindi first, with architecture prepared for Tamil, Telugu, Bengali, Marathi, and Kannada.

---

## 📁 Monorepo Layout

```text
RetinaGuard/
├── backend/                  # FastAPI service, clinical rules engine, crypto, DB
│   ├── src/retinaguard/      # Python source code
│   ├── tests/                # Pytest unit & integration tests
│   └── pyproject.toml
│
├── ml/                       # PyTorch training, evaluation (QWK, Sensitivity), ONNX export
│   ├── src/retinaguard_ml/   # Model architectures and dataset utilities
│   ├── tests/                # Model shape, export, and metric tests
│   └── pyproject.toml
│
├── frontend/                 # React 18, Vite, Tailwind CSS clinician UI
│   ├── src/                  # Components, screening intake, review queues
│   ├── tests/                # Component & intake smoke tests
│   └── package.json
│
├── configs/                  # Externalized clinical & hardware parameters
│   ├── clinical_rules.yaml   # AIOS/RSSDI thresholds (# NEEDS CLINICAL REVIEW)
│   ├── quality_thresholds.yaml # Camera quality profiles (# NEEDS CLINICAL REVIEW)
│   └── locales/              # Translation dictionaries (en, hi)
│
├── docs/                     # Regulatory, risk analysis, and architecture decisions
│   ├── DECISIONS.md          # Architectural Decision Records (ADRs)
│   └── regulatory/           # CDSCO SaMD traceability matrix & risk analysis
│
├── docker/                   # Dockerfiles for containerized deployment
├── docker-compose.yml        # Orchestration (API, Web, PostgreSQL)
├── Makefile                  # Developer automation targets
├── .env.example              # Environment variables template
└── .pre-commit-config.yaml   # Code quality and security git hooks
```

---

## 🚀 Quickstart & Local Setup

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional, for full containerized stack)

### 1. Environment Configuration
```bash
cp .env.example .env
```

### 2. Backend Setup & Run
```bash
# Install backend dependencies
pip install -e backend

# Start the FastAPI server (runs on http://localhost:8000)
uvicorn retinaguard.api.main:app --reload --port 8000
```
Verify via:
```bash
curl http://localhost:8000/health
```

### 3. ML Package
```bash
pip install -e ml
```

### 4. Frontend Setup & Run
```bash
cd frontend
npm install
npm run dev
# Accessible at http://localhost:3000
```

---

## 🧪 Running Tests

RetinaGuard requires all unit, clinical rules, and integration tests to pass across packages:

```bash
# Run all package tests
make test

# Or run per-package:
make test-backend    # Runs backend tests (pytest)
make test-ml         # Runs ML model & export tests (pytest)
make test-frontend   # Runs frontend smoke tests (node test runner)
```

---

## 📑 Architecture & Regulatory Documentation

- For architectural trade-offs and decisions, see [docs/DECISIONS.md](file:///d:/RetinaGuard/docs/DECISIONS.md).
- For CDSCO SaMD regulatory traceability, see [docs/regulatory/cdsco_traceability.md](file:///d:/RetinaGuard/docs/regulatory/cdsco_traceability.md).
