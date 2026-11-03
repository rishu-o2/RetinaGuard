# CDSCO SaMD Traceability Matrix (Software as a Medical Device)

| Requirement ID | Clinical / Regulatory Requirement | Module Implementation | Verification Test | Risk Hazard Ref |
|---|---|---|---|---|
| **REQ-CLIN-001** | Support ICDR Grades 0 to 4 classification with AIOS/RSSDI guideline citations. | `configs/clinical_rules.yaml`, `src/retinaguard/core/models.py` | `tests/test_config.py::test_load_clinical_rules` | HAZ-DIAG-001 |
| **REQ-CLIN-002** | Mark all clinical numbers with explicit `# NEEDS CLINICAL REVIEW` and source reference. | `configs/clinical_rules.yaml` | `tests/test_config.py::test_load_clinical_rules` | HAZ-PROC-001 |
| **REQ-CAM-001** | Differentiated quality thresholds for tabletop vs portable/smartphone fundus cameras. | `configs/quality_thresholds.yaml`, `src/retinaguard/core/config.py` | `tests/test_config.py::test_load_quality_thresholds` | HAZ-QUAL-001 |
| **REQ-PRIV-001** | DPDP Act 2023 compliant explicit, purpose-limited consent artefact with revocation. | `src/retinaguard/core/consent.py` | `tests/test_core_models.py::test_dpdp_consent_revocation` | HAZ-PRIV-001 |
| **REQ-PRIV-002** | Zero patient identifiers in logs (redact phone, email, Aadhaar, ABHA ID). | `src/retinaguard/core/logging.py` | `tests/test_phi_protection.py::test_logger_phi_scrubbing` | HAZ-PRIV-002 |
| **REQ-PRIV-003** | Field-level authenticated encryption (AES-256-GCM) for PHI at rest. | `src/retinaguard/core/encryption.py` | `tests/test_phi_protection.py::test_phi_encryptor_round_trip` | HAZ-PRIV-003 |
| **REQ-ABDM-001** | Forward compatibility with ABDM ABHA 14-digit format and PHR handle validation. | `src/retinaguard/core/models.py` | `tests/test_core_models.py::test_patient_context_valid` | HAZ-INTEG-001 |

