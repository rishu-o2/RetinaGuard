# RetinaGuard — DPDP Act 2023 Compliance Mapping

> [!CAUTION]
> **ENGINEERING PREPARATION ONLY — NOT LEGAL ADVICE.**
> This document maps the Digital Personal Data Protection Act 2023 (DPDP Act, Ministry of Electronics and Information Technology, India) obligations to what the RetinaGuard codebase does today and what is missing. It was prepared by the engineering team to structure a future review by qualified legal counsel and a certified Data Protection Officer. Every item is marked **NEEDS EXPERT REVIEW**. Do not treat this document as evidence of legal compliance.

**Reference:** Digital Personal Data Protection Act, 2023 (Act No. 22 of 2023). Rules not yet notified as of this writing (November 2026); obligations below follow the Act text directly.

---

## 1. Obligation: Notice and Consent (Sections 5–6)

**Act requirement:** Before or at the time of collecting personal data, the Data Fiduciary must give a notice in clear and plain language (in English and any scheduled language the Data Principal prefers) describing: what data is collected, the purpose, how to exercise rights, and how to file a grievance. Consent must be free, specific, informed, unconditional, and unambiguous (affirmative action).

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Notice before data collection | Frontend presents a consent checkbox with the text "I consent to RetinaGuard processing this retinal image for diabetic retinopathy screening under the DPDP Act 2023" before submission. | The notice text is embedded in `frontend/src/App.tsx` as hardcoded English prose. A formally reviewed multi-language privacy notice document does not exist. |
| Specific purpose stated | `core/consent.py::ConsentPurpose` enumerates `DIABETIC_RETINOPATHY_SCREENING`, `CLINICAL_DECISION_SUPPORT`, and `REGULATORY_AUDIT_AND_COMPLIANCE`. | Purposes are not surfaced verbatim to the patient in the UI notice; the UI checkbox text is a summary. |
| Informed consent | The checkbox is mandatory; the screening endpoint validates it in the request body. | The system does not display notice version or link to a full privacy notice document. |
| Unambiguous affirmative action | Checkbox is a boolean `consent_given: bool`; the backend rejects requests where `consent_given=false`. | Consent is captured as a boolean in the API request; there is no durable consent record stored independently of the screening record. |
| Consent language in Data Principal's language | `consent_language` field in `DPDPProcessingConsent`; locale system supports 11 Indian languages. | Consent notice text is English-only in the frontend; translations for the consent notice itself have not been professionally reviewed (all locales have `reviewed: false`). |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — A qualified lawyer must review: (1) whether the checkbox constitutes valid "free, specific, informed and unambiguous" consent under S.6; (2) whether a durable consent record (timestamped, versioned) must be retained; (3) the required content of the notice under notified Rules.

---

## 2. Obligation: Purpose Limitation (Section 6(5))

**Act requirement:** Personal data may be processed only for the purpose for which consent was given. It must not be reused for incompatible purposes.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Data processed only for stated purpose | The screening pipeline processes retinal images exclusively for DR grade inference and triage referral; no secondary analytics, advertising, or sharing endpoints exist. | There is no technical enforcement layer (e.g., purpose tag propagated through data flows) that programmatically prevents future code changes from repurposing data. |
| Data not shared with third parties for other purposes | Retinal images are never transmitted externally. Voice narration text (generic clinical advice only) is transmitted to Sarvam AI; this text contains no patient identifiers. | A Data Processing Agreement (DPA) with Sarvam AI for India-specific health data has not been confirmed. Under DPDP, Sarvam would be a "Data Processor" and a written contract is required. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Legal counsel must confirm: (1) whether transmitting generic voice narration text to Sarvam constitutes "processing" of personal data; (2) whether a formal DPA with Sarvam AI is required; (3) whether the audit ledger's pseudonym + outcome data constitutes personal data under the Act.

---

## 3. Obligation: Data Minimisation (Section 6(5))

**Act requirement:** Only the personal data necessary for the stated purpose may be collected.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Minimal identifiers collected | The system collects: (a) a patient pseudonym ID (not a legal name), (b) eye side, (c) camera type, (d) age (optional), (e) diabetes type (optional), (f) retinal image. No ABHA ID, Aadhaar, name, or address is requested. | Age and diabetes type are optional but not technically prevented from being omitted. The pseudonym ID format (`PAT-DEL-2026-XXXX`) is user-supplied and free-text; no pseudonymisation scheme is enforced by the system itself. |
| Image not retained beyond necessity | Retinal images are not persisted to disk or database by the backend. Only the SHA-256 hash of the image is stored in the audit ledger. | No explicit confirmation that in-memory buffers are cleared promptly after inference; garbage collection is relied upon. |
| Clinical notes minimal | The clinician override note (free text) is stored in the screening record. | No character limit or content filter is applied to clinician notes; notes could inadvertently contain patient names. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Legal counsel must assess: (1) whether SHA-256 image hash in the audit ledger constitutes personal data (it is a one-way hash, but could theoretically be used to confirm whether a specific image was processed); (2) whether age and diabetes type fields require a specific legal basis beyond consent.

---

## 4. Obligation: Accuracy (Section 8(3))

**Act requirement:** The Data Fiduciary must take reasonable steps to ensure personal data is accurate and complete where it is used to make a decision affecting the Data Principal.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| AI-generated results subject to clinician review | Every screening result carries `clinician_review_mandatory: true`; the UNCERTAIN pathway mandates clinician review before any clinical action. | The accuracy of the AI classification itself (ICDR grade) is a statistical estimate with documented confidence intervals. No mechanism prompts the screener to verify that the submitted image is of the correct patient's eye. |
| Clinician can correct (override) AI assessment | Clinicians can override grade and add clinical notes via `POST /screenings/{id}/review`. | Overridden records update the screening status but the original AI prediction is preserved in the audit ledger; this is correct behaviour. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Legal counsel must assess whether the obligation on "accuracy" imposes additional requirements on AI systems providing decision support (e.g., mandatory explainability disclosure, access to AI confidence intervals).

---

## 5. Obligation: Storage Limitation / Retention and Erasure (Section 8(7) and Section 12)

**Act requirement:** Personal data must be erased when: (a) the purpose for which it was collected is served and no legal requirement mandates retention, or (b) the Data Principal withdraws consent and there is no legal ground for continued processing. Data Fiduciaries must implement data retention policies.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Defined retention period | No retention period is defined in code or policy. | **GAP (critical):** No data retention schedule exists. Screening records (pseudonym, outcome) and audit ledger entries accumulate indefinitely. A legally reviewed retention period must be defined and implemented. |
| Automated erasure on consent withdrawal | `DPDPProcessingConsent.is_revoked` can be set to `True`; `is_valid_for_processing()` returns `False` for revoked consent. | Revoking the in-memory consent object does not delete stored screening records or audit ledger entries. No erasure workflow is implemented. |
| Data erasure API | No `DELETE` endpoint exists for screening records or audit entries. | **GAP (critical):** Right to erasure (implicit under S.12 Right to Withdraw Consent) has no implementation. This must be built before production. |
| Audit ledger retention | The audit JSONL ledger is append-only by design and grows indefinitely. | A legally reviewed decision is needed: health regulation (Clinical Establishments Act, MoHFW record-keeping rules) may require audit logs to be retained for 5–7 years; the DPDP Act may require patient data to be purged on request. These may conflict and require legal resolution. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — This is a critical gap. Legal counsel must specify: (1) mandatory retention period for DR screening records under applicable health regulations; (2) how the right to erasure interacts with the tamper-evident audit ledger; (3) the technical implementation of a compliant erasure workflow.

---

## 6. Obligation: Security Safeguards (Section 8(1))

**Act requirement:** Appropriate technical and organisational measures must be implemented to protect personal data against breaches.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Encryption at rest | AES-256-GCM field-level encryption for PHI (`core/encryption.py`, `PHIEncryptor`). | Encryption key is loaded from `RETINAGUARD_PHI_ENCRYPTION_KEY` env var; key management (rotation, HSM) is not implemented. |
| Encryption in transit | HTTPS required (not enforced at application layer; relies on deployment infrastructure). | Application does not enforce TLS; this must be confirmed at the infrastructure layer. |
| Access control | RBAC with JWT; screener/clinician/admin roles. | JWT uses symmetric HS256; no token revocation list; session expiry configurable via `JWT_EXPIRE_MINUTES`. |
| PHI not in logs | `PHIScrubbingFilter` applied to all loggers; phone, email, Aadhaar, ABHA redacted automatically. | PHI scrubbing patterns cover known formats; free-text clinician notes are not logged. |
| Zero PHI in voice narration | Voice text verified by test to contain only generic advice (`test_voice_text_generic_advice_no_phi`). | Sarvam AI DPA not confirmed (see Section 2). |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Security assessment by a CISO or certified information-security professional is required before production. The organisational measures (access logs, incident response plan, employee training) are entirely outside the codebase.

---

## 7. Obligation: Breach Notification (Section 8(6))

**Act requirement:** In the event of a personal data breach, the Data Fiduciary must notify the Data Protection Board of India and affected Data Principals in a prescribed manner and timeline (timelines to be specified in Rules; likely 72 hours by analogy with GDPR and CERT-In).

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Breach detection | No breach detection, alerting, or anomaly detection is implemented in the codebase. | **GAP (critical):** No SIEM integration, no intrusion detection, no automated alerting on abnormal access patterns. |
| Breach notification workflow | No breach notification workflow exists. | **GAP (critical):** A documented incident response plan and breach notification SOP must be created and tested. |
| Board notification | Not implemented. | **GAP (critical):** Requires legal and organisational preparation. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — This obligation is entirely unaddressed by the codebase. A qualified lawyer and incident response team must design and document a compliant breach notification programme.

---

## 8. Obligation: Rights of Data Principals (Sections 11–13)

**Act requirement:** Data Principals have the right to: (11) access information about their data; (12) correction and erasure; (13) grievance redressal; (14) nominate a person to exercise rights on death/incapacity.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Right to access | No patient-facing data access portal or API endpoint exists. | **GAP:** No data subject access request (DSAR) workflow. |
| Right to correction | Clinicians can correct the AI grade; no patient-facing correction mechanism exists. | **GAP:** Patient-initiated correction not implemented. |
| Right to erasure | Not implemented (see Section 5). | **GAP (critical).** |
| Right to grievance redressal | No grievance officer contact is defined in the application or documentation. | **GAP:** A named Grievance Officer with contact details must be designated and published. |
| Right to nominate | Not applicable to the current prototype scope. | To be addressed before production deployment. |

---

## 9. Obligation: Data Fiduciary Registration and Significant Data Fiduciary (Section 10)

**Act requirement:** Certain Data Fiduciaries may be designated as "Significant Data Fiduciaries" (SDFs) by the Central Government based on volume, sensitivity, or risk and must comply with additional obligations (data localisation, DPIA, appointment of DPO).

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Data localisation | `.env.example` specifies `CLOUD_REGION=asia-south1` (Mumbai) as default; documentation states "All personal health data strictly hosted in India data centers (DPDP Act 2023)". | Not technically enforced; relies on deployment configuration. SDF designation is not yet determined. |
| Data Protection Impact Assessment (DPIA) | Not conducted. | **GAP:** A DPIA is likely required given the sensitive health data and AI-decision-support nature of the system. |
| Appointment of DPO | No Data Protection Officer designated. | **GAP:** Required if system is designated an SDF; prudent in any case. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Legal counsel must determine whether RetinaGuard's operator will be classified as an SDF and what additional obligations follow.

---

## 10. Obligation: Cross-Border Data Transfer (Section 16)

**Act requirement:** Transfer of personal data outside India is permitted only to countries notified by the Central Government. The notified countries list has not been published as of this writing.

| Sub-obligation | What the Code Does Today | What Is Missing |
|---|---|---|
| Patient data not transferred abroad | Retinal images, pseudonym IDs, and clinical outcomes are not sent abroad. | Confirmed by code review: no patient data leaves the backend. |
| Voice text to Sarvam AI | Sarvam AI is an Indian company; API endpoint is `api.sarvam.ai`. | Sarvam AI's infrastructure location must be confirmed as India-only. If Sarvam routes requests through non-notified countries, this may constitute a cross-border transfer. |
| Edge-TTS (Microsoft Azure) | Edge-TTS routes audio through Microsoft Azure infrastructure, which may not be in India. | **GAP:** If voice text is ever treated as personal data (legal question — see Section 2), Edge-TTS usage may violate S.16 until notified countries include the US/EU. Consider limiting Edge-TTS to non-personal, generic text use cases. |

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — Whether generic clinical advice text constitutes "personal data" for the purpose of cross-border transfer restrictions must be determined by legal counsel once the notified-countries list is published.

---

## Summary of Critical Gaps (Must-Fix Before Production)

| # | Gap | Act Section |
|---|---|---|
| 1 | No data retention schedule or automated erasure | S.8(7), S.12 |
| 2 | No patient-facing Right to Erasure endpoint | S.12 |
| 3 | No breach detection or notification workflow | S.8(6) |
| 4 | No Grievance Officer designated with contact | S.13 |
| 5 | No DPIA conducted | S.10 (if SDF) |
| 6 | Consent notice text not professionally translated | S.5–6 |
| 7 | Sarvam AI DPA not confirmed for Indian health data | S.6(5) |
| 8 | Edge-TTS cross-border transfer status unresolved | S.16 |
| 9 | Encryption key management not implemented | S.8(1) |
| 10 | No patient data subject access request (DSAR) workflow | S.11 |
