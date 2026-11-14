# RetinaGuard Threat Model

> [!CAUTION]
> **ENGINEERING PREPARATION ONLY — NOT LEGAL OR SECURITY ADVICE.**
> This document was prepared by the engineering team to structure a future formal threat-model review by qualified information-security and regulatory personnel. It does not constitute a complete security assessment. All items marked "GAP" must be addressed before clinical or production deployment.

---

## 1. System Scope

This threat model covers the following components:

| Component | Description |
|---|---|
| **Screening endpoint** | `POST /screen/` — receives retinal fundus image, runs quality gate, OOD gate, ONNX inference, rules engine, and writes audit entry |
| **Clinician review queue** | `GET /screenings`, `POST /screenings/{id}/review` — RBAC-gated case queue for clinician sign-off and override |
| **Audit ledger** | Append-only SHA-256 hash-chained JSONL file (`data/audit/retinaguard.audit.jsonl`) |
| **Voice synthesis providers** | Sarvam AI Bulbul:v3 (Tier 1), Edge-TTS (Tier 2), Browser Web Speech API (Tier 3 fallback) |
| **PDF report generation** | `GET /screenings/{id}/report.pdf` — server-side PDF via ReportLab using locale translations |
| **Auth subsystem** | `POST /auth/login` (JWT), `POST /auth/users` (admin creates clinicians), rate-limited |
| **ML model runtime** | ONNX runtime (CPU), `models/retinaguard_dr_v1.onnx`, OOD statistics `models/ood_stats_v1.npz` |

**Out of scope (not modelled here):** Network perimeter, cloud infrastructure, operating system, physical clinic access control.

---

## 2. Trust Boundaries

```
[ PHC Screener / Browser ]
         |
         | HTTPS (TLS required in production)
         v
[ FastAPI Backend (port 8000) ]
         |
    +----+----+
    |         |
    v         v
[ SQLite /   [ JSONL Audit
  PostgreSQL ]  Ledger (filesystem) ]
         |
         v
[ External: Sarvam AI API (HTTPS, api.sarvam.ai) ]
[ External: Edge-TTS (HTTPS, Microsoft Azure TTS) ]
```

**Trust boundary crossings of concern:**
- Browser → Backend (unauthenticated screening if `ALLOW_ANONYMOUS_SCREENING=true`)
- Backend → Sarvam/Edge-TTS (voice narration text transmitted to third-party cloud)
- Filesystem → Backend (model files loaded at startup; tamper risk if filesystem is shared)

---

## 3. STRIDE Threat Table

### 3.1 Screening Flow (`POST /screen/`)

| STRIDE | Threat | Mitigation Implemented | Gap / Open Item |
|---|---|---|---|
| **S** Spoofing | Unauthenticated screener impersonates clinician | JWT Bearer tokens; `ALLOW_ANONYMOUS_SCREENING=false` enforced; role check on all clinical-action endpoints | GAP: `ALLOW_ANONYMOUS_SCREENING=true` mode (demo only) skips identity entirely; must be disabled in production |
| **T** Tampering | Attacker replaces ONNX model file or OOD stats to manipulate predictions | SHA-256 checksum of ONNX in `models/registry.json`; documented in MODEL_CARD.md | GAP: Checksum verified manually/documentally, not automatically at startup; no code-enforced startup integrity check |
| **R** Repudiation | Screener denies having submitted a screening | Append-only SHA-256 hash-chained audit ledger (`retinaguard.audit.jsonl`); every screening event is recorded with pseudonym, image hash, and outcome | GAP: Ledger is a local JSONL file; integrity verification endpoint `/screen/audit/verify` exists but ledger is not stored in an independent tamper-evident store (e.g., WORM storage) |
| **I** Information disclosure | Fundus image or clinical result exposed in transit | HTTPS required; image stored only as SHA-256 in audit ledger; screening result uses patient pseudonym ID, never plaintext name | GAP: TLS not enforced by application layer (relies on deployment infrastructure); no TLS pinning |
| **D** Denial of service | Flooding `/screen/` endpoint with large images | `slowapi` rate limiter applied; image size validated | GAP: Rate limit per IP is not tested under load; large file DoS not formally validated |
| **E** Elevation of privilege | Screener role invokes clinician-only `/screenings/{id}/review` | RBAC enforced; `Depends(require_clinician)` on review endpoint; role embedded in JWT | GAP: JWT secret from `JWT_SECRET_KEY` env var; if secret leaks, tokens can be forged; no token revocation list |

### 3.2 Clinician Review Queue (`GET /screenings`, `POST /screenings/{id}/review`)

| STRIDE | Threat | Mitigation Implemented | Gap / Open Item |
|---|---|---|---|
| **S** Spoofing | Attacker presents forged clinician JWT | RS256 or HS256 JWT with configurable secret; role validated on each request | GAP: Default HS256 is symmetric; consider asymmetric RS256 for production multi-service environments |
| **T** Tampering | Attacker modifies stored screening result before clinician review | Screening results written to database; audit ledger records outcome at time of screening | GAP: Database records are mutable (SQL UPDATE); no row-level tamper detection after initial write |
| **R** Repudiation | Clinician denies having confirmed or overridden a result | Review action appends a `CLINICIAN_REVIEW` audit entry with clinician ID and timestamp | GAP: Clinician ID in audit is the database UUID, not a verified digital signature |
| **I** Information disclosure | Queue lists all screenings to any authenticated user | Clinician sees only `pending_review` queue; role-based filter | GAP: No patient-level ownership check; one clinician can see another clinic's pending cases if on the same DB |
| **D** Denial of service | Clinician queue flooded with UNCERTAIN cases | UNCERTAIN cases are queued for review per design; no throttle on UNCERTAIN submissions specifically | GAP: No daily cap or circuit breaker on UNCERTAIN routing |
| **E** Elevation of privilege | Clinician account promotes itself to ADMIN | Admin role required for `POST /auth/users`; role check enforced | Mitigated |

### 3.3 Audit Ledger (`retinaguard.audit.jsonl`)

| STRIDE | Threat | Mitigation Implemented | Gap / Open Item |
|---|---|---|---|
| **S** Spoofing | Attacker injects a crafted audit entry | Ledger is append-only at application level; genesis hash and chain link each entry | GAP: Filesystem-level protections (file ACL, immutable bit) are deployment-specific; application cannot enforce them |
| **T** Tampering | Attacker modifies an existing entry | Hash-chain: tampering changes one entry hash, breaking all subsequent hashes; `GET /screen/audit/verify` detects this | GAP: Ledger is a local plaintext JSONL file; an attacker with filesystem write access could rebuild the chain from scratch |
| **R** Repudiation | Hospital disputes completeness of audit trail | Sequential entry_id and hash chain provides ordering proof | GAP: No external timestamping authority (RFC 3161) or WORM storage |
| **I** Information disclosure | PHI inadvertently written to audit log | Ledger stores only pseudonym_id, screening_id, image SHA-256, and outcome summary; no names or identifiers | Log scanner test (`test_log_scanning_covers_audit_ledger_zero_leak`) verifies secret note does not leak into log messages |
| **D** Denial of service | Disk-fill attack via excessive audit entries | Each JSONL line is bounded in size | GAP: No log rotation or size cap on the audit file |
| **E** Elevation of privilege | Access to audit file grants read access to pseudonym IDs | Pseudonym IDs are project-internal identifiers, not patient names | GAP: If pseudonym scheme is reversible, IDs could be re-linked to patients |

### 3.4 Voice Providers (Sarvam AI, Edge-TTS)

| STRIDE | Threat | Mitigation Implemented | Gap / Open Item |
|---|---|---|---|
| **S** Spoofing | MITM substitutes Sarvam API endpoint | HTTPS to `api.sarvam.ai`; httpx enforces TLS by default | GAP: No explicit CA pinning; a rogue CA could intercept |
| **T** Tampering | Attacker modifies voice narration text | Text is composed internally by `VoiceAgent`; not echoed from user input | Mitigated by design |
| **R** Repudiation | Sarvam claims they never received or logged the narration | No application-level proof; relies on Sarvam's terms of service | GAP: No receipt acknowledgement or logging (by design — zero voice text logging) |
| **I** Information disclosure | Generic advice text transmitted to Sarvam cloud | Voice text verified to contain only generic clinical advice — no pseudonym ID, image data, or clinician notes (`test_voice_text_generic_advice_no_phi`, `test_voice_endpoint_zero_logging`) | GAP: Sarvam's data processing agreement for Indian health data not confirmed; must be reviewed before production use |
| **I** Information disclosure | Audio response streamed from Sarvam stored on server | Audio returned directly to client; not persisted | Mitigated |
| **D** Denial of service | Sarvam/Edge-TTS unavailable | 3-tier fallback chain: Sarvam → Edge-TTS → HTTP 503 (browser Web Speech API); deterministically tested (`test_e2e_voice_tier1_sarvam_deterministic`, `test_e2e_voice_tier2_edge_tts_deterministic`, `test_e2e_voice_tier3_browser_fallback_deterministic`) | Mitigated |
| **E** Elevation of privilege | SARVAM_API_KEY leaked via environment | Key read from `SARVAM_API_KEY` env var; never logged | GAP: No secrets management (Vault, AWS Secrets Manager); key in `.env` file |

### 3.5 PDF Report Generation (`GET /screenings/{id}/report.pdf`)

| STRIDE | Threat | Mitigation Implemented | Gap / Open Item |
|---|---|---|---|
| **S** Spoofing | Unauthenticated user downloads another patient's PDF | Requires valid clinician JWT Bearer token | Mitigated |
| **T** Tampering | Attacker crafts locale translation to inject content | Locale JSON files in `configs/locales/`; `reviewed` flag prevents unreviewed locales in production (`ALLOW_UNREVIEWED_LOCALES=false`) | GAP: Locale files are on filesystem; server restart required to pick up changes; no content signature on locale files |
| **R** Repudiation | Printed PDF not traceable to system | PDF contains screening ID, model version, and generation timestamp | GAP: No digital signature on PDF output |
| **I** Information disclosure | PDF contains PHI | PDF uses patient pseudonym only; no ABHA, name, or Aadhaar | GAP: PDF is returned as a raw byte stream; no encryption-at-rest for PDF if saved by client |
| **D** Denial of service | Resource exhaustion via many simultaneous PDF renders | ReportLab render is synchronous; large clinician queues could block | GAP: No async PDF job queue or render timeout |
| **E** Elevation of privilege | Screener (not clinician) downloads PDF | Clinician JWT required; role enforced on endpoint | Mitigated |

---

## 4. Summary of Open Gaps (Priority Order)

| # | Gap | Risk | Before Production? |
|---|---|---|---|
| 1 | ONNX model file integrity not verified at startup | High — tampered model could cause systematic misdiagnosis | Yes |
| 2 | Audit ledger stored as local filesystem JSONL (not WORM) | High — filesystem access breaks tamper evidence | Yes |
| 3 | Sarvam AI DPA for Indian health data not confirmed | High — DPDP Act 2023 obligation on data processors | Yes |
| 4 | JWT symmetric secret (HS256); no token revocation | Medium | Yes |
| 5 | No TLS enforcement at application layer | Medium | Yes |
| 6 | `ALLOW_ANONYMOUS_SCREENING=true` removes all identity | Medium | Must be false in production |
| 7 | PDF not digitally signed | Low–Medium | Before regulatory submission |
| 8 | Locale files not cryptographically signed | Low | Before production |
| 9 | Audit file has no log rotation or size cap | Low | Before production |
| 10 | No rate limit load testing | Low | Before production |
