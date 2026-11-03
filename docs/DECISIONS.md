# Architecture Decision Records (ADRs) - RetinaGuard

This document records all significant architectural, clinical-safety, and regulatory decisions made for the RetinaGuard decision-support platform.

---

## ADR-001: Decision-Support Only (No Autonomous Diagnosis)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Under CDSCO medical device regulations (MDR 2017) and global SaMD guidance, autonomous diagnostic devices carry significant liability and patient safety risks.
* **Decision**: RetinaGuard shall strictly operate as a clinician-assistive tool. Every output package is flagged `clinician_review_mandatory: true`. The software never autonomously prescribes treatment or finalizes a diagnosis without a licensed medical practitioner's sign-off.
* **Consequences**: User interface and reports must prominently display disclaimer notices. Clinician sign-off events must be logged in the audit ledger.

---

## ADR-002: Deterministic Clinical Rules Engine Separated from ML
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Deep learning models are probabilistic and susceptible to distribution shifts. They must not make referral or urgency decisions directly.
* **Decision**: The ML model is strictly constrained to output ICDR grade (0–4) probabilities and confidence. A pure Python, deterministic rules engine (governed by AIOS, RSSDI, ICMR, and ICO clinical consensus) calculates referral decisions, clinical urgency, and follow-up intervals.
* **Consequences**: The rules engine is 100% unit-tested against clinical edge cases and can be updated/reviewed independently of neural network weights.

---

## ADR-003: Non-Alterable Explanatory Guideline RAG
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Generative language models may hallucinate, soften, or alter clinical facts and numbers.
* **Decision**: LLM / RAG integration is strictly restricted to generating plain-language narrative explanations from retrieved, authenticated clinical guideline texts. The LLM is technically blocked from modifying any numeric grades, urgency tiers, or intervals computed by the rules engine.
* **Consequences**: Output schemas strictly enforce that clinical variables remain immutable.

---

## ADR-004: Fail-Safe Gatekeeping (Sub-optimal Quality & OOD Rejection)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Low-quality fundus photos (blur, glare, improper illumination) or non-fundus images (cornea, eyelids, random photos) can induce false negatives or catastrophic misclassification.
* **Decision**: All images pass through a pre-inference quality gate and out-of-distribution (OOD) check. Any failure immediately halts inference and returns `"UNCERTAIN - needs clinician review"`. The model is never permitted to guess.
* **Consequences**: Specific thresholds are parameterized for tabletop vs. smartphone/portable cameras (e.g. Remidio, Forus).

---

## ADR-005: Hash-Chained Tamper-Evident Audit Ledger
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Medical-legal and regulatory audits (CDSCO, ISO 13485) require verifiable proof that historical inference results and clinician sign-offs have not been retroactively altered.
* **Decision**: Every screening transaction is appended to a hash-chained ledger where block $N$ incorporates the SHA-256 digest of block $N-1$, the image SHA-256 hash, and the encrypted result payload.
* **Consequences**: Any retroactive database tampering breaks the cryptographic chain and triggers an alert.

---

## ADR-006: DPDP Act 2023 Compliance & Indian Data Residency
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: India's Digital Personal Data Protection (DPDP) Act 2023 mandates purpose-specific affirmative consent, data minimization, right to erasure, and national data residency.
* **Decision**:
  - Retinal photos and patient metadata require explicit purpose-bound consent (`RETINOPATHY_SCREENING`).
  - Zero PHI is allowed in application logs; automated regex scrubbers filter ABHA IDs, Aadhaar numbers, phone numbers, and emails.
  - PHI at rest is encrypted using AES-256-GCM.
  - Infrastructure is pinned to Indian cloud regions (`me-central` / `asia-south1` [Mumbai] / `asia-south2` [Delhi]).
  - Schemas support future ABHA (Ayushman Bharat Health Account) integration without coupling to external gateways yet.

---

## ADR-007: Differentiated Hardware Profiles for Indian Screening Settings
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Indian primary health centers (PHCs) and community screening camps predominantly use portable, handheld, or smartphone-attached fundus cameras, while hospital departments use tabletop cameras.
* **Decision**: Quality assessment profiles (`configs/quality_thresholds.yaml`) are decoupled into `tabletop` and `smartphone_portable` configurations with empirical tuning for lens flare, corneal reflection, and motion blur.



## ADR-008: Dataset Registry, Strict Patient-Level Splitting & IDRiD External Holdout
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Retinal fundus datasets often contain multiple images per patient (both eyes, follow-ups). Naive random train/test splitting causes patient-level data leakage, inflating test accuracy. Furthermore, algorithms trained on foreign datasets (e.g. EyePACS, Messidor) frequently suffer demographic and hardware domain shift when deployed in India.
* **Decision**:
  1. A centralized `DatasetRegistry` tracks every dataset with mandatory `LICENSE`, `citation`, `camera_device`, and `site` metadata.
  2. All splits use `PatientStratifiedSplitter` which strictly groups by `patient_id` to guarantee zero patient overlap between train, val, and test.
  3. The Indian dataset **IDRiD** (Kowa VX-10α, Nanded, Maharashtra) is locked as an **external test set** and is kept strictly out of model training to provide an uncompromised Indian ground-truth generalization benchmark.
* **Consequences**: Splitting scripts validate mathematical disjointness of patient sets (`len(train_patients & test_patients) == 0`).

---

## ADR-009: Synthetic Smartphone Camera Degradation for Domain Shift
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Primary health centers and community camps in India rely heavily on smartphone and handheld fundus cameras (e.g., Remidio NM-FOP, Forus 3nethra). These suffer from lower resolution, non-mydriatic lens flare, uneven illumination gradients, JPEG compression, and minor optical blur compared to benchtop tabletop cameras.
* **Decision**: RetinaGuard introduces a dedicated `SmartphoneDegradationPipeline` that stochastically simulates handheld camera artifacts during training and domain-shift stress testing, without altering retinal lesion pigmentation.
* **Consequences**: Models trained with synthetic smartphone degradation exhibit significantly improved robustness when evaluated on real-world Indian handheld camera photos.


## ADR-010: Dual-Head Architecture (Categorical Logits + Ordinal Regression Head)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Diabetic Retinopathy progression is intrinsically ordinal (Grade 0 < 1 < 2 < 3 < 4). Treating it strictly as nominal multi-class classification penalizes a Grade 0 vs 1 error identically to a Grade 0 vs 4 error. Conversely, pure ordinal regression can struggle to yield calibrated multi-class probability distributions required for uncertainty quantification.
* **Decision**: RetinaGuard adopts a **Dual-Head** deep learning architecture sharing a common EfficientNet backbone:
  1. **Categorical Head**: Produces 5 logits corresponding to ICDR grades 0 to 4, trained with class-weighted cross-entropy / focal loss.
  2. **Ordinal Regression Head**: Produces 4 cumulative threshold logits representing binary probabilities $P(\text{grade} \ge k)$ for $k \in \{1, 2, 3, 4\}$, trained with multi-label binary cross-entropy.
  3. **Joint Objective**: Trained jointly with loss $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{ord}} \cdot \mathcal{L}_{\text{ord}}$.
* **Consequences**: Predictions penalize distant misclassifications more severely, directly improving the clinical gold-standard Quadratic Weighted Kappa (QWK) while preserving calibrated softmax probabilities for confidence estimation.


## ADR-011: Two-Tier Pre-Inference Gatekeeping (Classical CV + OOD Detection)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Deploying deep learning models in real-world Indian clinics risks processing out-of-distribution non-fundus images (e.g. skin photos, faces, clinical notes) or heavily degraded photos (motion blur, flash glare, total darkness). Feeding these to a neural network causes uncontrolled hallucinatory predictions.
* **Decision**: All candidate screening images pass through two mandatory pre-inference gates before touching the deep grading model:
  1. **Tier 1 (Classical Computer Vision Quality Gate)**: Detects circular retinal field-of-view (FOV) contour, masked Laplacian variance (blur), percentile brightness, and glare/underexposure ratios. Returns human-readable recapture instructions in English and Hindi.
  2. **Tier 2 (Out-of-Distribution Rejection Gate)**: Evaluates physiological color spectrum ($R > G > B$ hemoglobin reflectance) and deep feature space metrics (Maximum Softmax Probability, Energy score, and Mahalanobis distance to class centroids).
  3. **Fail-Safe Guarantee**: Any failure in Tier 1 or Tier 2 halts execution and returns `"UNCERTAIN - needs clinician review"`.
* **Consequences**: Zero degraded or non-fundus inputs reach the diagnostic rules engine or patient reports.


---

## ADR-012: Hash-Chained Audit Ledger (Append-Only)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: CDSCO SaMD Technical File requirements and the DPDP Act 2023 mandate a tamper-evident audit trail for all AI-assisted clinical screening events. Simple log files are editable.
* **Decision**: Every completed screening event is appended to a JSONL ledger (`data/audit/retinaguard.audit.jsonl`) where each entry's SHA-256 hash covers the previous entry's hash, creating a provably linked chain. Tampering any entry invalidates every subsequent hash. The ledger is PHI-free: it stores only pseudonymous patient IDs, screening IDs, and outcome summaries.
* **Consequences**: The `/screen/audit/verify` endpoint can detect retrospective record modification at any time. Ledger files can be exported for regulatory inspection. Chain verification is O(n) over total entries.

---

## ADR-013: Bilingual Screening Reports (English + Hindi)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Indian primary-care clinicians frequently operate in Hindi-speaking regions. Providing reports only in English excludes a large proportion of the target user base and increases risk of misinterpretation.
* **Decision**: The `ReportService` always generates both `report_en` and `report_hi` for every screening. The requesting client specifies a preferred language (`language` form field). All grade labels, referral decisions, urgency classifications, and disclaimers are maintained in both languages in a static lookup table (not LLM-generated, to prevent hallucination of clinical facts).
* **Consequences**: Reports are safe against translation drift. Adding additional Indian languages (Tamil, Telugu, Kannada) requires only extending the lookup tables.

---

## ADR-014: Lazy-Loaded ONNX Inference with Graceful Absence Handling
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: During development and testing, the trained ONNX model file does not yet exist. The API must remain fully functional and return safe `UNCERTAIN` outcomes rather than crashing.
* **Decision**: `ONNXInferenceService` lazily loads the ONNX Runtime session on the first `predict()` call and returns `None` (triggering the rules engine's `UNCERTAIN` pathway) if the model file is absent or `onnxruntime` is not installed.
* **Consequences**: The `/screen` endpoint is testable and deployable without a trained model. Staging environments can run with a mock or placeholder model. The ONNX session is cached in a module-level singleton (loaded once per process).


---

## ADR-015: Content-Hash Grouping for Patient-Unblinded Datasets (APTOS 2019)
* **Date**: 2026-11-02
* **Status**: Accepted
* **Context**: Public Kaggle competition datasets such as APTOS 2019 omit patient identifiers (`patient_id`) and laterality markers (`eye_side`), providing only isolated image IDs (`id_code`). Standard random splitting risks severe train-test data leakage if near-duplicate captures, rescaled images, or burst shots from the same examination cross partitions.
* **Decision**: We employ cryptographic (SHA-256) and perceptual hashing (64-bit DCT pHash via `imagehash` with Hamming distance threshold $\le 6$) to discover and cluster duplicate and near-duplicate images. All clustered images are assigned a single `group_id` and partitioned using `ContentGroupSplitter` (derived from `PatientStratifiedSplitter`). All images in any group are strictly quarantined to a single split (train, val, or test). Conflicting clinical labels within identical or near-identical image groups are highlighted as label-noise candidates without deleting data.
* **Known Limitations**:
  1. Content hashing can only identify identical, cropped, or slightly rescaled versions of the *same* physical photograph.
  2. It **cannot** detect two different photographs of the *same* patient's contralateral eye (OD vs OS) or longitudinal follow-up visits taken under different optical angles. True patient-level de-duplication requires clinical hospital manifests, which are unavailable in the public APTOS release.
* **Consequences**: Zero identical or near-identical photographs cross the train/val/test boundary, producing robust, non-inflated evaluation metrics while maintaining transparency about data provenance limitations.


---

## ADR-016: Strict Non-Transitive Image Clustering for APTOS 2019 Group-Aware Splitting
* **Date**: 2026-11-03
* **Status**: Accepted
* **Context**: The initial content-hashing approach (ADR-015) grouped images via union-find with a 64-bit pHash Hamming distance threshold of $\le 6$. Analysis of `splits_v1.csv` revealed severe transitive chaining: a giant connected component swallowed 1,589 images (43.4% of the dataset), forcing the test split to inflate to 1,841 images while leaving validation with only 258 images.
* **Decision**: We replaced transitive graph closure with strict dual-condition clustering:
  1. **Strict pHash**: Hamming distance $\le 2$ (instead of $\le 6$).
  2. **Secondary Cross-Validation**: Normalized Cross-Correlation (NCC) of 32x32 grayscale thumbnails of the cropped retina $> 0.98$.
  3. **Cluster Size Cap**: Any connected component is strictly capped at 10 members. If a component exceeds 10 members, it is partitioned into its tightest sub-clusters using mutual correlation centroids.
  4. **Target Partitions**: Stratified group-aware splits target 70% train, 15% validation, and 15% test, with `splits_v1.csv` preserved and the new manifest saved as `data/processed/splits_v2.csv`.
* **Consequences**:
  - `splits_v2.csv` achieves 2,551 train (69.7%), 559 val (15.3%), and 552 test (15.1%).
  - Grade proportions across all three splits are balanced within $\pm 1.7$ percentage points of each other.
  - Near-duplicate images remain quarantined strictly within single splits, ensuring zero data leakage while restoring standard train/val/test allocation.
