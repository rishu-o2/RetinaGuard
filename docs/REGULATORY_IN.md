# RetinaGuard — Regulatory Guidance Document (India)

> [!CAUTION]
> **ENGINEERING PREPARATION ONLY — NOT LEGAL OR REGULATORY ADVICE.**
> This document was prepared by the engineering team to structure a future consultation with a qualified CDSCO regulatory consultant and medical ethics committee. It does not constitute a regulatory filing, claim of compliance, or clearance of any kind. No statement in this document should be interpreted as an assertion that RetinaGuard is a licensed or approved medical device.

---

## 1. Intended Use Statement

**Product name:** RetinaGuard Diabetic Retinopathy Screening Aid (Prototype)

**Intended use:**
RetinaGuard is an investigational clinical **decision-support** software tool intended to assist trained clinical screeners and licensed ophthalmologists in the preliminary triage of diabetic retinopathy (DR) from retinal fundus images, using artificial intelligence. It is designed for use in primary health centres (PHCs) in India as a triage aid to prioritise which patients should be referred to a specialist ophthalmologist for a comprehensive dilated eye examination.

**What it is NOT:**
- RetinaGuard is **not** an autonomous diagnostic device.
- It **does not** replace the clinical judgment of a licensed ophthalmologist or physician.
- It **does not** provide a definitive diagnosis of diabetic retinopathy or any other ophthalmic condition.
- All screening findings produced by RetinaGuard are labelled "decision support only" and carry a mandatory requirement for clinician sign-off before any clinical action is taken.

**Operator:** The system is intended to be operated by trained healthcare workers under the supervision of a licensed ophthalmologist at a registered clinical establishment.

**Patient population:** Adult patients with a diagnosed history of diabetes mellitus presenting for routine DR screening at participating PHCs.

**Operating conditions:** Tabletop fundus cameras similar to those used in the APTOS 2019 Blindness Detection dataset (Aravind Eye Hospital). Performance on portable/smartphone cameras has not been validated.

---

## 2. Medical Device Classification and Regulatory Pathway (India)

**Reference regulation:** Medical Device Rules 2017 (MDR 2017) under the Drugs and Cosmetics Act, 1940 as amended. Administered by the Central Drugs Standard Control Organisation (CDSCO), Ministry of Health & Family Welfare.

**Software as a Medical Device (SaMD):** AI/ML-based software intended for diagnosis or decision support may qualify as a medical device under MDR 2017. Classification depends on the risk class and the nature of the intended use.

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — The following points must be evaluated by a qualified CDSCO regulatory consultant to determine the applicable classification, registration pathway, and clinical evidence requirements. The engineering team has not made a regulatory determination.

### Points to Raise with a CDSCO Regulatory Consultant

1. **SaMD classification under MDR 2017**: Does RetinaGuard meet the definition of a "medical device" under MDR 2017 (specifically, software that acquires, processes, or analyses medical images for clinical decision-making)? If so, what risk class applies (Class A, B, C, or D)?

2. **CDSCO notification requirement**: Has CDSCO issued formal guidance on AI/ML SaMD classification comparable to the FDA's SaMD guidance or IMDRF N12 framework? If not, which existing SaMD provisions apply by analogy?

3. **Performance data requirements**: What minimum sensitivity and specificity thresholds would CDSCO require for a DR screening aid? What reference standard (ETDRS 7-field photography or equivalent) would be accepted for the clinical validation dataset?

4. **Labelling requirements**: Does the software interface constitute labelling under MDR 2017? Are the current prototype disclaimers ("Not clinically validated", "Decision support only") sufficient for the prototype phase, and what additional labelling is required for a licensed product?

5. **Manufacturing site licence**: Is a manufacturing site licence required for software-only medical devices? What constitutes the "manufacturing site" for cloud-deployed software?

6. **Post-market surveillance (PMS)**: What PMS obligations apply to AI/ML SaMD after market authorisation? Does CDSCO require a Software Development and Maintenance Plan (SDMP)?

7. **Algorithm lock**: CDSCO and international SaMD guidance requires that the algorithm deployed in clinical use be "locked" (no further training or threshold changes without regulatory re-evaluation). The current architecture locks the model at deployment via `models/registry.json`. Confirm whether this design satisfies regulatory requirements.

8. **Paediatric use**: Is the absence of paediatric data (all training data is adult) a contraindication that must appear in the labelling?

---

## 3. Ethics Committee Checklist

If a clinical validation study is conducted with human participants (even retrospective chart review using patient images), approval from a duly constituted Institutional Ethics Committee (IEC) is required under the New Drugs and Clinical Trials Rules 2019 (NDCTR 2019) and ICMR National Ethical Guidelines for Biomedical and Health Research Involving Human Participants (2017).

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — This checklist is a starting-point template for discussion with an IEC chair or research ethics professional. It is not a substitute for a formal ethics submission.

| Item | Description | Status |
|---|---|---|
| Study design | Prospective or retrospective? Multi-site or single site? | To be defined by clinical investigator |
| IEC registration | Is the institution registered with CDSCO/ICMR for clinical research? | To be confirmed |
| Participant selection | Inclusion/exclusion criteria for the validation cohort | To be defined |
| Informed consent | Form in participant's language; voluntary; withdrawable | Template to be drafted by legal/clinical team |
| Data minimisation | Only data necessary for validation collected | Align with DPDP Act; see `docs/DPDP_COMPLIANCE.md` |
| Risk–benefit assessment | Risk of misclassification weighed against benefit of earlier referral | Requires clinical judgment |
| Reference standard | Gold-standard grading by certified ophthalmologist on ETDRS grading system | To be arranged |
| Conflict of interest | Declare funding sources, investigator affiliations | To be declared |
| Adverse event reporting | Protocol for reporting unexpected clinical outcomes | To be drafted |
| Data security plan | How participant images and grades are stored | Reference `docs/THREAT_MODEL.md` and `docs/DPDP_COMPLIANCE.md` |
| Study registration | CTRI registration (Clinical Trials Registry India) if prospective | To be completed before participant enrolment |
| IEC approval validity | Typically annual re-approval required | Calendar to be set |
| Protocol amendment process | How changes to the AI algorithm trigger protocol re-review | To be defined (see algorithm lock above) |

---

## 4. Clinical Validation Outline

A clinical validation study is required to generate the evidence needed for regulatory submission and to demonstrate that the system meets the intended-use performance claims.

> [!IMPORTANT]
> **NEEDS EXPERT REVIEW** — All sensitivity and specificity targets in the table below are **left blank**. The acceptable clinical performance thresholds are a clinical and regulatory judgment that must be made by qualified ophthalmologists and the CDSCO regulatory consultant, not by the engineering team. The engineering team must not fill in these cells.

### 4.1 Study Objectives

1. Establish the sensitivity and specificity of RetinaGuard for detecting referable DR (ICDR Grade ≥ 2) against a reference standard of expert ophthalmologist grading.
2. Demonstrate that the system's classification performance is maintained across camera equipment used in the target deployment environment.
3. Evaluate the abstention/UNCERTAIN rate and its clinical implications in the target population.

### 4.2 Reference Standard

| Aspect | Requirement |
|---|---|
| Grading system | ICDR Grading Scale (Grades 0–4) |
| Graders | ≥ 2 certified ophthalmologists; adjudication by a third senior grader on disagreement |
| Field | Minimum single-field (macula-centred); ETDRS 7-field preferred for the gold standard |
| Masking | Graders must be masked to the AI output |

### 4.3 Minimum Performance Targets

The following targets are intentionally left blank pending expert clinical and regulatory consultation.

| Metric | Minimum Acceptable Performance | Source of Threshold |
|---|---|---|
| Referable DR Sensitivity (Grade ≥ 2) | ___% (to be defined by clinical expert) | CDSCO / IEC guidance |
| Referable DR Specificity (Grade < 2) | ___% (to be defined by clinical expert) | CDSCO / IEC guidance |
| Grade ≥ 3 Sensitivity at SOON escalation tier | ___% (to be defined by clinical expert) | Clinical safety requirement |
| Acceptable UNCERTAIN (abstention) rate | ___% (to be defined by clinical expert) | Operational feasibility |
| AUC (Referable vs Non-Referable, ROC) | ___ (to be defined by clinical expert) | Clinical expert |

**Current prototype performance (from MODEL_CARD.md, NOT a clinical claim):**

| Cohort | Referable Sensitivity | Referable Specificity | ROC-AUC |
|---|---|---|---|
| APTOS 2019 VAL (N=559, same camera distribution as training) | 0.9641 | 0.9315 | 0.9890 |
| APTOS 2019 Held-Out Test (N=552) | 0.9602 | 0.9202 | 0.9745 |
| IDRiD External (N=516, different camera) | 0.7492 | 0.9585 | 0.9396 |

> [!WARNING]
> The IDRiD external benchmark demonstrates significant domain shift (referable sensitivity drops from 96% to 75%) when the camera equipment differs from training data. This is a known and documented limitation. A clinical validation study must use cameras representative of the actual deployment environment and must NOT rely solely on APTOS-sourced numbers.

### 4.4 Validation Dataset Requirements

| Requirement | Detail |
|---|---|
| Minimum sample size | To be calculated by a biostatistician based on target sensitivity/specificity and desired 95% CI width |
| DR prevalence | Should reflect the expected prevalence at target PHC sites (India-specific) |
| Grade distribution | Sufficient Grade 3 and Grade 4 cases required (rarest grades most critical for safety validation) |
| Camera equipment | Must match or represent cameras planned for deployment; a separate sub-study for each camera type |
| Geographic representation | Multi-site, multi-state if intended for national deployment |
| Exclusions | Poor-quality images should be reported separately (quality gate performance) |

### 4.5 Planned Evaluation Metrics

All metrics must be reported with 95% confidence intervals (bootstrap or exact binomial as appropriate):

- Sensitivity and specificity for referable DR detection (primary endpoint)
- Area under the ROC curve (secondary endpoint)
- Quadratic Weighted Kappa (QWK) for 5-class ICDR grade agreement
- Per-class recall (Grades 0–4 individually)
- Quality gate pass rate and characteristics of rejected images
- Abstention (UNCERTAIN) rate and characteristics of abstained cases
- Grade ≥ 3 sensitivity at the SOON escalation tier

---

## 5. What This Document Is Not

This document does not:
- Claim that RetinaGuard is approved, cleared, or licensed as a medical device.
- Constitute a regulatory submission to CDSCO or any other authority.
- Provide legal advice on compliance with any statute.
- Replace consultation with a qualified CDSCO regulatory consultant, Institutional Ethics Committee, or legal counsel specialising in Indian health law.

Before RetinaGuard is deployed in a clinical setting, operated on real patients, or submitted to any regulatory authority, the following must be completed by qualified professionals:
1. Formal CDSCO regulatory pathway determination
2. Qualified legal review of DPDP Act compliance (see `docs/DPDP_COMPLIANCE.md`)
3. Institutional Ethics Committee (IEC) approval for any human research
4. Formal threat model and penetration testing by qualified information-security professionals (see `docs/THREAT_MODEL.md`)
5. Clinical validation study with pre-specified performance targets approved by clinical experts
