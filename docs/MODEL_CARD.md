# RetinaGuard Model Card: EfficientNet-B0 Dual-Head DR Classifier

> [!CAUTION]
> **This software is a PROTOTYPE and is NOT clinically validated. It MUST NOT be used to make
> clinical decisions without regulatory clearance (CDSCO, FDA, or CE marking). All screening
> results must be reviewed by a certified ophthalmologist.**

## 1. Model Details

- **Model Name**: RetinaGuard DR Dual-Head Classifier (`retinaguard_dr_v1`)
- **Version**: `v1.0.0`
- **Architecture**: `EfficientNetDRDualHead`
  - Backbone: `efficientnet_b0` (feature dimension: 1,280)
  - Dropout rate: `0.30`
  - Head 1 (Categorical): Linear layer `[1280 -> 5]` outputting raw logits for 5 ICDR classes (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
  - Head 2 (Ordinal Thresholds): Linear layer `[1280 -> 4]` outputting cumulative logits for $P(\text{grade} \ge k)$ for $k \in \{1, 2, 3, 4\}$.
- **Prediction Head**:
  - **Grade prediction**: Frank & Hall ordinal threshold summation:
    $$\hat{y} = \text{clamp}\left(\text{round}\left(\sum_{k=1}^4 \sigma(z_k^{\text{ord}})\right), 0, 4\right)$$
  - **Probability / Confidence estimation**: Categorical logits with temperature scaling:
    $$P(y = c) = \frac{\exp(z_c^{\text{cat}} / T^*)}{\sum_{j=0}^4 \exp(z_j^{\text{cat}} / T^*)}$$
- **Training Framework**: PyTorch 2.9 (CPU execution mode)
- **Export Format**: ONNX opset 17 (`ml/exports/retinaguard_dr_v1.onnx`)
- **ONNX Checksum (SHA-256)**: `7d0c5d8bb1b448d09f48db586f2eff6627da5671e4abcc30c55c165d509c45aa`
- **Checkpoint**: `ml/runs/baseline_cpu_b0_256/best.pt` (Epoch 7)
- **Configuration**: `configs/baseline_cpu_b0_256.yaml`

---

## 2. Intended Use and Clinical Context

- **Primary Clinical Purpose**: Automated pre-screening triage for Diabetic Retinopathy in rural primary health centers (PHCs).
- **Referable DR Threshold**: $\text{ICDR Grade} \ge 2$ (Moderate NPDR, Severe NPDR, or PDR).
- **Clinical Governance Principle (ADR-003)**: The ML model strictly provides diagnostic probability estimates and an initial grade; referral recommendations and follow-up intervals are governed by the deterministic rules engine and **always** require clinician sign-off.
- **Fail-Safe Policy (ADR-004)**: When low confidence or out-of-distribution inputs occur, the system transitions to `UNCERTAIN` and requests expert clinical review rather than returning a confident false negative ("No DR").

---

## 3. Training and Evaluation Data

### Primary Dataset: APTOS 2019 Blindness Detection
- **Source**: Aravind Eye Hospital, Tamil Nadu, India.
- **Total Images**: 3,662 retinal fundus images.
- **Split Strategy**: Patient-stratified split manifest (`data/processed/splits_v2.csv`) with zero patient leakage:
  - **Train**: 2,551 images (69.7%)
  - **Validation**: 559 images (15.3%)
  - **Test (Held-out)**: 552 images (15.1%)
- **Preprocessing**:
  - Dark border cropping (`tol=15`)
  - Resized to $512 \times 512$ cache, loaded at $256 \times 256$
  - ImageNet normalization: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$

### External Benchmark: IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Source**: Dr. Ramanath Eye Hospital, Nanded, Maharashtra, India.
- **Role**: Strictly locked as external held-out test data (`is_external_test_only = True`). Never seen during training or validation.
- **Total Images**: 516 images (Training set: 413, Official Test set: 103).
- **Class Distribution in IDRiD**:
  - Grade 0: 168
  - Grade 1: 25 *(Small sample size; wide bootstrap confidence intervals)*
  - Grade 2: 168
  - Grade 3: 93
  - Grade 4: 62

---

## 4. Validation Performance (VAL Split, N=559)

*Measured on `ml/runs/baseline_cpu_b0_256/best.pt` (Epoch 7) using unaugmented VAL split:*

| Metric | Point Estimate | 95% Bootstrap CI (1,000 resamples) |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.9188** | [0.9009, 0.9363] |
| **Referable DR Sensitivity** ($\text{Grade} \ge 2$) | **0.9641** | [0.9384, 0.9868] |
| **Referable DR Specificity** ($\text{Grade} < 2$) | **0.9315** | [0.9037, 0.9575] |
| **Referable DR ROC-AUC** | **0.9889** | — |
| **Sensitivity @ 90% Specificity** | **0.9776** | — |
| **Sensitivity @ 95% Specificity** | **0.9462** | — |

### Per-Class Recall and Precision (VAL)

| Grade | Diagnostic Label | Recall (Sensitivity) | Precision (PPV) | Ground Truth $N$ |
|---|---|---|---|---|
| **0** | No DR | 0.9894 | 0.9929 | 282 |
| **1** | Mild NPDR | 0.5556 | 0.7500 | 54 |
| **2** | Moderate NPDR | 0.7500 | 0.7255 | 148 |
| **3** | Severe NPDR | 0.6000 | 0.2951 | 30 |
| **4** | Proliferative DR | 0.3778 | 0.7083 | 45 |

### Confusion Matrix (VAL)
```
            Predicted 0  Predicted 1  Predicted 2  Predicted 3  Predicted 4
True 0:             279            3            0            0            0
True 1:               1           30           23            0            0
True 2:               1            7          111           27            2
True 3:               0            0            7           18            5
True 4:               0            0           12           16           17
```

- **Weakest Grades by Recall**: Grade 4 (0.3778), Grade 1 (0.5556), Grade 3 (0.6000).
- **Primary Clinical Confusions**:
  1. Grade 2 $\rightarrow$ 3 (27 cases): Moderate classified as Severe. Clinically safe as both are referable.
  2. Grade 1 $\rightarrow$ 2 (23 cases): Mild classified as Moderate. Clinically safe (over-referral rather than under-referral).
  3. Grade 4 $\rightarrow$ 3 (16 cases): Proliferative classified as Severe NPDR (both referable).
  4. Grade 4 $\rightarrow$ 2 (12 cases): Proliferative classified as Moderate NPDR (both referable).
  5. Grade 3 $\rightarrow$ 2 (7 cases): Severe classified as Moderate NPDR (both referable).

---

## 5. Calibration (VAL Split)

- **Method**: Post-hoc Temperature Scaling on raw categorical logits:
  $$T^* = \arg\min_T \text{NLL}\left(\text{Softmax}\left(\frac{z}{T}\right), y\right)$$
- **Expected Calibration Error (ECE)** (10 uniform confidence bins):
  - **Before Calibration ($T=1.0000$)**: `0.0453` (4.53%)
  - **Optimal Temperature ($T^*$)**: `1.2962`
  - **After Calibration ($T=1.2962$)**: `0.0336` (3.36%)
  - **ECE Reduction**: `0.0117` (1.17 percentage points)
- **Artifact**: `ml/evaluation/val_report/reliability_diagram.png`
- **Deployment**: Integrated directly into `backend/src/retinaguard/services/inference.py` reading $T^*$ from `models/registry.json`.

---

## 6. Abstention Policy

- **Configuration File**: `configs/abstention.yaml` (all thresholds marked `NEEDS CLINICAL REVIEW`).
- **Policy**: Referable-band abstention on calibrated $P(\text{referable}) = P_2 + P_3 + P_4$.
  - If $P(\text{referable}) \in [0.30, 0.70]$, return `UNCERTAIN` (genuine model ambiguity near the referral boundary).
  - **Safety invariant**: never return a confident "No DR" when $P(\text{referable}) \ge 0.30$. The system fails to `UNCERTAIN`, never to a false negative.
- **Chosen Band**: `[0.30, 0.70]` — chosen on VAL only. Covers the decision boundary at 0.50 with ±0.20 margin.

### Measured on APTOS VAL (N=559)

| Metric | Value |
|---|---|
| `UNCERTAIN` rate | **7.33%** (41/559) |
| Automated (retained by abstention) | **92.67%** (518/559) |
| Retained referable sensitivity | **97.54%** (198/203 referable retained cases) |
| Retained referable specificity | **96.83%** (305/315 non-referable retained cases) |

- **Artifact**: `ml/evaluation/val_report/abstention_coverage.png`
- **Legacy thresholds** (confidence-based, superseded): recorded in `configs/abstention.yaml` for reference.

---

## 7. Out-of-Distribution (OOD) Gate

- **Feature Embeddings**: 1,280-dimensional penultimate representations from TRAIN split ($N=2,551$).
- **Algorithm**: Class centroids ($\mu_c$) and pooled precision matrix ($\Sigma^{-1}$) with Ledoit-Wolf shrinkage ($10^{-4}$).
- **Statistics Location**: `models/ood_stats_v1.npz` (SHA-256 prefix: `affa702776f5`).
- **OOD Threshold**: $84.82$ ($2 \times 99\text{th percentile of TRAIN Mahalanobis distance}$).
- **Empirical Rejection Rates**:
  - **In-Distribution Retinal Fundus (VAL, $N=100$)**: **0.0% rejected** (Mean distance: 49.47, Max: 82.63).
  - **Non-Fundus Synthetic Images**:
    - Blank White image: **100% REJECTED** (Optical spectrum: $R/B = 1.00$)
    - Black image: **100% REJECTED** (Optical spectrum: $R/B = 0.00$)
    - Natural Outdoor Scene: **100% REJECTED** (Optical spectrum: $R/B = 1.34$, Green-dominant)
    - Document / Text photo: **100% REJECTED** (Optical spectrum: $R/B = 1.00$)
    - Face / Skin photo: Passes optical spectrum filter ($R/B = 1.38$), flagged by Tier 2 deep feature distance.

---

## 8. Quality Gate

> [!NOTE]
> The blur check was updated to compute Laplacian variance on a fixed 512×512 crop of the
> retina (resolution-independent). The previous implementation computed on the full source
> image at variable resolution, causing a 0% pass rate on APTOS images from tabletop cameras.

- **Service**: Classical CV image quality filter (`backend/src/retinaguard/services/quality.py`).
- **Checks**:
  - **FOV**: Optical-spectrum filter (R/B ratio in expected fundus range).
  - **Resolution**: Minimum $1024 \times 1024$ source image.
  - **Blur**: Laplacian variance on fixed $512 \times 512$ crop of cropped retina. Threshold: **12.21** (flags bottom ~4% of TRAIN distribution, N=2,551). Config: `configs/quality_thresholds.yaml` (marked `NEEDS CLINICAL REVIEW`).

### VAL Pass Rates (New 512px Blur Check, N=559)

| Grade | Diagnostic Label | Pass / Total | Pass Rate |
|---|---|---|---|
| **0** | No DR | 230 / 282 | 81.6% |
| **1** | Mild NPDR | 48 / 54 | 88.9% |
| **2** | Moderate NPDR | 121 / 148 | 81.8% |
| **3** | Severe NPDR | 24 / 30 | 80.0% |
| **4** | Proliferative DR | 31 / 45 | 68.9% |
| **Overall** | | **454 / 559** | **81.2%** |

- **Threshold Derivation**: TRAIN blur distribution (512px): p1=7.55, p3=10.64, p4=12.21, p5=13.67. Threshold 12.21 = 4th percentile (flags ~4% of TRAIN).
- **Visual Verification**: 24 flagged TRAIN images shown in `ml/evaluation/val_report/flagged_train_blurry_grid.png` and visually confirmed to be genuinely soft-focus or hazy images.

---

## 9. Tiered Sight-Threatening Escalation Rule

> [!IMPORTANT]
> This two-tier safety rule ensures cases with elevated P(Grade ≥ 3) are always escalated,
> even if the top-1 prediction is Grade 2. The rule is **monotone** — it never lowers an
> urgency that was already set higher (verified by property tests over 50 × random Dirichlet
> draws × 4 context scenarios × 5 predicted grades).

**Two tiers, both chosen on VAL only** (`configs/clinical_rules.yaml`, marked `NEEDS CLINICAL REVIEW`):

| Tier | Condition | Resulting Minimum Urgency | Follow-up |
|---|---|---|---|
| **SOON** | $P(\text{Grade} \ge 3) \ge 0.05$ | At least `SOON` | ≤ 3 months |
| **URGENT** | $P(\text{Grade} \ge 3) \ge 0.30$ | At least `URGENT` | ≤ 1 month |

**Threshold derivation** (VAL N=559, true Grade≥3 = 75 cases):

| Threshold | Grade≥3 Sensitivity | Share of VAL escalated |
|---|---|---|
| 0.01 | 100.0% (75/75) | 46.7% |
| 0.02 | 100.0% (75/75) | 42.8% |
| **0.05 → SOON** | **97.3% (73/75)** | **36.1%** |
| 0.08 | 96.0% (72/75) | 33.6% |
| **0.30 → URGENT** | **90.7% (68/75)** | **22.9%** |
| 0.35 | 90.7% (68/75) | 21.8% |

- **SOON tier** (`threshold_soon = 0.05`): highest threshold achieving ≥95% Grade≥3 sensitivity (97.33%, 73/75).
- **URGENT tier** (`threshold_urgent = 0.30`): manageable escalation rate (22.9%) while catching 90.7% of Grade≥3 at the highest urgency.
- **Implementation**: `backend/src/retinaguard/rules/engine.py` Modifier 4 (tiered).
- **Tests**: 6-row table-driven test (`test_sight_threatening_escalation_table`) + property test (`test_property_escalation_never_lowers_urgency`) in `backend/tests/test_rules_engine.py`.

---

## 10. Gating & Abstention Metrics (Reporting Only — New Settings)

*All numbers below are measured with the new settings (referable-band abstention [0.30, 0.70] and tiered escalation thresholds 0.05/0.30). No thresholds were chosen from these results.*

| Cohort | N | QG Pass | OOD Reject | UNCERTAIN | Retained | Ret. Ref. Sens | Ret. Ref. Spec | G≥3 Sens (SOON) | G≥3 Sens (URGENT) |
|---|---|---|---|---|---|---|---|---|---|
| **APTOS VAL** | 559 | 80.86% (452) | 1.07% (6) | 7.33% (41) | 74.06% (414) | **0.9684** (153/158) | **0.9688** (248/256) | **0.9733** (73/75) | **0.9067** (68/75) |
| **APTOS Test** | 552 | 81.52% (450) | 0.91% (5) | 6.16% (34) | 76.09% (420) | **0.9576** (158/165) | **0.9255** (236/255) | **0.9726** (71/73) | **0.8630** (63/73) |
| **IDRiD All** | 516 | 98.64% (509) | 0.00% (0) | 7.95% (41) | 90.70% (468) | **0.7456** (211/283) | **0.9838** (182/185) | **0.8903** (138/155) | **0.6968** (108/155) |
| **IDRiD Test** | 103 | 98.06% (101) | 0.00% (0) | 10.68% (11) | 87.38% (90) | **0.7222** (39/54) | **0.9722** (35/36) | **0.8750** (28/32) | **0.6250** (20/32) |

> [!NOTE]
> IDRiD images pass the OOD gate (0.0% rejection) because IDRiD fundus images fall within
> the Mahalanobis feature distribution learned from APTOS TRAIN. The domain shift (QWK drop,
> referable sensitivity drop) is captured by per-class metrics, not by OOD rejection.
> IDRiD passes the quality gate at a higher rate (98.6%) than APTOS (80.9%) because IDRiD
> images are higher resolution and sharper than the legacy APTOS scans.

> [!WARNING]
> **IDRiD retained referable sensitivity (74.6% all / 72.2% test) is substantially lower
> than APTOS VAL (96.8%).** This confirms significant domain shift from APTOS tabletop
> cameras to the IDRiD camera system. Grade≥3 sensitivity at the SOON tier recovers to
> 90.7% (IDRiD all) / 87.5% (IDRiD test) for the most sight-threatening cases.

---

## 11. Final Test & External Benchmark Results

> [!WARNING]
> **DOMAIN SHIFT — CRITICAL LIMITATION**: Referable sensitivity drops from **96.4% on APTOS
> VAL** to **74.9% on IDRiD (all 516 images)** and **75.0% on IDRiD Official Test (N=103)**.
> QWK drops from **0.9188 to 0.7483**. IDRiD was captured at a different hospital
> (Dr. Ramanath Eye Hospital, Nanded) on different camera equipment. This demonstrates
> significant generalizability limitations. The model **must not** be deployed at sites
> using cameras and imaging protocols not represented in the training data.

*Evaluated once using `ml/scripts/final_test.py --confirm` on held-out test sets (one-time evaluation, no re-tuning permitted):*

### A. APTOS Held-Out Test Split ($N=552$)

| Metric | Measured Value | 95% Bootstrap CI |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.9003** | [0.8769, 0.9235] |
| **Referable DR Sensitivity** | **0.9602** | [0.9345, 0.9825] |
| **Referable DR Specificity** | **0.9202** | [0.8889, 0.9479] |
| **Referable DR ROC-AUC** | **0.9745** | [0.9612, 0.9846] |

**Per-Class Test Performance (APTOS)**:
- Grade 0: Recall = 0.9814, Precision = 0.9851
- Grade 1: Recall = 0.5263, Precision = 0.7317
- Grade 2: Recall = 0.8170, Precision = 0.7530
- Grade 3: Recall = 0.6000, Precision = 0.3333
- Grade 4: Recall = 0.3488, Precision = 0.6522

### B. External Benchmark: IDRiD All ($N=516$)

| Metric | Measured Value | 95% Bootstrap CI |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.7483** | [0.7013, 0.7862] |
| **Referable DR Sensitivity** | **0.7492** | [0.7045, 0.7965] |
| **Referable DR Specificity** | **0.9585** | [0.9271, 0.9846] |
| **Referable DR ROC-AUC** | **0.9396** | [0.9199, 0.9584] |

**Per-Class Performance (IDRiD All)**:
- Grade 0 ($N=168$): Recall = 0.6190, Precision = 0.8525
- Grade 1 ($N=25$): Recall = 0.7200, Precision = 0.1250 *(Wide CI due to $N=25$)*
- Grade 2 ($N=168$): Recall = 0.5119, Precision = 0.5478
- Grade 3 ($N=93$): Recall = 0.3226, Precision = 0.4918
- Grade 4 ($N=62$): Recall = 0.3871, Precision = 0.7500

### C. External Benchmark: IDRiD Official Test Split ($N=103$)

| Metric | Measured Value | 95% Bootstrap CI |
|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | **0.6317** | [0.5087, 0.7267] |
| **Referable DR Sensitivity** | **0.7500** | [0.6349, 0.8485] |
| **Referable DR Specificity** | **0.8974** | [0.7941, 0.9773] |
| **Referable DR ROC-AUC** | **0.8886** | [0.8229, 0.9421] |

---

## 12. Known Limitations & Clinical Warnings

> [!CAUTION]
> **THIS MODEL MUST NOT BE USED FOR CLINICAL DECISIONS WITHOUT REGULATORY CLEARANCE.**
> It is an experimental AI research prototype. All screening determinations must be reviewed
> by a certified ophthalmologist.

1. **Patient Leakage Mitigation**: APTOS 2019 contains no original patient IDs. While content-group perceptual hash stratification was applied to eliminate duplicate image leakage across splits, true patient-level segregation cannot be guaranteed without unblinded clinical records.

2. **Hardware Domain Shift — Major Limitation**: Training data was captured exclusively on hospital-grade tabletop fundus cameras (Aravind Eye Hospital). As measured on IDRiD (Dr. Ramanath Eye Hospital, different camera):
   - QWK drops from **0.9003** (APTOS test) to **0.7483** (IDRiD all).
   - Referable sensitivity drops from **96.0%** (APTOS test) to **74.9%** (IDRiD all).
   - This is a **known, measured limitation** that restricts deployment to sites using similar imaging equipment and protocols.

3. **No Smartphone or Portable Fundus Camera Validation**: The model has not been validated on low-cost handheld or smartphone-attachment cameras (e.g., Remidio, Volk iNview). Performance on such devices is unknown and could be substantially worse than the IDRiD domain-shift results.

4. **Severe Class Under-Representation**: ICDR Grade 4 (PDR) and Grade 1 (Mild) exhibit significantly lower sensitivity (0.35–0.53 on VAL) than Grade 0 (0.99), driven by severe dataset class imbalance. The sight-threatening escalation rule (Section 9) partially mitigates Grade 3/4 misses but does not eliminate them.

5. **Tiered Escalation Creates High Workload at SOON Tier**: The SOON tier (P(≥3) ≥ 0.05) escalates 36.1% of all VAL images (chosen for 97.3% Grade≥3 sensitivity). The URGENT tier (P(≥3) ≥ 0.30) escalates 22.9%. At deployment, this will increase referral workload substantially. Both thresholds are marked `NEEDS CLINICAL REVIEW` and must be calibrated against site-specific referral capacity.

6. **IDRiD Grade≥3 Sensitivity Drops at SOON Tier**: SOON-tier Grade≥3 sensitivity is 90.7% on IDRiD All (vs 97.3% on APTOS VAL) and 87.5% on IDRiD Test (103 images), reflecting domain shift. The URGENT tier (≥0.30) drops further to 69.7%/62.5%. Sites using different cameras than APTOS should not rely on these escalation thresholds without re-calibration.

6. **Quality Gate Passes Lower-Quality Grade 4 Images at a Reduced Rate**: Grade 4 images have the lowest QG pass rate (68.9% on VAL). Some PDR images with severe disc neovascularization may appear diffuse/hazy and be flagged as blurry. These images should still receive clinical review.

7. **Calibration on APTOS Only**: Temperature scaling ($T^* = 1.2962$) was calibrated on the APTOS VAL split. Its ECE on IDRiD images has not been directly measured.

8. **Regulatory Status**: This software is an experimental AI research prototype. It is **not** approved as a medical device by the Central Drugs Standard Control Organisation (CDSCO, India), the US FDA, or under European CE marking regulations. All screening determinations must be reviewed by a certified ophthalmologist.

---

## 13. Evaluation Artifacts

| Artifact | Path |
|---|---|
| Confusion Matrix (VAL) | `ml/evaluation/val_report/confusion_matrix.png` |
| Reliability Diagram (VAL) | `ml/evaluation/val_report/reliability_diagram.png` |
| Abstention Coverage Plot (VAL) | `ml/evaluation/val_report/abstention_coverage.png` |
| Flagged TRAIN Images (Blur Grid) | `ml/evaluation/val_report/flagged_train_blurry_grid.png` |
| VAL Predictions CSV | `ml/evaluation/val_report/predictions_val.csv` |
| APTOS Test Predictions CSV | `ml/evaluation/test_report/predictions_aptos_test.csv` |
| IDRiD All Predictions CSV | `ml/evaluation/test_report/predictions_idrid_all.csv` |
| IDRiD Test Predictions CSV | `ml/evaluation/test_report/predictions_idrid_test.csv` |
| Gating & Abstention JSON (old settings) | `ml/evaluation/gating_and_abstention_report.json` |
| New Settings Report JSON | `ml/evaluation/new_settings_report.json` |
| Final Test Results JSON | `ml/evaluation/test_report/final_test_results.json` |
| OOD Statistics | `models/ood_stats_v1.npz` |
| Model Registry | `models/registry.json` |
| ONNX Export | `ml/exports/retinaguard_dr_v1.onnx` |
