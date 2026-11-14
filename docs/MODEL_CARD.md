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
| v1 vs v2 Comparison JSON | `ml/evaluation/comparison_v1_v2/comparison_v1_v2.json` |

---

## 14. v2 (robust augmentation): evaluated, not promoted

> [!NOTE]
> **Status**: Evaluated and committed (`ml/scripts/compare_v1_v2.py`, commit `9472995`).
> **Decision**: Not promoted. **Model `v1` remains the active model** in `models/registry.json`.

A robustly trained checkpoint (`ml/runs/robust_cpu_b0_256/best.pt`, Epoch 20) with camera-invariance augmentations (color jitter, blur, zoom, gamma adjustments) was evaluated as candidate version `v2` without modifying any `v1` artifact, threshold, or registry entry.

An operating-point-free and pipeline-level comparison was performed across the APTOS Validation split ($N=559$), IDRiD All ($N=516$), and IDRiD Official Test ($N=103$). All metrics reported below are directly measured numbers from `compare_v1_v2.py` (with 1,000-resample bootstrap 95% confidence intervals).

### A. Operating-Point-Free Comparison (Referable ROC-AUC & Matched Points)

#### 1. In-Domain APTOS VAL ($N = 559$, 223 Referable, 336 Non-Referable)
- **Referable ROC-AUC**:
  - Model v1: **0.9890** [0.9826, 0.9940]
  - Model v2: **0.9865** [0.9786, 0.9926]
  - Paired difference $\Delta$ (v2 − v1): **−0.0025** [−0.0088, +0.0028] (CI crosses 0; difference is not statistically significant)
- **Sensitivity @ Matched Specificity**:
  - @ Specificity = 0.80: v1 = **1.0000** [1.0000, 1.0000], v2 = **1.0000** [1.0000, 1.0000]
  - @ Specificity = 0.90: v1 = **0.9776** [0.9488, 1.0000], v2 = **0.9821** [0.9417, 1.0000]
- **Specificity @ Matched Sensitivity**:
  - @ Sensitivity = 0.90: v1 = **0.9643** [0.9395, 0.9874], v2 = **0.9643** [0.9326, 0.9799]
  - @ Sensitivity = 0.95: v1 = **0.9524** [0.8969, 0.9732], v2 = **0.9315** [0.8929, 0.9660]

#### 2. External Benchmark: IDRiD All ($N = 516$, 323 Referable, 193 Non-Referable)
- **Referable ROC-AUC**:
  - Model v1: **0.9396** [0.9197, 0.9570]
  - Model v2: **0.9270** [0.9041, 0.9462]
  - Paired difference $\Delta$ (v2 − v1): **−0.0126** [−0.0265, +0.0022] (CI crosses 0)
- **Sensitivity @ Matched Specificity**:
  - @ Specificity = 0.80: v1 = **0.8978** [0.8600, 0.9385], v2 = **0.8762** [0.8217, 0.9152]
  - @ Specificity = 0.90: v1 = **0.8483** [0.7714, 0.8900], v2 = **0.8080** [0.7605, 0.8616]
- **Specificity @ Matched Sensitivity**:
  - @ Sensitivity = 0.90: v1 = **0.7979** [0.7050, 0.8867], v2 = **0.7513** [0.6138, 0.8351]
  - @ Sensitivity = 0.95: v1 = **0.5855** [0.4793, 0.7578], v2 = **0.4767** [0.3711, 0.6802]

#### 3. External Benchmark: IDRiD Official Test Split ($N = 103$, 64 Referable, 39 Non-Referable)
- **Referable ROC-AUC**:
  - Model v1: **0.8886** [0.8202, 0.9469]
  - Model v2: **0.8558** [0.7822, 0.9249]
  - Paired difference $\Delta$ (v2 − v1): **−0.0329** [−0.0812, +0.0091]
- **Sensitivity @ Matched Specificity**:
  - @ Specificity = 0.80: v1 = **0.7969** [0.6769, 0.9091], v2 = **0.7969** [0.6764, 0.8871]
  - @ Specificity = 0.90: v1 = **0.7188** [0.5538, 0.8750], v2 = **0.7344** [0.6229, 0.8621]
- **Specificity @ Matched Sensitivity**:
  - @ Sensitivity = 0.90: v1 = **0.5385** [0.2703, 0.8183], v2 = **0.3846** [0.0555, 0.7255]
  - @ Specificity = 0.95: v1 = **0.3846** [0.2272, 0.6906], v2 = **0.1282** [0.0286, 0.5111]

---

### B. IDRiD (All 516): 5x5 Confusion Matrices and Direction of Extra Referable Calls

#### Model v1 Confusion Matrix & Recall
```
               Pred 0   Pred 1   Pred 2   Pred 3   Pred 4    Recall
True Grade 0: [  104       56        5        2        1 ]   61.90%  (104/168)
True Grade 1: [    7       18        0        0        0 ]   72.00%  (18/25)
True Grade 2: [   10       61       86       10        1 ]   51.19%  (86/168)
True Grade 3: [    1        6       50       30        6 ]   32.26%  (30/93)
True Grade 4: [    0        3       16       19       24 ]   38.71%  (24/62)
```

#### Model v2 Confusion Matrix & Recall
```
               Pred 0   Pred 1   Pred 2   Pred 3   Pred 4    Recall
True Grade 0: [   65       67       34        1        1 ]   38.69%  (65/168)
True Grade 1: [    6       16        3        0        0 ]   64.00%  (16/25)
True Grade 2: [    4       28      131        5        0 ]   77.98%  (131/168)
True Grade 3: [    0        2       69       18        4 ]   19.35%  (18/93)
True Grade 4: [    0        1       27       25        9 ]   14.52%  (9/62)
```

#### Breakdown of Extra Referable Calls
- **True Grade 0 predicted as Referable ($\ge 2$)**:
  - Model v1: **8 / 168** (4.76%)
  - Model v2: **36 / 168** (21.43%) — **+28 false positives**
- **True Grade 1 predicted as Referable ($\ge 2$)**:
  - Model v1: **0 / 25** (0.00%)
  - Model v2: **3 / 25** (12.00%) — **+3 false positives**
- **Total False Referable Calls on Grade 0–1**:
  - Model v1: **8 / 193** (4.15%)
  - Model v2: **39 / 193** (20.21%) — **+31 net false positives** (near 5-fold increase)
- **True Referable Cases Correctly Identified (Grade 2–4 predicted $\ge 2$)**:
  - Model v1: **242 / 323** (74.92%)
  - Model v2: **288 / 323** (89.16%) — **+46 additional true referable cases captured**

---

### C. Tier Loads and Over-Triage Rates

*Tiers evaluated at $P(\text{Grade} \ge 3) \ge 0.05$ (SOON) and $P(\text{Grade} \ge 3) \ge 0.30$ (URGENT):*

| Cohort | Metric | Model v1 | Model v2 |
|---|---|---|---|
| **APTOS VAL ($N=559$)** | Share Escalated @ SOON | 36.1% (202/559) | 38.1% (213/559) |
| | Share Escalated @ URGENT | 22.9% (128/559) | 20.8% (116/559) |
| | Grade $\ge 3$ Sensitivity @ SOON | **97.3%** (73/75) | **96.0%** (72/75) |
| | Grade $\ge 3$ Sensitivity @ URGENT | **90.7%** (68/75) | **90.7%** (68/75) |
| | Over-Triage @ SOON (Grade 0–1 escalated) | **5.7%** (19/336) | **7.1%** (24/336) |
| | Over-Triage @ URGENT (Grade 0–1 escalated) | **0.3%** (1/336) | **0.9%** (3/336) |
| **IDRiD All ($N=516$)** | Share Escalated @ SOON | 45.2% (233/516) | 57.9% (299/516) |
| | Share Escalated @ URGENT | 28.5% (147/516) | 24.6% (127/516) |
| | Grade $\ge 3$ Sensitivity @ SOON | **89.0%** (138/155) | **95.5%** (148/155) |
| | Grade $\ge 3$ Sensitivity @ URGENT | **69.7%** (108/155) | **61.3%** (95/155) |
| | Over-Triage @ SOON (Grade 0–1 escalated) | **7.8%** (15/193) | **25.4%** (49/193) |
| | Over-Triage @ URGENT (Grade 0–1 escalated) | **2.1%** (4/193) | **3.6%** (7/193) |
| **IDRiD Test ($N=103$)** | Share Escalated @ SOON | 47.6% (49/103) | 63.1% (65/103) |
| | Share Escalated @ URGENT | 25.2% (26/103) | 21.4% (22/103) |
| | Grade $\ge 3$ Sensitivity @ SOON | **87.5%** (28/32) | **96.9%** (31/32) |
| | Grade $\ge 3$ Sensitivity @ URGENT | **62.5%** (20/32) | **50.0%** (16/32) |
| | Over-Triage @ SOON (Grade 0–1 escalated) | **15.4%** (6/39) | **35.9%** (14/39) |
| | Over-Triage @ URGENT (Grade 0–1 escalated) | **0.0%** (0/39) | **2.6%** (1/39) |

---

### D. Explanation of v1 IDRiD Discrepancies and Code Path Reconciliation

During early evaluation scripts, two metrics for model v1 on IDRiD All showed apparent numerical differences between reporting scripts:
1. **Retained Referable Sensitivity: `0.7378` vs `0.7456`**:
   - **Original Pipeline Path (`0.7456`)**: Applied the frontend Quality Gate filter (where 7 poor illumination/focus images failed, leaving 509 images) and evaluated clinical abstention with the safety override `((0.30 <= p_ref <= 0.70) | ((p_ref >= 0.30) & (y_pred == 0)))`. On the retained cases ($N=468$, 283 true referable), referable decisions were determined by the model's categorical grade prediction: $\hat{y} \ge 2$. This yielded $211 / 283 = \mathbf{0.7456}$.
   - **Shortcut Path (`0.7378`)**: Evaluated raw images without the Quality Gate ($N=516$), evaluated abstention only as `0.30 <= p_ref <= 0.70` without the safety override, and evaluated referable strictly as $P(\text{ref}) > 0.70$ rather than $\hat{y} \ge 2$. This yielded $211 / 286 = \mathbf{0.7378}$.
2. **Grade $\ge 3$ Sensitivity at SOON: `0.9091` vs `0.8903`**:
   - **Original Pipeline Path (`0.8903`)**: Evaluated the escalation tier over the entire cohort of true Grade $\ge 3$ cases ($N=155$). With 138 cases escalated at $P(\text{Grade} \ge 3) \ge 0.05$, sensitivity is $138 / 155 = \mathbf{0.8903}$.
   - **Shortcut Path (`0.9091`)**: Evaluated sensitivity solely on the retained subset, inadvertently excluding the 12 Grade $\ge 3$ cases that landed in the abstention band. The denominator shrank to 143, yielding $130 / 143 = \mathbf{0.9091}$.

*Reconciliation*: `ml/scripts/compare_v1_v2.py` now runs both the exact end-to-end clinical pipeline path and the standalone cohort path side-by-side, explicitly labeling denominators.

---

### E. Clinical Takeaway and Promotion Decision

**Plain Summary of Findings**:
1. **No Operating-Point-Free Advantage**: Model v2 exhibits essentially identical ROC-AUC to v1 on in-domain VAL ($0.9865$ vs $0.9890$, $\Delta = -0.0025$, 95% CI: [−0.0088, +0.0028]) and slightly lower ROC-AUC on IDRiD All ($0.9270$ vs $0.9396$, $\Delta = -0.0126$, 95% CI: [−0.0265, +0.0022]).
2. **Aggressive Upward Shift Rather Than Better Discrimination**: While v2 increases raw referable recall on IDRiD (referable sensitivity rises from 74.9% to 89.2% due to Grade 2 recall jumping from 51.2% to 78.0%), this gain is achieved via an upward calibration shift rather than superior lesion discrimination.
3. **Severe Over-Triage Burden**: As a direct consequence of this shift, v2 incurs a near 5-fold increase in false positives on IDRiD Grade 0–1 (rising from 4.15% to 20.21%). At the SOON escalation tier, over-triage of non-referable eyes surges from 7.77% to 25.39%, causing 57.9% of all screened patients on IDRiD to be escalated. In a resource-constrained rural Indian PHC, this volume of false escalations would overwhelm tertiary ophthalmology referral pathways.

**Decision**:
Model **v2 is NOT promoted**. **Model `v1` remains the active model** in `models/registry.json`. No thresholds, abstention bands, or active endpoints have been changed.

