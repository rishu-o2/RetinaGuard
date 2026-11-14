# Eye Care Facility Directory Maintenance Guide

## Overview
RetinaGuard provides localized referral direction for patients flagged as **Referable** (Moderate NPDR or worse) or **Uncertain** (requiring clinical evaluation). The facility directory is stored in `configs/facilities.csv`.

In accordance with clinical safety principles and data integrity requirements:
- The initial directory is committed with **headers only**; **no artificial or hallucinated facilities** may be added.
- All entries must be verified against official health authority registries before inclusion.
- No GPS coordinates are stored, and no patient location is ever persisted to patient records or the audit ledger.

---

## Schema (`configs/facilities.csv`)

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `state` | string | Indian State or Union Territory | `Maharashtra` |
| `district` | string | Administrative district name | `Pune` |
| `name` | string | Official name of the medical institution | `District Hospital Aundh` |
| `type` | string | Institution classification (`PHC`, `CHC`, `DH`, `Tertiary`, `Private NGO`) | `DH` |
| `address` | string | Physical postal address | `Aundh, Pune, Maharashtra 411027` |
| `phone` | string | Verified emergency or outpatient contact number | `+91-20-27282929` |
| `source_url` | string | Official public portal / directory source link | `https://nhm.gov.in/...` |

---

## Approved Data Sources

Only authoritative government registries or verified ophthalmological societies may be used:

1. **National Health Mission (NHM) Health Facility Registry (HFR)**
   - Portal: [https://facility.abdm.gov.in](https://facility.abdm.gov.in)
   - Filter by district, facility type (`DH`, `SDH`, `CHC`), and active ophthalmology department.

2. **National Programme for Control of Blindness & Visual Impairment (NPCB&VI)**
   - Portal: [https://npcbvi.mohfw.gov.in](https://npcbvi.mohfw.gov.in)
   - Annual reports and state cell directory listings of designated eye care centers and vision centers.

3. **All India Ophthalmological Society (AIOS) Accredited Eye Hospitals**
   - Portal: [https://aios.org](https://aios.org)
   - Registered charitable and tertiary eye hospitals with diabetic retinopathy laser and vitrectomy capabilities.

---

## Verification & Addition Workflow

1. **Verify Facility Existence & Services**:
   - Ensure the center has an operational ophthalmic unit capable of diabetic retinopathy evaluation (slit-lamp biomicroscopy, indirect ophthalmoscopy, or fundus fluorescein angiography/laser).
2. **Standardize Naming**:
   - Use official district spelling recognized by the Government of India Census / LGD (Local Government Directory).
3. **Validate Phone Numbers**:
   - Test or confirm active telephone numbers via official portal records.
4. **Pull Request & Audit**:
   - Submit changes via a signed Git commit referencing the source document or government notification.
