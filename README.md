# 🧠 Advancing Continuous Prediction for Acute Kidney Injury via Multi-task Learning: Towards Better Clinical Applicability (TOBCA)

(Early Accepted -> https://ieeexplore.ieee.org/abstract/document/10834563)

# ⚠️ Dataset Version & Reproducibility

This repository contains the official, fully updated implementation built on MIMIC‑IV v3.1.

* Version migration.  
  -The study was originally prototyped on MIMIC‑IV v2.0 and re‑run on v3.1 during peer‑review.  
  -The analytic pipeline (feature engineering, splitting strategy, evaluation) is unchanged, but patient‑level membership inevitably differs because the two releases do not contain identical records.

* Why the cohorts diverge.  
  -A subset of patients used for the temporal validation split appears only in v3.1. They were initially treated as 2020‑2022 admissions; later inspection showed that some are simply newly published records and not strictly time‑delimited.

* Impact on results.  
  - Main experiments & ablation studies: fully reproduced on v3.1 and unaffected by the version change.  
  - Temporal validation in the Appendix: cohort shift means the exact numbers in the paper cannot be recreated. Interpret those figures with caution; they are illustrative, not central to the conclusions.

> ✅ Author‑endorsed codebase. Use this repository for any replication or extension—every primary result has been verified under MIMIC‑IV v3.1.  
> ❗ Caveat. Only the Appendix’s temporal‑validation metrics are version‑sensitive and are not directly comparable to the headline results.

# 🏥 MIMIC-IV v3.1 Database

This project is based on the **MIMIC-IV v3.1** dataset, a large-scale, publicly available electronic health record (EHR) database developed by the MIT Laboratory for Computational Physiology.

### 📌 What is MIMIC-IV?
- **MIMIC (Medical Information Mart for Intensive Care)** is a comprehensive EHR dataset containing de-identified data from **over 300,000 ICU and hospital admissions** at the Beth Israel Deaconess Medical Center.
- **Version 4 (MIMIC-IV)** separates hospital-level and ICU-level data for cleaner integration and research.
- **v3.1** is the latest stable release as of this repository, with expanded coverage and fixed inconsistencies from prior versions.

### 📦 Key Features
- Demographics, vital signs, lab tests, medications, diagnoses, and procedures
- Structured in **PostgreSQL-style relational tables**
- Separated into modules: `hosp`, `icu`, `note`, `cxr`, and more
- Supports longitudinal studies and risk modeling in clinical settings

### 🔐 Data Access Requirements
To access and use MIMIC-IV:
1. Complete CITI “Data or Specimens Only Research” training (HIPAA compliance)
2. Register and sign a data use agreement on PhysioNet

➡️ **Access the dataset**: [https://physionet.org/content/mimiciv/3.1](https://physionet.org/content/mimiciv/3.1)

> ⚠️ Note: All preprocessing and cohort selection in this project assumes **MIMIC-IV v3.1 structure**. Earlier versions (e.g., v2.0) are **not compatible**.

# Stack  
 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>

# Contents
AKI - Optuna - Model_C.ipynb is including an AKI - Model C.pt development by Optuna. <br/>
AKI - Calibration - Model C.csv summarizes outcome of AKI - Calibration - Model C.ipynb.
