# Advancing Continuous Prediction for Acute Kidney Injury via Multi-task Learning: Towards Better Clinical Applicability (TOBCA)

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

# Stack  
 <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/pytorch-EE4C2C?style=flat&logo=pytorch&logoColor=white"/>

# Database
MIMIC-IV : https://physionet.org/content/mimiciv/3.1

# Software
Python 3.11.7 <br/>
PyTorch 2.1.2 <br/>
CUDA version 12.1 <br/>
CUDNN 8.9.7 <br/>
IBM SPSS Statistics 27

# Contents
AKI - Optuna - Model_C.ipynb is including an AKI - Model C.pt development by Optuna. <br/>
AKI - Calibration - Model C.csv summarizes outcome of AKI - Calibration - Model C.ipynb.
