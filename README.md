# 🧠 Advancing Continuous Prediction for Acute Kidney Injury via Multi-task Learning: Towards Better Clinical Applicability (TOBCA)

📝 Early Accepted to *IEEE Journal of Biomedical and Health Informatics*  
🔗 [View on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10960515)  
📌 DOI: [10.1109/JBHI.2025.3559677](https://doi.org/10.1109/JBHI.2025.3559677)

---

## 🏥 MIMIC-IV v3.1 Database

This project is based on the **MIMIC-IV v3.1** dataset, a large-scale, publicly available electronic health record (EHR) database developed by the MIT Laboratory for Computational Physiology.

### 📌 What is MIMIC-IV?
- **MIMIC (Medical Information Mart for Intensive Care)** is a comprehensive EHR dataset containing de-identified data from over 300,000 ICU and hospital admissions.
- **MIMIC-IV** separates hospital-level and ICU-level data for cleaner integration.
- **v3.1** is the latest stable release used in this repository.

### 📦 Key Features
- Demographics, vital signs, lab tests, medications, diagnoses, and procedures
- Structured in PostgreSQL-style relational tables
- Modules include: `hosp`, `icu`, `note`, `cxr`, etc.
- Suitable for longitudinal and real-world clinical risk modeling

### 🔐 Data Access
To use MIMIC-IV:
1. Complete CITI “Data or Specimens Only Research” training
2. Sign a data use agreement via PhysioNet

➡️ Access the dataset: [https://physionet.org/content/mimiciv/3.1](https://physionet.org/content/mimiciv/3.1)

> 🔎 All preprocessing assumes **v3.1** structure. Other versions (e.g., v2.0) are not compatible.

---

## ⚠️ Dataset Version & Reproducibility

This repository contains the **official, fully updated implementation based on MIMIC-IV v3.1**.

### 🧪 Version migration
- The study was originally prototyped on **MIMIC-IV v2.0**, and updated to **v3.1** during peer-review.
- The core logic (feature engineering, splitting, evaluation) is identical, but **patient cohort composition differs** due to dataset updates.

### 🧬 Why cohorts differ
- The *temporal validation* cohort included patients present in v3.1 but not in v2.0.
- Some were incorrectly assumed to represent 2020–2022 admissions but were simply newly added records.

### 🧾 Impact on results
- **Main experiments & ablation studies** are **unchanged** and fully reproducible under v3.1.
- **Temporal validation in Appendix** is sensitive to version shifts and **should be interpreted with caution**.

> ✅ This repository reflects the author-endorsed final version on MIMIC-IV v3.1.  
> ❗ Only the Appendix’s temporal validation results are version-sensitive and not directly comparable to headline results.

---

## 🧱 Stack

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/> <img src="https://img.shields.io/badge/Optuna-70AADB?style=flat"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/Numpy-013243?style=flat&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/> <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=flat&logo=matplotlib&logoColor=white"/> <img src="https://img.shields.io/badge/Seaborn-0C5A6B?style=flat"/> <img src="https://img.shields.io/badge/WandB-FFBE00?style=flat&logo=wandb&logoColor=black"/> <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/> <img src="https://img.shields.io/badge/SPSS-005BAB?style=flat&logo=ibm&logoColor=white"/>

---

## 📄 Citation

If this work was helpful in your research, please consider citing the following paper:

<details>
<summary><strong>📚 Click to show BibTeX</strong></summary>

```bibtex
@article{kim2025advancing,
  title={Advancing Continuous Prediction for Acute Kidney Injury via Multi-task Learning: Towards Better Clinical Applicability},
  author={Kim, Hyunwoo and Lee, Sung Woo and Kim, Su Jin and Han, Kap Su and Lee, Sijin and Song, Juhyun and Lee, Hyo Kyung},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE},
  doi={10.1109/JBHI.2025.3559677},
  url={https://ieeexplore.ieee.org/abstract/document/10960515}
}
