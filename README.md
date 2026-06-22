# BioNLP Project: Reproducing Dosing Error Risk Stratification in Clinical Trials

This repository contains the replication study and codebase for reproducing the baseline results of the paper **"Early Risk Stratification of Dosing Errors in Clinical Trials Using Machine Learning"** ([arXiv:2602.22285](https://arxiv.org/abs/2602.22285)), presented as part of the **CT-DEB 2026** (Clinical Trial Dosing Errors Benchmark) shared task at the CL4Health workshop (LREC-COLING 2026).

---

## 📖 Scientific Context & Problem Definition

### The BioNLP Challenge
Medication and dosing errors in clinical trials (CTs) represent a significant patient safety risk and can invalidate trial results. Traditionally, safety monitoring happens retrospectively or during the trial. This work focuses on **early, trial-level risk stratification**—predicting the likelihood of a trial exhibiting high rates of dosing errors *before* the trial begins, using only information available in the protocol registry.

### Dataset & Class Imbalance
The CT-DEB dataset contains over **42,000 clinical trial protocols** curated from ClinicalTrials.gov. Dosing error labels are derived by mapping Adverse Event terms to MedDRA terminology (specifically looking for medication error and overdose/underdose categories).
* **Imbalance**: Only about **5%** of trials contain dosing errors (positive class `wilson_label = 1`), making this a highly imbalanced binary classification problem.
* **Input Modalities**:
  * **Textual**: Protocol narratives (brief summaries, detailed descriptions, primary/secondary outcome measures).
  * **Categorical/Numerical Metadata**: Trial phase, enrollment count, study type, allocation, masking, etc.

---

## 🔄 BioNLP Workflow

This project implements a complete BioNLP workflow:
1. **Understand**: Analyze the clinical protocol text processing, MedDRA hierarchy mappings, and the formulation of the `wilson_label` target.
2. **Reproduce**: Run the authors' baseline models (XGBoost, ClinicalModernBERT, and Late Fusion) on the dataset.
3. **Evaluate**: Evaluate the models using ranking and classification metrics (ROC-AUC, Precision-Recall AUC/AP, F1-Score).
4. **Analyze**: Compare reproduced metrics with the paper's reported numbers and analyze performance discrepancies or subgroup vulnerabilities (e.g., risk stratification performance across different phases and enrollment bins).
5. **Report**: Synthesize findings in a scientific paper and presentation slides.

---

## 🛠️ Codebase Structure

We have built a custom, enterprise-grade machine learning pipeline to run efficiently on local hardware. The core scripts are:

* `custom_xgboost.py`: Processes structured metadata, handles missing values, optimizes hyperparameters via Optuna, and trains an XGBoost classifier with Isotonic Regression probability calibration.
* `custom_transformer.py`: Fine-tunes the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` transformer on the unstructured clinical text with weighted cross-entropy to handle the 5% class imbalance.
* `custom_early_fusion.py`: A novel multimodal architecture that extracts 768-dimensional textual embeddings from PubMedBERT and fuses them directly with the structured metadata, passing the combined feature space into a unified XGBoost model.
* `generate_shap.py`: Evaluates the trained metadata XGBoost model using SHAP (SHapley Additive exPlanations) to provide Explainable AI (XAI) insights into exactly how variables like "Phase 4" or "Enrollment Count" drive the final risk score.
* `scratch.py`: Sandbox script for local testing and validation.

---

## 🚀 Execution Guide

Activate your virtual environment and run the custom scripts directly. Ensure that the CT-DEB dataset is available locally in the `temp_clone/resources` directory (note: the `temp_clone` dataset submodule is excluded from this repository to keep it lightweight, you must download it yourself).

### Training and Evaluating
Run the independent pipelines:
```bash
python custom_xgboost.py
python custom_transformer.py
```

### Multimodal Early Fusion
After running the transformer, run the Early Fusion architecture to combine text embeddings with structured data:
```bash
python custom_early_fusion.py
```

### Explainable AI (SHAP)
Generate the SHAP summary plot (`shap_summary.png`) to understand feature importances:
```bash
python generate_shap.py
```

---

## 📈 Final Model Performance (Test Set Results)

Our custom implementations successfully replicated the literature baselines and applied rigorous evaluation metrics to combat the 5% class imbalance. Notably, Isotonic Regression was used to achieve excellent Brier Scores.

| Model Architecture | Modality | ROC-AUC | Brier Score | F1-Score | Recall | Precision | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PubMedBERT** | Text Only | 0.821 | 0.078 | 0.261 | 0.461 | 0.182 | 0.872 |
| **XGBoost** | Metadata Only | 0.846 | **0.042** | 0.297 | 0.477 | 0.216 | 0.889 |
| **Early-Fusion** | Text Embeddings + Metadata | 0.847 | 0.042 | 0.290 | **0.538** | 0.198 | 0.870 |
| **Late-Fusion** | Averaged Predictions | **0.848** | 0.049 | **0.311** | 0.332 | **0.292** | **0.927** |

*Note: Late-Fusion proved to be the most robust architecture, as the Early-Fusion model experienced dimensionality challenges when injecting 768 continuous text features into the XGBoost split-finding algorithms.*
