# BioNLP Project: Reproducing Dosing Error Risk Stratification in Clinical Trials

This repository contains the replication study and codebase for reproducing the baseline results of the paper **"Early Risk Stratification of Dosing Errors in Clinical Trials Using Machine Learning"** (arXiv:2602.22285), presented as part of the **CT-DEB 2026** (Clinical Trial Dosing Errors Benchmark) shared task at the CL4Health workshop (LREC-COLING 2026).

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

All the core logic is implemented in the [temp_clone](file:///home/david/Desktop/Developer/BioNLP/temp_clone) subdirectory, which is structured as follows:

* **[aidose/baselines](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines)**: Python scripts containing baseline models.
  * [main.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/main.py): Entry point to run training, optimization, and evaluation.
  * [our_xgboost.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/our_xgboost.py): XGBoost classifier using categorical metadata features.
  * [our_clinicalModernBERT.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/our_clinicalModernBERT.py): ModernBERT encoder fine-tuned on clinical trial protocol texts.
  * [LateFusionMultimodal.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/LateFusionMultimodal.py): Model that combines text and metadata predictions using late fusion.
  * [preprocessing.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/preprocessing.py): Preprocessing logic for text and tabular attributes.
  * [CustomTrainer.py](file:///home/david/Desktop/Developer/BioNLP/temp_clone/aidose/baselines/CustomTrainer.py): Subclassed Hugging Face `Trainer` implementing balanced batch sampling.
* **[Figures](file:///home/david/Desktop/Developer/BioNLP/temp_clone/Figures)**: Jupyter Notebooks for figures and risk analysis.
  * [Paper_figures.ipynb](file:///home/david/Desktop/Developer/BioNLP/temp_clone/Figures/Paper_figures.ipynb): Distribution and exploratory data analysis.
  * [Stratification.ipynb](file:///home/david/Desktop/Developer/BioNLP/temp_clone/Figures/Stratification.ipynb): Risk stratification computations across phases and enrollment.
* **[resources](file:///home/david/Desktop/Developer/BioNLP/temp_clone/resources)**: Caches for local data and model runs.
  * [CT-DOSING-ERRORS/0.2.3](file:///home/david/Desktop/Developer/BioNLP/temp_clone/resources/CT-DOSING-ERRORS/0.2.3): Pre-processed train, validation, and test datasets.
  * [baselines](file:///home/david/Desktop/Developer/BioNLP/temp_clone/resources/baselines): Cached model checkpoints, Optuna parameters, and prediction outputs.

---

## 🚀 Execution Guide

Make sure the virtual environment [venv](file:///home/david/Desktop/Developer/BioNLP/.venv) is activated. The `aidose` package has been installed in editable mode (`pip install -e temp_clone`).

### 1. Verification & Dry Runs
To verify that the code and dependencies work without executing full long-running experiments, run a short Optuna search for XGBoost:
```bash
PYTHONPATH=temp_clone python temp_clone/aidose/baselines/main.py --model XGBoost --num_trials 3
```

### 2. Full Reproductions
To retrain or re-evaluate the baselines from scratch:

* **XGBoost Baseline (Metadata)**:
  ```bash
  PYTHONPATH=temp_clone python temp_clone/aidose/baselines/main.py --model XGBoost --num_trials 200
  ```

* **ClinicalModernBERT Baseline (Text)**:
  ```bash
  PYTHONPATH=temp_clone python temp_clone/aidose/baselines/main.py --model ClinicalModernBERT --num_epoch 10
  ```

* **Late Fusion Multimodal (Combined)**:
  ```bash
  PYTHONPATH=temp_clone python temp_clone/aidose/baselines/main.py --model LateFusionModel --late_fusion_num_trials 100
  ```

---

## 📈 Baseline Metrics (Test Set Results)

Based on the cached baseline outputs inside `temp_clone/resources/baselines/LateFusionMultimodal/wilson_label/test_predictions_after_calibration.csv`:

| Baseline Model | Feature Modality | Test ROC-AUC | Test PR-AUC (Average Precision) |
| :--- | :--- | :---: | :---: |
| **ClinicalModernBERT** | Text Only | **0.8124** | **0.1651** |
| **XGBoost** | Tabular/Metadata Only | **0.8499** | **0.2159** |
| **LateFusionMultimodal** | Multimodal (Text + Tabular) | **0.8600** | **0.2390** |
