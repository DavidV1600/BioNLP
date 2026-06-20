# Dosing Error Detection in Clinical Trial Protocols

## 1. Introduction
The CT-DEB (Clinical Trial Dosing Error Benchmark) shared task focuses on identifying medication dosing errors in clinical trial protocols using Natural Language Processing (NLP) and Machine Learning techniques. The dataset contains roughly 42,000 clinical trial narratives from ClinicalTrials.gov. This task is framed as a highly imbalanced binary classification problem, where only about ~5% of the trials contain dosing errors.

### Objective
The goal of this project is to implement a complete BioNLP workflow from scratch to identify these dosing errors, bypassing the original source code, and directly applying models to the dataset to establish custom baselines. We compare traditional ML approaches against modern Transformer-based NLP models.

## 2. Methodology
Our workflow approaches the problem via two distinct paradigms:
1. **Metadata & Tabular Features (XGBoost)**
2. **Unstructured Text Processing (Transformer - PubMedBERT)**

### 2.1 XGBoost Baseline
The tabular and structural metadata from the dataset were preprocessed:
- Features such as `healthyVolunteers` and `oversightHasDmc` were cast to booleans.
- Categorical list features (e.g., `armGroupTypes`, `phases`, `interventionTypes`) were expanded using multi-hot (count) encoding.
- Text features were dropped to focus purely on structured inputs.
- To handle the heavy class imbalance (1,352 positive vs. 28,126 negative instances), the `scale_pos_weight` parameter was initialized based on inverse class frequencies (approximately `20.8`).
- Hyperparameters were optimized using Optuna (TPE Sampler) over the validation split, tuning metrics such as `n_estimators`, `max_depth`, `learning_rate`, and `scale_pos_weight`.
- Post-processing included Isotonic Calibration to convert the raw XGBoost outputs into well-calibrated probabilities.

### 2.2 Transformer Baseline (PubMedBERT)
To leverage the unstructured protocol descriptions, we utilized a pretrained domain-specific language model: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`.
- **Feature Extraction:** Key textual columns were concatenated into a single descriptive document per trial: `Brief Summary`, `Detailed Description`, `Conditions`, `Arm Descriptions`, `Intervention Descriptions`, and `Intervention Names`.
- **Training Strategy:** The text was tokenized up to a maximum sequence length of 512. The model was trained using the Hugging Face `Trainer` API.
- **Handling Imbalance:** A custom `CrossEntropyLoss` function was implemented in the trainer, explicitly passing inverse class weights to heavily penalize false negatives for the minority class. 

## 3. Results
Below are the experimental results obtained on the test dataset.

### 3.1 XGBoost Results
*Metrics generated following Optuna tuning and isotonic calibration.*

| Metric | Uncalibrated Model | Calibrated Model |
|--------|--------------------|------------------|
| **ROC-AUC** | 0.8506 | 0.8466 |
| **PR-AUC (AP)**| 0.2212 | 0.2085 |
| **F1-Score** | 0.2902 | 0.2972 |
| **Precision** | 0.2321 | 0.2239 |
| **Recall** | 0.3871 | 0.4419 |
| **Accuracy** | 0.9071 | 0.8974 |

*Note: The uncalibrated threshold was optimized to `0.720`, while the calibrated threshold settled at `0.130`. The F1-score on the highly skewed test set reached `~0.30`.*

### 3.2 Transformer Results
*Metrics generated following 1 epoch of training on the concatenated free-text fields.*

| Metric | PubMedBERT |
|--------|------------|
| **ROC-AUC** | 0.8210 |
| **PR-AUC (AP)**| 0.1961 |
| **F1-Score** | 0.2526 |
| **Precision** | 0.1867 |
| **Recall** | 0.3903 |
| **Accuracy** | 0.8867 |

## 4. Analysis and Discussion
### 4.1 XGBoost Feature Importance
To understand what drives the metadata-based predictions, we extracted the feature importances from the trained XGBoost model. The top predictive features were:
1. **Phase 4 Trials (`FEATURE_phases_4`):** The strongest predictor. Phase 4 (post-marketing surveillance) trials might have structurally different protocols or distinct dosing error frequencies compared to earlier phases.
2. **Enrollment Count (`FEATURE_enrollmentCount`):** Trial size is highly correlated with outcomes; larger trials might be more complex, increasing the risk of dosing errors or, conversely, might have stricter oversight.
3. **Number of Locations (`FEATURE_numLocations`):** Multi-center trials inherently involve more logistical complexity in drug distribution and dosing standardizations.
4. **Healthy Volunteers (`FEATURE_healthyVolunteers`):** Whether the trial accepts healthy volunteers strongly impacts the baseline risk profile.

![Top 15 Feature Importances](/home/david/.gemini/antigravity-cli/brain/8a309fda-4897-4e6c-b6a7-65a1aa3b8f37/feature_importance.png)

### 4.2 Baseline Performance Comparison
- **XGBoost Performance:** The metadata-only model achieved a robust ROC-AUC of `~0.85`. However, the PR-AUC of `~0.21` highlights the severe difficulty of the dataset imbalance. While it can rank trials decently well, precision remains low when predicting the positive class.
- **Calibration Impact:** Isotonic calibration successfully adjusted the decision threshold, allowing for a noticeable jump in recall while preserving the overall F1-score.
- **Transformer Performance:** The PubMedBERT model achieved a ROC-AUC of `~0.82` and an F1-score of `~0.25` after a single epoch. This is slightly lower than the XGBoost model. 
- **Modality Comparison:** It appears that structural metadata (e.g., trial phases, enrollment size, locations) contains stronger or more accessible predictive signals for dosing errors than the raw, unstructured clinical narrative. Alternatively, learning complex linguistic cues from noisy, 512-token segments simply requires more epochs and potentially larger batch sizes to reach parity with the metadata approach.

## 5. Conclusion
This project successfully implemented a complete pipeline to establish independent machine learning and NLP baselines for the CT-DEB dataset. We found that predicting dosing errors is a complex challenge heavily hindered by the 5% class imbalance. While our metadata-based XGBoost model outperformed the early-stage PubMedBERT approach (achieving an F1 of `0.30` and ROC-AUC of `0.85`), both methods struggled to maintain high precision, indicating that early risk stratification for dosing errors remains an open and challenging area of research. Future improvements could involve ensembling both models, utilizing Large Language Models (LLMs) for few-shot extraction, or employing extensive data augmentation techniques.
