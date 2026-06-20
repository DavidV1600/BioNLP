---
marp: true
theme: default
paginate: true
header: 'Detecting Dosing Errors in Clinical Trials'
footer: 'CT-DEB BioNLP Task'
---

# Detecting Medication Dosing Errors in Clinical Trial Protocols
**A Multimodal Machine Learning Approach**

---

# 1. The Challenge
- **The Problem:** Inadvertent medication dosing errors during clinical trials compromise patient safety and study validity.
- **The Goal:** Identify protocols with a high inherent risk of errors *before* the trial begins.
- **Dataset:** CT-DEB (over 42,000 real trial narratives from ClinicalTrials.gov).
- **Difficulty:** Highly imbalanced data—only ~5% of historical trials contain reported dosing errors.

---

# 2. Our Approach (Replicating Literature)
We bypassed traditional heuristics and built two independent models to extract signals from different parts of the protocol:

1. **Structured Data Approach (The "Stats" Model)**
   - Extracted metadata (phases, patient count, multi-center status).
   - Trained an optimized **XGBoost Classifier**.
   
2. **Unstructured Data Approach (The "Text" Model)**
   - Aggregated free-text protocol fields (Brief Summary, Arm descriptions).
   - Fine-tuned **PubMedBERT**, a state-of-the-art biomedical language model.

---

# 3. Beyond the Literature: Modality Fusion
To advance the state-of-the-art and combine the predictive power of both text and statistics, we engineered two fusion architectures:

1. **Late-Fusion Ensemble:**
   - Averaged the final probability predictions from the independent XGBoost and PubMedBERT models.
2. **Early-Fusion Super Model (Novel):**
   - Used PubMedBERT to extract 768-dimensional mathematical representations (embeddings) of the text.
   - Concatenated these embeddings directly with the structural metadata.
   - Trained a single, unified XGBoost model to learn complex cross-modal interactions simultaneously.

---

# 4. Results: Comparative Performance
*Because of the 5% class imbalance, ROC-AUC is our primary evaluation metric.*

| Model Architecture | Data Used | ROC-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| **1. Text-Only (Transformer)** | Clinical text only | `0.8210` | `0.2526` |
| **2. Stats-Only (XGBoost)** | Metadata only | `0.8466` | `0.2972` |
| **3. Early-Fusion** | Fused embeddings + stats | `0.8475` | `0.2904` |
| **4. Late-Fusion (Winner)** | Averaged models #1 and #2 | **`0.8486`** | **`0.3112`** |

---

# 5. Model Comparison Chart

*(Insert `model_comparison.png` here on your slide to visually display the table metrics!)*

---

# 6. Feature Impact (Why do errors happen?)
Analysis of our decision trees revealed that the most critical drivers of dosing error risk are structural:

1. **Trial Phase (Phase 4):** Post-marketing surveillance trials exhibit significantly different error profiles than early-stage trials.
2. **Enrollment Count:** Massive patient volumes inherently increase the statistical likelihood of administrative/dosing errors.
3. **Number of Locations:** Multi-center trials involve complex logistical distributions, increasing standardization risks across different hospitals.

---

# 7. Conclusions
- **Stats beat Text:** The structural metadata of a trial provides more accessible predictive signals than the unstructured clinical jargon.
- **The Curse of Dimensionality:** Forcing a single AI to look at both the text embeddings and the stats simultaneously (Early-Fusion) caused it to slightly overfit to the massive text data.
- **Simple is Best:** The most robust way to predict dosing errors is to build two separate expert models (one for text, one for stats) and simply average their predictions (Late-Fusion).
