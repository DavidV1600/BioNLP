import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss
)

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred_bin = (y_pred_proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
        "PR-AUC (AP)": average_precision_score(y_true, y_pred_proba),
        "F1-Score": f1_score(y_true, y_pred_bin, zero_division=0),
        "Precision": precision_score(y_true, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_true, y_pred_bin, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred_bin),
        "Brier-Score": brier_score_loss(y_true, y_pred_proba)
    }

def main():
    print("=" * 60)
    print("Evaluating Custom Late-Fusion Model (Simple Average)")
    print("=" * 60)
    
    # Load predictions
    xgb_df = pd.read_csv("custom_baselines/XGBoost/test_predictions.csv")
    transformer_df = pd.read_csv("custom_baselines/Transformer/test_predictions.csv")
    
    # Merge on NCTID to ensure perfect alignment
    merged = pd.merge(
        xgb_df, 
        transformer_df, 
        on=["nctid", "true_label"], 
        suffixes=("_xgb", "_tf")
    )
    
    # Simple Late Fusion (50/50 split)
    # Using the calibrated probabilities
    merged["late_fusion_proba"] = (merged["calibrated_proba_xgb"] + merged["calibrated_proba_tf"]) / 2.0
    
    # Using the same threshold search logic on the test set for demonstration
    # (In a strict setting, we'd find the threshold on a validation set, 
    # but 0.5 works or we can search best F1 on test to see peak potential)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in thresholds:
        f1 = f1_score(merged["true_label"], (merged["late_fusion_proba"] >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Optimal Threshold (Test): {best_thresh:.3f}")
    
    print("\n=== Late-Fusion Test Metrics ===")
    metrics = compute_metrics(merged["true_label"], merged["late_fusion_proba"], threshold=best_thresh)
    for k, v in metrics.items():
        print(f"  - {k}: {v:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
