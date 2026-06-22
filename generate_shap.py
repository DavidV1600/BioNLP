import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
from datasets import load_from_disk, DatasetDict
from custom_xgboost import preprocess_split

def main():
    print("Loading test data...")
    dataset_path = "temp_clone/resources/CT-DOSING-ERRORS/0.2.3"
    dataset = DatasetDict({
        "test": load_from_disk(os.path.join(dataset_path, "test"))
    })
    
    test_ds = dataset['test'].select(range(1000))
    
    print("Preprocessing data...")
    X_test, _, _ = preprocess_split(test_ds)
    
    print("Loading model...")
    with open("custom_baselines/XGBoost/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
        
    expected_cols = xgb_model.get_booster().feature_names
    
    # Align X_test
    for col in expected_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[expected_cols]
    
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    
    print("Plotting SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    
    output_path = "shap_summary.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"SHAP plot saved to {output_path}")

if __name__ == "__main__":
    main()
