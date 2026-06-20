import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def main():
    model_path = "custom_baselines/XGBoost/xgb_model.pkl"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Get feature importances
    importances = model.feature_importances_
    
    # We need the feature names. Since we don't have them saved directly in the model, 
    # we can get them from the model object if it's an XGBClassifier
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"f{i}" for i in range(len(importances))]
    
    # Create DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    
    # Sort and get top 15
    df = df.sort_values("Importance", ascending=False).head(15)
    
    print("Top 15 Feature Importances:")
    for _, row in df.iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")
        
    # Generate a plot
    plt.figure(figsize=(10, 6))
    plt.barh(df["Feature"][::-1], df["Importance"][::-1])
    plt.xlabel("Importance (Gain)")
    plt.title("Top 15 XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig("custom_baselines/XGBoost/feature_importance.png")
    print("Saved feature importance plot to custom_baselines/XGBoost/feature_importance.png")

if __name__ == "__main__":
    main()
