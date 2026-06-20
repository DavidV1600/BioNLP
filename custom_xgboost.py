import os
import pickle
import argparse
import pandas as pd
import numpy as np
import optuna
from datasets import load_from_disk, Value
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV

# Suppress Optuna logging to clean up output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom XGBoost baseline on CT-dosing-errors metadata.")
    parser.add_argument("--dataset_path", type=str, default="temp_clone/resources/CT-DOSING-ERRORS/0.2.3",
                        help="Path to the Hugging Face dataset split directory.")
    parser.add_argument("--output_dir", type=str, default="custom_baselines/XGBoost",
                        help="Directory to save model checkpoints and predictions.")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of Optuna trials for hyperparameter tuning.")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()

def preprocess_split(dataset_split, label_name="wilson_label"):
    # Extract metadata columns
    nct_ids = dataset_split["METADATA_nctId"]
    
    # Identify features to drop (all text/string fields, metadata fields, other labels)
    string_cols = [name for name, feat in dataset_split.features.items() 
                   if isinstance(feat, Value) and feat.dtype == "string"]
    metadata_cols = [name for name in dataset_split.column_names if name.startswith("METADATA_")]
    label_cols_to_drop = [name for name in dataset_split.column_names 
                          if name.startswith("LABEL_") and name != f"LABEL_{label_name}"]
    
    cols_to_drop = set(string_cols) | set(metadata_cols) | set(label_cols_to_drop)
    dataset_reduced = dataset_split.remove_columns(list(cols_to_drop))
    
    # Convert to pandas DataFrame
    df = dataset_reduced.to_pandas()
    
    # Multi-hot/Count encoding for lists of categoricals (armGroupTypes, phases, interventionTypes)
    need_to_encode = ['FEATURE_armGroupTypes', 'FEATURE_phases', 'FEATURE_interventionTypes']
    for col in need_to_encode:
        if col in df.columns:
            exploded = df[col].explode()
            counts = pd.crosstab(exploded.index, exploded)
            counts = counts.add_prefix(col + '_').astype(float)
            df = pd.concat([df.drop(columns=[col]), counts], axis=1)
            
    # Handle boolean features
    if 'FEATURE_healthyVolunteers' in df.columns:
        df['FEATURE_healthyVolunteers'] = df['FEATURE_healthyVolunteers'].astype(float)
    if 'FEATURE_oversightHasDmc' in df.columns:
        df['FEATURE_oversightHasDmc'] = df['FEATURE_oversightHasDmc'].astype(float)
        
    df = df.infer_objects(copy=False)
    
    # Separate features and labels
    label_col = f"LABEL_{label_name}"
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    
    return X, y, nct_ids

def align_columns(X_train, X_val, X_test):
    # Get reference columns from training set
    train_cols = list(X_train.columns)
    
    # Reindex validation and test sets to match training columns
    # missing columns are filled with 0.0, and extra columns are dropped
    X_val_aligned = X_val.reindex(columns=train_cols, fill_value=0.0)
    X_test_aligned = X_test.reindex(columns=train_cols, fill_value=0.0)
    
    return X_train, X_val_aligned, X_test_aligned

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
    args = parse_args()
    print("=" * 60)
    print("Starting Custom XGBoost Training Pipeline")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Optuna trials: {args.num_trials}")
    print(f"Seed: {args.random_seed}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading local datasets...")
    dataset = load_from_disk(args.dataset_path)
    
    # Preprocess splits
    print("Preprocessing training split...")
    X_train, y_train, train_nct = preprocess_split(dataset["train"])
    print("Preprocessing validation split...")
    X_val, y_val, val_nct = preprocess_split(dataset["validation"])
    print("Preprocessing test split...")
    X_test, y_test, test_nct = preprocess_split(dataset["test"])
    
    # Align features across splits
    print("Aligning feature columns across splits...")
    X_train, X_val, X_test = align_columns(X_train, X_val, X_test)
    print(f"Preprocessed features count: {X_train.shape[1]}")
    
    # Compute scale_pos_weight to address class imbalance
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos
    print(f"Class imbalance: {num_pos} positive trials, {num_neg} negative trials.")
    print(f"Base scale_pos_weight = {scale_pos_weight:.4f}")
    
    # Hyperparameter Optimization with Optuna
    hyperparam_path = os.path.join(args.output_dir, "xgb_best_hyperparam.pkl")
    if os.path.exists(hyperparam_path):
        print(f"Loading best hyperparameters from {hyperparam_path}...")
        with open(hyperparam_path, "rb") as f:
            best_params = pickle.load(f)
    else:
        print(f"Starting Optuna hyperparameter optimization for {args.num_trials} trials...")
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
                "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5 * scale_pos_weight, 2.0 * scale_pos_weight),
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": args.random_seed,
                "n_jobs": -1
            }
            
            model = XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds_val = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, preds_val)
            
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.random_seed))
        study.optimize(objective, n_trials=args.num_trials)
        
        print("\nOptuna Optimization Complete!")
        print(f"Best Validation ROC-AUC: {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")
            
        # Save best hyperparameters
        best_params = study.best_params
        with open(hyperparam_path, "wb") as f:
            pickle.dump(best_params, f)
        print(f"Saved best hyperparameters to: {hyperparam_path}")
    
    # Train final model using best hyperparameters
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "auc"
    best_params["random_state"] = args.random_seed
    best_params["n_jobs"] = -1
    
    final_model = XGBClassifier(**best_params)
    print("\nTraining final model...")
    final_model.fit(X_train, y_train)
    
    # Calibrate probability predictions using isotonic calibration on validation split
    print("Calibrating model using isotonic calibration on validation split...")
    from sklearn.frozen import FrozenEstimator
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(final_model), method="isotonic")
    calibrated_model.fit(X_val, y_val)
    
    # Save models
    model_path = os.path.join(args.output_dir, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    calibrated_model_path = os.path.join(args.output_dir, "xgb_calibrated_model.pkl")
    with open(calibrated_model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    print(f"Saved trained models to {args.output_dir}")
    
    # Predict and evaluate uncalibrated model
    print("\nEvaluating uncalibrated model...")
    train_proba = final_model.predict_proba(X_train)[:, 1]
    val_proba = final_model.predict_proba(X_val)[:, 1]
    test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Search threshold for maximum F1-score on validation set
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = -1.0
    best_thresh = 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (val_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"Best classification threshold found on validation set: {best_thresh:.3f} (F1 = {best_f1:.4f})")
    
    print("\n=== Uncalibrated Test Metrics ===")
    test_metrics = compute_metrics(y_test, test_proba, threshold=best_thresh)
    for k, v in test_metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    # Predict and evaluate calibrated model
    print("\nEvaluating calibrated model...")
    cal_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    cal_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Search threshold for maximum F1-score on validation set (calibrated)
    best_f1_cal = -1.0
    best_thresh_cal = 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (cal_val_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1_cal:
            best_f1_cal = f1
            best_thresh_cal = t
    print(f"Best calibrated classification threshold found on validation set: {best_thresh_cal:.3f} (F1 = {best_f1_cal:.4f})")
    
    print("\n=== Calibrated Test Metrics ===")
    cal_test_metrics = compute_metrics(y_test, cal_test_proba, threshold=best_thresh_cal)
    for k, v in cal_test_metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    # Save predictions
    predictions_df = pd.DataFrame({
        "nctid": test_nct,
        "true_label": y_test,
        "uncalibrated_proba": test_proba,
        "calibrated_proba": cal_test_proba
    })
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    predictions_df.to_csv(preds_path, index=False)
    print(f"\nSaved test set predictions to: {preds_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
