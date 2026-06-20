import os
import pickle
import argparse
import pandas as pd
import numpy as np
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from tqdm import tqdm

from custom_xgboost import preprocess_split, align_columns, compute_metrics
from custom_transformer import create_text_input
from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Train Early-Fusion Super Model.")
    parser.add_argument("--dataset_path", type=str, default="temp_clone/resources/CT-DOSING-ERRORS/0.2.3")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--output_dir", type=str, default="custom_baselines/EarlyFusion")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def extract_embeddings(dataset_split, tokenizer, model, batch_size, max_length):
    text_dataset = dataset_split.map(create_text_input, remove_columns=dataset_split.column_names, num_proc=4)
    all_embeddings = []
    
    device = model.device
    
    for i in tqdm(range(0, len(text_dataset), batch_size), desc="Extracting Embeddings"):
        batch_texts = text_dataset[i:i+batch_size]["text"]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    return np.vstack(all_embeddings)

def main():
    args = parse_args()
    print("=" * 60)
    print("Starting Early-Fusion Super Model Pipeline")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading local datasets...")
    dataset = load_from_disk(args.dataset_path)
    
    # 1. Process Structured Features
    print("\n[1/3] Preprocessing structured metadata...")
    X_train_s, y_train, train_nct = preprocess_split(dataset["train"])
    X_val_s, y_val, val_nct = preprocess_split(dataset["validation"])
    X_test_s, y_test, test_nct = preprocess_split(dataset["test"])
    
    X_train_s, X_val_s, X_test_s = align_columns(X_train_s, X_val_s, X_test_s)
    
    # 2. Process Text Embeddings
    print(f"\n[2/3] Extracting text embeddings using {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Extract and cache embeddings if not already cached
    cache_path = os.path.join(args.output_dir, "embeddings_cache.pkl")
    if os.path.exists(cache_path):
        print("Loading cached embeddings...")
        with open(cache_path, "rb") as f:
            emb_train, emb_val, emb_test = pickle.load(f)
    else:
        print("Extracting train embeddings...")
        emb_train = extract_embeddings(dataset["train"], tokenizer, model, args.batch_size, args.max_length)
        print("Extracting val embeddings...")
        emb_val = extract_embeddings(dataset["validation"], tokenizer, model, args.batch_size, args.max_length)
        print("Extracting test embeddings...")
        emb_test = extract_embeddings(dataset["test"], tokenizer, model, args.batch_size, args.max_length)
        with open(cache_path, "wb") as f:
            pickle.dump((emb_train, emb_val, emb_test), f)
            
    # Free VRAM
    del model
    torch.cuda.empty_cache()
    
    # Create combined datasets
    print("Merging embeddings with structured features...")
    emb_cols = [f"EMB_{i}" for i in range(emb_train.shape[1])]
    
    X_train = pd.concat([X_train_s.reset_index(drop=True), pd.DataFrame(emb_train, columns=emb_cols)], axis=1)
    X_val = pd.concat([X_val_s.reset_index(drop=True), pd.DataFrame(emb_val, columns=emb_cols)], axis=1)
    X_test = pd.concat([X_test_s.reset_index(drop=True), pd.DataFrame(emb_test, columns=emb_cols)], axis=1)
    
    # 3. Train Early-Fusion XGBoost
    print("\n[3/3] Training Early-Fusion XGBoost Model...")
    
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos
    
    # Using reasonably robust hyperparameters
    params = {
        "n_estimators": 250,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "random_state": args.random_seed,
        "n_jobs": -1
    }
    
    final_model = XGBClassifier(**params)
    final_model.fit(X_train, y_train)
    
    print("Calibrating model...")
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(final_model), method="isotonic")
    calibrated_model.fit(X_val, y_val)
    
    # Evaluate
    print("\nEvaluating calibrated model on Test Set...")
    cal_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    cal_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Search threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1_cal = -1.0
    best_thresh_cal = 0.5
    for t in thresholds:
        f1 = f1_score(y_val, (cal_val_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1_cal:
            best_f1_cal = f1
            best_thresh_cal = t
            
    print("\n=== Early-Fusion Test Metrics ===")
    test_metrics = compute_metrics(y_test, cal_test_proba, threshold=best_thresh_cal)
    for k, v in test_metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()
