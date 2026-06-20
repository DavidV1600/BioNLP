import os
import argparse
import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset, Features, Value, ClassLabel
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train custom Transformer baseline on CT-dosing-errors text.")
    parser.add_argument("--dataset_path", type=str, default="temp_clone/resources/CT-DOSING-ERRORS/0.2.3")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--output_dir", type=str, default="custom_baselines/Transformer")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Sigmoid for probabilities
    # We only have a single output unit since we can treat it as binary class,
    # or two if we use AutoModelForSequenceClassification with num_labels=2.
    # Let's assume num_labels=2.
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    
    # Avoid zero division errors
    roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "accuracy": accuracy_score(labels, preds),
        "brier": brier_score_loss(labels, probs)
    }

def create_text_input(example):
    # Combine relevant textual fields
    fields = [
        ("Brief Summary", example.get("FEATURE_briefSummary", "")),
        ("Detailed Description", example.get("FEATURE_detailedDescription", "")),
        ("Conditions", example.get("FEATURE_conditions", "")),
        ("Arm Descriptions", example.get("FEATURE_armDescriptions", "")),
        ("Intervention Descriptions", example.get("FEATURE_interventionDescriptions", "")),
        ("Intervention Names", example.get("FEATURE_interventionNames", ""))
    ]
    
    text_parts = []
    for title, content in fields:
        if content and str(content).strip() and str(content).strip() != "None":
            text_parts.append(f"{title}: {content}")
            
    combined_text = "\n\n".join(text_parts)
    return {"text": combined_text, "label": example["LABEL_wilson_label"], "nctid": example["METADATA_nctId"]}

def main():
    args = parse_args()
    print("=" * 60)
    print("Starting Custom Transformer Training Pipeline")
    print(f"Model: {args.model_name}")
    print("=" * 60)
    
    # Load raw dataset
    print("Loading raw datasets...")
    dataset = load_from_disk(args.dataset_path)
    
    # Process each split
    processed_dataset = {}
    for split in ["train", "validation", "test"]:
        print(f"Processing text fields for {split} split...")
        ds_split = dataset[split]
        ds_split = ds_split.map(create_text_input, remove_columns=ds_split.column_names, num_proc=4)
        processed_dataset[split] = ds_split
        
    # Check class imbalance in training set to compute class weights
    train_labels = processed_dataset["train"]["label"]
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    print(f"Class imbalance: {num_pos} positive trials, {num_neg} negative trials.")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding=False, truncation=True, max_length=args.max_length)
    
    print("Tokenizing datasets...")
    tokenized_dataset = {}
    for split in ["train", "validation", "test"]:
        tokenized_dataset[split] = processed_dataset[split].map(
            tokenize_function, 
            batched=True, 
            num_proc=4,
            remove_columns=["text"]
        )
        
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2
    )
    
    # Handle class imbalance using a custom trainer
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Compute weights based on inverse class frequency
            # weight = [pos_weight for class 1, neg_weight for class 0]
            # Since ratio is highly skewed, we can set pos_weight = num_neg/num_pos
            weight = torch.tensor([1.0, num_neg / num_pos]).to(model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
            
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        seed=args.random_seed,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    
    print("\n=== Test Metrics ===")
    for k, v in test_results.items():
        if k.startswith("eval_"):
            print(f"  - {k[5:]}: {v:.4f}")
            
    # Save test predictions
    print("\nGenerating test predictions...")
    test_preds = trainer.predict(tokenized_dataset["test"])
    logits = test_preds.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    preds_df = pd.DataFrame({
        "nctid": tokenized_dataset["test"]["nctid"],
        "true_label": tokenized_dataset["test"]["label"],
        "calibrated_proba": probs  # Just treating output probs as predictions
    })
    
    os.makedirs(args.output_dir, exist_ok=True)
    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved test set predictions to: {preds_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
