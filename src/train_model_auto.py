import argparse
import os
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding, 
                          TrainerCallback)
from datasets import Dataset
import torch
from accelerate import Accelerator
import json

#python src/train_model_auto.py --train_path data/train.csv --val_path data/val.csv

LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}

def get_device_info():
    """Get information about available compute devices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"\n{'='*50}")
    print(f"Using device: {device}")
    print(f"Device name: {device_name}")
    if device == "cuda":
        print(f"Number of GPUs available: {device_count}")
        print(f"CUDA version: {torch.version.cuda}")
        # Print GPU memory
        try:
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        except:
            print("Could not retrieve GPU memory information")
    print(f"{'='*50}\n")
    
    return device

def load_csv_as_dataset(path: str) -> Dataset:
    """Load a CSV file as a Hugging Face Dataset."""
    df = pd.read_csv(path)
    # prefer clean_text if exists
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    return Dataset.from_pandas(df)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text examples."""
    return tokenizer(examples['text'], truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    # Also compute and print confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(f"            Predicted")
    print(f"           Fake  Real")
    print(f"Actual Fake {cm[0][0]:4d} {cm[0][1]:4d}")
    print(f"      Real {cm[1][0]:4d} {cm[1][1]:4d}")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def check_gpu_utilization():
    """Check GPU utilization if available."""
    if torch.cuda.is_available():
        try:
            # Get current GPU utilization percentage
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  stdout=subprocess.PIPE, text=True)
            util = int(result.stdout.strip())
            return f"{util}%"
        except:
            return "Error checking GPU"
    return "No GPU"

class GPUUtilizationCallback(TrainerCallback):
    """Custom callback to monitor GPU utilization during training."""
    def __init__(self, check_interval=50):
        self.check_interval = check_interval
        self.step = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.check_interval == 0:
            print(f"Step {self.step} - GPU Utilization: {check_gpu_utilization()}")

def main():
    parser = argparse.ArgumentParser(description="Train a fake news detection model with automatic GPU/CPU detection")
    parser.add_argument('--train_path', default='data/train.csv', help='Path to training data CSV')
    parser.add_argument('--val_path', default='data/val.csv', help='Path to validation data CSV')
    parser.add_argument('--model_name', default='distilbert-base-uncased', help='Pretrained model name')
    parser.add_argument('--out_dir', default='model', help='Output directory for saved model')
    parser.add_argument('--epochs', type=float, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size (None for auto)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--no_fp16', action='store_true', help='Disable mixed precision training')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check for GPU and set up accelerator
    device = get_device_info()
    accelerator = Accelerator(mixed_precision='fp16' if (not args.no_fp16 and torch.cuda.is_available()) else 'no')
    
    # Set batch size based on available hardware if not specified
    if args.batch_size is None:
        if torch.cuda.is_available():
            # For GPU, use larger batch size
            if torch.cuda.get_device_properties(0).total_memory > 10e9:  # >10GB VRAM
                args.batch_size = 32
            else:
                args.batch_size = 16
        else:
            # For CPU, use smaller batch size
            args.batch_size = 8
    print(f"Using batch size: {args.batch_size}")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    # Load and preprocess data
    print(f"Loading training data from {args.train_path}")
    ds_train = load_csv_as_dataset(args.train_path)
    print(f"Loading validation data from {args.val_path}")
    ds_val = load_csv_as_dataset(args.val_path)
    
    print(f"Training set size: {len(ds_train)}")
    print(f"Validation set size: {len(ds_val)}")
    
    print("Tokenizing datasets...")
    start_time = time.time()
    tokenized_train = ds_train.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length), 
        batched=True, 
        remove_columns=['text']
    )
    tokenized_val = ds_val.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length), 
        batched=True, 
        remove_columns=['text']
    )
    print(f"Tokenization completed in {time.time() - start_time:.2f} seconds")

    # Optimize data loading with efficient collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    # Set up training arguments with GPU optimizations if available
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    gradient_accum_steps = 2 if torch.cuda.is_available() else 1
    
    training_args = TrainingArguments(
        output_dir=f"{args.out_dir}/checkpoints",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        save_total_limit=2,
        fp16=use_fp16,
        gradient_accumulation_steps=gradient_accum_steps,
        report_to="none"  # Disable wandb/tensorboard reporting for speed
    )

    # Add GPU utilization monitoring if GPU is available
    callbacks = []
    if torch.cuda.is_available():
        from transformers.trainer_callback import TrainerCallback
        callbacks.append(GPUUtilizationCallback(check_interval=50))

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Train the model
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

    # Evaluate the model
    print("\nEvaluating model...")
    metrics = trainer.evaluate()
    print("\nValidation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save model, tokenizer, and label map
    print(f"\nSaving model to {args.out_dir}...")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, 'label_map.json'), 'w') as f:
        json.dump(ID2LABEL, f)
    
    # Save training arguments for reference
    with open(os.path.join(args.out_dir, 'training_args.json'), 'w') as f:
        # Convert args to dictionary and save
        args_dict = vars(args)
        args_dict["device"] = device
        args_dict["fp16_used"] = use_fp16
        args_dict["gradient_accumulation_steps"] = gradient_accum_steps
        args_dict["train_time_seconds"] = train_time
        json.dump(args_dict, f, indent=2)

    print(f"\nTraining complete! Model saved to {args.out_dir}")
    print(f"Total training time: {train_time/60:.2f} minutes")

if __name__ == '__main__':
    main()