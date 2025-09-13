import argparse
import os
import numpy as np
import pandas as pd
import time
import subprocess
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
import psutil  # For system memory monitoring

#python src/train_model_auto.py --train_path data/train.csv --val_path data/val.csv --out-dir model

LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}

def check_gpu_utilization():
    """Get current GPU utilization percentage."""
    if not torch.cuda.is_available():
        return "N/A"
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                stdout=subprocess.PIPE, text=True, check=False)
        if result.returncode == 0:
            return f"{result.stdout.strip()}%"
        return "N/A"
    except Exception:
        return "N/A"

def get_device_info():
    """Get detailed information about available compute devices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"🔍 HARDWARE DETECTION")
    print(f"{'='*60}")
    print(f"📊 Using device: {device.upper()}")
    
    if device == "cuda":
        # Get GPU details
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        
        # Memory information
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory_mb = torch.cuda.memory_allocated(0) / 1024**2
        reserved_memory_mb = torch.cuda.memory_reserved(0) / 1024**2
        
        print(f"🖥️  GPU detected: {device_name}")
        print(f"📈 CUDA version: {cuda_version}")
        print(f"📊 Number of GPUs available: {device_count}")
        print(f"💾 Total GPU memory: {total_memory_gb:.2f} GB")
        print(f"💾 Initial memory allocated: {allocated_memory_mb:.2f} MB")
        print(f"💾 Initial memory reserved: {reserved_memory_mb:.2f} MB")
        
        # Try to get utilization from nvidia-smi
        gpu_util = check_gpu_utilization()
        if gpu_util != "N/A":
            print(f"⚡ Initial GPU utilization: {gpu_util}")
        
        # Simple test to verify GPU is working
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"✅ Test tensor successfully placed on GPU: {test_tensor.device}")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            
        # Print additional GPU capabilities
        try:
            for i in range(device_count):
                gpu_properties = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i} Details:")
                print(f"  CUDA Capability: {gpu_properties.major}.{gpu_properties.minor}")
                print(f"  Max allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Error retrieving detailed GPU information: {str(e)}")
    else:
        print(f"⚠️ No GPU detected - using CPU only")
        print(f"⚠️ CUDA is not available")
        
        # CPU information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"💻 CPU cores: {cpu_physical} physical, {cpu_count} logical")
        print(f"💾 System memory: {memory_gb:.2f} GB")
    
    print(f"{'='*60}\n")
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

class GPUMonitorCallback(TrainerCallback):
    """Enhanced callback to monitor GPU utilization and memory during training."""
    def __init__(self, check_interval=50):
        self.check_interval = check_interval
        self.step = 0
        self.start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step += 1
        if self.step % self.check_interval == 0:
            # Calculate speed
            elapsed = time.time() - self.start_time
            steps_per_second = self.step / elapsed
            
            # Memory info
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
                reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
                utilization = check_gpu_utilization()
                
                print(f"Step {self.step} | {steps_per_second:.2f} steps/sec | " 
                      f"GPU: {utilization} | Mem: {allocated_gb:.2f}GB / {reserved_gb:.2f}GB")
                
                # If utilization is very low, print a warning
                try:
                    util_pct = int(utilization.strip('%'))
                    if util_pct < 30:
                        print("⚠️ GPU utilization is very low. Consider increasing batch size or model complexity.")
                except:
                    pass
            else:
                print(f"Step {self.step} | {steps_per_second:.2f} steps/sec | CPU mode")

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
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.9, help='Fraction of GPU memory to use (0.0-1.0)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Check for GPU and set up accelerator
    device = get_device_info()
    
    # Set GPU memory fraction if using CUDA
    if torch.cuda.is_available():
        # Reserve specific fraction of GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = int(total_memory * args.gpu_memory_fraction)
            # This is a trick to reserve memory upfront to prevent OOM errors
            torch.cuda.empty_cache()
            print(f"Setting GPU memory fraction to {args.gpu_memory_fraction*100:.0f}% ({reserved_memory/(1024**3):.2f} GB)")
            
            # You can optionally uncomment the next line to limit GPU memory usage in some cases
            # But it doesn't work with all GPUs/CUDA versions, so use with caution
            # torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
        except Exception as e:
            print(f"Failed to set GPU memory fraction: {e}")
    
    accelerator = Accelerator(mixed_precision='fp16' if (not args.no_fp16 and torch.cuda.is_available()) else 'no')
    
    # Set batch size based on available hardware if not specified
    if args.batch_size is None:
        if torch.cuda.is_available():
            # For GPU, use larger batch size based on available VRAM
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # VRAM in GB
            
            if gpu_vram > 10:  # >10GB VRAM (like RTX 3080/3090/4070/4080/4090)
                args.batch_size = 32
            elif gpu_vram > 8:  # 8-10GB VRAM (like RTX 3070/2080)
                args.batch_size = 24
            elif gpu_vram > 6:  # 6-8GB VRAM (like RTX 3060/2070)
                args.batch_size = 16
            else:  # <6GB VRAM
                args.batch_size = 8
        else:
            # For CPU, use smaller batch size
            args.batch_size = 8
    print(f"Using batch size: {args.batch_size}")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    start_time = time.time()
    
    # Clear CUDA cache before model loading if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache before model loading")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    # Display model size
    model_size_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size_params:,} parameters")

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

    # Get example sequence length stats to help with optimization
    seq_lengths = [len(x["input_ids"]) for x in tokenized_train]
    avg_seq_length = sum(seq_lengths) / len(seq_lengths)
    max_seq = max(seq_lengths)
    print(f"Average sequence length: {avg_seq_length:.1f}, Maximum: {max_seq}")
    
    if max_seq < args.max_length * 0.75:
        print(f"⚠️ Most sequences are much shorter than max_length ({args.max_length})")
        print(f"💡 Consider reducing max_length to {int(max_seq * 1.1)} to improve performance")

    # Optimize data loading with efficient collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

    # Set up training arguments with GPU optimizations if available
    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    
    # Determine gradient accumulation steps based on hardware
    # This helps compensate for smaller batch sizes on limited hardware
    gradient_accum_steps = 1
    if args.batch_size < 16 and torch.cuda.is_available():
        # Smaller effective batch size, use gradient accumulation to compensate
        gradient_accum_steps = max(1, 16 // args.batch_size)
        print(f"Using gradient accumulation steps: {gradient_accum_steps} (equivalent to batch size {args.batch_size * gradient_accum_steps})")
    
    if use_fp16:
        print("Using mixed precision training (FP16)")
    else:
        print("Using full precision training (FP32)")
    
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

    # Add GPU monitoring callbacks
    callbacks = []
    if torch.cuda.is_available():
        # Clear CUDA cache before loading model
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache before model loading")
        
        # Add GPU monitoring
        callbacks.append(GPUMonitorCallback(check_interval=50))

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

    # Final GPU memory status
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
        reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
        print(f"💾 Final GPU memory: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
        
        # Clear cache to free memory
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache after training")

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