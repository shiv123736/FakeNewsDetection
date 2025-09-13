import argparse
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import Dataset
import torch
import json

LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}

def load_csv_as_dataset(path: str) -> Dataset:
    df = pd.read_csv(path)
    # prefer clean_text if exists
    text_col = 'clean_text' if 'clean_text' in df.columns else 'text'
    df = df[[text_col, 'label']].rename(columns={text_col: 'text'})
    return Dataset.from_pandas(df)

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(examples['text'], truncation=True, max_length=max_length)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train.csv')
    parser.add_argument('--val_path', default='data/val.csv')
    parser.add_argument('--model_name', default='distilbert-base-uncased')
    parser.add_argument('--out_dir', default='model')
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    ds_train = load_csv_as_dataset(args.train_path)
    ds_val = load_csv_as_dataset(args.val_path)

    tokenized_train = ds_train.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])
    tokenized_val = ds_val.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=['text'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\nValidation metrics:", metrics)

    # Save model + tokenizer + label map
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    with open(os.path.join(args.out_dir, 'label_map.json'), 'w') as f:
        json.dump(ID2LABEL, f)
    print(f"Saved trained model to {args.out_dir}")

if __name__ == '__main__':
    main()
