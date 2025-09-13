import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_model(model_dir='model'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        id2label = json.load(f)
    return tokenizer, model, {int(k):v for k,v in id2label.items()}

def predict_text(texts, tokenizer, model, max_length=512):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    return preds, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', help='Input text (quote if contains spaces)')
    ap.add_argument('--input_file', help='Path to input CSV file (e.g., test.csv)')
    ap.add_argument('--output_file', help='Path to output CSV file for predictions')
    ap.add_argument('--model_dir', default='model', help='Directory containing the trained model')
    args = ap.parse_args()

    tokenizer, model, id2label = load_model(args.model_dir)
    
    # Process single text
    if args.text:
        preds, probs = predict_text(args.text, tokenizer, model)
        label_id = int(preds[0])
        label = id2label[label_id]
        confidence = float(np.max(probs[0]))
        print(f"Prediction: {label.upper()} | Confidence: {confidence:.2%}")
    
    # Process input file
    elif args.input_file:
        if not args.output_file:
            args.output_file = 'predictions.csv'
            
        print(f"Processing file: {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        # Check if 'text' column exists
        text_col = 'text' if 'text' in df.columns else df.columns[0]
        
        results = []
        texts = df[text_col].tolist()
        
        # Process in batches for efficiency
        batch_size = 8
        all_preds = []
        all_probs = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            preds, probs = predict_text(batch_texts, tokenizer, model)
            all_preds.extend(preds)
            all_probs.extend([p.max() for p in probs])
            
        # Create results DataFrame
        results_df = pd.DataFrame({
            'text': texts,
            'prediction': [id2label[int(p)] for p in all_preds],
            'confidence': all_probs
        })
        
        # Add original label if it exists
        if 'label' in df.columns:
            results_df['true_label'] = df['label']
            # Calculate accuracy
            accuracy = (results_df['prediction'] == results_df['true_label']).mean()
            print(f"Accuracy: {accuracy:.2%}")
        
        # Save results
        results_df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    
    else:
        print("Error: Either --text or --input_file must be provided")
        ap.print_help()

if __name__ == '__main__':
    main()
