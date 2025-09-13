import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

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
    ap.add_argument('--text', required=True, help='Input text (quote if contains spaces)')
    ap.add_argument('--model_dir', default='model')
    args = ap.parse_args()

    tokenizer, model, id2label = load_model(args.model_dir)
    preds, probs = predict_text(args.text, tokenizer, model)

    label_id = int(preds[0])
    label = id2label[label_id]
    confidence = float(np.max(probs[0]))
    print(f"Prediction: {label.upper()} | Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main()
