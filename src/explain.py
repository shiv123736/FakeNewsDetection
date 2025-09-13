import argparse, os, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer

CLASS_NAMES = ['fake', 'real']

def load_model(model_dir='model'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        id2label = json.load(f)
    return tokenizer, model, {int(k):v for k,v in id2label.items()}

def predict_proba(texts, tokenizer, model, max_length=512):
    import torch
    enc = tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

def explain_text(text, tokenizer, model, num_features=8):
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    prob_fn = lambda xs: predict_proba(xs, tokenizer, model)
    exp = explainer.explain_instance(text, prob_fn, num_features=num_features)
    return exp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', required=True)
    ap.add_argument('--model_dir', default='model')
    ap.add_argument('--num_features', type=int, default=8)
    args = ap.parse_args()

    tokenizer, model, id2label = load_model(args.model_dir)
    exp = explain_text(args.text, tokenizer, model, num_features=args.num_features)
    print("Top features (word -> weight):")
    for word, weight in exp.as_list():
        print(f"{word:>20s}  {weight:+.3f}")
    # Save HTML
    html_path = os.path.join(args.model_dir, 'lime_explanation.html')
    exp.save_to_file(html_path)
    print(f"Saved interactive HTML to: {html_path}")

if __name__ == '__main__':
    main()
