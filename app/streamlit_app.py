import streamlit as st
import os, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_model(model_dir='model'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        id2label = json.load(f)
    id2label = {int(k):v for k,v in id2label.items()}
    return tokenizer, model, id2label

def predict(texts, tokenizer, model, max_length=512):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
    return preds, probs

st.title("📰 Fake News Detection (BERT)")
st.write("Enter a news headline or article text to classify as **FAKE** or **REAL**.")

tokenizer, model, id2label = load_model('model')

user_text = st.text_area("Paste news text here:", height=200)

if st.button("Check News"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        preds, probs = predict(user_text, tokenizer, model)
        label_id = int(preds[0])
        label = id2label[label_id].upper()
        confidence = float(np.max(probs[0]))

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2%}")
        st.caption("Model: loaded from ./model. You can retrain with src/train_model.py")


st.markdown("""
---
**Tips**
- Retrain with `src/train_model.py` to improve accuracy (domain data).
- For multilingual, retrain with `--model_name bert-base-multilingual-cased`.
- To add explainability (LIME), run:  
  `python src/explain.py --text "your news text here"`
""")
