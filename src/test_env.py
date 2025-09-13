# test_env.py
import torch
import transformers
import sklearn
import pandas as pd
import numpy as np

print("✅ Torch version:", torch.__version__)
print("✅ Transformers version:", transformers.__version__)
print("✅ Scikit-learn version:", sklearn.__version__)
print("✅ Pandas version:", pd.__version__)
print("✅ NumPy version:", np.__version__)

# Quick torch test
x = torch.rand(3, 3)
print("✅ Torch tensor works:\n", x)

# Quick transformers test
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Fake news detection project test!", return_tensors="pt")
outputs = model(**inputs)

print("✅ Transformers forward pass success, output shape:", outputs.last_hidden_state.shape)
