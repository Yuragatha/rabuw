#!/usr/bin/env python
# coding: utf-8

# text → preprocess → tokenizer → tensor → model → prediksi

# In[1]:


from tqdm import tqdm
import os, time, random, csv
import os, json, re, random, time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch import amp
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel
import nltk
from nltk.corpus import wordnet

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DistilBERT_BiGRU(nn.Module):
    def __init__(self, bert_model="distilbert-base-uncased", hidden_dim=256, num_classes=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(bert_model)
        embedding_dim = self.bert.config.hidden_size
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*4, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state
        gru_out, _ = self.gru(last_hidden_state)
        avg_pool = torch.mean(gru_out, 1)
        max_pool, _ = torch.max(gru_out, 1)
        feat = torch.cat((avg_pool, max_pool), 1)
        return self.fc(self.dropout(feat))

model = DistilBERT_BiGRU().to(DEVICE)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = MyModel(vocab_size, embed_dim, hidden_dim, num_classes)
model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/TESIS FIX/HYBRID/3_R/best_model.pt", map_location=device))
model.to(device)
model.eval()
#vocab = torch.load("vocab.pt")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#fungsi preprocessing
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()

    filtered_tokens = [
        word for word in tokens
        if word.lower() not in stop_words
    ]

    return " ".join(filtered_tokens)
"""
#masukkan teks testing untuk dichobaw
texts = [
    "I love this film",
    "This is terrible"
]
"""
import pandas as pd

# load dataset
df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/DATA/IMDB Dataset.csv")

# ambil 100 sample (acak tapi reproducible)
df_sample = df.sample(n=100, random_state=42)

texts = df_sample["review"].astype(str).tolist()
true_labels_str = df_sample["sentiment"].tolist() # optional (kalau mau evaluasi)
#===========================================
clean_text = [remove_stopwords(t) for t in texts]

print(clean_text)

inputs = tokenizer(
    clean_text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#inputs = torch.stack([encode(t, vocab) for t in texts]).to(device)

with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    preds = torch.argmax(outputs, dim=1)

true_labels = [str(l) for l in true_labels_str]

label_map = {0: "negative", 1: "positive"}

pred_labels = [label_map[p.item()] for p in preds]

for t, p in zip(texts, preds):
    print(t, "->", label_map[p.item()])

for i in range(5):  # tampilkan 5 saja biar tidak spam
    print("TEXT:", texts[i][:100])
    print("PRED:", pred_labels[i])
    print("-----")

# =========================
# 1. Buat dataframe hasil
# =========================
df_result = pd.DataFrame({
    "review": texts,
    "true_label": true_labels,
    "pred_label": pred_labels
})

# =========================
# 2. Match status
# =========================
df_result["match_status"] = np.where(
    df_result["true_label"] == df_result["pred_label"],
    "match",
    "not_match"
)

# =========================
# 3. Error type
# =========================
def error_type(row):
    if row["true_label"] == row["pred_label"]:
        return "correct"
    elif row["true_label"] == "positive" and row["pred_label"] == "negative":
        return "false_negative"
    elif row["true_label"] == "negative" and row["pred_label"] == "positive":
        return "false_positive"

df_result["error_type"] = df_result.apply(error_type, axis=1)

# =========================
# 4. Statistik
# =========================
match_percent = (df_result["match_status"] == "match").mean() * 100
not_match_percent = 100 - match_percent

print(f"Match: {match_percent:.2f}%")
print(f"Not Match: {not_match_percent:.2f}%")

# =========================
# 5. Summary row
# =========================
summary = pd.DataFrame({
    "review": ["SUMMARY"],
    "true_label": [""],
    "pred_label": [""],
    "match_status": [f"match={match_percent:.2f}%, not_match={not_match_percent:.2f}%"],
    "error_type": [""]
})

df_final = pd.concat([df_result, summary], ignore_index=True)

# =========================
# 6. Save (rapih & aman)
# =========================
SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/TESIS FIX"
os.makedirs(SAVE_PATH, exist_ok=True)

file_path = os.path.join(SAVE_PATH, "hasil_prediksi.tsv")

df_final.to_csv(file_path, index=False, sep="\t", encoding="utf-8")

print(f"✅ Tersimpan di: {file_path}")

# In[2]:


from sklearn.metrics import classification_report
print(classification_report(true_labels, pred_labels))

print(df_result["error_type"].value_counts())
