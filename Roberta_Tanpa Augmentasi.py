#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 08:33:23 2026

@author: indri
"""
# =========================================================
# CLEAN RESEARCH PIPELINE — DISTILBERT BASELINE
# =========================================================
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')
import os, json, random, time, csv
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm


# ================= CONFIG =================
CONFIG = {
    "model_name":"roberta-base",
    "max_len":128,
    "batch_size":32,
    "epochs":50,
    "lr":1e-5,
    "patience":4,
    "seed":42,
    "num_labels":2
}
OUTPUT_DIR = rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/ROBERTA_BASELINE/ROBERTA_BASELINE_1.RAW.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
json.dump(CONFIG, open(f"{OUTPUT_DIR}/config.json","w"), indent=4)

DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/1.RAW.csv"

# ================= REPRO =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TOKENIZER =================
tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model_name"])
# ============================================================
# DATASET & TOKENIZER
# ============================================================
BATCH_SIZE = 32

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=(CONFIG["max_len"])):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx])

        )

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH).dropna()

all_reviews = df['review'].astype(str).values
all_labels = df["sentiment"].replace({"0":0, "1":1}).values # Pastikan integer

train_idx = np.load(r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/SPLIT IDX/train.npy")
val_idx   = np.load(r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/SPLIT IDX/val.npy")
test_idx  = np.load(r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/SPLIT IDX/test.npy")

# 2. Distribusikan berdasarkan index yang sudah di-load (Anti-Leakage)
train_texts = [all_reviews[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]

val_texts = [all_reviews[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]

test_texts = [all_reviews[i] for i in test_idx]
test_labels = [all_labels[i] for i in test_idx]

# 3. Verifikasi Kebocoran (Tambahkan ini untuk bukti di Tesis)
assert len(set(train_idx) & set(test_idx)) == 0, "LEAKAGE DETECTED: Train & Test overlap!"
assert len(set(train_idx) & set(val_idx)) == 0, "LEAKAGE DETECTED: Train & Val overlap!"

#==============================================
#KALAU MAU PAKE SAMPLE
#==============================================

DEBUG = False
DEBUG_RATIO = 0.05

if DEBUG:
    train_idx = train_idx[:int(len(train_idx)*DEBUG_RATIO)]
    val_idx   = val_idx[:int(len(val_idx)*DEBUG_RATIO)]
    test_idx  = test_idx[:int(len(test_idx)*DEBUG_RATIO)]


# 3. Distribusikan data (Gunakan List Comprehension seperti di BiGRU)
train_texts = [all_reviews[i] for i in train_idx]
train_labels = [all_labels[i] for i in train_idx]

val_texts = [all_reviews[i] for i in val_idx]
val_labels = [all_labels[i] for i in val_idx]

test_texts = [all_reviews[i] for i in test_idx]
test_labels = [all_labels[i] for i in test_idx]

# 4. Inisialisasi Dataset yang BENAR (Hapus vocab, samakan nama class)
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset   = TextDataset(val_texts, val_labels, tokenizer)
test_dataset  = TextDataset(test_texts, test_labels, tokenizer)

g = torch.Generator()
g.manual_seed(CONFIG["seed"])

def seed_worker(worker_id):
    worker_seed = (CONFIG["seed"]) + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

train_loader = DataLoader(train_dataset,
                          batch_size=(CONFIG["batch_size"]),
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g)
val_loader = DataLoader(val_dataset, batch_size=(CONFIG["batch_size"]))
test_loader = DataLoader(test_dataset, batch_size=(CONFIG["batch_size"]))

# ================= MODEL =================

class RoBERTaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(CONFIG["model_name"])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(768, CONFIG["num_labels"])

    def forward(self, input_ids, attn):
        out = self.bert(input_ids=input_ids, attention_mask=attn)

        # RoBERTa tidak punya token [CLS] eksplisit seperti BERT,
        # tapi tetap pakai index 0 sebagai representasi
        cls = out.last_hidden_state[:, 0]

        cls = self.dropout(cls)
        return self.fc(cls)

model = RoBERTaClassifier().to(device)

# ================= OPTIM =================
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-3)
current_lr = optimizer.param_groups[0]["lr"]
print(f"LR = {current_lr:.2e}")
scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler(enabled=torch.cuda.is_available())

# ================= TRAIN =================
def train_epoch():
    model.train()
    total_loss, preds, gold = 0, [], []

    for ids,attn,labels in tqdm(train_loader, desc="Training", leave=False):
        ids,attn,labels = ids.to(device), attn.to(device), labels.to(device)

        optimizer.zero_grad()

        with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(ids,attn)
            loss = criterion(logits,labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds.extend(torch.argmax(logits,1).cpu().numpy())
        gold.extend(labels.cpu().numpy())

    return total_loss/len(train_loader), preds, gold

# ================= EVAL =================
def evaluate(loader):
    model.eval()
    total_loss, preds, gold, probs = 0, [], [], []

    with torch.no_grad():
        for ids,attn,labels in tqdm(loader, desc="Evaluating", leave=False):
            ids,attn,labels = ids.to(device), attn.to(device), labels.to(device)

            logits = model(ids,attn)
            loss = criterion(logits,labels)

            total_loss += loss.item()

            p = torch.softmax(logits,1)
            preds.extend(torch.argmax(p,1).cpu().numpy())
            gold.extend(labels.cpu().numpy())
            probs.extend(p[:,1].cpu().numpy())

    return total_loss/len(loader), preds, gold, probs


CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoint.pt"

start_epoch = 0
best_acc = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_acc"]

    random.setstate(ckpt["python_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
# ================= TRAIN LOOP =================
best_acc = 0
patience = 0
history = []
total_start = time.time()
train_start = time.time()
for epoch in range(start_epoch, CONFIG["epochs"]):

    tr_loss,tr_p,tr_g = train_epoch()
    val_loss,val_p,val_g,_ = evaluate(val_loader)

    #print(f"Epoch {epoch+1} | train {tr_loss:.4f} | val {val_loss:.4f}")
    tr_acc = accuracy_score(tr_g, tr_p)
    val_acc = accuracy_score(val_g, val_p)
    history.append([epoch+1,tr_loss,val_loss,tr_acc,val_acc])

    scheduler.step(val_acc)

    print(f"Epoch {epoch+1} | "
          f"train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | "
          f"train_acc {tr_acc:.4f} | val_acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience = 0
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best.pt")
    else:
        patience += 1
        if patience >= CONFIG["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break # INI YANG KURANG!
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_acc,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state()
    }, CHECKPOINT_PATH)

train_time = time.time() - train_start
# ================= TEST =================
test_start = time.time()
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best.pt"))

test_loss,test_p,test_g,test_prob = evaluate(test_loader)

acc  = accuracy_score(test_g,test_p)
prec = precision_score(test_g,test_p)
rec  = recall_score(test_g,test_p)
f1   = f1_score(test_g,test_p)

print("\nTEST RESULT")
print(acc,prec,rec,f1)
test_time = time.time() - test_start
total_time = time.time() - total_start
# ================= TEST =================
# ... (kode evaluasi Anda)

# 1. HITUNG WAKTU SEBELUM SIMPAN
inference_time_total = test_time
num_samples = len(test_texts)
avg_inference_per_sample = (inference_time_total / num_samples) * 1000

# 2. SIMPAN METRICS (SATU KALI SAJA - JANGAN DOUBLE)
metrics_data = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "loss": test_loss,
    "time": {
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "avg_inference_ms": avg_inference_per_sample,
        "total_time_sec": total_time
    }
}

with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)




# ================= CURVES =================
epochs=[x[0] for x in history]

plt.plot(epochs,[x[1] for x in history],label="train")
plt.plot(epochs,[x[2] for x in history],label="val")
plt.legend(); plt.grid()
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
plt.close()

plt.plot(epochs,[x[3] for x in history],label="train acc")
plt.plot(epochs,[x[4] for x in history],label="val acc")
plt.legend(); plt.grid()
plt.savefig(f"{OUTPUT_DIR}/accuracy_curve.png")
plt.close()


# ================= ROC =================
fpr,tpr,_ = roc_curve(test_g,test_prob)
roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label=f"AUC={roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.legend(); plt.grid()
plt.savefig(f"{OUTPUT_DIR}/roc.png")
plt.close()


# ================= CM =================
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(test_g, test_p)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative","Positive"]
)

plt.figure(figsize=(6,6))
disp.plot(values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# ================= SAVE TRAINING LOGS (CSV) =================
log_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
log_df.to_csv(f"{OUTPUT_DIR}/training_log.csv", index=False)

# ================= SAVE DETAILED CLASSIFICATION REPORT =================
from sklearn.metrics import classification_report
report = classification_report(test_g, test_p, target_names=["Negative", "Positive"], output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{OUTPUT_DIR}/classification_report.csv")

# ================= SAVE PREDICTIONS (Lebih Lengkap untuk Analisis) =================
pred_df = pd.DataFrame({
    "text": test_texts,        # Tambahkan ini agar bisa dibaca manusia
    "actual": test_g,
    "predicted": test_p,
    "probability": test_prob
})
pred_df.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)
