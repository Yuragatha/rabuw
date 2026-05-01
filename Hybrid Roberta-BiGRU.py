#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
classification report	✅
confusion matrix csv	✅
roc auc csv	✅
runtime total	✅
avg epoch time	✅
epoch time log	✅
early stopping


@author: indri
"""

# ============================================================
# FINAL PIPELINE – ROBERTA + BiGRU (IMDB)
# Reviewer-grade, logging + plot + timing
# ============================================================
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
from transformers import RobertaTokenizer, RobertaModel
import nltk
from nltk.corpus import wordnet
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
nltk.download("wordnet")
nltk.download("omw-1.4")

# ============================================================
# SETUP
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = rf"D:\LATIAN\ROBERTA-BIGRU.1.RAW_{datetime.now():%Y%m%d_%H%M%S}"
DATA_PATH = r"D:\LATIAN\PREPROCESS\1.RAW.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.csv")
CONFIG_FILE = os.path.join(OUTPUT_DIR, "config.csv")
PLOT_LC = os.path.join(OUTPUT_DIR, "learning_curve.png")
PLOT_ROC = os.path.join(OUTPUT_DIR, "roc_auc.png")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
torch.cuda.manual_seed_all(SEED)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")


# ============================================================
# DATASET & TOKENIZER
# ============================================================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_LEN = 128
BATCH_SIZE = 32

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def synonym_replacement(self, text, p=0.3):
        words = text.split()
        new_words = []
        for w in words:
            if random.random() < p:
                syns = wordnet.synsets(w)
                if syns:
                    lemmas = syns[0].lemma_names()
                    if lemmas:
                        new_words.append(lemmas[0].replace("_"," "))
                        continue
            new_words.append(w)
        return " ".join(new_words)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.augment and random.random() < 0.3:
            text = self.synonym_replacement(text)
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

# ============================================================
# DATASETS
# ============================================================


df = pd.read_csv(DATA_PATH)
df["sentiment"] = (
    df["sentiment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"0":"negative","1":"positive"})
    .map({"negative":0,"positive":1})
)

#df["sentiment"] = df["sentiment"].map({"negative":0,"positive":1})
#texts = df['review'].astype(str).values
#labels = df["sentiment"].astype(int).values

texts = df['review'].astype(str).tolist()
labels = df['sentiment'].tolist()

#=======================================================
train_idx = np.load(r"D:\LATIAN\SPLIT IDX\train.npy")
val_idx   = np.load(r"D:\LATIAN\SPLIT IDX\val.npy")
test_idx  = np.load(r"D:\LATIAN\SPLIT IDX\test.npy")

train_texts = [texts[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]

val_texts = [texts[i] for i in val_idx]
val_labels = [labels[i] for i in val_idx]

test_texts = [texts[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]

train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN, augment=True)
val_dataset = IMDBDataset(val_texts, val_labels, tokenizer, MAX_LEN, augment=False)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LEN, augment=False)

g = torch.Generator()
g.manual_seed(SEED)

"""
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
"""
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          worker_init_fn=seed_worker,
                          generator=g)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# ============================================================
# MODEL
# ============================================================
class RoBERTa_BiGRU(nn.Module):
    def __init__(self, bert_model="roberta-base", hidden_dim=256, num_classes=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(bert_model)

        embedding_dim = self.bert.config.hidden_size

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 4, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = out.last_hidden_state

        gru_out, _ = self.gru(last_hidden_state)

        avg_pool = torch.mean(gru_out, 1)
        max_pool, _ = torch.max(gru_out, 1)

        feat = torch.cat((avg_pool, max_pool), 1)

        return self.fc(self.dropout(feat))

model = RoBERTa_BiGRU().to(DEVICE)


# ============================================================
# LOGGING CONFIG & SAVE
# ============================================================
config_dict = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "max_len": MAX_LEN,
    "optimizer": "AdamW",
    "lr": 1.22e-06,
    "weight_decay": 0.01,
    "epochs": 50,
    "model": "ROBERTA_BiGRU",
    "device": str(DEVICE)
}
pd.DataFrame([config_dict]).to_csv(CONFIG_FILE, index=False)

# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                    best_val_acc, early_stopper,
                    train_losses, val_losses, train_accs, val_accs,
                    path):

    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_acc": best_val_acc,

        # RNG STATES
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),

        # EARLY STOP
        "early_best": early_stopper.best,
        "early_counter": early_stopper.counter,

        # HISTORY
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs

    }, path)

# ============================================================
# TRAINING SETUP
# ============================================================
optimizer = optim.AdamW(model.parameters(), lr=config_dict["lr"], weight_decay=config_dict["weight_decay"])
criterion = nn.CrossEntropyLoss()
scaler = amp.GradScaler(enabled=torch.cuda.is_available())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",        # karena loss harus turun
    patience=4,
    factor=0.7
)
EPOCHS = config_dict["epochs"]

# ============================================================
# TRAIN / EVAL FUNCTIONS
# ============================================================
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for input_ids, attn, labels in tqdm(loader):
        input_ids, attn, labels = input_ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with amp.autocast('cuda'):
            logits = model(input_ids, attn)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        correct += (torch.argmax(logits, 1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for input_ids, attn, labels in loader:
            input_ids, attn, labels = input_ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)
            logits = model(input_ids, attn)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
    return total_loss / len(loader), correct / len(loader.dataset), np.array(all_preds), np.array(all_labels), np.array(all_probs)
# ============================================================
# EARLY STOPPING CLASS
# ============================================================
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stop = True

        return self.stop

early_stopper = EarlyStopping(patience=4)

# ============================================================
#  RESUME CHECKPOINT
# ============================================================
start_epoch = 0
best_val_acc = 0

if os.path.exists(CHECKPOINT_PATH):
    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    random.setstate(ckpt["python_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    torch.set_rng_state(ckpt["torch_rng"])
    torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    early_stopper.best = ckpt["early_best"]
    early_stopper.counter = ckpt["early_counter"]

    train_losses = ckpt["train_losses"]
    val_losses   = ckpt["val_losses"]
    train_accs   = ckpt["train_accs"]
    val_accs     = ckpt["val_accs"]

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_val_acc = ckpt["best_val_acc"]

    print(f"Resumed from epoch {start_epoch} | best_val_acc={best_val_acc:.4f}")


# ============================================================
# TRAINING LOOP + TIMING + LOGGING
# ============================================================
train_start_time = time.time()

if not os.path.exists(CHECKPOINT_PATH):
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

mode = "a" if start_epoch > 0 else "w"
with open(LOG_FILE, mode, newline="", encoding="utf-8") as f:

    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()

        # ===== TRAIN =====
        model.train()
        train_loss, correct = 0,0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True, dynamic_ncols=True)

        for input_ids, attn, labels in loop:
            input_ids, attn, labels = input_ids.to(DEVICE), attn.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            with amp.autocast('cuda'):
                logits = model(input_ids, attn)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            correct += (logits.argmax(1)==labels).sum().item()

            #loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)


        # ===== VALIDATION =====
        val_loss, val_acc, _, _, _ = evaluate(model,val_loader)


        # ===== LR Scheduler FIX =====
        scheduler.step(val_acc)


        # ===== BEST MODEL SAVE =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)


        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]['lr']

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # ===== CHECKPOINT SAVE =====
        save_checkpoint(
        epoch, model, optimizer, scheduler, scaler,
        best_val_acc, early_stopper,
        train_losses, val_losses, train_accs, val_accs,
        CHECKPOINT_PATH
        )

        # ===== LOG =====
        with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch+1,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                lr_now,
                round(epoch_time,2),
                round(time.time()-train_start_time,2)
            ])


        print(f"Epoch {epoch+1} | Train {train_acc:.4f} | Val {val_acc:.4f} | LR {lr_now:.2e} | {epoch_time:.1f}s")


        # ===== EARLY STOP =====
        if early_stopper.step(val_acc):
            print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
            break




total_runtime = time.time() - train_start_time
avg_epoch_time = avg_epoch_time = total_runtime / (epoch+1)
runtime_file= os.path.join(OUTPUT_DIR,"runtime.csv")

# save runtime info
pd.DataFrame([{"total_runtime_s":total_runtime,
               "avg_epoch_s":avg_epoch_time}]).to_csv(runtime_file,index=False)

# ============================================================
# FINAL TEST EVAL + ROC
# ============================================================
best_model = RoBERTa_BiGRU().to(DEVICE)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
#best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
test_loss, test_acc, y_pred, y_true, y_probs = evaluate(best_model, test_loader)
#_, _, y_pred, y_true, y_probs = evaluate(best_model, test_loader)

# ============================================================
# PLOT LEARNING CURVE
# ============================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.plot([test_loss]*EPOCHS, '--', label="Test Loss (final)")  # garis putus-putus
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(); plt.grid(True)
plt.savefig(PLOT_LC,dpi=300)
plt.close()

plt.figure(figsize=(8,5))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.plot([test_acc]*EPOCHS, '--', label="Test Acc (final)")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(); plt.grid(True)
plt.savefig(PLOT_LC.replace(".png","_acc.png"),dpi=300)
plt.close()


print("\n===== CLASSIFICATION REPORT =====")
report_dict = classification_report(
    y_true,
    y_pred,
    target_names=["Negative","Positive"],
    output_dict=True
)

report_df = pd.DataFrame(report_dict).T
report_df.index.name = "class"

cols = ["precision","recall","f1-score","support"]
report_df = report_df[cols]

report_df.to_csv(
    os.path.join(OUTPUT_DIR,"classification_report.csv"),
    float_format="%.6f"
)

print("\n===== CLASSIFICATION REPORT =====")
print(report_df)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
plt.title("Confusion Matrix - Final Test")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.savefig(os.path.join(OUTPUT_DIR,"confusion_matrix.png"),dpi=300)
plt.close()
pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR,"confusion_matrix.csv"), index=False)


fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve - Test")
plt.legend(); plt.grid(True)
plt.savefig(PLOT_ROC,dpi=300)
plt.close()
pd.DataFrame([{"roc_auc":roc_auc}]).to_csv(os.path.join(OUTPUT_DIR,"roc_auc.csv"),index=False)


runtime_file = os.path.join(OUTPUT_DIR,"runtime.csv")
pd.DataFrame([{
    "total_runtime_s": total_runtime,
    "avg_epoch_s": avg_epoch_time
}]).to_csv(runtime_file,index=False)


print(f"\n✅ Total runtime: {total_runtime/60:.2f} min | Avg epoch: {avg_epoch_time:.2f}s")
print(f"✅ All outputs saved to {OUTPUT_DIR}")


# ============================================================
# EXPERIMENT SUMMARY SAVE
# ============================================================

summary_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")

best_epoch = np.argmax(val_accs) + 1 if len(val_accs) > 0 else 0


summary_dict = {
    "model": "ROBERTA_BiGRU",
    "best_epoch": best_epoch,
    "best_val_acc": max(val_accs),
    "final_test_acc": test_acc,
    "final_test_loss": test_loss,
    "roc_auc": roc_auc,
    "total_runtime_s": total_runtime,
    "avg_epoch_s": avg_epoch_time,
    "batch_size": BATCH_SIZE,
    "max_len": MAX_LEN,
    "lr": config_dict["lr"],
    "weight_decay": config_dict["weight_decay"],
    "epochs": EPOCHS,
    "seed": SEED
}

summary_df = pd.DataFrame([summary_dict])

# append jika file sudah ada
if os.path.exists(summary_path):
    summary_df.to_csv(summary_path, mode="a", header=False, index=False)
else:
    summary_df.to_csv(summary_path, index=False)

print("✅ Experiment summary saved.")

