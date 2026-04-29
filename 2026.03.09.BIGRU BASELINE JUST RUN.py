#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import zipfile

zip_path = "/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BIGRU_BASELINE/glove.6B.zip"
extract_path = "/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction finished")

# # **1**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/1.RAW.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/1.RAW.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **2**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/2.L{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/2.L.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **3**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/3.R.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/3.R.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **4**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/4.S.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/4.S.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **5**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/5.R.L{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/5.R.L.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **6.R.S**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/6.R.S.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/6.R.S.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **7**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/7.L.S.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/7.L.S.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()

# # **8.R.L.S**

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on 6 mar 2026

- split index
- embedding false
- tanpa preprocess lagi. murni model aja
- optimizer adamw aja.

@author: indri
"""

# =========================================================
# REVIEWER-GRADE FULL PIPELINE: BiGRU + GloVe
# =========================================================

import os, json, re, random, time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "max_len": 128,
    "batch_size": 32,
    "epochs": 50,
    "lr": 2.03e-4,
    "embed_dim": 300,
    "hidden": 256,
    "min_freq": 2,
    "max_vocab": 30000,
    "seed": 42,
    "patience": 4,
    "test_size": 0.2,
    "val_size": 0.1,
    "glove_path": r"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/GLOVE/glove.6B.300d.txt"
}

# =========================================================
# SEED + DEVICE
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TOKENIZER
# =========================================================
"""
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"[^a-z0-9<>!?.,']+", " ", t)
    return t.split()
"""
# =========================================================
# TOKENIZER
# =========================================================

def clean_text(t):
    return t.split()
# =========================================================
# VOCAB
# =========================================================

def build_vocab(texts):
    c = Counter()
    for t in texts:
        c.update(clean_text(t))
    vocab = {"<PAD>":0, "<UNK>":1}
    for w, f in c.most_common(CONFIG["max_vocab"]):
        if f >= CONFIG["min_freq"]:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    toks = clean_text(text)
    ids = [vocab.get(w, 1) for w in toks][:CONFIG["max_len"]]
    l = len(ids)
    ids += [0]*(CONFIG["max_len"]-l)
    return ids, l


# =========================================================
# DATASET
# =========================================================

def synonym_replace(tokens, p=0.1):
    new = tokens.copy()
    for i in range(len(new)):
        if random.random() < p:
            new[i] = new[i]  # placeholder kalau belum pakai wordnet
    return new

def random_delete(tokens, p=0.1):
    if len(tokens) == 1:
        return tokens
    return [t for t in tokens if random.random() > p]

def random_swap(tokens, n=1):
    new = tokens.copy()
    for _ in range(n):
        i, j = random.sample(range(len(new)), 2)
        new[i], new[j] = new[j], new[i]
    return new

def augment(tokens):
    r = random.random()
    if r < 0.33:
        return synonym_replace(tokens)
    elif r < 0.66:
        return random_delete(tokens)
    else:
        return random_swap(tokens)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, augment=False):
        self.texts = texts
        self.labels = labels.values
        self.vocab = vocab
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = clean_text(self.texts[idx])

        if self.augment:
            if random.random() < 0.5:
                tokens = augment(tokens)

        ids = [self.vocab.get(w,1) for w in tokens][:CONFIG["max_len"]]
        l = len(ids)
        ids += [0]*(CONFIG["max_len"]-l)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "len": torch.tensor(l, dtype=torch.long),
            "y": torch.tensor(self.labels[idx], dtype=torch.long)
        }
# =========================================================
# GLOVE EMBEDDINGS
# =========================================================
def load_glove_embeddings(vocab):
    embeddings = np.random.normal(scale=0.1, size=(len(vocab), CONFIG["embed_dim"]))
    found = 0
    with open(CONFIG["glove_path"], encoding="utf8") as f:
        for line in f:
            sp=line.split()
            w=sp[0]
            if w in vocab:
                embeddings[vocab[w]]=np.asarray(sp[1:],dtype=np.float32)
                found+=1
    print(f"GloVe: found {found}/{len(vocab)} words")
    return torch.tensor(embeddings, dtype=torch.float32)

# =========================================================
# MODEL
# =========================================================
class BiGRU(nn.Module):
    def __init__(self, vocab_size, embeddings=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, CONFIG["embed_dim"], padding_idx=0)
        if embeddings is not None:
            self.emb.weight.data.copy_(embeddings)
            self.emb.weight.requires_grad = False

        self.gru = nn.GRU(
            CONFIG["embed_dim"],
            CONFIG["hidden"],
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3 if 2 > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.4)
        self.gru_dropout = 0.3
        self.fc = nn.Linear(CONFIG["hidden"]*2, 2)

    def forward(self, x, lengths):
        x = self.emb_dropout(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

# =========================================================
# LOAD DATA
# =========================================================


def main():
    RUN_DIR =rf"/content/drive/MyDrive/Colab Notebooks/TESIS FIX/BiGRU_BASELINE/8.R.L.S.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RUN_DIR, exist_ok=True)
    json.dump(CONFIG, open(f"{RUN_DIR}/config.json","w"), indent=4)

    #OUTPUT_DIR = r"FINAL TESIS\BIGRUBASELINE_0_20260224_141853"
    #DATA_PATH = r"D:\TOPIK RISET\LATIAN\PREPROCESSING\PREPROCESSING_20260215_135559\11.R-L-S.csv"
    #os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_PATH = r"/content/drive/MyDrive/Colab Notebooks/PREPROCESS/8.R.L.S.csv"
    df = pd.read_csv(DATA_PATH)
    """
    texts = df["review"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()"""

    #===============================================
    # KHUSUS KALAU MAU SAMPLE 10% DATASETS
    #===============================================

    """
    subset = 0.05

    train_idx = train_idx[:int(len(train_idx)*subset)]
    val_idx   = val_idx[:int(len(val_idx)*subset)]
    test_idx  = test_idx[:int(len(test_idx)*subset)]

    """
    #===============================================

    # 1. Ambil data mentah
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

    # 4. Bangun vocab HANYA dari train_texts
    vocab = build_vocab(train_texts)
    embeddings = load_glove_embeddings(vocab)

    # 5. Dataset (Gunakan list langsung, lebih stabil)
    train_dataset = IMDBDataset(train_texts, pd.Series(train_labels), vocab, augment=True)
    val_dataset   = IMDBDataset(val_texts, pd.Series(val_labels), vocab, augment=False)
    test_dataset  = IMDBDataset(test_texts, pd.Series(test_labels), vocab, augment=False)
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

    # =========================================================
    # =========================================================
    model = BiGRU(len(vocab), embeddings).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="max",        # karena monitor val_acc
        factor=0.5,        # lr dikali 0.5 saat stagnan
        patience=2,        # tunggu 1 epoch stagnan
        min_lr=1e-6,
        #verbose=True
    )
    crit = nn.CrossEntropyLoss()

    # =========================================================
    # CHECKPOINT
    # =========================================================
    CKPT=os.path.join(RUN_DIR,"last.pt")
    start_ep=1
    best_acc=-1
    pat=0

    if os.path.exists(CKPT):
        ck=torch.load(CKPT,map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_acc=ck["best"]
        start_ep=ck["ep"]+1
        pat=ck["pat"]
        torch.set_rng_state(ck["torch_rng"])
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        print("Resumed from epoch",start_ep)

    def test_clean_text_independence():
        text = "This is a TEST!"
        result_1 = clean_text(text)

        # Bayangkan kita punya dataset raksasa
        fake_global_stats = [random.random() for _ in range(1000)]

        result_2 = clean_text(text)

        assert result_1 == result_2, "Fungsi clean_text tidak konsisten!"
        print("Clean_text aman: Tidak terpengaruh data eksternal.")

    test_clean_text_independence()

    # =========================================================
    # TRAIN / EVAL FUNCTION
    # =========================================================
    def run(loader, train=True):
        model.train() if train else model.eval()
        preds, ys, probs = [], [], []
        loss_sum = 0
        for b in loader:
            x = b["ids"].long().to(device)
            l = b["len"].to(device)
            y = b["y"].to(device)
            if train: opt.zero_grad()
            with torch.set_grad_enabled(train):
                out = model(x, l)
                loss = crit(out, y)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            loss_sum += loss.item()*y.size(0)
            p = torch.softmax(out,1)[:,1]
            preds += torch.argmax(out,1).cpu().tolist()
            ys += y.cpu().tolist()
            probs += p.detach().cpu().tolist()
        return loss_sum/len(ys), accuracy_score(ys,preds), ys, preds, probs

    print("========== DEBUG ==========")
    print("Device:", device)
    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))
    print("Test size:", len(test_dataset))
    print("Train batches:", len(train_loader))
    print("Vocab size:", len(vocab))
    print("===========================")

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    patience = pat
    log = []
    epoch_times=[]
    start_time = time.time()

    from tqdm import tqdm
    for ep in tqdm(range(start_ep, CONFIG["epochs"]+1)):

        t0=time.time()

        tr_loss, tr_acc, _, _, _ = run(train_loader, True)
        va_loss, va_acc, _, _, _ = run(val_loader, False)
        scheduler.step(va_acc)
        current_lr = opt.param_groups[0]["lr"]
        print("LR:", current_lr)
        epoch_times.append(time.time()-t0)
        log.append([ep, tr_loss, va_loss, tr_acc, va_acc])
        print(f"Ep{ep} | TL {tr_loss:.3f} VL {va_loss:.3f} | TA {tr_acc:.3f} VA {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            patience = 0
            torch.save(model.state_dict(), f"{RUN_DIR}/best_model.pt")
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                print("Early stop")
                break

        # save checkpoint
        torch.save({
            "ep":ep,
            "model":model.state_dict(),
            "opt":opt.state_dict(),
            "sched": scheduler.state_dict(),   # ← tambah ini
            "best":best_acc,
            "pat":patience,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate()
        },CKPT)

    # =========================================================
    # TEST EVAL
    # =========================================================
    model.load_state_dict(torch.load(f"{RUN_DIR}/best_model.pt"))
    test_loss, test_acc, ys, preds, probs = run(test_loader, False)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)

    print("\nTEST RESULTS")
    print(f"Acc={test_acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    # =========================================================
    # REVISED INFERENCE TIME CALCULATION
    # =========================================================
    model.eval()
    num_samples = len(test_dataset)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0

    with torch.no_grad():
        for b in test_loader:
            x = b["ids"].to(device)
            l = b["len"].to(device)

            # SINKRONISASI WAJIB UNTUK GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()

            starter.record()
            _ = model(x, l)
            ender.record()

            if device.type == 'cuda':
                torch.cuda.synchronize()
                total_ms += starter.elapsed_time(ender)
            else:
                # Jika pakai CPU tetap pakai time.time
                pass

    inference_per_sample_ms = total_ms / num_samples
    # =========================================================
    # TIME LOGGING (FINAL REVISED)
    # =========================-================================
    avg_epoch_dist = np.mean(epoch_times)
    total_train_time = time.time() - start_time

    # Simpan dalam satu dictionary terpusat
    timing_stats = {
        "avg_time_per_epoch_sec": float(avg_epoch_dist),
        "total_training_time_sec": float(total_train_time),
        "total_epochs_completed": int(len(epoch_times)),
        "inference_latency_per_sample_ms": float(inference_per_sample_ms),
        "total_test_samples": int(num_samples),
        "device": str(device),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Simpan sebagai JSON
    with open(os.path.join(RUN_DIR, "timing_statistics.json"), "w") as f:
        json.dump(timing_stats, f, indent=4)

    print(f"\nEFFICIENCY REPORT:")
    print(f"Avg Time/Epoch: {avg_epoch_dist:.2f}s")
    print(f"Inference Latency: {inference_per_sample_ms:.4f} ms/sample")
    print(f"Stats saved to: {RUN_DIR}/timing_statistics.json")

    # =========================================================
    # SAVE LOGS
    # =========================================================
    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])\
      .to_csv(f"{RUN_DIR}/training_log.csv", index=False)

    pd.DataFrame([{
        "accuracy": test_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "loss": test_loss
    }]).to_csv(f"{RUN_DIR}/test_metrics.csv", index=False)

    pd.DataFrame(classification_report(ys,preds,output_dict=True))\
    .T.to_csv(os.path.join(RUN_DIR,"classification_report.csv"))

    cm=confusion_matrix(ys,preds)
    pd.DataFrame(cm).to_csv(os.path.join(RUN_DIR,"confusion_matrix.csv"),index=False)

    fpr,tpr,_=roc_curve(ys,probs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(RUN_DIR,"roc_curve.csv"),index=False)


    # =========================================================
    # ROC + AUC
    # =========================================================
    fpr, tpr, _ = roc_curve(ys, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(f"{RUN_DIR}/roc.png")
    plt.close()

    # =========================================================
    # CONFUSION MATRIX
    # =========================================================
    cm = confusion_matrix(ys, preds)

    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center",
                     color="black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{RUN_DIR}/cm.png")
    plt.close()

    # =========================================================
    # LEARNING CURVE
    # =========================================================
    df_log = pd.DataFrame(log, columns=["epoch","train_loss","val_loss","train_acc","val_acc"])

    plt.figure()
    plt.plot(df_log["train_acc"], label="Train")
    plt.plot(df_log["val_acc"], label="Val")
    plt.axhline(test_acc, linestyle="--", label="Test")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(f"{RUN_DIR}/acc_curve.png")
    plt.close()

    plt.figure()
    plt.plot(df_log["train_loss"], label="Train")
    plt.plot(df_log["val_loss"], label="Val")
    plt.axhline(test_loss, linestyle="--", label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{RUN_DIR}/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
