#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# In[6]:


rm /content/drive/MyDrive/Colab\ Notebooks/TESIS\ FIX/SPLIT\ IDX/cache/*.npy

# In[7]:


# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 2026

NEW PREPROCESSING WITHOUT LOWERCASING
@author: indri
"""

import os
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer

# =========================================================
# CONFIG
# =========================================================
SEED = 42
DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/DATA/IMDB Dataset.csv"
SAVE_DIR = "/content/drive/MyDrive/Colab Notebooks/PREPROCESS"

CACHE_DIR = "/content/drive/MyDrive/Colab Notebooks/TESIS FIX/SPLIT IDX/cache"
SPLIT_DIR = "/content/drive/MyDrive/Colab Notebooks/TESIS FIX/SPLIT IDX"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

np.random.seed(SEED)

# =========================================================
# DOWNLOAD NLTK
# =========================================================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(DATA_PATH)

df["sentiment"] = df["sentiment"].map({
    "positive":1,
    "negative":0
})

texts = df["review"].astype(str).tolist()
labels = df["sentiment"].tolist()

# =========================================================
# SPLIT INDEX (ONCE ONLY)
# =========================================================
if not os.path.exists(f"{SPLIT_DIR}/train.npy"):

    idx = np.arange(len(df))

    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=SEED,
        stratify=np.array(labels)[temp_idx]
    )

    np.save(f"{SPLIT_DIR}/train.npy", train_idx)
    np.save(f"{SPLIT_DIR}/val.npy", val_idx)
    np.save(f"{SPLIT_DIR}/test.npy", test_idx)

    print("✔ Split index saved")

else:
    print("✔ Split index already exists")

# =========================================================
# NLP TOOLS
# =========================================================
stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ

    if tag.startswith('V'):
        return wordnet.VERB

    if tag.startswith('N'):
        return wordnet.NOUN

    if tag.startswith('R'):
        return wordnet.ADV

    return wordnet.NOUN


# =========================================================
# PREPROCESSING FUNCTIONS
# =========================================================
def remove_stop(tokens):

    return [
        w for w in tokens
        if w not in stop_words
    ]


def lemmatize(pos_tokens):

    return [
        lemmatizer.lemmatize(w, get_wordnet_pos(p))
        for w,p in pos_tokens
    ]


def stem(tokens):

    return [
        stemmer.stem(w)
        for w in tokens
    ]


def retag(tokens):

    return nltk.pos_tag(tokens)


# =========================================================
# TOKEN CACHE
# =========================================================
TOK_PATH = f"{CACHE_DIR}/tokens.npy"
POS_PATH = f"{CACHE_DIR}/pos.npy"

if not os.path.exists(TOK_PATH):

    print("🔹 Tokenizing once...")

    tokens_all = [
        word_tokenize(t)
        for t in tqdm(texts)
    ]

    np.save(TOK_PATH, np.array(tokens_all, dtype=object))

else:

    print("✔ Loading token cache...")
    tokens_all = np.load(TOK_PATH, allow_pickle=True)


if not os.path.exists(POS_PATH):

    print("🔹 POS tagging once...")

    pos_all = [
        retag(tokens)
        for tokens in tqdm(tokens_all)
    ]

    np.save(POS_PATH, np.array(pos_all, dtype=object))

else:

    print("✔ Loading POS cache...")
    pos_all = np.load(POS_PATH, allow_pickle=True)


# =========================================================
# PROCESS ENGINE
# =========================================================
def process_config(steps, name):

    out = []

    for i in tqdm(range(len(tokens_all)), desc=name):

        tokens = list(tokens_all[i])
        pos = list(pos_all[i])

        for s in steps:

            if s == "R":

                tokens = remove_stop(tokens)
                pos = retag(tokens)

            elif s == "L":

                tokens = lemmatize(pos)
                pos = retag(tokens)

            elif s == "S":

                tokens = stem(tokens)
                pos = retag(tokens)

        out.append(" ".join(tokens))

    return out


# =========================================================
# PREPROCESSING CONFIGS (NO LOWERCASING)
# =========================================================
configs = {

    "2.L.csv": ["L"],

    "3.R.csv": ["R"],

    "4.S.csv": ["S"],

    "6.R.L.csv": ["R","L"],

    "7.R.S.csv": ["R","S"],

    "9.L.S.csv": ["L","S"],

    "11.R.L.S.csv": ["R","L","S"]

}

# =========================================================
# RUN PREPROCESSING
# =========================================================
for filename, steps in configs.items():

    save_path = os.path.join(SAVE_DIR, filename)

    if os.path.exists(save_path):

        print(f"✔ Exists → {filename}")
        continue

    print(f"\n🚀 Processing {filename} : {steps}")

    processed = process_config(steps, filename)

    df_out = pd.DataFrame({

        "review": processed,
        "sentiment": labels

    })

    df_out.to_csv(
        save_path,
        index=False,
        encoding="utf-8"
    )

    print(f"✅ Saved → {filename}")

print("\n🎯 ALL DATASETS GENERATED SUCCESSFULLY")

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
