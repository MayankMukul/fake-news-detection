"""
Fake News Detection - Hybrid Model (Sentence-BERT + Classical ML)
Dataset: Kaggle Fake and Real News Dataset (Fake.csv, True.csv)

TRAIN:
    python fake_news_project.py train --data_dir path/to/dataset

RUN APP:
    streamlit run fake_news_project.py

It will load model.joblib automatically.
"""

import os
import sys
import argparse
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# try sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SBER_AVAILABLE = True
except:
    SBER_AVAILABLE = False

MODEL_PATH = "model.joblib"
EMB_MODEL = "all-mpnet-base-v2"


# ✅ LOAD & MERGE KAGGLE DATASET
def load_kaggle_dataset(data_dir):
    fake_path = os.path.join(data_dir, "Fake.csv")
    real_path = os.path.join(data_dir, "True.csv")

    if not os.path.exists(fake_path) or not os.path.exists(real_path):
        raise FileNotFoundError("Fake.csv or True.csv not found in provided folder")

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 0
    real_df["label"] = 1

    # Normalize column names
    for df in [fake_df, real_df]:
        df.rename(columns={"title": "title", "text": "text"}, inplace=True)
        df["text"] = df["text"].astype(str)
        df["title"] = df["title"].astype(str)

    df = pd.concat([fake_df, real_df], ignore_index=True)
    df["full_text"] = df["title"] + ". " + df["text"]

    df = df.dropna(subset=["full_text"])
    return df


# ✅ EMBEDDINGS
def compute_embeddings(texts):
    if not SBER_AVAILABLE:
        raise RuntimeError("Install sentence-transformers")
    model = SentenceTransformer(EMB_MODEL)
    return model.encode(texts, batch_size=32, show_progress_bar=True)


# ✅ TRAIN TF-IDF BASELINE
def train_tfidf_baseline(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    params = {"clf__C": [0.1, 1, 10]}
    gs = GridSearchCV(pipeline, params, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs


# ✅ TRAIN EMBEDDING MODEL
def train_embedding_model(X_emb_train, y_train):
    clf = LogisticRegression(max_iter=2000)
    params = {"C": [0.1, 1, 10]}
    gs = GridSearchCV(clf, {"C": [0.1, 1, 10]}, scoring="f1", cv=3, n_jobs=-1)
    gs.fit(X_emb_train, y_train)
    return gs


# ✅ EVALUATION
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        print("\nROC-AUC:", roc_auc_score(y_test, y_proba))


# ✅ SAVE MODEL
def save_model(model):
    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


# ✅ TRAIN MODE
def run_training(args):
    print("\n📥 Loading Kaggle Fake–Real Dataset...")
    df = load_kaggle_dataset(args.data_dir)
    print("Dataset size:", len(df))

    X = df["full_text"].tolist()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ✅ Baseline TF-IDF
    print("\n🔧 Training TF-IDF baseline...")
    tfidf_model = train_tfidf_baseline(X_train, y_train)
    print("\n✅ Best TF-IDF:", tfidf_model.best_params_)
    print("\n📊 Baseline Results:")
    evaluate(tfidf_model, X_test, y_test)

    # ✅ Hybrid embedding model
    if SBER_AVAILABLE:
        print("\n🔧 Generating sentence embeddings...")
        X_emb_train = compute_embeddings(X_train)
        X_emb_test = compute_embeddings(X_test)

        print("\n🔧 Training Embedding + Logistic Regression model...")
        emb_model = train_embedding_model(X_emb_train, y_train)

        print("\n✅ Best Embedding model:", emb_model.best_params_)
        print("\n📊 Embedding Model Results:")
        evaluate(emb_model, X_emb_test, y_test)

        # save best (embedding)
        best = {"type": "emb", "model": emb_model}
    else:
        print("\n⚠️ sentence-transformers not installed → saving TF-IDF model only")
        best = {"type": "tfidf", "model": tfidf_model}

    save_model(best)
    print("\n✅ Training Complete!")


# ✅ STREAMLIT APP
def run_streamlit():
    import streamlit as st

    st.title(" Fake News Detector ")
    
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Train first!")
        return

    pack = joblib.load(MODEL_PATH)
    model_type = pack["type"]
    model = pack["model"]

    title = st.text_input("News Title")
    text = st.text_area("News Body", height=200)

    if st.button("Predict"):
        full = title + ". " + text

        if model_type == "emb":
            # compute embedding
            sb = SentenceTransformer(EMB_MODEL)
            emb = sb.encode([full])
            pred = model.predict(emb)[0]
            prob = model.predict_proba(emb)[0][pred]
        else:
            pred = model.predict([full])[0]
            prob = model.predict_proba([full])[0][pred]

        st.subheader("Prediction:")
        st.write("✅ REAL NEWS" if pred == 1 else "❌ FAKE NEWS")
        st.write(f"Confidence: {prob:.3f}")


# ✅ ENTRY POINT
if __name__ == "__main__":
    if 'streamlit' in sys.modules:
        run_streamlit()
        sys.exit(0)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    train_p = sub.add_parser("train")
    train_p.add_argument("--data_dir", required=True, help="Folder containing Fake.csv & True.csv")

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    else:
        print("Use:")
        print(" python fake_news_project.py train --data_dir <folder>")
        print(" streamlit run fake_news_project.py")
