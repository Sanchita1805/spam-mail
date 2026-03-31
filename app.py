from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

try:
    import nltk
    from nltk.corpus import stopwords
except:
    nltk = None
    stopwords = None


# ---------------- CONFIG ----------------
DATASET_NAME = "spam_ham_dataset.csv"
NB_ALPHA = 0.3

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 2,
}

SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9\s]+")
WHITESPACE_PATTERN = re.compile(r"\s+")


# ---------------- DATASET ----------------
def load_dataset():
    path = Path(DATASET_NAME)
    if not path.exists():
        st.error("❌ Dataset not found. Upload spam_ham_dataset.csv to your GitHub repo.")
        st.stop()

    return pd.read_csv(path)


# ---------------- STOPWORDS ----------------
def get_stop_words():
    fallback = set(ENGLISH_STOP_WORDS)

    if nltk is None or stopwords is None:
        return fallback

    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

    return set(stopwords.words("english"))


# ---------------- CLEAN TEXT ----------------
def clean_text(text, stop_words):
    text = str(text).lower()
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])


# ---------------- TRAIN MODEL ----------------
@dataclass
class ModelArtifacts:
    model: MultinomialNB
    vectorizer: TfidfVectorizer
    accuracy: float
    stop_words: set


@st.cache_resource
def train():
    df = load_dataset()

    # detect columns
    if "text" in df.columns:
        text_col = "text"
    elif "message" in df.columns:
        text_col = "message"
    else:
        st.error("❌ Dataset must have 'text' or 'message' column")
        st.stop()

    if "label" not in df.columns:
        st.error("❌ Dataset must have 'label' column")
        st.stop()

    # encode labels
    df["label"] = df["label"].astype(str).str.lower()
    df["label"] = df["label"].map({"ham": 0, "spam": 1, "0": 0, "1": 1})

    stop_words = get_stop_words()

    df["clean"] = df[text_col].apply(lambda x: clean_text(x, stop_words))

    X = df["clean"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    return ModelArtifacts(model, vectorizer, acc, stop_words)


# ---------------- PREDICT ----------------
def predict(text, artifacts):
    cleaned = clean_text(text, artifacts.stop_words)
    vec = artifacts.vectorizer.transform([cleaned])

    pred = artifacts.model.predict(vec)[0]
    prob = artifacts.model.predict_proba(vec)[0]

    return pred, max(prob)


# ---------------- UI ----------------
def main():
    st.set_page_config(page_title="Spam Detection", layout="wide")

    st.title("📧 Spam Email Detection using NLP")

    artifacts = train()

    st.write(f"📊 Accuracy: **{artifacts.accuracy:.2%}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Email")
        text = st.text_area("Type email here...")

        if st.button("Predict"):
            if not text.strip():
                st.warning("Enter text first")
            else:
                pred, conf = predict(text, artifacts)

                if pred == 1:
                    st.error(f"🚫 Spam (Confidence: {conf:.2%})")
                else:
                    st.success(f"✅ Not Spam (Confidence: {conf:.2%})")

    with col2:
        st.subheader("Examples")

        if st.button("Normal Email"):
            st.info("Hi, let’s meet tomorrow at 10 AM.")

        if st.button("Spam Email"):
            st.info("Congratulations! You won a free iPhone!")

        if st.button("Phishing Email"):
            st.info("URGENT: Verify your bank account immediately!")

    st.markdown("---")
    st.caption("Built using NLP + Machine Learning")


if __name__ == "__main__":
    main()
