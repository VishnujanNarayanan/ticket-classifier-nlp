"""Feature pipeline shared by training, inference and the notebook.

Everything that turns raw ticket text into a model-ready row lives here, once.

Why this module exists: the notebook used to build its feature vector twice — once
column-by-column during training, and again by hand inside ``predict_ticket``. The two
constructions disagreed (different column order, and inference skipped the scaler
entirely), so a served prediction was not the prediction the trained model would have
made. ``FEATURE_ORDER`` and ``build_meta_frame`` are now the single definition, used by
training and serving alike.
"""
from __future__ import annotations

import re

import nltk
import numpy as np
import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

NLTK_PACKAGES = ("punkt", "averaged_perceptron_tagger", "stopwords", "wordnet", "vader_lexicon")

PRODUCT_LIST = ["laptop", "phone", "charger", "headphones", "battery"]
COMPLAINT_KEYWORDS = [
    "broken", "late", "error", "issue", "crash", "not working", "damaged", "fail",
]

#: The exact column order the models were trained on. Training and inference both build
#: their matrix from this list, so the two cannot drift apart.
FEATURE_ORDER = (
    ["num_products", "num_dates", "num_complaints", "has_product", "has_date", "has_complaint"]
    + [f"complaint_{kw.replace(' ', '_')}" for kw in COMPLAINT_KEYWORDS]
    + ["ticket_length", "sentiment", "exclamation_count", "question_count",
       "all_caps_count", "char_length"]
)

_lemmatizer = WordNetLemmatizer()
_stop_words = None
_sentiment = None


def ensure_nltk() -> None:
    """Download the corpora the pipeline needs, quietly and only once."""
    for pkg in NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


def _stopwords() -> set:
    global _stop_words
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
    return _stop_words


def _analyzer() -> SentimentIntensityAnalyzer:
    """One VADER analyser for the process; it used to be rebuilt on every call."""
    global _sentiment
    if _sentiment is None:
        _sentiment = SentimentIntensityAnalyzer()
    return _sentiment


def get_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess_text(text: str) -> str:
    """Lowercase, strip punctuation, drop stopwords, POS-aware lemmatisation."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(
        _lemmatizer.lemmatize(w, get_wordnet_pos(t))
        for w, t in pos_tag(tokens)
        if w not in _stopwords()
    )


def extract_entities(text: str) -> dict:
    """Products, dates and complaint wording, matched literally against fixed lists."""
    text = str(text)
    lowered = text.lower()
    return {
        "products": [p for p in PRODUCT_LIST if p in lowered],
        "dates": re.findall(
            r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", text
        ),
        "complaint_keywords": [
            kw for kw in COMPLAINT_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", lowered)
        ],
    }


def meta_features(text: str, clean: str, entities: dict) -> dict:
    """The handcrafted signal columns for one ticket, keyed by name."""
    row = {
        "num_products": len(entities["products"]),
        "num_dates": len(entities["dates"]),
        "num_complaints": len(entities["complaint_keywords"]),
        "has_product": int(len(entities["products"]) > 0),
        "has_date": int(len(entities["dates"]) > 0),
        "has_complaint": int(len(entities["complaint_keywords"]) > 0),
        "ticket_length": len(clean.split()),
        "sentiment": _analyzer().polarity_scores(text)["compound"],
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "all_caps_count": sum(1 for w in text.split() if w.isupper()),
        "char_length": len(text),
    }
    for kw in COMPLAINT_KEYWORDS:
        row[f"complaint_{kw.replace(' ', '_')}"] = int(kw in entities["complaint_keywords"])
    return row


def build_meta_frame(texts) -> pd.DataFrame:
    """Handcrafted features for many tickets, in FEATURE_ORDER."""
    rows = []
    for text in texts:
        clean = preprocess_text(text)
        rows.append(meta_features(text, clean, extract_entities(text)))
    return pd.DataFrame(rows, columns=FEATURE_ORDER)


def build_matrix(clean_texts, meta_frame, tfidf, scaler) -> np.ndarray:
    """Join sparse TF-IDF columns to the scaled handcrafted ones, training-side order."""
    tfidf_part = tfidf.transform(clean_texts).toarray()
    meta_part = scaler.transform(meta_frame[FEATURE_ORDER].astype(np.float32).values)
    return np.hstack([tfidf_part, meta_part])
