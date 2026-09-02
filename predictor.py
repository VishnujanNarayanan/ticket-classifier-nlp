"""Load the persisted artifacts and score one ticket.

Both the Gradio app and the notebook call this, so a served prediction is built by the
exact code path training used -- same column order, same scaler.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib

import pipeline

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


@lru_cache(maxsize=1)
def load_artifacts(artifacts: Path = ARTIFACTS) -> dict:
    missing = [
        n for n in ("tfidf", "scaler", "issue_model", "urgency_model")
        if not (artifacts / f"{n}.joblib").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing artifacts {missing} in {artifacts} — run `python train.py` first"
        )
    pipeline.ensure_nltk()
    return {n: joblib.load(artifacts / f"{n}.joblib")
            for n in ("tfidf", "scaler", "issue_model", "urgency_model")}


def predict_ticket(text: str, artifacts: Path = ARTIFACTS) -> dict:
    """Classify one ticket and pull its entities out."""
    art = load_artifacts(artifacts)
    clean = pipeline.preprocess_text(text)
    entities = pipeline.extract_entities(text)
    meta = pipeline.build_meta_frame([text])
    matrix = pipeline.build_matrix([clean], meta, art["tfidf"], art["scaler"])
    return {
        "issue_type": str(art["issue_model"].predict(matrix)[0]),
        "urgency_level": str(art["urgency_model"].predict(matrix)[0]),
        "entities": entities,
    }
