"""Train both classifiers and persist everything inference needs.

Running this writes artifacts/ so the app can serve predictions without retraining --
the notebook kept the vectoriser, scaler and models in kernel memory only, which is why
restarting meant retraining and why nothing could be deployed.

    python train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import pipeline
from tickets_db import DEFAULT_DB, build_database, load_clean_tickets

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42
K_VALUES = [3, 5, 7, 10, 15]


def train(excel_path: str | None = None, db_path=DEFAULT_DB, out_dir: Path = ARTIFACTS) -> dict:
    pipeline.ensure_nltk()
    if excel_path:
        build_database(excel_path, db_path)
    frame = load_clean_tickets(db_path)

    clean = [pipeline.preprocess_text(t) for t in frame["ticket_text"]]
    meta = pipeline.build_meta_frame(frame["ticket_text"])

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    tfidf.fit(clean)
    scaler = MinMaxScaler()
    scaler.fit(meta[pipeline.FEATURE_ORDER].astype(np.float32).values)

    matrix = pipeline.build_matrix(clean, meta, tfidf, scaler)

    x_train, x_test, issue_train, issue_test, urg_train, urg_test = train_test_split(
        matrix, frame["issue_type"], frame["urgency_level"],
        test_size=0.2, random_state=RANDOM_STATE,
    )

    issue_model = RandomForestClassifier(random_state=RANDOM_STATE).fit(x_train, issue_train)
    issue_report = classification_report(issue_test, issue_model.predict(x_test), output_dict=True)

    sweep, best_score, best_model, best_k = {}, -1.0, None, None
    for k in K_VALUES:
        knn = KNeighborsClassifier(n_neighbors=k).fit(x_train, urg_train)
        score = accuracy_score(urg_test, knn.predict(x_test))
        sweep[k] = round(float(score), 4)
        if score > best_score:
            best_score, best_model, best_k = score, knn, k

    # compress=3 takes the bundle from ~7.2 MB to ~0.3 MB — small enough to commit,
    # which is what lets a host that deploys straight from GitHub serve without
    # retraining. The KNN shrinks most, since it stores its training data.
    out_dir.mkdir(exist_ok=True)
    joblib.dump(tfidf, out_dir / "tfidf.joblib", compress=3)
    joblib.dump(scaler, out_dir / "scaler.joblib", compress=3)
    joblib.dump(issue_model, out_dir / "issue_model.joblib", compress=3)
    joblib.dump(best_model, out_dir / "urgency_model.joblib", compress=3)

    metrics = {
        "tickets": len(frame),
        "features": int(matrix.shape[1]),
        "issue_accuracy": round(float(issue_report["accuracy"]), 4),
        "urgency_sweep": sweep,
        "urgency_best_k": best_k,
        "urgency_best_accuracy": round(float(best_score), 4),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    return metrics


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else None
    print(json.dumps(train(source), indent=2))
