"""Publish the Gradio app to a Hugging Face Space.

NOTE: Hugging Face now requires a PRO subscription to host Gradio or Docker Spaces --
only static Spaces are free -- so this returns 402 Payment Required on a free account.
The live deployment uses Streamlit Community Cloud instead (see streamlit_app.py and the
README). This script is kept because it works as-is the moment an account has PRO.

    huggingface-cli login          # once, needs a WRITE token
    python train.py ai_dev_assignment_tickets_complex_1000.xlsx
    python deploy_space.py         # optional: --space-id user/name

The Space gets the app, the pipeline and the trained artifacts — but never the ticket
spreadsheet. Artifacts are the whole point of persisting the models: the Space loads
them and serves immediately instead of retraining on every cold start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DEFAULT_SPACE = "ticket-classifier-nlp"

#: Uploaded as-is. requirements-space.txt is renamed on the way up, because a Space
#: installs from a file literally called requirements.txt.
FILES = {
    "app.py": "app.py",
    "pipeline.py": "pipeline.py",
    "predictor.py": "predictor.py",
    "requirements-space.txt": "requirements.txt",
}
ARTIFACT_FILES = ["tfidf.joblib", "scaler.joblib", "issue_model.joblib",
                  "urgency_model.joblib", "metrics.json"]

SPACE_README = """---
title: Customer Support Ticket Classifier
emoji: 🎫
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Customer Support Ticket Classifier

Predicts **issue type** and **urgency level** for a free-text support ticket, and
extracts the product, any dates, and the complaint wording.

Classical NLP only — TF-IDF over unigrams and bigrams plus handcrafted signal features
(VADER sentiment, lengths, punctuation, all-caps), a Random Forest for issue type and a
K-nearest-neighbour model for urgency. No LLMs.

**Urgency is not reliable.** On the held-out split it scores below the majority-class
baseline: the ticket text does not carry that label. It is shown rather than hidden,
because reporting a negative result is the honest thing to do.

Source: https://github.com/VishnujanNarayanan/ticket-classifier-nlp
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-id", help="owner/name; defaults to <your-username>/" + DEFAULT_SPACE)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    missing = [f for f in ARTIFACT_FILES if not (ARTIFACTS / f).exists()]
    if missing:
        print(f"missing artifacts {missing} — run `python train.py <export.xlsx>` first",
              file=sys.stderr)
        return 1

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        print("not logged in — run `huggingface-cli login` with a WRITE token",
              file=sys.stderr)
        return 1

    space_id = args.space_id or f"{who['name']}/{DEFAULT_SPACE}"
    api.create_repo(space_id, repo_type="space", space_sdk="gradio",
                    private=args.private, exist_ok=True)
    print(f"space: {space_id}")

    readme = ROOT / ".space_readme.md"
    readme.write_text(SPACE_README, encoding="utf8")
    try:
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=space_id, repo_type="space")
        for src, dest in FILES.items():
            api.upload_file(path_or_fileobj=str(ROOT / src), path_in_repo=dest,
                            repo_id=space_id, repo_type="space")
        for name in ARTIFACT_FILES:
            api.upload_file(path_or_fileobj=str(ARTIFACTS / name),
                            path_in_repo=f"artifacts/{name}",
                            repo_id=space_id, repo_type="space")
    finally:
        readme.unlink(missing_ok=True)

    print(f"deployed: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
