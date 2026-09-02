"""Gradio app for the ticket classifier.

Loads the persisted artifacts and serves them; it does not train. This is the entry
point Hugging Face Spaces runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from predictor import predict_ticket

METRICS = Path(__file__).resolve().parent / "artifacts" / "metrics.json"

DESCRIPTION = """
Paste a customer support ticket. The app predicts the **issue type** and the
**urgency level**, and pulls out the product, any dates and the complaint wording.

**Read the urgency with caution.** On the held-out split, urgency scores below the
majority-class baseline — the text simply does not carry that label. It is shown here
because hiding a negative result would be worse than reporting it.
"""

EXAMPLES = [
    "My phone crashed on 25/05/2024 and it hasn't worked since. Very frustrated!",
    "I was charged twice for order #29224. Can you refund the duplicate?",
    "Where is my delivery? It was due on 03/06/2024 and still hasn't arrived.",
    "Can you tell me more about the laptop warranty? Also, is it available in white?",
]


def classify(ticket_text: str):
    if not ticket_text or not ticket_text.strip():
        return "—", "—", {"note": "enter a ticket first"}
    result = predict_ticket(ticket_text)
    return result["issue_type"], result["urgency_level"], result["entities"]


def build_interface() -> gr.Interface:
    article = ""
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf8"))
        article = (
            f"Trained on {m['tickets']} de-duplicated tickets, {m['features']} features. "
            f"Issue-type accuracy {m['issue_accuracy']}; urgency best "
            f"{m['urgency_best_accuracy']} at k={m['urgency_best_k']}."
        )
    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(lines=5, placeholder="Enter a support ticket...", label="Ticket"),
        outputs=[
            gr.Text(label="Issue Type"),
            gr.Text(label="Urgency Level"),
            gr.JSON(label="Extracted Entities"),
        ],
        title="Customer Support Ticket Classifier",
        description=DESCRIPTION,
        article=article,
        examples=EXAMPLES,
        allow_flagging="never",
    )


if __name__ == "__main__":  # pragma: no cover - manual entry point
    build_interface().launch()
