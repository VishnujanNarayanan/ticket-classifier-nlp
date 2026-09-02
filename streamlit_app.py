"""Streamlit front end for the ticket classifier.

Deployed on Streamlit Community Cloud, which serves straight from the GitHub repo.
It loads the committed artifacts through predictor.py, so the browser demo and the
notebook produce identical predictions from identical code.

`app.py` is the Gradio equivalent, kept for local use.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from predictor import predict_ticket

METRICS = Path(__file__).resolve().parent / "artifacts" / "metrics.json"

EXAMPLES = {
    "Broken product": "My phone crashed on 25/05/2024 and it hasn't worked since. Very frustrated!",
    "Billing": "I was charged twice for order #29224. Can you refund the duplicate?",
    "Late delivery": "Where is my delivery? It was due on 03/06/2024 and still hasn't arrived.",
    "General question": "Can you tell me more about the laptop warranty? Is it available in white?",
}

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫", layout="centered")
st.title("Customer Support Ticket Classifier")
st.write(
    "Paste a support ticket. The model predicts the **issue type** and the "
    "**urgency level**, and pulls out the product, any dates, and the complaint wording."
)

with st.sidebar:
    st.header("About")
    st.write(
        "Classical NLP only — TF-IDF over unigrams and bigrams plus handcrafted signals "
        "(VADER sentiment, lengths, punctuation, all-caps), a Random Forest for issue "
        "type and K-nearest neighbours for urgency. No LLMs."
    )
    if METRICS.exists():
        m = json.loads(METRICS.read_text(encoding="utf8"))
        st.metric("Tickets trained on", m["tickets"])
        st.metric("Features", m["features"])
        st.metric("Issue-type accuracy", m["issue_accuracy"])
        st.metric(f"Urgency accuracy (k={m['urgency_best_k']})", m["urgency_best_accuracy"])
    st.warning(
        "**Urgency is not reliable.** It scores below the majority-class baseline — the "
        "ticket text does not carry that label. Shown rather than hidden, because "
        "reporting a negative result beats burying it."
    )
    st.caption("[Source on GitHub](https://github.com/VishnujanNarayanan/ticket-classifier-nlp)")

choice = st.selectbox("Load an example", ["(write my own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "")
ticket = st.text_area("Ticket text", value=default, height=140,
                      placeholder="Enter a support ticket...")

if st.button("Classify", type="primary"):
    if not ticket.strip():
        st.info("Enter a ticket first.")
    else:
        with st.spinner("Classifying..."):
            result = predict_ticket(ticket)
        left, right = st.columns(2)
        left.success(f"**Issue type**\n\n{result['issue_type']}")
        right.info(f"**Urgency**\n\n{result['urgency_level']}")
        st.subheader("Extracted entities")
        st.json(result["entities"])
