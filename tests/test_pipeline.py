"""Tests for the shared feature pipeline.

The bug these exist to prevent: training built its feature vector in one column order
and `predict_ticket` rebuilt it in another, while also skipping the scaler. Nothing
caught it, because nothing compared the two paths.
"""
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

import pipeline

TICKETS = [
    "My laptop is broken and the charger failed on 25/05/2024!",
    "Where is my order? It was late again.",
    "Can you tell me about the battery warranty?",
    "PHONE CRASHED. Not working at all!!!",
]


@pytest.fixture(scope="module", autouse=True)
def _corpora():
    pipeline.ensure_nltk()


def test_feature_order_has_no_duplicates_and_expected_width():
    assert len(pipeline.FEATURE_ORDER) == len(set(pipeline.FEATURE_ORDER))
    assert len(pipeline.FEATURE_ORDER) == 20


def test_meta_frame_columns_are_exactly_feature_order():
    frame = pipeline.build_meta_frame(TICKETS)
    assert list(frame.columns) == list(pipeline.FEATURE_ORDER)
    assert len(frame) == len(TICKETS)


def test_entity_extraction_finds_products_dates_and_complaints():
    ents = pipeline.extract_entities(TICKETS[0])
    assert ents["products"] == ["laptop", "charger"]
    assert ents["dates"] == ["25/05/2024"]
    assert "broken" in ents["complaint_keywords"]


def test_preprocess_is_deterministic_and_strips_punctuation():
    once = pipeline.preprocess_text(TICKETS[3])
    assert once == pipeline.preprocess_text(TICKETS[3])
    assert "!" not in once and once == once.lower()


def test_all_caps_and_exclamations_are_counted_from_raw_text():
    frame = pipeline.build_meta_frame([TICKETS[3]])
    assert frame.loc[0, "exclamation_count"] == 3
    assert frame.loc[0, "all_caps_count"] >= 2


def test_build_matrix_is_column_order_independent_of_input_frame():
    """A frame whose columns are shuffled must still produce the same matrix.

    This is the regression guard: build_matrix selects FEATURE_ORDER explicitly, so a
    caller cannot silently feed the columns in a different order.
    """
    clean = [pipeline.preprocess_text(t) for t in TICKETS]
    meta = pipeline.build_meta_frame(TICKETS)

    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2)).fit(clean)
    scaler = MinMaxScaler().fit(meta[pipeline.FEATURE_ORDER].astype(np.float32).values)

    straight = pipeline.build_matrix(clean, meta, tfidf, scaler)
    shuffled = pipeline.build_matrix(clean, meta[list(reversed(pipeline.FEATURE_ORDER))],
                                     tfidf, scaler)
    np.testing.assert_allclose(straight, shuffled)


def test_matrix_width_is_tfidf_plus_handcrafted():
    clean = [pipeline.preprocess_text(t) for t in TICKETS]
    meta = pipeline.build_meta_frame(TICKETS)
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2)).fit(clean)
    scaler = MinMaxScaler().fit(meta[pipeline.FEATURE_ORDER].astype(np.float32).values)
    matrix = pipeline.build_matrix(clean, meta, tfidf, scaler)
    assert matrix.shape == (len(TICKETS), len(tfidf.vocabulary_) + 20)


def test_scaled_columns_land_in_unit_range():
    clean = [pipeline.preprocess_text(t) for t in TICKETS]
    meta = pipeline.build_meta_frame(TICKETS)
    tfidf = TfidfVectorizer(max_features=50).fit(clean)
    scaler = MinMaxScaler().fit(meta[pipeline.FEATURE_ORDER].astype(np.float32).values)
    tail = pipeline.build_matrix(clean, meta, tfidf, scaler)[:, -20:]
    # float32 scaling lands a hair outside [0, 1]; the point is the columns are scaled
    # at inference at all, which the old predict_ticket path skipped entirely.
    assert tail.min() >= -1e-6 and tail.max() <= 1 + 1e-6
