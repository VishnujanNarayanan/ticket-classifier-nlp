"""Tests for the SQL storage layer.

These run on a synthetic export rather than the real spreadsheet, because the
dataset is gitignored and never reaches CI. The fixture is built to exercise
each rule in sql/clean_tickets.sql at least once.
"""
from pathlib import Path

import pandas as pd
import pytest

from tickets_db import build_database, class_distribution, load_clean_tickets

ROWS = [
    # id, text, issue_type, urgency, product
    (1, "My laptop is broken", "Product Defect", "High", "laptop"),
    (2, "My laptop is broken", "Product Defect", "High", "laptop"),   # duplicate body
    (3, "Where is my order", "Late Delivery", "Medium", "phone"),
    (4, None, "Billing Problem", "Low", "charger"),                   # no text
    (5, "Charger never arrived", None, "Low", "charger"),             # no issue label
    (6, "Cannot sign in", "Account Access", None, "laptop"),          # no urgency label
    (7, "   ", "General Inquiry", "Low", "phone"),                    # whitespace-only text
    (8, "Battery swollen", "Product Defect", "High", "battery"),
]


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    export = tmp_path / "export.xlsx"
    pd.DataFrame(
        ROWS,
        columns=["ticket_id", "ticket_text", "issue_type", "urgency_level", "product"],
    ).to_excel(export, index=False)

    db_path = tmp_path / "tickets.db"
    assert build_database(export, db_path) == len(ROWS)
    return db_path


def test_unlabelled_and_empty_rows_are_dropped(db: Path) -> None:
    clean = load_clean_tickets(db)
    assert set(clean["ticket_id"]) == {1, 3, 8}


def test_duplicate_bodies_collapse_to_the_lowest_id(db: Path) -> None:
    clean = load_clean_tickets(db)
    laptop = clean[clean["ticket_text"] == "My laptop is broken"]
    assert len(laptop) == 1
    assert laptop.iloc[0]["ticket_id"] == 1


def test_clean_set_is_deterministic(db: Path) -> None:
    first = load_clean_tickets(db)
    second = load_clean_tickets(db)
    pd.testing.assert_frame_equal(first, second)


def test_rebuilding_does_not_accumulate_rows(db: Path, tmp_path: Path) -> None:
    export = tmp_path / "export.xlsx"
    assert build_database(export, db) == len(ROWS)
    assert len(load_clean_tickets(db)) == 3


def test_class_distribution_sums_to_one_hundred_percent(db: Path) -> None:
    dist = class_distribution(db)
    assert dist["tickets"].sum() == 3
    assert round(dist["pct_of_total"].sum()) == 100


def test_missing_columns_are_rejected(tmp_path: Path) -> None:
    export = tmp_path / "bad.xlsx"
    pd.DataFrame({"ticket_id": [1], "ticket_text": ["hi"]}).to_excel(export, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        build_database(export, tmp_path / "bad.db")
