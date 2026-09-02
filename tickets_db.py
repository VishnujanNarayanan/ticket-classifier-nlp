"""SQL storage layer for the ticket dataset.

The notebook used to read the spreadsheet straight into pandas and do its
cleaning with ``dropna`` and ``drop_duplicates``. That worked, but it left the
cleaning rules buried in a notebook cell where nothing else could reuse or test
them. Here the export is loaded into a SQLite database once, and every consumer
asks SQL for the rows it wants.

Typical use::

    from tickets_db import build_database, load_clean_tickets

    build_database("ai_dev_assignment_tickets_complex_1000.xlsx")
    df = load_clean_tickets()
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent / "sql"
DEFAULT_DB = Path(__file__).resolve().parent / "tickets.db"
COLUMNS = ["ticket_id", "ticket_text", "issue_type", "urgency_level", "product"]


def _read_sql(name: str) -> str:
    return (SQL_DIR / name).read_text(encoding="utf8")


def connect(db_path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the ticket database."""
    return sqlite3.connect(str(db_path))


def build_database(
    excel_path: str | Path,
    db_path: str | Path = DEFAULT_DB,
) -> int:
    """Load the spreadsheet export into the ``tickets`` table.

    Returns the number of raw rows written. The table is recreated from
    ``sql/schema.sql`` on every call, so re-running against a fresh export is
    safe and leaves no rows behind from the previous one.
    """
    frame = pd.read_excel(excel_path)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"export is missing expected columns: {missing}")

    with connect(db_path) as conn:
        conn.executescript(_read_sql("schema.sql"))
        frame[COLUMNS].to_sql("tickets", conn, if_exists="append", index=False)
        (rows,) = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()
    return rows


def load_clean_tickets(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Return the modelling set: labelled rows, one per distinct ticket body.

    The filtering and de-duplication both happen in ``sql/clean_tickets.sql``,
    so the definition of "a usable ticket" lives in one place instead of being
    restated wherever the data is loaded.
    """
    with connect(db_path) as conn:
        return pd.read_sql_query(_read_sql("clean_tickets.sql"), conn)


def class_distribution(db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Ticket counts per issue type and urgency level, straight from SQL."""
    with connect(db_path) as conn:
        body = _read_sql("clean_tickets.sql").rstrip().rstrip(";")
        conn.executescript(
            "DROP VIEW IF EXISTS clean_tickets;\n"
            "CREATE TEMP VIEW clean_tickets AS " + body
        )
        return pd.read_sql_query(_read_sql("class_distribution.sql"), conn)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "ai_dev_assignment_tickets_complex_1000.xlsx"
    raw = build_database(source)
    clean = load_clean_tickets()
    print(f"loaded {raw} raw rows -> {len(clean)} usable tickets")
    print(class_distribution().to_string(index=False))
