-- Raw landing table for the support-ticket export.
-- Loaded verbatim from the spreadsheet; no cleaning happens here, so the raw
-- rows stay inspectable after the cleaning query has run.
DROP TABLE IF EXISTS tickets;

CREATE TABLE tickets (
    ticket_id     INTEGER PRIMARY KEY,
    ticket_text   TEXT,
    issue_type    TEXT,
    urgency_level TEXT,
    product       TEXT
);

-- The cleaning query filters on these three columns and de-duplicates on
-- ticket_text, so index what it actually scans.
CREATE INDEX idx_tickets_text    ON tickets (ticket_text);
CREATE INDEX idx_tickets_labels  ON tickets (issue_type, urgency_level);
