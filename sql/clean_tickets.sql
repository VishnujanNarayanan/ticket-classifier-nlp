-- The training set, defined in SQL rather than rebuilt in pandas.
--
-- Two rules, both of which used to live in notebook cell 2:
--   1. a row is unusable unless it has ticket text AND both labels
--   2. duplicate ticket bodies are collapsed to their lowest ticket_id
--
-- ROW_NUMBER() gives rule 2 a deterministic winner, so the row that survives
-- de-duplication is the same one on every run and the split stays reproducible.
SELECT ticket_id,
       ticket_text,
       issue_type,
       urgency_level,
       product
FROM (
    SELECT ticket_id,
           ticket_text,
           issue_type,
           urgency_level,
           product,
           ROW_NUMBER() OVER (
               PARTITION BY ticket_text
               ORDER BY ticket_id
           ) AS body_rank
    FROM tickets
    WHERE ticket_text   IS NOT NULL AND TRIM(ticket_text)   <> ''
      AND issue_type    IS NOT NULL AND TRIM(issue_type)    <> ''
      AND urgency_level IS NOT NULL AND TRIM(urgency_level) <> ''
)
WHERE body_rank = 1
ORDER BY ticket_id;
