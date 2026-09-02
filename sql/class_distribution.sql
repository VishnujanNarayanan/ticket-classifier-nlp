-- Class balance per issue type and urgency level, for the README's results
-- table and for spotting a target that is too skewed to learn.
SELECT issue_type,
       urgency_level,
       COUNT(*)                                        AS tickets,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
FROM clean_tickets
GROUP BY issue_type, urgency_level
ORDER BY tickets DESC, issue_type, urgency_level;
