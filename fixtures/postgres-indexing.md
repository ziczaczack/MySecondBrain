# PostgreSQL B-tree indexing and query planning notes

B-tree indexes are the default in PostgreSQL and work well for equality and range queries on ordered data.
The query planner uses table statistics from ANALYZE to estimate row counts and pick between index and sequential scans.
Run EXPLAIN ANALYZE to see the actual plan, including whether an index scan or a slower seq scan was chosen.
Composite indexes help when columns are queried together, but column order matters for which predicates can use them.
A partial index on a filtered subset keeps the index small and fast for common WHERE conditions.
Watch for index bloat after many updates and consider REINDEX or autovacuum tuning to keep performance steady.
