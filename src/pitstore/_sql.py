"""SQL used by the store. Kept in one place so the queries can be read and reviewed together."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    name        VARCHAR PRIMARY KEY,
    description VARCHAR NOT NULL DEFAULT '',
    unit        VARCHAR NOT NULL DEFAULT '',
    created_at  TIMESTAMP NOT NULL DEFAULT current_timestamp
);
CREATE TABLE IF NOT EXISTS feature_values (
    feature      VARCHAR   NOT NULL,
    entity       VARCHAR   NOT NULL,
    event_time   TIMESTAMP NOT NULL,  -- the period the value describes (valid time)
    available_at TIMESTAMP NOT NULL,  -- when this version of the value was published
    value        DOUBLE    NOT NULL,
    ingested_at  TIMESTAMP NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (feature, entity, event_time, available_at)
);
-- Derived, read-optimised: for each (feature, entity), the value a point-in-time query
-- returns from each publication time onwards. Maintained by ingest for the entities a batch
-- touches, so queries are a single ASOF join.
CREATE TABLE IF NOT EXISTS timeline (
    feature      VARCHAR   NOT NULL,
    entity       VARCHAR   NOT NULL,
    available_at TIMESTAMP NOT NULL,
    event_time   TIMESTAMP NOT NULL,
    value        DOUBLE    NOT NULL
);
"""

# One lookup of the incoming batch against stored values. Rows with a stored value are either
# exact repeats (ignored) or conflicts (rejected); rows without one are new. Everything
# downstream works from this table, so re-sent rows cost nothing.
STAGE = """
CREATE OR REPLACE TEMP TABLE _staged AS
SELECT i.entity, i.event_time, i.available_at, i.value, f.value AS stored
FROM _incoming i
LEFT JOIN feature_values f
  ON f.feature = $feature AND f.entity = i.entity
 AND f.event_time = i.event_time AND f.available_at = i.available_at
"""

CONFLICTS = """
SELECT entity, event_time, available_at, stored, value AS incoming
FROM _staged
WHERE stored IS NOT NULL AND stored <> value
LIMIT 5
"""

COUNT_NEW = "SELECT count(*) FROM _staged WHERE stored IS NULL"

INSERT_NEW = """
INSERT INTO feature_values (feature, entity, event_time, available_at, value)
SELECT $feature, entity, event_time, available_at, value FROM _staged WHERE stored IS NULL
"""

# Point-in-time semantics. At time t the answer is the value of the latest event known at t,
# in its latest version known at t. Sorted by publication time, the latest known event is a
# running maximum, and only rows that reach it ("defining" rows) can ever be the answer;
# revisions of older events cannot. The defining rows form a step function of publication
# time (the timeline), which an ASOF join evaluates at each spine time.
_TIMELINE_ROWS = """
WITH ordered AS (
    SELECT entity, event_time, available_at, value,
           max(event_time) OVER (
               PARTITION BY entity ORDER BY available_at, event_time
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS latest_known_event
    FROM feature_values
    WHERE feature = $feature
)
SELECT $feature AS feature, entity, available_at,
       max(event_time) AS event_time,
       arg_max(value, event_time) AS value
FROM ordered
WHERE event_time = latest_known_event
GROUP BY entity, available_at
"""

# Incremental refresh. New rows can only change the timeline from their own publication time
# onwards, so for each touched entity the rows published before its earliest new row are kept.
# The rows after it are recomputed with the running maximum seeded by the last kept row: by
# construction that row's event time is the latest event known just before the change.
# An append-only batch therefore costs a scan of the entity, not a sort of its whole history.
STAGE_STARTS = """
CREATE OR REPLACE TEMP TABLE _starts AS
SELECT entity, min(available_at) AS start FROM _staged WHERE stored IS NULL GROUP BY entity
"""

_DELETE_FROM_STARTS = """
DELETE FROM timeline t
USING _starts s
WHERE t.feature = $feature AND t.entity = s.entity AND t.available_at >= s.start
"""

_TIMELINE_ROWS_FROM_STARTS = """
WITH seed AS (
    SELECT s.entity, s.start, max(t.event_time) AS prior_latest
    FROM _starts s
    LEFT JOIN timeline t
      ON t.feature = $feature AND t.entity = s.entity AND t.available_at < s.start
    GROUP BY s.entity, s.start
),
ordered AS (
    SELECT v.entity, v.event_time, v.available_at, v.value, d.prior_latest,
           max(v.event_time) OVER (
               PARTITION BY v.entity ORDER BY v.available_at, v.event_time
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS latest_since_start
    FROM feature_values v
    JOIN seed d ON v.entity = d.entity AND v.available_at >= d.start
    WHERE v.feature = $feature
)
SELECT $feature AS feature, entity, available_at,
       max(event_time) AS event_time,
       arg_max(value, event_time) AS value
FROM ordered
WHERE event_time = latest_since_start
  AND (prior_latest IS NULL OR event_time >= prior_latest)
GROUP BY entity, available_at
"""

# DuckDB binds parameters only in the last statement of a query, so each step is a statement.
REFRESH_TIMELINE = (
    _DELETE_FROM_STARTS,
    "INSERT INTO timeline " + _TIMELINE_ROWS_FROM_STARTS,
)

# The full rebuild deliberately uses the unseeded query above: the tests check that the two
# paths produce identical tables.
REBUILD_TIMELINE = (
    "DELETE FROM timeline WHERE feature = $feature",
    "INSERT INTO timeline " + _TIMELINE_ROWS,
)

POINT_IN_TIME = """
SELECT s.row_id, t.value, t.event_time, t.available_at
FROM _spine s
ASOF LEFT JOIN (SELECT * FROM timeline WHERE feature = $feature) t
  ON s.entity = t.entity AND s.ts >= t.available_at
"""

# What a join on event time against a downloaded (final, revised) data set produces: the
# latest event up to t, in its final version, regardless of when it was published.
NAIVE = """
WITH final AS (
    SELECT entity, event_time,
           max(available_at) AS available_at,
           arg_max(value, available_at) AS value
    FROM feature_values
    WHERE feature = $feature
    GROUP BY entity, event_time
)
SELECT s.row_id, f.value, f.event_time, f.available_at
FROM _spine s
ASOF LEFT JOIN final f
  ON s.entity = f.entity AND s.ts >= f.event_time
"""

SNAPSHOT = """
SELECT entity, event_time,
       arg_max(value, available_at) AS value,
       max(available_at) AS available_at,
       count(*) AS versions_known
FROM feature_values
WHERE feature = $feature AND available_at <= $as_of
GROUP BY entity, event_time
ORDER BY entity, event_time
"""

HISTORY = """
SELECT entity, event_time, available_at, value
FROM feature_values
WHERE feature = $feature
ORDER BY entity, event_time, available_at
"""

DESCRIBE = """
WITH per_event AS (
    SELECT feature, entity, event_time,
           count(*) AS releases,
           min(available_at) AS first_release,
           max(available_at) AS last_release,
           arg_min(value, available_at) AS first_value,
           arg_max(value, available_at) AS final_value
    FROM feature_values
    GROUP BY feature, entity, event_time
)
SELECT p.feature,
       count(DISTINCT p.entity)                      AS entities,
       count(*)                                      AS events,
       sum(p.releases)::BIGINT                       AS rows,
       sum(p.releases - 1)::BIGINT                   AS revisions,
       min(p.event_time)                             AS first_event,
       max(p.event_time)                             AS last_event,
       max(p.last_release)                           AS last_update,
       median(epoch(p.first_release) - epoch(p.event_time)) / 86400 AS median_lag_days,
       max(epoch(p.first_release) - epoch(p.event_time)) / 86400    AS max_lag_days,
       avg(abs(p.final_value - p.first_value))       AS mean_abs_revision,
       any_value(f.description)                      AS description,
       any_value(f.unit)                             AS unit
FROM per_event p
LEFT JOIN features f ON f.name = p.feature
GROUP BY p.feature
ORDER BY p.feature
"""
