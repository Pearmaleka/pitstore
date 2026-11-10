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

# Values in an incoming batch whose key already exists with a different value.
CONFLICTS = """
SELECT i.entity, i.event_time, i.available_at, f.value AS stored, i.value AS incoming
FROM _incoming i
JOIN feature_values f
  ON f.feature = $feature AND f.entity = i.entity
 AND f.event_time = i.event_time AND f.available_at = i.available_at
WHERE f.value <> i.value
LIMIT 5
"""

INSERT_NEW = """
INSERT INTO feature_values (feature, entity, event_time, available_at, value)
SELECT $feature, i.entity, i.event_time, i.available_at, i.value
FROM _incoming i
WHERE NOT EXISTS (
    SELECT 1 FROM feature_values f
    WHERE f.feature = $feature AND f.entity = i.entity
      AND f.event_time = i.event_time AND f.available_at = i.available_at
)
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
    WHERE feature = $feature {entity_filter}
)
SELECT $feature AS feature, entity, available_at,
       max(event_time) AS event_time,
       arg_max(value, event_time) AS value
FROM ordered
WHERE event_time = latest_known_event
GROUP BY entity, available_at
"""

_TOUCHED = "AND entity IN (SELECT DISTINCT entity FROM _incoming)"

# DuckDB binds parameters only in the last statement of a query, so each refresh is a pair.
REFRESH_TIMELINE_TOUCHED = (
    "DELETE FROM timeline WHERE feature = $feature " + _TOUCHED,
    "INSERT INTO timeline " + _TIMELINE_ROWS.format(entity_filter=_TOUCHED),
)

REBUILD_TIMELINE = (
    "DELETE FROM timeline WHERE feature = $feature",
    "INSERT INTO timeline " + _TIMELINE_ROWS.format(entity_filter=""),
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
