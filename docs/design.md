# Design notes

## Data model

```sql
CREATE TABLE feature_values (
    feature      VARCHAR,
    entity       VARCHAR,
    event_time   TIMESTAMP,   -- the period the value describes (valid time)
    available_at TIMESTAMP,   -- when this version was published (knowledge time)
    value        DOUBLE,
    ingested_at  TIMESTAMP,
    PRIMARY KEY (feature, entity, event_time, available_at)
);
```

This is a bitemporal table in the sense of Snodgrass (1999), with one simplification:
`available_at` is the publication time of the source, not the time the row reached the
database (that is `ingested_at`, kept for auditing). Backtests care about when the world could
have known a number, not when our pipeline happened to load it.

A revision is a new row with the same `event_time` and a later `available_at`. Rows are never
updated or deleted, so any past state of the data can be reconstructed. Series without an
entity (a single macro indicator, say) are stored under the entity `"_"`.

All timestamps are naive UTC. Time-zone-aware input is converted on the way in, and every
timestamp is cut to microseconds, DuckDB's `TIMESTAMP` resolution. Batches and spines are passed
to DuckDB at that resolution too: a nanosecond column makes DuckDB cast the stored side of every
join, which cost more than the rest of a small update.

## Point-in-time semantics

For an entity and a time *t*, let *K(t)* be the rows with `available_at <= t`. The point-in-time
value is

1. the largest `event_time` in *K(t)*, then
2. among the rows of *K(t)* for that event, the one with the largest `available_at`.

This is what a person looking at the latest release on day *t* would have seen. Two
alternatives are wrong in ways that are easy to miss:

- **As-of join on `available_at`** (take the most recently published row). Q2 GDP is first
  published on July 25; an annual revision of Q1 comes out on September 26. On October 1 this
  join returns the Q1 revision, although Q2 is the latest quarter anyone could see.
- **As-of join on `event_time` against final values** (`mode="naive"`). This is the usual
  join on a downloaded CSV. It returns values that had not been published, and the final
  version of values that had.

## The timeline

Answering the definition row by row costs a filter per spine row. pitstore precomputes, for each
(feature, entity), the sequence of answers and when each one starts to hold.

Sort an entity's rows by `(available_at, event_time)` and let *m* be the running maximum of
`event_time`. A row can only be an answer if its `event_time` equals *m* at its position:

- if its `event_time` is larger than everything published before, it is a new latest event;
- if it equals the running maximum, it is a revision of the latest event, which replaces the
  previous answer;
- if it is smaller, it revises an older event and can never be the answer again, because a
  later event is already public and stays public.

Several rows can share an `available_at` (a batch publishing several periods at once). Among
those, the answer is the row with the largest `event_time`, so the timeline keeps one row per
(entity, `available_at`): `max(event_time)` and `arg_max(value, event_time)`. The primary key
guarantees one value per (`event_time`, `available_at`), so this is well defined.

By construction, `event_time` never decreases along the timeline, and the answer at *t* is the
last timeline row with `available_at <= t`. That is a standard as-of join:

```sql
SELECT s.row_id, t.value, t.event_time, t.available_at
FROM _spine s
ASOF LEFT JOIN (SELECT * FROM timeline WHERE feature = $feature) t
  ON s.entity = t.entity AND s.ts >= t.available_at
```

DuckDB executes this with a sort-merge, so a query is O((n + m) log(n + m)) for n timeline rows
and m spine rows, regardless of how many revisions were dropped from the timeline.

### Keeping the timeline current

The timeline is derived data and has to change whenever `feature_values` does. A new row can
only change answers from its own publication time onwards, so for each entity in a batch, the
timeline rows published before the entity's earliest new row stay as they are. Inside the
ingest transaction, after the new rows are inserted:

```sql
-- per touched entity, the earliest publication time among the new rows
CREATE TEMP TABLE _starts AS
SELECT entity, min(available_at) AS start FROM _staged WHERE stored IS NULL GROUP BY entity;

DELETE FROM timeline t USING _starts s
WHERE t.feature = $feature AND t.entity = s.entity AND t.available_at >= s.start;

INSERT INTO timeline
SELECT ...  -- the running-maximum query over rows with available_at >= start,
            -- seeded with the latest event known just before start
```

The seed is the `event_time` of the last timeline row that was kept. That is the running
maximum just before `start`: along the timeline `event_time` never decreases, and between two
timeline rows no row reaches a new maximum. A recomputed row is defining if it reaches the
running maximum taken from `start` and is not older than the seed.

The cost of an update therefore depends on how far back the batch reaches, not on how long the
history is. On a single series of 4 million rows, appending 1,000 rows takes 0.04 s (1.2 s
when the whole entity was recomputed); a backfill of 1,000 rows into the middle of the history
still recomputes the half after it, in 0.5 s.

`rebuild_timeline()` recomputes everything with the unseeded query. The two paths are separate
implementations of the same definition, and the tests check them against each other: a property
test ingests random histories in shuffled batches and compares the maintained table with a
rebuild, and a unit test covers appends, a backfill before every stored row and a late revision
of an older event.

### Versions that were tried first

1. A correlated subquery per spine row. Correct, and the basis of `reference.py`, but quadratic.
2. The running-maximum window computed inside every query. Correct, but each query paid for a
   sort of the whole history; on a million rows it was slower than pandas.
3. The materialised timeline, recomputed in full for every entity a batch touched. Correct,
   but a daily update of one long series paid for a sort of its entire history.
4. The materialised timeline with the seeded refresh above.

## Ingestion guarantees

`ingest` registers the batch as a DuckDB view and runs, in one transaction:

1. one left join of the batch against stored values into a temporary table `_staged`;
2. a conflict check on it: if any incoming (feature, entity, `event_time`, `available_at`) is
   already stored with a different value, the transaction is rolled back and nothing is written;
3. an insert of the rows not yet stored;
4. the timeline refresh, if anything was inserted.

Before that, in pandas:

- rows with a missing value are dropped (or rejected with `dropna=False`); infinite and
  non-numeric values are rejected;
- rows published before their event time are rejected, since that would be a look-ahead
  in the source itself;
- duplicates within the batch collapse if identical and raise if they disagree.

Loading the same file twice therefore changes nothing, and a failed load leaves the store as it
was. `IngestReport` says how many rows were inserted, already present and dropped.

## Release calendars

Many sources give no publication dates. The release rules in `calendars.py` map event times to
an assumed `available_at`:

| Rule | Example |
|---|---|
| `fixed_lag(days=15)` | daily data released with a fixed delay |
| `business_days(2)` | T+2 settlement-style lags |
| `monthly_release(day=15)` | CPI: the 15th of the following month |
| `after_quarter_end(days=90)` | quarterly filings, 90 days after the quarter |

A rule should be conservative: assuming data arrive later than they did costs a little
freshness, while assuming they arrive earlier brings back the look-ahead.

## Not in scope

- Concurrent writers. DuckDB allows one writing process per file.
- Non-numeric values.
- Deletes. A wrong value is corrected by publishing a new version, which keeps the history
  honest about what was believed and when.
