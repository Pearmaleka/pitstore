"""Data-quality checks on the stored features.

``check(store)`` returns one row per problem found, so it can be printed, saved, or used to
fail a pipeline (``assert check(store).empty``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pitstore.store import FeatureStore

COLUMNS = ["feature", "entity", "check", "detail"]

_GAPS = """
WITH events AS (
    SELECT DISTINCT feature, entity, event_time FROM feature_values
), spaced AS (
    SELECT feature, entity, event_time,
           epoch(event_time) - epoch(lag(event_time) OVER w) AS step
    FROM events
    WINDOW w AS (PARTITION BY feature, entity ORDER BY event_time)
), typical AS (
    SELECT feature, entity, median(step) AS usual, count(step) AS n
    FROM spaced WHERE step IS NOT NULL GROUP BY feature, entity
)
SELECT s.feature, s.entity, s.event_time, s.step / 86400 AS gap_days, t.usual / 86400 AS usual_days
FROM spaced s JOIN typical t USING (feature, entity)
WHERE t.n >= 2 AND s.step > $gap_factor * t.usual
ORDER BY s.feature, s.entity, s.event_time
"""

_LAG_OUTLIERS = """
WITH first_release AS (
    SELECT feature, entity, event_time,
           (epoch(min(available_at)) - epoch(event_time)) / 86400 AS lag_days
    FROM feature_values GROUP BY feature, entity, event_time
), typical AS (
    SELECT feature, entity, median(lag_days) AS usual FROM first_release GROUP BY feature, entity
)
SELECT f.feature, f.entity, f.event_time, f.lag_days, t.usual AS usual_days
FROM first_release f JOIN typical t USING (feature, entity)
WHERE f.lag_days > $lag_factor * greatest(t.usual, 1)
ORDER BY f.feature, f.entity, f.event_time
"""

_REVISIONS = """
WITH per_event AS (
    SELECT feature, entity, event_time,
           arg_min(value, available_at) AS first_value,
           arg_max(value, available_at) AS final_value
    FROM feature_values GROUP BY feature, entity, event_time
), scale AS (
    SELECT feature, entity, stddev_samp(first_value) AS sd FROM per_event GROUP BY feature, entity
)
SELECT p.feature, p.entity, p.event_time, p.first_value, p.final_value, s.sd
FROM per_event p JOIN scale s USING (feature, entity)
WHERE s.sd > 0 AND abs(p.final_value - p.first_value) > $revision_sd * s.sd
ORDER BY p.feature, p.entity, p.event_time
"""

_STALE = """
SELECT feature, entity, max(available_at) AS last_update
FROM feature_values GROUP BY feature, entity
HAVING max(available_at) < $cutoff
ORDER BY feature, entity
"""


def _rows(df: pd.DataFrame, check: str, detail: Any) -> list[dict[str, Any]]:
    return [
        {"feature": r.feature, "entity": r.entity, "check": check, "detail": detail(r)}
        for r in df.itertuples(index=False)
    ]


def check(
    store: FeatureStore,
    *,
    gap_factor: float = 1.5,
    lag_factor: float = 3.0,
    revision_sd: float = 2.0,
    stale_after: pd.Timedelta | None = None,
    now: Any = None,
) -> pd.DataFrame:
    """Look for gaps, unusually late releases, large revisions and stale series.

    Parameters
    ----------
    gap_factor
        Flag a gap when consecutive events are more than this many times the usual spacing
        apart.
    lag_factor
        Flag a first release that came more than this many times later than usual.
    revision_sd
        Flag a revision (final minus first release) larger than this many standard
        deviations of the first releases.
    stale_after
        Flag series whose latest release is older than this at ``now`` (default: today).
    """
    rows: list[dict[str, Any]] = []
    gaps = store.raw(_GAPS, {"gap_factor": gap_factor})
    rows += _rows(
        gaps,
        "gap",
        lambda r: (
            f"{r.gap_days:.0f} days before {r.event_time:%Y-%m-%d} (usually {r.usual_days:.0f})"
        ),
    )
    lags = store.raw(_LAG_OUTLIERS, {"lag_factor": lag_factor})
    rows += _rows(
        lags,
        "late_release",
        lambda r: (
            f"event {r.event_time:%Y-%m-%d} first published after {r.lag_days:.0f} days "
            f"(usually {r.usual_days:.0f})"
        ),
    )
    revisions = store.raw(_REVISIONS, {"revision_sd": revision_sd})
    rows += _rows(
        revisions,
        "large_revision",
        lambda r: (
            f"event {r.event_time:%Y-%m-%d} revised from {r.first_value:.4g} "
            f"to {r.final_value:.4g} (sd {r.sd:.3g})"
        ),
    )
    if stale_after is not None:
        reference = pd.Timestamp.now() if now is None else pd.Timestamp(now)
        stale = store.raw(_STALE, {"cutoff": reference - stale_after})
        rows += _rows(stale, "stale", lambda r: f"last update {r.last_update:%Y-%m-%d}")
    return pd.DataFrame(rows, columns=COLUMNS)
