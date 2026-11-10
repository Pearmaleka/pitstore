"""Release calendars: when does an observation about a period become known?

Each helper returns a function that maps a Series of event times (the end of the period an
observation describes) to the times the observation was first published. Pass it to
``FeatureStore.ingest(..., lag=...)`` when a data set records values but not release dates.

The helpers are deliberately simple and conservative. Real publication schedules vary from
release to release; when the actual dates are known, store them in an ``available_at``
column instead.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import pandas as pd

ReleaseRule = Callable[[pd.Series], pd.Series]


def _as_datetime(event_time: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.Series(event_time)).astype("datetime64[ns]")


def fixed_lag(**timedelta_kwargs: float) -> ReleaseRule:
    """Known a fixed time after the event, e.g. ``fixed_lag(days=15)``."""
    delta = pd.Timedelta(timedelta(**timedelta_kwargs))
    if delta < pd.Timedelta(0):
        raise ValueError("a release lag cannot be negative")

    def rule(event_time: pd.Series) -> pd.Series:
        return _as_datetime(event_time) + delta

    rule.__name__ = f"fixed_lag({delta})"
    return rule


def business_days(n: int) -> ReleaseRule:
    """Known ``n`` business days (Monday to Friday, no holidays) after the event."""
    if n < 0:
        raise ValueError("n must be non-negative")
    offset = pd.offsets.BDay(n)

    def rule(event_time: pd.Series) -> pd.Series:
        return _as_datetime(event_time) + offset

    rule.__name__ = f"business_days({n})"
    return rule


def monthly_release(day: int, months_after: int = 1) -> ReleaseRule:
    """Known on day ``day`` of the month ``months_after`` months after the event's month.

    ``monthly_release(15)`` puts the January figure (event time 31 January) on 15 February,
    a typical schedule for consumer prices.
    """
    if not 1 <= day <= 28:
        raise ValueError("day must be between 1 and 28")
    if months_after < 0:
        raise ValueError("months_after must be non-negative")

    def rule(event_time: pd.Series) -> pd.Series:
        et = _as_datetime(event_time)
        month_start = et.dt.to_period("M").dt.to_timestamp()
        out = month_start + pd.DateOffset(months=months_after) + pd.Timedelta(days=day - 1)
        # never before the event itself (e.g. months_after=0 and a late event time)
        return out.where(out >= et, et)

    rule.__name__ = f"monthly_release(day={day}, months_after={months_after})"
    return rule


def after_quarter_end(days: int) -> ReleaseRule:
    """Known ``days`` days after the end of the calendar quarter containing the event.

    Suits quarterly accounting data reported with a delay, including monthly series that
    are interpolated from quarterly reports (every month of the quarter waits for the
    report).
    """
    if days < 0:
        raise ValueError("days must be non-negative")

    def rule(event_time: pd.Series) -> pd.Series:
        et = _as_datetime(event_time)
        quarter_end = et.dt.to_period("Q").dt.end_time.dt.normalize()
        return quarter_end + pd.Timedelta(days=days)

    rule.__name__ = f"after_quarter_end(days={days})"
    return rule
