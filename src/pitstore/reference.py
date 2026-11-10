"""Slow, obviously correct implementations of the as-of joins, used as a test oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_in_time(history: pd.DataFrame, entity: str, ts: pd.Timestamp) -> float:
    """Value of the latest event published by ``ts``, in its latest version published by ``ts``.

    ``history`` has columns ``entity, event_time, available_at, value``.
    """
    known = history[(history["entity"] == entity) & (history["available_at"] <= ts)]
    if known.empty:
        return float("nan")
    latest = known.sort_values(["event_time", "available_at"]).iloc[-1]
    return float(latest["value"])


def naive(history: pd.DataFrame, entity: str, ts: pd.Timestamp) -> float:
    """Final version of the latest event up to ``ts``, ignoring publication times."""
    rows = history[(history["entity"] == entity) & (history["event_time"] <= ts)]
    if rows.empty:
        return float("nan")
    latest = rows.sort_values(["event_time", "available_at"]).iloc[-1]
    return float(latest["value"])


def as_of(history: pd.DataFrame, spine: pd.DataFrame, *, mode: str = "pit") -> np.ndarray:
    """Apply :func:`point_in_time` or :func:`naive` to every ``(entity, ts)`` row of ``spine``."""
    fn = point_in_time if mode == "pit" else naive
    return np.array([fn(history, e, t) for e, t in zip(spine["entity"], spine["ts"], strict=True)])
