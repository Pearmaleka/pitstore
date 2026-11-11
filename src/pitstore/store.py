"""A bitemporal feature store on DuckDB.

Every stored value has two times:

``event_time``
    The period the value describes (for monthly CPI, the end of the month).
``available_at``
    When this version of the value was published. A revision of an earlier figure is a new
    row with the same ``event_time`` and a later ``available_at``; nothing is overwritten.

Training data built with :meth:`FeatureStore.as_of` therefore contains, for each row, only
what had been published by that row's timestamp.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd

from pitstore import _sql

Mode = Literal["pit", "naive"]
LagLike = pd.Timedelta | pd.DateOffset | Callable[[pd.Series], pd.Series]
GLOBAL_ENTITY = "_"


@dataclass(frozen=True)
class IngestReport:
    """What :meth:`FeatureStore.ingest` did with a batch."""

    feature: str
    received: int
    dropped_missing: int
    inserted: int
    unchanged: int

    def __str__(self) -> str:
        return (
            f"{self.feature}: {self.inserted} rows inserted, {self.unchanged} already stored, "
            f"{self.dropped_missing} dropped (missing values) out of {self.received}"
        )


def _to_timestamps(values: Any, what: str) -> pd.Series:
    ts = pd.to_datetime(pd.Series(values), errors="raise")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    # DuckDB's TIMESTAMP has microsecond resolution. Matching it here means joins compare like
    # with like; a nanosecond column would make DuckDB cast the stored side on every query.
    ts = ts.astype("datetime64[us]")
    if ts.isna().any():
        raise ValueError(f"{what} contains missing timestamps")
    return ts.reset_index(drop=True)


def _apply_lag(event_time: pd.Series, lag: LagLike) -> pd.Series:
    if callable(lag) and not isinstance(lag, pd.DateOffset | pd.Timedelta):
        out = lag(event_time)
        return _to_timestamps(out, "the result of the lag function")
    if isinstance(lag, pd.Timedelta) and lag < pd.Timedelta(0):
        raise ValueError("lag cannot be negative")
    return (event_time + lag).reset_index(drop=True)


class FeatureStore:
    """Point-in-time feature store backed by a DuckDB database.

    Parameters
    ----------
    path
        Database file, created if needed. ``":memory:"`` (the default) keeps everything in
        memory.
    read_only
        Open an existing database without write access.

    Examples
    --------
    >>> store = FeatureStore()
    >>> cpi = pd.DataFrame({"month_end": ["2024-01-31"], "cpi": [308.4]})
    >>> store.ingest("cpi", cpi, event_time="month_end", value="cpi", lag=pd.Timedelta(days=13))
    IngestReport(feature='cpi', received=1, dropped_missing=0, inserted=1, unchanged=0)
    """

    def __init__(self, path: str | Path = ":memory:", *, read_only: bool = False):
        self.path = str(path)
        self.con = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self.con.execute(_sql.SCHEMA)

    # -- lifecycle -----------------------------------------------------------------------

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"FeatureStore({self.path!r}, features={self.feature_names()})"

    # -- registry ------------------------------------------------------------------------

    def register(self, name: str, *, description: str = "", unit: str = "") -> None:
        """Add a feature to the registry, or update its description and unit."""
        if not name or not isinstance(name, str):
            raise ValueError("feature name must be a non-empty string")
        self.con.execute(
            """
            INSERT INTO features (name, description, unit) VALUES ($name, $description, $unit)
            ON CONFLICT (name) DO UPDATE SET
                description = CASE WHEN excluded.description <> '' THEN excluded.description
                                   ELSE features.description END,
                unit = CASE WHEN excluded.unit <> '' THEN excluded.unit ELSE features.unit END
            """,
            {"name": name, "description": description, "unit": unit},
        )

    def feature_names(self) -> list[str]:
        rows = self.con.execute("SELECT name FROM features ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def features(self) -> pd.DataFrame:
        """The registry: name, description, unit and creation time of every feature."""
        return self.con.execute("SELECT * FROM features ORDER BY name").df()

    def _require(self, names: Iterable[str]) -> None:
        known = set(self.feature_names())
        missing = [n for n in names if n not in known]
        if missing:
            raise KeyError(f"unknown features {missing}; registered: {sorted(known)}")

    # -- writing -------------------------------------------------------------------------

    def ingest(
        self,
        name: str,
        data: pd.DataFrame,
        *,
        event_time: str,
        value: str,
        entity: str | None = None,
        available_at: str | None = None,
        lag: LagLike | None = None,
        dropna: bool = True,
    ) -> IngestReport:
        """Append observations of feature ``name``.

        Parameters
        ----------
        data
            One row per observation, or per version of an observation.
        event_time, value
            Columns holding the period each value describes and the value itself.
        entity
            Column identifying the entity (ticker, country, user). ``None`` stores a single
            series.
        available_at
            Column with the publication time of each row.
        lag
            Instead of ``available_at``: how long after ``event_time`` the value became
            known, as a ``Timedelta``, a ``DateOffset`` or a function such as
            :func:`pitstore.calendars.monthly_release`. ``lag=pd.Timedelta(0)`` declares the
            value known at its event time.
        dropna
            Drop rows with a missing value (the default) instead of raising.

        Ingestion is atomic and idempotent: rows already stored are skipped, and a row that
        would change a stored value raises ``ValueError``, because a revision must be a new
        row with a later ``available_at``.
        """
        if (available_at is None) == (lag is None):
            raise ValueError(
                "say when the values became known: pass exactly one of available_at= or lag="
            )
        for col in (event_time, value, entity, available_at):
            if col is not None and col not in data.columns:
                raise KeyError(f"column {col!r} not found; columns are {list(data.columns)}")

        received = len(data)
        frame = data.reset_index(drop=True)
        missing = frame[value].isna()
        if missing.any() and not dropna:
            raise ValueError(f"{int(missing.sum())} missing values in column {value!r}")
        frame = frame.loc[~missing].reset_index(drop=True)

        et = _to_timestamps(frame[event_time], f"column {event_time!r}")
        if available_at is not None:
            av = _to_timestamps(frame[available_at], f"column {available_at!r}")
        else:
            assert lag is not None
            av = _apply_lag(et, lag)
        if entity is None:
            ent = pd.Series([GLOBAL_ENTITY] * len(frame), dtype="object")
        else:
            if frame[entity].isna().any():
                raise ValueError(f"column {entity!r} contains missing entities")
            ent = frame[entity].astype(str).reset_index(drop=True)
        try:
            vals = pd.to_numeric(frame[value], errors="raise").astype(float).reset_index(drop=True)
        except (TypeError, ValueError) as err:
            raise ValueError(f"column {value!r} must be numeric") from err
        if not np.all(np.isfinite(vals.to_numpy())):
            raise ValueError(f"column {value!r} contains infinite values")

        early = av < et
        if early.any():
            raise ValueError(
                f"{int(early.sum())} rows are published before their event time "
                f"(first: event {et[early].iloc[0]}, available {av[early].iloc[0]})"
            )
        batch = pd.DataFrame({"entity": ent, "event_time": et, "available_at": av, "value": vals})
        dup = batch.duplicated(["entity", "event_time", "available_at"], keep=False)
        if dup.any():
            same = batch[dup].groupby(["entity", "event_time", "available_at"])["value"].nunique()
            if (same > 1).any():
                raise ValueError(
                    f"{int((same > 1).sum())} keys appear in the batch with different values; "
                    "versions of the same event need different available_at times"
                )
            batch = batch.drop_duplicates(["entity", "event_time", "available_at"])

        self.register(name)
        params = {"feature": name}
        self.con.register("_incoming", batch)
        try:
            self.con.execute("BEGIN TRANSACTION")
            self.con.execute(_sql.STAGE, params)
            conflicts = self.con.execute(_sql.CONFLICTS).df()
            if len(conflicts):
                self.con.execute("ROLLBACK")
                first = conflicts.iloc[0]
                raise ValueError(
                    f"{name}: stored values would change (e.g. entity {first.entity}, event "
                    f"{first.event_time}, available {first.available_at}: stored {first.stored}, "
                    f"incoming {first.incoming}); add revisions as rows with a later available_at"
                )
            row = self.con.execute(_sql.COUNT_NEW).fetchone()
            inserted = int(row[0]) if row else 0
            if inserted:
                self.con.execute(_sql.INSERT_NEW, params)
                self.con.execute(_sql.STAGE_STARTS)
                for statement in _sql.REFRESH_TIMELINE:
                    self.con.execute(statement, params)
            self.con.execute("COMMIT")
        except duckdb.Error:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("_incoming")
            self.con.execute("DROP TABLE IF EXISTS _staged; DROP TABLE IF EXISTS _starts")
        return IngestReport(
            feature=name,
            received=received,
            dropped_missing=int(missing.sum()),
            inserted=inserted,
            unchanged=len(batch) - inserted,
        )

    def rebuild_timeline(self, name: str | None = None) -> None:
        """Recompute the derived point-in-time timeline (all features by default).

        ``ingest`` keeps it up to date; this is only needed after writing to
        ``feature_values`` by other means.
        """
        names = self.feature_names() if name is None else [name]
        self._require(names)
        for n in names:
            for statement in _sql.REBUILD_TIMELINE:
                self.con.execute(statement, {"feature": n})

    # -- reading -------------------------------------------------------------------------

    def as_of(
        self,
        spine: pd.DataFrame,
        features: Sequence[str],
        *,
        time: str = "ts",
        entity: str | None = None,
        mode: Mode = "pit",
        max_age: pd.Timedelta | None = None,
        metadata: bool = False,
    ) -> pd.DataFrame:
        """Attach feature values to every row of ``spine``.

        Parameters
        ----------
        spine
            The rows to build features for, e.g. one row per (entity, prediction time).
        features
            Feature names; each becomes a column of the result.
        time, entity
            Columns of ``spine`` with the prediction time and the entity (``None`` for a
            single series).
        mode
            ``"pit"``: the latest event published by ``time``, in its latest version
            published by ``time``. ``"naive"``: the latest event up to ``time`` in its final
            version, ignoring publication times. This is what a plain join against a
            downloaded data set gives and leaks information from the future; it exists for
            comparison.
        max_age
            Treat values whose event is older than this at ``time`` as missing.
        metadata
            Also return ``<feature>__event_time`` and ``<feature>__available_at`` columns.

        Returns
        -------
        A copy of ``spine`` with one column per feature, in the original row order.
        """
        if mode not in ("pit", "naive"):
            raise ValueError("mode must be 'pit' or 'naive'")
        names = list(features)
        if not names:
            raise ValueError("ask for at least one feature")
        if len(set(names)) != len(names):
            raise ValueError("features must not repeat")
        self._require(names)
        if time not in spine.columns:
            raise KeyError(f"time column {time!r} not in spine")
        if entity is not None and entity not in spine.columns:
            raise KeyError(f"entity column {entity!r} not in spine")
        clash = [n for n in names if n in spine.columns]
        if clash:
            raise ValueError(f"spine already has columns named {clash}")

        ts = _to_timestamps(spine[time], f"spine column {time!r}")
        ent = (
            pd.Series([GLOBAL_ENTITY] * len(spine), dtype="object")
            if entity is None
            else spine[entity].astype(str).reset_index(drop=True)
        )
        keys = pd.DataFrame({"row_id": np.arange(len(spine)), "entity": ent, "ts": ts})
        out = spine.copy()
        query = _sql.POINT_IN_TIME if mode == "pit" else _sql.NAIVE
        self.con.register("_spine", keys)
        try:
            for name in names:
                res = self.con.execute(query, {"feature": name}).df()
                res = res.set_index("row_id").reindex(keys["row_id"])
                values = res["value"].to_numpy(dtype=float)
                if max_age is not None:
                    age = keys["ts"].to_numpy() - res["event_time"].to_numpy()
                    values = np.where(age <= np.timedelta64(max_age), values, np.nan)
                out[name] = values
                if metadata:
                    out[f"{name}__event_time"] = res["event_time"].to_numpy()
                    out[f"{name}__available_at"] = res["available_at"].to_numpy()
        finally:
            self.con.unregister("_spine")
        return out

    def snapshot(self, name: str, as_of: Any, *, entity: str | None = None) -> pd.DataFrame:
        """The whole series as it was known at ``as_of``: latest version of every event."""
        self._require([name])
        ts = _to_timestamps([as_of], "as_of").iloc[0]
        df = self.con.execute(_sql.SNAPSHOT, {"feature": name, "as_of": ts}).df()
        if entity is not None:
            df = df[df["entity"] == str(entity)].reset_index(drop=True)
        return df

    def history(self, name: str, *, entity: str | None = None) -> pd.DataFrame:
        """Every stored version of every event of feature ``name``."""
        self._require([name])
        df = self.con.execute(_sql.HISTORY, {"feature": name}).df()
        if entity is not None:
            df = df[df["entity"] == str(entity)].reset_index(drop=True)
        return df

    def describe(self) -> pd.DataFrame:
        """One row per feature: coverage, revisions and publication lags."""
        return self.con.execute(_sql.DESCRIBE).df()

    def raw(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a read query against the store's tables (``features``, ``feature_values``)."""
        return self.con.execute(sql, params or {}).df()
