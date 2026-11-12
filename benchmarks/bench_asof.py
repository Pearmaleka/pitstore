"""How fast is the point-in-time join, and how does it compare with pandas?

Synthetic revision histories: 1,000 entities, daily events, each published 1-10 days late
and revised up to three times. The spine asks for one value per (entity, random time).

Three implementations of the same point-in-time semantics:
* ``pitstore``: the DuckDB query used by ``FeatureStore.as_of``;
* ``pandas``: the same algorithm with ``groupby().cummax()`` and ``merge_asof``;
* ``brute force``: ``pitstore.reference``, one filter per spine row (small sizes only).

pitstore pays for its derived timeline at ingest time; the benchmark therefore reports the bulk
load, a small incremental update (1,000 rows for 50 entities) and the query separately.

A second scenario is a single long series (one entity, minute data), where the cost of an update
depends on how far back its earliest publication time reaches: an append only touches the end
of the timeline, a backfill into the middle of the history recomputes everything after it.

Run from the repository root:  python benchmarks/bench_asof.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from pitstore import FeatureStore, reference

ROOT = Path(__file__).resolve().parents[1]
BASE = np.datetime64("2015-01-01")


def make_history(n_rows: int, n_entities: int, rng: np.random.Generator) -> pd.DataFrame:
    n_events = max(1, int(n_rows / 2.5))
    entity = rng.integers(0, n_entities, n_events)
    event_day = rng.integers(0, 3650, n_events)
    events = pd.DataFrame({"entity": entity, "event_day": event_day}).drop_duplicates()
    versions = rng.integers(1, 5, len(events))  # 1 release + up to 3 revisions
    rep = events.loc[events.index.repeat(versions)].reset_index(drop=True)
    first_lag = rng.integers(1, 11, len(events)).repeat(versions)
    step = rng.integers(1, 31, len(rep))
    k = rep.groupby(["entity", "event_day"]).cumcount().to_numpy()
    step[k == 0] = 0
    lag = first_lag + pd.Series(step).groupby([rep["entity"], rep["event_day"]]).cumsum().to_numpy()
    return pd.DataFrame(
        {
            "entity": rep["entity"].astype(str).to_numpy(),
            "event_time": BASE + rep["event_day"].to_numpy().astype("timedelta64[D]"),
            "available_at": BASE + (rep["event_day"].to_numpy() + lag).astype("timedelta64[D]"),
            "value": rng.standard_normal(len(rep)),
        }
    )


def make_series(n_rows: int, rng: np.random.Generator, first_event: int = 0) -> pd.DataFrame:
    """One entity, an event every minute, published 1-60 minutes later, revised up to twice."""
    n_events = max(1, n_rows // 2)
    events = np.arange(first_event, first_event + n_events)
    versions = rng.integers(1, 4, n_events)
    ev = np.repeat(events, versions)
    k = np.concatenate([np.arange(v) for v in versions])
    # version k is published in the k-th day after the first release
    lag = np.repeat(rng.integers(1, 61, n_events), versions) + k * 1440
    lag[k > 0] += rng.integers(0, 1440, int((k > 0).sum()))
    return pd.DataFrame(
        {
            "event_time": BASE + ev.astype("timedelta64[m]"),
            "available_at": BASE + (ev + lag).astype("timedelta64[m]"),
            "value": rng.standard_normal(len(ev)),
        }
    )


def make_spine(n: int, n_entities: int, rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity": rng.integers(0, n_entities, n).astype(str),
            "ts": BASE + rng.integers(0, 3700 * 24, n).astype("timedelta64[h]"),
        }
    )


def pitstore_point_in_time(store: FeatureStore, spine: pd.DataFrame) -> np.ndarray:
    return store.as_of(spine, ["x"], entity="entity")["x"].to_numpy()


def pandas_point_in_time(history: pd.DataFrame, spine: pd.DataFrame) -> np.ndarray:
    """The same algorithm as the SQL query, in pandas."""
    h = history.sort_values(["available_at", "event_time"], kind="stable")
    latest = h.groupby("entity")["event_time"].cummax()
    timeline = h[h["event_time"] == latest]
    timeline = timeline.sort_values(["entity", "available_at", "event_time"]).drop_duplicates(
        ["entity", "available_at"], keep="last"
    )
    s = spine.reset_index().rename(columns={"index": "row_id"}).sort_values("ts")
    merged = pd.merge_asof(
        s,
        timeline[["entity", "available_at", "value"]].sort_values("available_at"),
        left_on="ts",
        right_on="available_at",
        by="entity",
        direction="backward",
    )
    return merged.sort_values("row_id")["value"].to_numpy()


def timed(fn, repeat: int = 3) -> tuple[float, object]:
    best, out = float("inf"), None
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - start)
    return best, out


def many_entities(sizes: list[int], rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in sizes:
        history = make_history(n, 1000, rng)
        spine = make_spine(n // 5, 1000, rng)
        store = FeatureStore()
        cols = {"event_time": "event_time", "value": "value", "entity": "entity"}
        start = time.perf_counter()
        store.ingest("x", history, available_at="available_at", **cols)
        ingest_s = time.perf_counter() - start
        # A daily update: 1,000 new releases for 50 entities, after the bulk load.
        update = make_history(1_000, 50, rng)
        update["available_at"] = update["available_at"] + np.timedelta64(4000, "D")
        start = time.perf_counter()
        store.ingest("x", update, available_at="available_at", **cols)
        update_s = time.perf_counter() - start
        history = pd.concat([history, update], ignore_index=True)
        pit_s, pit = timed(partial(pitstore_point_in_time, store, spine))
        pd_s, pdv = timed(partial(pandas_point_in_time, history, spine))
        assert np.array_equal(pit, pdv, equal_nan=True), "implementations disagree"
        row: dict[str, object] = {
            "history_rows": len(history),
            "spine_rows": len(spine),
            "ingest_seconds": ingest_s,
            "update_1000_rows_seconds": update_s,
            "pitstore_seconds": pit_s,
            "pandas_seconds": pd_s,
            "brute_force_seconds": None,
        }
        if n <= 10_000:
            small = spine.head(500)
            bf_s, bf = timed(partial(reference.as_of, history, small), repeat=1)
            assert np.array_equal(bf, pit[:500], equal_nan=True)
            row["brute_force_seconds"] = bf_s * len(spine) / len(small)  # extrapolated
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        store.close()
    return rows


def one_long_series(sizes: list[int], rng: np.random.Generator) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in sizes:
        history = make_series(n, rng)
        store = FeatureStore()
        cols = {"event_time": "event_time", "value": "value", "available_at": "available_at"}
        start = time.perf_counter()
        store.ingest("x", history, **cols)
        ingest_s = time.perf_counter() - start
        last_event = int((history["event_time"].max() - BASE) / np.timedelta64(1, "m"))
        appends = []
        for _ in range(3):  # three appends of 1,000 rows each, after everything stored
            update = make_series(1_000, rng, first_event=last_event + 1)
            last_event += 500
            start = time.perf_counter()
            store.ingest("x", update, **cols)
            appends.append(time.perf_counter() - start)
            history = pd.concat([history, update], ignore_index=True)
        # A backfill: 1,000 new rows published halfway through the stored history.
        mid = history["available_at"].sort_values().iloc[len(history) // 2]
        backfill = make_series(1_000, rng, first_event=last_event + 10_000)
        backfill["event_time"] = (
            mid
            - np.timedelta64(30, "s")
            - backfill["event_time"]
            .rank(method="first")
            .astype("int64")
            .to_numpy()
            .astype("timedelta64[s]")
        )
        backfill["available_at"] = mid + np.arange(len(backfill)).astype("timedelta64[s]")
        start = time.perf_counter()
        store.ingest("x", backfill, **cols)
        backfill_s = time.perf_counter() - start
        history = pd.concat([history, backfill], ignore_index=True)
        spine = pd.DataFrame(
            {"ts": BASE + rng.integers(0, n, 10_000).astype("timedelta64[m]")}
        ).assign(entity="_")
        got = store.as_of(spine[["ts"]], ["x"])["x"].to_numpy()
        want = pandas_point_in_time(history.assign(entity="_"), spine)
        assert np.array_equal(got, want, equal_nan=True), "implementations disagree"
        row: dict[str, object] = {
            "history_rows": len(history),
            "ingest_seconds": ingest_s,
            "append_1000_rows_seconds": float(np.median(appends)),
            "backfill_1000_rows_seconds": backfill_s,
        }
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        store.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", choices=["many", "single"], default=None)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    out: dict[str, object] = {"cpus": os.cpu_count()}
    if args.only != "single":
        sizes = [10_000, 100_000] if args.quick else [10_000, 100_000, 1_000_000, 5_000_000]
        out["results"] = many_entities(sizes, rng)
    if args.only != "many":
        sizes = [100_000] if args.quick else [100_000, 1_000_000, 4_000_000]
        out["single_entity"] = one_long_series(sizes, rng)
    if args.only is None:
        suffix = "-quick" if args.quick else ""
        path = ROOT / "experiments" / "results" / f"benchmark{suffix}.json"
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
