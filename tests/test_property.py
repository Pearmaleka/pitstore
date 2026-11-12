"""Randomised revision histories: the SQL joins must agree with the brute-force reference."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pitstore import FeatureStore, reference

BASE = pd.Timestamp("2024-01-01")


@st.composite
def histories(draw):
    rows = []
    for entity in draw(st.lists(st.sampled_from("ABC"), min_size=1, max_size=3, unique=True)):
        n_events = draw(st.integers(1, 12))
        events = sorted(
            draw(st.lists(st.integers(0, 40), min_size=n_events, max_size=n_events, unique=True))
        )
        for e in events:
            first_lag = draw(st.integers(0, 10))
            n_versions = draw(st.integers(1, 4))
            gaps = draw(
                st.lists(st.integers(1, 15), min_size=n_versions - 1, max_size=n_versions - 1)
            )
            avail = e + first_lag
            for k in range(n_versions):
                if k:
                    avail += gaps[k - 1]
                value = draw(st.floats(-100, 100, allow_nan=False, width=32))
                rows.append(
                    (
                        entity,
                        BASE + pd.Timedelta(days=e),
                        BASE + pd.Timedelta(days=avail),
                        float(value),
                    )
                )
    hist = pd.DataFrame(rows, columns=["entity", "event_time", "available_at", "value"])
    return hist.drop_duplicates(["entity", "event_time", "available_at"]).reset_index(drop=True)


@st.composite
def spines(draw):
    n = draw(st.integers(1, 25))
    ents = draw(st.lists(st.sampled_from("ABCD"), min_size=n, max_size=n))
    days = draw(st.lists(st.integers(-5, 90), min_size=n, max_size=n))
    return pd.DataFrame({"entity": ents, "ts": [BASE + pd.Timedelta(days=d) for d in days]})


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(hist=histories(), sp=spines())
def test_sql_matches_reference(hist, sp):
    store = FeatureStore()
    store.ingest(
        "x",
        hist,
        event_time="event_time",
        value="value",
        entity="entity",
        available_at="available_at",
    )
    for mode in ("pit", "naive"):
        got = store.as_of(sp, ["x"], entity="entity", mode=mode)["x"].to_numpy()
        want = reference.as_of(hist, sp, mode=mode)
        np.testing.assert_array_equal(got, want, err_msg=mode)
    store.close()


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(hist=histories(), sp=spines())
def test_point_in_time_never_uses_unpublished_data(hist, sp):
    store = FeatureStore()
    store.ingest(
        "x",
        hist,
        event_time="event_time",
        value="value",
        entity="entity",
        available_at="available_at",
    )
    out = store.as_of(sp, ["x"], entity="entity", metadata=True)
    known = out["x__available_at"].notna()
    assert (out.loc[known, "x__available_at"] <= out.loc[known, "ts"]).all()
    assert (out.loc[known, "x__event_time"] <= out.loc[known, "ts"]).all()
    store.close()


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    hist=histories(),
    sp=spines(),
    cuts=st.lists(st.integers(0, 10**6), max_size=4),
    seed=st.integers(0, 2**31),
)
def test_incremental_batches_give_the_same_answers(hist, sp, cuts, seed):
    """Ingest the history in shuffled batches; the maintained timeline must stay correct."""
    shuffled = hist.sample(frac=1.0, random_state=seed % (2**32)).reset_index(drop=True)
    bounds = sorted({0, len(shuffled), *(c % (len(shuffled) + 1) for c in cuts)})
    store = FeatureStore()
    for lo, hi in itertools.pairwise(bounds):
        store.ingest(
            "x",
            shuffled.iloc[lo:hi],
            event_time="event_time",
            value="value",
            entity="entity",
            available_at="available_at",
        )
    got = store.as_of(sp, ["x"], entity="entity")["x"].to_numpy()
    np.testing.assert_array_equal(got, reference.as_of(hist, sp))
    before = store.raw("SELECT * FROM timeline ORDER BY ALL")
    store.rebuild_timeline()
    after = store.raw("SELECT * FROM timeline ORDER BY ALL")
    pd.testing.assert_frame_equal(before, after)
    store.close()
