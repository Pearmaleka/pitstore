from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitstore import FeatureStore, monthly_release


@pytest.fixture
def gdp() -> pd.DataFrame:
    # First release, two revisions of Q1, then Q2 and one revision.
    return pd.DataFrame(
        {
            "quarter_end": ["2024-03-31"] * 3 + ["2024-06-30"] * 2,
            "released": ["2024-04-25", "2024-05-30", "2024-06-27", "2024-07-25", "2024-08-29"],
            "growth": [1.6, 1.3, 1.4, 2.8, 3.0],
        }
    )


@pytest.fixture
def store(gdp: pd.DataFrame) -> FeatureStore:
    s = FeatureStore()
    s.ingest("gdp", gdp, event_time="quarter_end", value="growth", available_at="released")
    return s


def spine(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"ts": pd.to_datetime(list(dates))})


def test_point_in_time_follows_releases_and_revisions(store):
    out = store.as_of(
        spine("2024-04-01", "2024-04-25", "2024-05-01", "2024-06-28", "2024-08-01", "2024-12-31"),
        ["gdp"],
    )
    expected = [np.nan, 1.6, 1.6, 1.4, 2.8, 3.0]
    np.testing.assert_array_equal(out["gdp"].to_numpy(), expected)


def test_naive_join_leaks_final_values(store):
    out = store.as_of(spine("2024-04-01", "2024-07-01"), ["gdp"], mode="naive")
    # On 1 April nothing about Q1 had been published, yet the naive join returns its final value.
    np.testing.assert_array_equal(out["gdp"].to_numpy(), [1.4, 3.0])


def test_revision_of_an_older_event_does_not_replace_the_latest_event(store):
    late_revision = pd.DataFrame(
        {"quarter_end": ["2024-03-31"], "released": ["2024-07-30"], "growth": [1.0]}
    )
    store.ingest(
        "gdp", late_revision, event_time="quarter_end", value="growth", available_at="released"
    )
    out = store.as_of(spine("2024-08-01"), ["gdp"], metadata=True)
    assert out.loc[0, "gdp"] == 2.8  # still Q2, not the newer revision of Q1
    assert out.loc[0, "gdp__event_time"] == pd.Timestamp("2024-06-30")
    assert out.loc[0, "gdp__available_at"] == pd.Timestamp("2024-07-25")


def test_metadata_and_max_age(store):
    out = store.as_of(
        spine("2024-05-01", "2025-06-01"), ["gdp"], metadata=True, max_age=pd.Timedelta(days=180)
    )
    assert out.loc[0, "gdp__event_time"] == pd.Timestamp("2024-03-31")
    assert out.loc[0, "gdp__available_at"] == pd.Timestamp("2024-04-25")
    assert np.isnan(out.loc[1, "gdp"])  # Q2 2024 is too old by June 2025


def test_entities_and_row_order_are_preserved():
    s = FeatureStore()
    prices = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "day": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
            "close": [10.0, 11.0, 20.0, 21.0],
        }
    )
    s.ingest("close", prices, event_time="day", value="close", entity="ticker", lag=pd.Timedelta(0))
    sp = pd.DataFrame(
        {
            "ticker": ["B", "A", "C", "A", "B"],
            "ts": pd.to_datetime(
                ["2024-01-02", "2024-01-05", "2024-01-05", "2023-12-31", "2024-01-03"]
            ),
            "label": [1, 2, 3, 4, 5],
        }
    )
    out = s.as_of(sp, ["close"], entity="ticker")
    np.testing.assert_array_equal(out["close"].to_numpy(), [20.0, 11.0, np.nan, np.nan, 21.0])
    assert list(out["label"]) == [1, 2, 3, 4, 5]
    assert list(out.columns) == ["ticker", "ts", "label", "close"]


def test_several_features_at_once(store):
    cpi = pd.DataFrame(
        {"month": pd.date_range("2024-01-31", periods=6, freq="ME"), "index": [1.0, 2, 3, 4, 5, 6]}
    )
    store.ingest("cpi", cpi, event_time="month", value="index", lag=monthly_release(15))
    out = store.as_of(spine("2024-05-20"), ["gdp", "cpi"])
    assert out.loc[0, "gdp"] == 1.6
    assert out.loc[0, "cpi"] == 4.0  # April CPI was published on 15 May


def test_ingest_is_idempotent_and_reports(store, gdp):
    report = store.ingest(
        "gdp", gdp, event_time="quarter_end", value="growth", available_at="released"
    )
    assert report.inserted == 0 and report.unchanged == 5 and report.received == 5
    assert "0 rows inserted" in str(report)
    assert len(store.history("gdp")) == 5


def test_changing_a_stored_value_is_rejected_atomically(store, gdp):
    changed = gdp.copy()
    changed.loc[0, "growth"] = 9.9
    changed = pd.concat(
        [
            changed,
            pd.DataFrame(
                {"quarter_end": ["2024-09-30"], "released": ["2024-10-30"], "growth": [2.0]}
            ),
        ]
    )
    with pytest.raises(ValueError, match="stored values would change"):
        store.ingest(
            "gdp", changed, event_time="quarter_end", value="growth", available_at="released"
        )
    assert len(store.history("gdp")) == 5  # nothing from the failed batch was written


def test_ingest_validation():
    s = FeatureStore()
    df = pd.DataFrame({"t": ["2024-01-31"], "v": [1.0], "a": ["2024-01-15"]})
    with pytest.raises(ValueError, match="exactly one"):
        s.ingest("x", df, event_time="t", value="v")
    with pytest.raises(ValueError, match="exactly one"):
        s.ingest("x", df, event_time="t", value="v", available_at="a", lag=pd.Timedelta(0))
    with pytest.raises(KeyError):
        s.ingest("x", df, event_time="missing", value="v", lag=pd.Timedelta(0))
    with pytest.raises(ValueError, match="before their event time"):
        s.ingest("x", df, event_time="t", value="v", available_at="a")
    with pytest.raises(ValueError, match="negative"):
        s.ingest("x", df, event_time="t", value="v", lag=pd.Timedelta(days=-1))
    words = pd.DataFrame({"t": ["2024-01-31"], "v": ["high"]})
    with pytest.raises(ValueError, match="numeric"):
        s.ingest("x", words, event_time="t", value="v", lag=pd.Timedelta(0))
    twice = pd.DataFrame({"t": ["2024-01-31"] * 2, "v": [1.0, 2.0]})
    with pytest.raises(ValueError, match="different values"):
        s.ingest("x", twice, event_time="t", value="v", lag=pd.Timedelta(0))
    gaps = pd.DataFrame({"t": ["2024-01-31", "2024-02-29"], "v": [1.0, None]})
    with pytest.raises(ValueError, match="missing values"):
        s.ingest("x", gaps, event_time="t", value="v", lag=pd.Timedelta(0), dropna=False)
    report = s.ingest("x", gaps, event_time="t", value="v", lag=pd.Timedelta(0))
    assert report.dropped_missing == 1 and report.inserted == 1


def test_exact_duplicates_in_a_batch_are_collapsed():
    s = FeatureStore()
    df = pd.DataFrame({"t": ["2024-01-31"] * 2, "v": [1.0, 1.0]})
    report = s.ingest("x", df, event_time="t", value="v", lag=pd.Timedelta(0))
    assert report.inserted == 1


def test_timezone_aware_times_are_stored_in_utc():
    s = FeatureStore()
    df = pd.DataFrame(
        {"t": pd.to_datetime(["2024-01-31 23:00"]).tz_localize("Asia/Hong_Kong"), "v": [1.0]}
    )
    s.ingest("x", df, event_time="t", value="v", lag=pd.Timedelta(0))
    assert s.history("x").loc[0, "event_time"] == pd.Timestamp("2024-01-31 15:00")


def test_snapshot_time_travel(store):
    snap = store.snapshot("gdp", "2024-06-01")
    assert len(snap) == 1 and snap.loc[0, "value"] == 1.3 and snap.loc[0, "versions_known"] == 2
    final = store.snapshot("gdp", "2030-01-01")
    np.testing.assert_array_equal(final["value"].to_numpy(), [1.4, 3.0])
    assert store.snapshot("gdp", "2024-01-01").empty
    assert store.snapshot("gdp", "2030-01-01", entity="other").empty


def test_registry_and_describe(store):
    store.register("gdp", description="Real GDP growth, annualised", unit="%")
    store.register("gdp")  # empty fields do not erase the description
    reg = store.features()
    assert reg.loc[0, "description"] == "Real GDP growth, annualised"
    assert reg.loc[0, "unit"] == "%"
    d = store.describe().iloc[0]
    assert (d.events, d.rows, d.revisions) == (2, 5, 3)
    assert d.median_lag_days == 25.0
    assert d.mean_abs_revision == pytest.approx(0.2)
    with pytest.raises(ValueError):
        store.register("")


def test_as_of_validation(store):
    with pytest.raises(KeyError, match="unknown features"):
        store.as_of(spine("2024-05-01"), ["nope"])
    with pytest.raises(ValueError, match="mode"):
        store.as_of(spine("2024-05-01"), ["gdp"], mode="future")  # type: ignore[arg-type]
    with pytest.raises(KeyError):
        store.as_of(spine("2024-05-01"), ["gdp"], time="when")
    with pytest.raises(KeyError):
        store.as_of(spine("2024-05-01"), ["gdp"], entity="country")
    clash = spine("2024-05-01").assign(gdp=1.0)
    with pytest.raises(ValueError, match="already has columns"):
        store.as_of(clash, ["gdp"])
    with pytest.raises(ValueError):
        store.as_of(spine("2024-05-01"), [])
    with pytest.raises(ValueError):
        store.as_of(spine("2024-05-01"), ["gdp", "gdp"])


def test_persists_on_disk_and_opens_read_only(tmp_path, gdp):
    path = tmp_path / "features.duckdb"
    with FeatureStore(path) as s:
        s.ingest("gdp", gdp, event_time="quarter_end", value="growth", available_at="released")
    with FeatureStore(path, read_only=True) as s:
        assert s.feature_names() == ["gdp"]
        assert s.as_of(spine("2024-05-01"), ["gdp"]).loc[0, "gdp"] == 1.6
        assert "gdp" in repr(s)
