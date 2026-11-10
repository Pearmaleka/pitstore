from __future__ import annotations

import pandas as pd
import pytest

from pitstore import (
    FeatureStore,
    after_quarter_end,
    business_days,
    check,
    fixed_lag,
    monthly_release,
)
from pitstore.cli import main


def ts(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


def test_fixed_lag_and_business_days():
    out = fixed_lag(days=15)(ts("2024-01-31"))
    assert out.iloc[0] == pd.Timestamp("2024-02-15")
    friday = business_days(1)(ts("2024-01-05"))  # Friday -> Monday
    assert friday.iloc[0] == pd.Timestamp("2024-01-08")
    with pytest.raises(ValueError):
        fixed_lag(days=-1)
    with pytest.raises(ValueError):
        business_days(-2)


def test_monthly_release():
    rule = monthly_release(15)
    out = rule(ts("2024-01-31", "2024-12-31", "2024-02-29"))
    assert list(out) == list(pd.to_datetime(["2024-02-15", "2025-01-15", "2024-03-15"]))
    same_month = monthly_release(1, months_after=0)(ts("2024-01-31"))
    assert same_month.iloc[0] == pd.Timestamp("2024-01-31")  # never before the event
    with pytest.raises(ValueError):
        monthly_release(31)
    with pytest.raises(ValueError):
        monthly_release(5, months_after=-1)


def test_after_quarter_end():
    out = after_quarter_end(45)(ts("2024-01-31", "2024-03-31", "2024-04-30"))
    assert list(out) == list(pd.to_datetime(["2024-05-15", "2024-05-15", "2024-08-14"]))
    with pytest.raises(ValueError):
        after_quarter_end(-1)


@pytest.fixture
def messy() -> FeatureStore:
    s = FeatureStore()
    months = pd.date_range("2023-01-31", periods=12, freq="ME").delete([5])  # June missing
    df = pd.DataFrame({"m": months, "v": range(11)}).astype({"v": float})
    df["a"] = df["m"] + pd.Timedelta(days=10)
    df.loc[8, "a"] = df.loc[8, "m"] + pd.Timedelta(days=90)  # one very late release
    s.ingest("x", df, event_time="m", value="v", available_at="a")
    revision = pd.DataFrame(
        {"m": [months[2]], "v": [50.0], "a": [months[2] + pd.Timedelta(days=40)]}
    )
    s.ingest("x", revision, event_time="m", value="v", available_at="a")
    return s


def test_checks_find_each_problem(messy):
    issues = check(messy, stale_after=pd.Timedelta(days=30), now="2024-06-01")
    found = issues.groupby("check")["detail"].apply(list).to_dict()
    assert set(found) == {"gap", "late_release", "large_revision", "stale"}
    assert "2023-07-31" in found["gap"][0]
    assert "revised from 2 to 50" in found["large_revision"][0]
    assert check(messy).query("check == 'stale'").empty  # stale only when asked


def test_clean_series_has_no_issues():
    s = FeatureStore()
    df = pd.DataFrame({"m": pd.date_range("2024-01-31", periods=6, freq="ME"), "v": 1.0})
    s.ingest("x", df, event_time="m", value="v", lag=pd.Timedelta(days=10))
    assert check(s).empty


def test_cli_end_to_end(tmp_path, capsys):
    db = tmp_path / "f.duckdb"
    data = tmp_path / "gdp.csv"
    pd.DataFrame(
        {"q": ["2024-03-31", "2024-03-31"], "r": ["2024-04-25", "2024-05-30"], "g": [1.6, 1.3]}
    ).to_csv(data, index=False)
    assert (
        main(
            [
                "ingest",
                str(db),
                "gdp",
                str(data),
                "--event-time",
                "q",
                "--value",
                "g",
                "--available-at",
                "r",
                "--description",
                "GDP growth",
            ]
        )
        == 0
    )
    assert "2 rows inserted" in capsys.readouterr().out

    sp = tmp_path / "spine.csv"
    pd.DataFrame({"ts": ["2024-05-01", "2024-06-01"]}).to_csv(sp, index=False)
    out = tmp_path / "train.csv"
    assert main(["asof", str(db), str(sp), "--features", "gdp", "-o", str(out)]) == 0
    assert list(pd.read_csv(out)["gdp"]) == [1.6, 1.3]
    assert main(["asof", str(db), str(sp), "--features", "gdp", "--mode", "naive"]) == 0
    assert "1.3" in capsys.readouterr().out

    assert main(["snapshot", str(db), "gdp", "--as-of", "2024-05-01"]) == 0
    assert "1.6" in capsys.readouterr().out
    assert main(["describe", str(db)]) == 0
    assert "GDP growth" in capsys.readouterr().out
    assert main(["check", str(db)]) == 0
    assert main(["check", str(db), "--stale-days", "1"]) == 1

    csv2 = tmp_path / "cpi.csv"
    pd.DataFrame({"m": ["2024-01-31"], "v": [3.1]}).to_csv(csv2, index=False)
    assert (
        main(
            [
                "ingest",
                str(db),
                "cpi",
                str(csv2),
                "--event-time",
                "m",
                "--value",
                "v",
                "--lag-days",
                "15",
            ]
        )
        == 0
    )
