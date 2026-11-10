"""Real data: return predictability on Shiller's S&P 500 series, with and without release lags.

Shiller's monthly file (``data/shiller_sp500_monthly.csv``) is the standard dataset for
predictive regressions in the style of Goyal and Welch (2008). Each row is stamped with the
month it describes, not the date the numbers were public:

* earnings are four-quarter totals, linearly interpolated within the quarter, so the value
  for October needs the December quarter-end figure, reported up to about 90 days later;
* dividends are interpolated the same way but known sooner (30 days after the quarter);
* CPI for a month is published around the 15th of the next month;
* prices and the long rate are monthly averages, known at the end of the month.

Features are ingested once into a ``FeatureStore`` with these release rules, then joined to
month-end decision dates twice: ``mode="naive"`` (event-time join, i.e. the CSV as is) and
point-in-time. Five univariate predictors of 1-month and 12-month real total returns are
evaluated out of sample from 1970 with an expanding window, against the historical mean.

Run from the repository root:  python experiments/shiller.py
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import SHILLER, oos_r2, save_figure, style_axes, write_csv, write_json

from pitstore import FeatureStore, after_quarter_end, monthly_release

START = "1927-12-31"
OOS_START = pd.Timestamp("1970-01-31")
HORIZONS = (1, 12)
PREDICTORS = {
    "log_ep": "log earnings/price",
    "log_dp": "log dividend/price",
    "earnings_growth": "12m earnings growth",
    "inflation": "12m CPI inflation",
    "long_rate": "10y Treasury yield",
}


def load() -> pd.DataFrame:
    raw = pd.read_csv(SHILLER, parse_dates=["Date"])
    raw = raw.rename(
        columns={
            "SP500": "price",
            "Dividend": "dividend",
            "Earnings": "earnings",
            "Consumer Price Index": "cpi",
            "Long Interest Rate": "long_rate",
        }
    )
    raw["month"] = raw["Date"] + pd.offsets.MonthEnd(0)
    cols = ["price", "dividend", "earnings", "cpi", "long_rate"]
    raw[cols] = raw[cols].replace(0.0, np.nan)  # the file uses 0 for "not reported"
    return raw[["month", *cols]].set_index("month")


def build_store(df: pd.DataFrame) -> FeatureStore:
    """Ingest the raw series and two derived ones, each with the release rule of its input."""
    d = df.reset_index()
    d["earnings_growth"] = np.log(d["earnings"] / d["earnings"].shift(12))
    d["inflation"] = np.log(d["cpi"] / d["cpi"].shift(12))
    store = FeatureStore()
    rules: dict[str, Any] = {
        "earnings": after_quarter_end(90),
        "earnings_growth": after_quarter_end(90),
        "dividend": after_quarter_end(30),
        "inflation": monthly_release(15),
        "price": pd.Timedelta(0),
        "long_rate": pd.Timedelta(0),
    }
    for name, rule in rules.items():
        store.ingest(name, d, event_time="month", value=name, lag=rule)
    return store


def real_total_return(df: pd.DataFrame, h: int) -> pd.Series:
    """Log real total return from the end of month t to the end of month t + h."""
    one = np.log((df["price"] + df["dividend"] / 12) / df["price"].shift(1)) - np.log(
        df["cpi"] / df["cpi"].shift(1)
    )
    forward = sum(one.shift(-k) for k in range(1, h + 1))
    return forward.rename(f"ret_{h}m")


def predictors(store: FeatureStore, months: pd.DatetimeIndex, mode: str) -> pd.DataFrame:
    spine = pd.DataFrame({"ts": months})
    names = ["earnings", "dividend", "earnings_growth", "inflation", "price", "long_rate"]
    f = store.as_of(spine, names, mode=mode, metadata=True)  # type: ignore[arg-type]
    out = pd.DataFrame(index=months)
    out["log_ep"] = np.log(f["earnings"] / f["price"]).to_numpy()
    out["log_dp"] = np.log(f["dividend"] / f["price"]).to_numpy()
    for name in ("earnings_growth", "inflation", "long_rate"):
        out[name] = f[name].to_numpy()
    for name in ("earnings", "dividend", "inflation"):
        out[f"{name}__available_at"] = f[f"{name}__available_at"].to_numpy()
        out[f"{name}__event_time"] = f[f"{name}__event_time"].to_numpy()
    return out


def clark_west(y: np.ndarray, pred: np.ndarray, bench: np.ndarray, h: int) -> float:
    """Clark and West (2007) MSPE-adjusted t-statistic; > 1.645 rejects at 5% (one-sided)."""
    f = (y - bench) ** 2 - (y - pred) ** 2 + (bench - pred) ** 2
    centred = f - f.mean()
    n = len(f)
    lags = max(h - 1, 0)
    var = centred @ centred / n
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (centred[k:] @ centred[:-k]) / n
    return float(f.mean() / np.sqrt(var / n))


def forecast(x: pd.Series, y: pd.Series, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expanding-window OLS forecasts from OOS_START; training uses only realised targets."""
    months = y.index
    ys, preds, benches = [], [], []
    for i, t in enumerate(months):
        if t < OOS_START or np.isnan(y.iloc[i]) or np.isnan(x.iloc[i]):
            continue
        # The target for month s is known at s + h months, so it can be used at t if s + h <= t.
        train = slice(0, max(i - h + 1, 0))
        xt, yt = x.iloc[train].to_numpy(), y.iloc[train].to_numpy()
        ok = ~(np.isnan(xt) | np.isnan(yt))
        xt, yt = xt[ok], yt[ok]
        slope, intercept = np.polyfit(xt, yt, 1)
        ys.append(y.iloc[i])
        preds.append(intercept + slope * x.iloc[i])
        benches.append(yt.mean())
    return np.array(ys), np.array(preds), np.array(benches)


def plot_earnings_yield(naive: pd.DataFrame, pit: pd.DataFrame) -> None:
    window = slice("2007-07-01", "2010-06-30")
    a, b = naive.loc[window, "log_ep"], pit.loc[window, "log_ep"]

    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        style_axes(ax, theme)
        naive_c, pit_c = theme["series"][1], theme["series"][0]
        ax.plot(a.index, np.exp(a) * 100, color=naive_c, lw=2)
        ax.plot(b.index, np.exp(b) * 100, color=pit_c, lw=2)
        ax.text(
            pd.Timestamp("2008-01-15"), 3.3, "Naive join (CSV as is)", color=naive_c, fontsize=9
        )
        ax.text(pd.Timestamp("2008-05-01"), 5.3, "Point-in-time", color=pit_c, fontsize=9)
        ax.set_ylabel("Earnings yield, E/P (%)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_title("The CSV knows about the 2008 earnings collapse months early", loc="left")
        return fig

    save_figure(draw, "shiller-earnings-yield-2008")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    start = time.perf_counter()
    df = load()
    store = build_store(df)
    end = df[["earnings", "dividend", "cpi"]].dropna().index.max()
    months = df.loc[START:end].index
    views = {mode: predictors(store, months, mode) for mode in ("naive", "pit")}

    rows = []
    for h in HORIZONS:
        y = real_total_return(df, h).loc[months]
        for name, label in PREDICTORS.items():
            for mode, frame in views.items():
                ys, preds, benches = forecast(frame[name], y, h)
                rows.append(
                    {
                        "horizon_months": h,
                        "predictor": name,
                        "label": label,
                        "mode": "point-in-time" if mode == "pit" else "naive",
                        "n_forecasts": len(ys),
                        "first": OOS_START,
                        "oos_r2": oos_r2(ys, preds, benches),
                        "clark_west_t": clark_west(ys, preds, benches, h),
                    }
                )
    results = pd.DataFrame(rows)

    # How far ahead of publication is the naive join?
    naive = views["naive"].loc[OOS_START:]
    lead = {}
    for name in ("earnings", "dividend", "inflation"):
        days = (naive[f"{name}__available_at"] - naive.index).dt.days
        lead[name] = {
            "share_of_months_unpublished": float((days > 0).mean()),
            "median_days_before_publication": float(days[days > 0].median()),
            "max_days_before_publication": float(days.max()),
        }
    pit = views["pit"].loc[OOS_START:]
    staleness = {
        name: float((pit.index - pit[f"{name}__event_time"]).dt.days.median())
        for name in ("earnings", "dividend", "inflation")
    }

    write_csv(results, "shiller_oos.csv")
    write_json(
        {
            "data": "data/shiller_sp500_monthly.csv",
            "sample": [months[0], months[-1]],
            "oos_start": OOS_START,
            "naive_lead_over_publication": lead,
            "point_in_time_median_age_days": staleness,
            "describe": store.describe().to_dict(orient="records"),
            "seconds": time.perf_counter() - start,
        },
        "shiller_meta.json",
    )
    plot_earnings_yield(views["naive"], views["pit"])
    table = results.pivot_table(
        index=["horizon_months", "label"], columns="mode", values=["oos_r2", "clark_west_t"]
    )
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(table.round(4))
    print(lead)
    print(staleness)
    store.close()


if __name__ == "__main__":
    main()
