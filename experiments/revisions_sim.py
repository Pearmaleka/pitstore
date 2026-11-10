"""Simulation: how much does a backtest overstate live performance when training data leak?

A monthly macro indicator x_t (AR(1), phi = 0.6) drives next month's return:
r_{t+1} = beta * x_t + noise, with beta chosen so that knowing x_t explains 10% of the
variance. Like an advance GDP estimate, x_t is published with a lag and revised:

    t + 15 days   first release   x_t + N(0, 0.6^2)
    t + 45 days   second release  x_t + N(0, 0.3^2)
    t + 400 days  final release   x_t

A model is trained on 20 years of monthly data and evaluated on the next 20, with features
built three ways from the same store:

* naive:      join the final series on event time (what a downloaded CSV gives you);
* lag-aware:  shift the final series by the publication lag but keep the revised values;
* point-in-time: ``FeatureStore.as_of``, the vintage that was actually available.

"Backtest" R² evaluates each model with its own kind of features. "Live" R² evaluates the
same model with point-in-time features, which is all a model has in production.

Run from the repository root:  python experiments/revisions_sim.py [--quick]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import oos_r2, save_figure, style_axes, write_csv, write_json

from pitstore import FeatureStore

PHI = 0.6
TARGET_R2 = 0.10
RELEASES = [(15, 0.6), (45, 0.3), (400, 0.0)]  # (days after event, noise sd)
METHODS = ["naive", "lag-aware", "point-in-time"]


def simulate(n_months: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Releases of the indicator and the returns, both indexed by month end."""
    x = np.empty(n_months)
    x[0] = rng.standard_normal()
    for t in range(1, n_months):
        x[t] = PHI * x[t - 1] + np.sqrt(1 - PHI**2) * rng.standard_normal()
    beta = np.sqrt(TARGET_R2 / (1 - TARGET_R2))
    months = pd.date_range("1990-01-31", periods=n_months, freq="ME")
    ret = np.full(n_months, np.nan)
    ret[1:] = beta * x[:-1] + rng.standard_normal(n_months - 1)  # return over month t+1
    releases = pd.concat(
        [
            pd.DataFrame(
                {
                    "month": months,
                    "available_at": months + pd.Timedelta(days=lag),
                    "value": x + sd * rng.standard_normal(n_months),
                }
            )
            for lag, sd in RELEASES
        ],
        ignore_index=True,
    )
    returns = pd.DataFrame({"month": months, "next_return": np.roll(ret, -1)})
    return releases, returns.iloc[:-1]


def features(releases: pd.DataFrame, returns: pd.DataFrame) -> dict[str, np.ndarray]:
    store = FeatureStore()
    store.ingest("x", releases, event_time="month", value="value", available_at="available_at")
    final = releases[
        releases["available_at"] == releases.groupby("month")["available_at"].transform("max")
    ]
    store.ingest(
        "x_final_lagged",
        final,
        event_time="month",
        value="value",
        lag=pd.Timedelta(days=RELEASES[0][0]),
    )
    spine = returns[["month"]].rename(columns={"month": "ts"})
    out = {
        "naive": store.as_of(spine, ["x"], mode="naive")["x"].to_numpy(),
        "lag-aware": store.as_of(spine, ["x_final_lagged"])["x_final_lagged"].to_numpy(),
        "point-in-time": store.as_of(spine, ["x"])["x"].to_numpy(),
    }
    store.close()
    return out


def one_replication(n_months: int, rng: np.random.Generator) -> dict[str, float]:
    releases, returns = simulate(n_months, rng)
    feats = features(releases, returns)
    y = returns["next_return"].to_numpy()
    half = len(y) // 2
    ok = np.ones(len(y), dtype=bool)
    for v in feats.values():
        ok &= ~np.isnan(v)
    train = np.flatnonzero(ok[:half])
    test = half + np.flatnonzero(ok[half:])
    bench = np.full(len(test), y[train].mean())
    out: dict[str, float] = {}
    for method in METHODS:
        X = np.column_stack([np.ones(len(y)), feats[method]])
        coef, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
        own = X[test] @ coef
        live = np.column_stack([np.ones(len(test)), feats["point-in-time"][test]]) @ coef
        out[f"{method}|backtest"] = oos_r2(y[test], own, bench)
        out[f"{method}|live"] = oos_r2(y[test], live, bench)
        out[f"{method}|slope"] = float(coef[1])
    return out


def plot(summary: pd.DataFrame) -> None:
    def draw(theme: dict[str, Any]) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(7.4, 2.9))
        style_axes(ax, theme)
        ax.grid(axis="y", visible=False)
        rows = list(reversed(METHODS))
        back_c, live_c = theme["series"][0], theme["series"][1]
        for i, method in enumerate(rows):
            b = summary.loc[method, "backtest_mean"] * 100
            lv = summary.loc[method, "live_mean"] * 100
            ax.plot([lv, b], [i, i], color=theme["axis"], lw=2.5, zorder=1, solid_capstyle="round")
            ax.scatter(
                [b],
                [i],
                s=80,
                color=back_c,
                zorder=3,
                edgecolors=theme["surface"],
                linewidths=1.5,
                label="Backtest" if i == 0 else None,
            )
            ax.scatter(
                [lv],
                [i],
                s=80,
                color=live_c,
                zorder=3,
                edgecolors=theme["surface"],
                linewidths=1.5,
                label="Live" if i == 0 else None,
            )
            gap = b - lv
            ax.annotate(
                f"gap {gap:+.1f} pts" if abs(gap) >= 0.05 else "backtest = live",
                (max(b, lv), i),
                xytext=(10, 0),
                textcoords="offset points",
                va="center",
                color=theme["ink2"],
                fontsize=9,
            )
        ax.axvline(0, color=theme["muted"], lw=1)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("Out-of-sample R² (%), mean of simulations")
        ax.set_xlim(min(-3.0, summary["live_mean"].min() * 100 - 1), 13)
        ax.set_title("Leaky training data inflate the backtest", loc="left")
        leg = ax.legend(loc="center right", frameon=False, ncol=2)
        for text in leg.get_texts():
            text.set_color(theme["ink2"])
        return fig

    save_figure(draw, "revisions-backtest-vs-live")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    reps = 40 if args.quick else 500
    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    raw = pd.DataFrame([one_replication(480, rng) for _ in range(reps)])
    rows = []
    for method in METHODS:
        b, lv = raw[f"{method}|backtest"], raw[f"{method}|live"]
        rows.append(
            {
                "method": method,
                "backtest_mean": b.mean(),
                "backtest_q05": b.quantile(0.05),
                "backtest_q95": b.quantile(0.95),
                "live_mean": lv.mean(),
                "live_q05": lv.quantile(0.05),
                "live_q95": lv.quantile(0.95),
                "gap_mean": (b - lv).mean(),
                "share_backtest_above_live": (b > lv).mean(),
                "slope_mean": raw[f"{method}|slope"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    suffix = "-quick" if args.quick else ""
    write_csv(summary, f"revisions_sim{suffix}.csv")
    write_json(
        {
            "replications": reps,
            "months": 480,
            "phi": PHI,
            "target_r2": TARGET_R2,
            "releases": RELEASES,
            "seed": args.seed,
            "seconds": time.perf_counter() - start,
        },
        f"revisions_sim_meta{suffix}.json",
    )
    if not args.quick:
        plot(summary.set_index("method"))
    with pd.option_context("display.width", 160):
        print(summary.round(4).to_string(index=False))
    print(f"{time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
