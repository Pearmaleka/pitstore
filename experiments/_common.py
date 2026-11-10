"""Paths, output helpers and plotting theme shared by the experiment scripts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "docs" / "figures"
SHILLER = ROOT / "data" / "shiller_sp500_monthly.csv"

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}


def _default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"cannot serialise {type(obj)}")


def write_json(obj: Any, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n", encoding="utf-8")
    return path


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    frame.to_csv(path, index=False, float_format="%.6g")
    return path


def style_axes(ax: plt.Axes, theme: dict[str, Any]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])


def save_figure(draw: Callable[[dict[str, Any]], plt.Figure], name: str) -> None:
    """Render ``draw(theme)`` as ``docs/figures/<name>-light.png`` and ``-dark.png``."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for theme_name, theme in THEMES.items():
        rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "savefig.dpi": 200,
            "figure.facecolor": theme["surface"],
            "axes.facecolor": theme["surface"],
            "text.color": theme["ink"],
            "axes.titlecolor": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "legend.labelcolor": theme["ink2"],
        }
        with plt.rc_context(rc):
            fig = draw(theme)
            fig.savefig(
                FIGURES / f"{name}-{theme_name}.png",
                facecolor=theme["surface"],
                bbox_inches="tight",
                pad_inches=0.15,
            )
            plt.close(fig)


def oos_r2(y: np.ndarray, pred: np.ndarray, benchmark: np.ndarray) -> float:
    """Out-of-sample R² against a benchmark forecast (Campbell and Thompson, 2008)."""
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - benchmark) ** 2))
