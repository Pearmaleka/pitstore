"""Command-line interface.

pitstore ingest  DB FEATURE data.csv --event-time month --value cpi --lag-days 15
pitstore asof    DB spine.csv --features cpi,rate --time ts -o train.csv
pitstore snapshot DB FEATURE --as-of 2009-01-01
pitstore describe DB
pitstore check   DB
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import pandas as pd

from pitstore import __version__
from pitstore.checks import check
from pitstore.store import FeatureStore


def _emit(df: pd.DataFrame, output: str | None) -> None:
    if output:
        df.to_csv(output, index=False)
        print(f"wrote {len(df)} rows to {output}")
    else:
        with pd.option_context("display.width", 160, "display.max_columns", 50):
            print(df.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pitstore", description="Point-in-time feature store.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="load a CSV of observations into a feature")
    ing.add_argument("db")
    ing.add_argument("feature")
    ing.add_argument("csv")
    ing.add_argument("--event-time", required=True, help="column with the period described")
    ing.add_argument("--value", required=True, help="column with the value")
    ing.add_argument("--entity", help="column identifying the entity")
    when = ing.add_mutually_exclusive_group(required=True)
    when.add_argument("--available-at", help="column with the publication time")
    when.add_argument("--lag-days", type=float, help="publication lag in days")
    ing.add_argument("--description", default="")
    ing.add_argument("--unit", default="")

    asof = sub.add_parser("asof", help="attach point-in-time features to a spine CSV")
    asof.add_argument("db")
    asof.add_argument("spine")
    asof.add_argument("--features", required=True, help="comma-separated feature names")
    asof.add_argument("--time", default="ts")
    asof.add_argument("--entity")
    asof.add_argument("--mode", choices=["pit", "naive"], default="pit")
    asof.add_argument("--max-age-days", type=float)
    asof.add_argument("-o", "--output")

    snap = sub.add_parser("snapshot", help="a feature as it was known at a given time")
    snap.add_argument("db")
    snap.add_argument("feature")
    snap.add_argument("--as-of", required=True)
    snap.add_argument("--entity")
    snap.add_argument("-o", "--output")

    desc = sub.add_parser("describe", help="coverage, revisions and lags of every feature")
    desc.add_argument("db")

    chk = sub.add_parser("check", help="data-quality checks; exit code 1 if any fail")
    chk.add_argument("db")
    chk.add_argument("--stale-days", type=float)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    read_only = args.command != "ingest"
    with FeatureStore(args.db, read_only=read_only) as store:
        if args.command == "ingest":
            data = pd.read_csv(args.csv)
            report = store.ingest(
                args.feature,
                data,
                event_time=args.event_time,
                value=args.value,
                entity=args.entity,
                available_at=args.available_at,
                lag=None if args.lag_days is None else pd.Timedelta(days=args.lag_days),
            )
            store.register(args.feature, description=args.description, unit=args.unit)
            print(report)
        elif args.command == "asof":
            spine = pd.read_csv(args.spine)
            out = store.as_of(
                spine,
                [f.strip() for f in args.features.split(",") if f.strip()],
                time=args.time,
                entity=args.entity,
                mode=args.mode,
                max_age=None if args.max_age_days is None else pd.Timedelta(days=args.max_age_days),
            )
            _emit(out, args.output)
        elif args.command == "snapshot":
            _emit(store.snapshot(args.feature, args.as_of, entity=args.entity), args.output)
        elif args.command == "describe":
            _emit(store.describe(), None)
        else:
            stale = None if args.stale_days is None else pd.Timedelta(days=args.stale_days)
            issues = check(store, stale_after=stale)
            if issues.empty:
                print("no issues found")
                return 0
            _emit(issues, None)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
