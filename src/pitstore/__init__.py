"""pitstore: a point-in-time feature store on DuckDB.

Training data built from a pitstore only contains what had been published at each row's
timestamp, including the version of revised figures that was current at the time.
"""

from pitstore.calendars import after_quarter_end, business_days, fixed_lag, monthly_release
from pitstore.checks import check
from pitstore.store import FeatureStore, IngestReport

__version__ = "0.1.0"

__all__ = [
    "FeatureStore",
    "IngestReport",
    "__version__",
    "after_quarter_end",
    "business_days",
    "check",
    "fixed_lag",
    "monthly_release",
]
