"""FOMC meeting dates loader.

Reads from a hand-curated CSV committed to the repo at
data/calendars_source/fomc_dates_source.csv. FRED has no clean FOMC series.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from src.data.exceptions import DataLoadError

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ("meeting_date", "decision_date", "is_unscheduled")
_MIN_EXPECTED_ROWS = 130  # ~8/yr * 17 yrs (2007-2024) - some tolerance


def load_fomc_dates(csv_path: Path, start: date, end: date) -> pd.DataFrame:
    """Read the committed FOMC CSV and return a validated frame.

    Args:
        csv_path: Path to data/calendars_source/fomc_dates_source.csv.
        start: Inclusive start date for filtering by meeting_date.
        end: Inclusive end date for filtering by meeting_date.

    Returns:
        DataFrame with columns [meeting_date, decision_date, is_unscheduled],
        sorted by meeting_date.

    Raises:
        DataLoadError: file missing, schema mismatch, fewer rows than expected,
            or duplicate meeting_dates.
    """
    if not csv_path.exists():
        raise DataLoadError(
            f"FOMC source CSV not found at {csv_path}. "
            "Compile from federalreserve.gov/monetarypolicy/fomccalendars.htm "
            "with columns: meeting_date,decision_date,is_unscheduled"
        )

    df = pd.read_csv(csv_path, encoding="utf-8")

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"FOMC CSV missing columns: {missing}")

    df["meeting_date"] = pd.to_datetime(df["meeting_date"], errors="coerce")
    df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

    if df["meeting_date"].isna().any():
        bad = int(df["meeting_date"].isna().sum())
        raise DataLoadError(f"FOMC CSV has {bad} unparseable meeting_date values")

    df["is_unscheduled"] = df["is_unscheduled"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    )
    if df["is_unscheduled"].isna().any():
        raise DataLoadError("FOMC CSV is_unscheduled column contains values that aren't true/false")

    if df["meeting_date"].duplicated().any():
        dups = df.loc[df["meeting_date"].duplicated(), "meeting_date"].tolist()
        raise DataLoadError(f"FOMC CSV has duplicate meeting_dates: {dups}")

    df = df.sort_values("meeting_date").reset_index(drop=True)

    if len(df) < _MIN_EXPECTED_ROWS:
        raise DataLoadError(f"FOMC CSV has only {len(df)} rows; expected >= {_MIN_EXPECTED_ROWS}")

    mask = (df["meeting_date"] >= pd.Timestamp(start)) & (df["meeting_date"] <= pd.Timestamp(end))
    df_filtered = df.loc[mask].reset_index(drop=True)

    logger.info("Loaded FOMC dates: %d total in CSV, %d in window %s..%s",
                len(df), len(df_filtered), start, end)
    return df_filtered
