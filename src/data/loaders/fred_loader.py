"""FRED loader for macro release dates (as-of timing aware).

Critical: Stage 1.1's biggest lookahead-bias risk is using FRED's reference_date
(period the data describes) instead of release_date (when it was published).
CPI for January 2024 reports refer to January but were published ~2024-02-13.
This module returns release_date and a derived available_from = release_date + 1 BDay.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
from fredapi import Fred

from src.data.cache import is_cached, read_cache, write_cache
from src.data.exceptions import DataLoadError

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SECONDS = (1, 2, 4)


def load_fred_release_dates(
    series_id: str,
    start: date,
    end: date,
    cache_dir: Path,
    api_key: str,
    refresh: bool = False,
) -> pd.DataFrame:
    """Pull all releases of a FRED series and reduce to first-publication dates.

    Args:
        series_id: FRED series identifier (e.g. "CPIAUCSL").
        start: Inclusive start of the reference period range to keep.
        end: Inclusive end of the reference period range to keep.
        cache_dir: Directory for parquet cache.
        api_key: FRED API key.
        refresh: If True, bypass cache and re-fetch.

    Returns:
        DataFrame with columns:
            reference_date  (Timestamp)  — period the value describes
            release_date    (Timestamp)  — first publication of that value
            available_from  (Timestamp)  — release_date + 1 business day

    Raises:
        DataLoadError: empty result, missing api_key, or repeated API failure.
    """
    if not api_key:
        raise DataLoadError("FRED_API_KEY missing; cannot call FRED API")

    cache_path = cache_dir / f"{series_id}_releases_{start}_{end}.parquet"
    if not refresh:
        cached = read_cache(cache_path)
        if cached is not None:
            return cached

    raw = _fetch_with_retry(series_id, api_key)
    if raw is None or raw.empty:
        raise DataLoadError(f"FRED returned no releases for series {series_id}")

    df = _reduce_to_first_release(raw)
    df = _filter_reference_window(df, start, end)
    df["available_from"] = df["release_date"] + pd.tseries.offsets.BDay(1)

    df = df.sort_values("reference_date").reset_index(drop=True)

    write_cache(df, cache_path, source="fred", key=series_id)
    logger.info("Loaded FRED %s releases: %d rows, %s..%s",
                series_id, len(df), df["reference_date"].min().date(), df["reference_date"].max().date())
    return df


def load_fred_value_series(*_args, **_kwargs):
    """Reserved for future use; not in scope of Stage 1.1."""
    raise NotImplementedError("Reserved for future use; not in scope of 1.1")


def _fetch_with_retry(series_id: str, api_key: str) -> pd.DataFrame | None:
    fred = Fred(api_key=api_key)
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_BACKOFF_SECONDS, start=1):
        try:
            raw = fred.get_series_all_releases(series_id)
            if raw is None or len(raw) == 0:
                logger.warning("FRED empty response for %s attempt %d", series_id, attempt)
                if attempt < len(_RETRY_BACKOFF_SECONDS):
                    time.sleep(delay)
                continue
            return raw
        except Exception as e:
            last_exc = e
            logger.warning("FRED fetch error for %s attempt %d: %s", series_id, attempt, e)
            if attempt < len(_RETRY_BACKOFF_SECONDS):
                time.sleep(delay)

    if last_exc is not None:
        logger.warning("FRED gave up for %s after retries: %s", series_id, last_exc)
    return None


def _reduce_to_first_release(raw: pd.DataFrame) -> pd.DataFrame:
    """Group by reference date, keep earliest realtime_start (first publication)."""
    required = {"date", "realtime_start"}
    missing = required - set(raw.columns)
    if missing:
        raise DataLoadError(f"FRED response missing columns: {missing}")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])

    first_release = (
        df.groupby("date", as_index=False)["realtime_start"].min()
        .rename(columns={"date": "reference_date", "realtime_start": "release_date"})
    )
    return first_release


def _filter_reference_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df["reference_date"] >= pd.Timestamp(start)) & (df["reference_date"] <= pd.Timestamp(end))
    return df.loc[mask].copy()
