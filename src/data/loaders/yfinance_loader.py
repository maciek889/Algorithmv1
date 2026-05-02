"""yfinance OHLCV loader with cache, retry, and OHLC sanity validation."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.cache import is_cached, read_cache, write_cache
from src.data.exceptions import DataLoadError

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_SECONDS = (1, 2, 4)
_DXY_FALLBACK = "DXY=F"
_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def load_yfinance_series(
    name: str,
    ticker: str,
    start: date,
    end: date,
    cache_dir: Path,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download and validate OHLCV for a single ticker.

    Args:
        name: Logical name (e.g. "us100"); used for cache filename + log context.
        ticker: yfinance ticker symbol (e.g. "^NDX").
        start: Inclusive start date.
        end: Exclusive end date (yfinance convention).
        cache_dir: Directory for parquet cache.
        refresh: If True, bypass cache and re-download.

    Returns:
        DataFrame with DatetimeIndex (tz-naive, monotonic, unique) and
        columns [open, high, low, close, volume] (float64 except volume int64).

    Raises:
        DataLoadError: empty response after retries, or OHLC validation failed.
    """
    cache_path = cache_dir / f"{name}_{start}_{end}.parquet"
    if not refresh:
        cached = read_cache(cache_path)
        if cached is not None:
            return cached

    df = _download_with_retry(name, ticker, start, end)

    if name == "dxy" and df is None:
        logger.warning("DXY primary ticker '%s' empty; falling back to '%s'", ticker, _DXY_FALLBACK)
        df = _download_with_retry(name, _DXY_FALLBACK, start, end)

    if df is None or df.empty:
        raise DataLoadError(f"yfinance returned no data for {name} ({ticker}) {start}..{end}")

    df = _normalize(df)
    _validate_ohlc(df, name=name, allow_zero_volume=(name == "vix"))

    write_cache(df, cache_path, source="yfinance", key=name)
    logger.info("Loaded %s (%s): %d rows, %s..%s", name, ticker, len(df), df.index.min().date(), df.index.max().date())
    return df


def _download_with_retry(name: str, ticker: str, start: date, end: date) -> pd.DataFrame | None:
    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_BACKOFF_SECONDS, start=1):
        try:
            df = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or len(df) == 0:
                logger.warning("yfinance empty response for %s (%s) attempt %d", name, ticker, attempt)
                if attempt < len(_RETRY_BACKOFF_SECONDS):
                    time.sleep(delay)
                continue
            return df
        except Exception as e:
            last_exc = e
            logger.warning("yfinance download error for %s (%s) attempt %d: %s", name, ticker, attempt, e)
            if attempt < len(_RETRY_BACKOFF_SECONDS):
                time.sleep(delay)

    if last_exc is not None:
        logger.warning("yfinance gave up for %s (%s) after retries: %s", name, ticker, last_exc)
    return None


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(f"yfinance response missing columns: {missing}")

    df = df[list(_REQUIRED_COLUMNS)]

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[~df.index.duplicated(keep="first")].sort_index()

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].fillna(0).astype("int64")

    return df


def _validate_ohlc(df: pd.DataFrame, name: str, allow_zero_volume: bool) -> None:
    if df["close"].isna().any():
        raise DataLoadError(f"{name}: NaN in close column")

    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        raise DataLoadError(f"{name}: {bad_hl.sum()} rows where high < low")

    bad_high = (df["high"] < df["close"]) | (df["high"] < df["open"])
    if bad_high.any():
        raise DataLoadError(f"{name}: {bad_high.sum()} rows where high < max(open, close)")

    bad_low = (df["low"] > df["close"]) | (df["low"] > df["open"])
    if bad_low.any():
        raise DataLoadError(f"{name}: {bad_low.sum()} rows where low > min(open, close)")

    if (df["volume"] < 0).any():
        raise DataLoadError(f"{name}: negative volume detected")

    if not allow_zero_volume and (df["volume"] == 0).all():
        raise DataLoadError(f"{name}: all-zero volume (unexpected for non-index ticker)")
