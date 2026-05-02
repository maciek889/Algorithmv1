"""CBOE put/call ratio loader with yfinance ^CPCE fallback.

Best-effort: CBOE has changed historical CSV URLs multiple times. We try a
small list of known endpoints, and on failure fall back to ^CPCE via yfinance.
The fallback is logged at WARNING so the QA report can flag the source used.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
import yfinance as yf

from src.data.cache import is_cached, read_cache, write_cache
from src.data.exceptions import DataLoadError

logger = logging.getLogger(__name__)

_CBOE_DIRECT_URLS = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/CPC_History.csv",
    "https://www.cboe.com/us/options/market_statistics/daily/csv/?dt=ALL",
)
_FALLBACK_TICKER = "^CPCE"
_HTTP_TIMEOUT_SECONDS = 30


def load_cboe_putcall(
    start: date,
    end: date,
    cache_dir: Path,
    source: Literal["cboe_direct", "yfinance_cpc"] = "cboe_direct",
    refresh: bool = False,
) -> pd.Series:
    """Return daily CBOE put/call ratio as a Series.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.
        cache_dir: Directory for parquet cache.
        source: Primary source attempt; falls back to yfinance ^CPCE on failure.
        refresh: If True, bypass cache.

    Returns:
        Series with name 'putcall_ratio', tz-naive DatetimeIndex, float64.

    Raises:
        DataLoadError: both primary and fallback failed.
    """
    cache_path = cache_dir / f"putcall_{start}_{end}.parquet"
    if not refresh:
        cached = read_cache(cache_path)
        if cached is not None:
            return cached["putcall_ratio"]

    series: pd.Series | None = None
    used_source: str | None = None

    if source == "cboe_direct":
        series = _try_cboe_direct(start, end)
        if series is not None:
            used_source = "cboe_direct"
        else:
            logger.warning("CBOE direct unavailable, falling back to yfinance %s", _FALLBACK_TICKER)

    if series is None:
        series = _try_yfinance_cpc(start, end)
        if series is not None:
            used_source = "yfinance_cpc"

    if series is None or series.empty:
        raise DataLoadError("Both CBOE direct and yfinance ^CPCE fallback returned no data")

    series = series.sort_index()
    series = series[~series.index.duplicated(keep="first")]
    series = series.loc[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
    series.name = "putcall_ratio"

    write_cache(series.to_frame(), cache_path, source="cboe", key=used_source or "unknown")
    logger.info("Loaded put/call: %d rows via %s, %s..%s",
                len(series), used_source, series.index.min().date(), series.index.max().date())
    return series


def _try_cboe_direct(start: date, end: date) -> pd.Series | None:
    for url in _CBOE_DIRECT_URLS:
        try:
            logger.debug("Trying CBOE URL: %s", url)
            resp = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200 or not resp.content:
                continue
            series = _parse_cboe_csv(resp.content)
            if series is not None and not series.empty:
                return series
        except (requests.RequestException, ValueError) as e:
            logger.debug("CBOE URL %s failed: %s", url, e)
            continue
    return None


def _parse_cboe_csv(content: bytes) -> pd.Series | None:
    """Parse a CBOE CSV. Schema varies; best-effort detection of date + ratio columns."""
    for skip in (0, 1, 2, 3):
        try:
            df = pd.read_csv(io.BytesIO(content), skiprows=skip)
        except Exception:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        date_col = next((cols_lower[k] for k in cols_lower if "date" in k), None)
        ratio_col = next((cols_lower[k] for k in cols_lower if "p/c" in k or "putcall" in k or "ratio" in k), None)

        if date_col is None or ratio_col is None:
            continue

        try:
            idx = pd.to_datetime(df[date_col], errors="coerce")
            vals = pd.to_numeric(df[ratio_col], errors="coerce")
            series = pd.Series(vals.values, index=idx, name="putcall_ratio").dropna()
            series.index = series.index.tz_localize(None) if series.index.tz else series.index
            if not series.empty:
                return series
        except Exception:
            continue

    return None


def _try_yfinance_cpc(start: date, end: date) -> pd.Series | None:
    try:
        df = yf.download(
            _FALLBACK_TICKER,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Close" if "Close" in df.columns else "close"
        series = df[col].astype("float64")
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.name = "putcall_ratio"
        return series
    except Exception as e:
        logger.warning("yfinance ^CPCE fallback failed: %s", e)
        return None
