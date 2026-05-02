"""Tests for src.data.assembler."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.assembler import build_master_dataset
from src.data.config import CboeSpec, DataConfig, FredSeriesSpec, TickerSpec
from src.data.exceptions import ValidationError


def _ohlcv(idx: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    n = len(idx)
    return pd.DataFrame({
        "open":   [base] * n,
        "high":   [base + 1] * n,
        "low":    [base - 1] * n,
        "close":  [base + 0.5] * n,
        "volume": [1_000_000] * n,
    }, index=idx)


def _config(training_start: date = date(2010, 1, 1)) -> DataConfig:
    return DataConfig(
        start=date(2007, 1, 1),
        end=date(2025, 12, 31),
        training_start=training_start,
        yfinance_tickers=(
            TickerSpec("us100", "^NDX", "primary"),
            TickerSpec("vix",   "^VIX", "context"),
            TickerSpec("spx",   "^GSPC", "context"),
            TickerSpec("dxy",   "DX-Y.NYB", "context"),
        ),
        fred_series=(FredSeriesSpec("cpi", "CPIAUCSL", "release_dates_only"),),
        cboe=CboeSpec("cboe_direct", "yfinance_cpc"),
        fred_api_key="x",
    )


def test_index_matches_us100_anchor() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    yf = {"us100": _ohlcv(idx), "vix": _ohlcv(idx, 15), "spx": _ohlcv(idx, 4500), "dxy": _ohlcv(idx, 100)}
    pc = pd.Series([0.85] * len(idx), index=idx, name="putcall_ratio")

    out = build_master_dataset(yf, pc, _config(training_start=date(2024, 1, 1)))

    assert out.index.equals(idx)
    assert list(out.columns) == [
        "us100_open", "us100_high", "us100_low", "us100_close", "us100_volume",
        "vix_open", "vix_high", "vix_low", "vix_close",
        "spx_close", "dxy_close", "putcall_ratio",
    ]


def test_context_series_gap_one_day_ffilled() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    vix_idx = idx.delete(3)  # gap on day 3
    yf = {
        "us100": _ohlcv(idx),
        "vix":   _ohlcv(vix_idx, 15),
        "spx":   _ohlcv(idx, 4500),
        "dxy":   _ohlcv(idx, 100),
    }
    pc = pd.Series([0.85] * len(idx), index=idx)

    out = build_master_dataset(yf, pc, _config(training_start=date(2024, 1, 1)))
    assert not out["vix_close"].isna().any()


def test_pre_training_nan_tolerated() -> None:
    idx_full = pd.date_range("2007-01-02", periods=200, freq="B")
    cutoff = pd.Timestamp("2007-06-01")
    early_only = idx_full[idx_full < cutoff]

    yf = {
        "us100": _ohlcv(idx_full),
        "vix":   _ohlcv(idx_full[idx_full >= cutoff], 15),  # missing first half
        "spx":   _ohlcv(idx_full, 4500),
        "dxy":   _ohlcv(idx_full, 100),
    }
    pc = pd.Series([0.85] * len(idx_full), index=idx_full)

    cfg = _config(training_start=date(2007, 7, 1))  # cutoff inside the present-data range
    out = build_master_dataset(yf, pc, cfg)

    post = out[out.index >= pd.Timestamp(cfg.training_start)]
    assert not post[["vix_close", "spx_close", "dxy_close"]].isna().any().any()


def test_post_training_nan_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    spx_idx = idx[:5]  # SPX missing for second half — gap larger than ffill limit
    yf = {
        "us100": _ohlcv(idx),
        "vix":   _ohlcv(idx, 15),
        "spx":   _ohlcv(spx_idx, 4500),
        "dxy":   _ohlcv(idx, 100),
    }
    pc = pd.Series([0.85] * len(idx), index=idx)

    with pytest.raises(ValidationError, match="NaN found in core columns"):
        build_master_dataset(yf, pc, _config(training_start=date(2024, 1, 1)))


def test_missing_required_series_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=5, freq="B")
    yf = {"us100": _ohlcv(idx)}  # missing vix, spx, dxy
    pc = pd.Series([0.85] * len(idx), index=idx)

    with pytest.raises(ValidationError, match="missing required series"):
        build_master_dataset(yf, pc, _config())


def test_putcall_long_gap_ffilled_within_limit() -> None:
    idx = pd.date_range("2024-01-02", periods=10, freq="B")
    pc_idx = idx[:1].union(idx[6:])  # 5-day gap
    pc = pd.Series([0.85] * len(pc_idx), index=pc_idx)

    yf = {"us100": _ohlcv(idx), "vix": _ohlcv(idx, 15), "spx": _ohlcv(idx, 4500), "dxy": _ohlcv(idx, 100)}

    out = build_master_dataset(yf, pc, _config(training_start=date(2024, 1, 1)))
    # Within 5-day ffill limit -> no NaN
    assert not out["putcall_ratio"].isna().any()
