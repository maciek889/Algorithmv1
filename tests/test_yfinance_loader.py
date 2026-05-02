"""Tests for src.data.loaders.yfinance_loader (mocked)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data import loaders
from src.data.exceptions import DataLoadError
from src.data.loaders import yfinance_loader
from src.data.loaders.yfinance_loader import load_yfinance_series


def _good_df(n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":   [100.0] * n,
            "High":   [101.0] * n,
            "Low":    [ 99.0] * n,
            "Close":  [100.5] * n,
            "Volume": [1_000_000] * n,
        },
        index=idx,
    )


def _bad_high_low_df() -> pd.DataFrame:
    df = _good_df(3)
    df.loc[df.index[1], "High"] = 50.0  # high < low
    return df


def test_validates_ohlc_high_below_low(tmp_path: Path, mocker) -> None:
    mocker.patch.object(yfinance_loader.yf, "download", return_value=_bad_high_low_df())
    with pytest.raises(DataLoadError, match="high < low"):
        load_yfinance_series("us100", "^NDX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)


def test_handles_empty_response(tmp_path: Path, mocker) -> None:
    mocker.patch.object(yfinance_loader.yf, "download", return_value=pd.DataFrame())
    mocker.patch.object(yfinance_loader.time, "sleep")  # speed up retries
    with pytest.raises(DataLoadError, match="no data"):
        load_yfinance_series("us100", "^NDX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)


def test_caches_correctly_second_call_no_api(tmp_path: Path, mocker) -> None:
    mock_download = mocker.patch.object(yfinance_loader.yf, "download", return_value=_good_df())
    args = ("us100", "^NDX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)

    df1 = load_yfinance_series(*args)
    df2 = load_yfinance_series(*args)

    assert mock_download.call_count == 1
    pd.testing.assert_frame_equal(df1, df2, check_freq=False)


def test_refresh_bypasses_cache(tmp_path: Path, mocker) -> None:
    mock_download = mocker.patch.object(yfinance_loader.yf, "download", return_value=_good_df())
    args = ("us100", "^NDX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)

    load_yfinance_series(*args)
    load_yfinance_series(*args, refresh=True)

    assert mock_download.call_count == 2


def test_dxy_fallback_to_secondary_ticker(tmp_path: Path, mocker, caplog) -> None:
    call_count = {"n": 0}

    def fake_download(ticker, **_kwargs):
        call_count["n"] += 1
        if ticker == "DX-Y.NYB":
            return pd.DataFrame()
        return _good_df()

    mocker.patch.object(yfinance_loader.yf, "download", side_effect=fake_download)
    mocker.patch.object(yfinance_loader.time, "sleep")

    with caplog.at_level("WARNING"):
        df = load_yfinance_series("dxy", "DX-Y.NYB", date(2024, 1, 1), date(2024, 2, 1), tmp_path)

    assert len(df) > 0
    assert any("falling back" in rec.message for rec in caplog.records)


def test_vix_zero_volume_accepted(tmp_path: Path, mocker) -> None:
    df = _good_df()
    df["Volume"] = 0
    mocker.patch.object(yfinance_loader.yf, "download", return_value=df)

    out = load_yfinance_series("vix", "^VIX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)
    assert (out["volume"] == 0).all()


def test_tz_aware_index_forced_naive(tmp_path: Path, mocker) -> None:
    df = _good_df()
    df.index = df.index.tz_localize("UTC")
    mocker.patch.object(yfinance_loader.yf, "download", return_value=df)

    out = load_yfinance_series("us100", "^NDX", date(2024, 1, 1), date(2024, 2, 1), tmp_path)
    assert out.index.tz is None
