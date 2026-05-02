"""Tests for src.data.loaders.cboe_loader (mocked)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.exceptions import DataLoadError
from src.data.loaders import cboe_loader
from src.data.loaders.cboe_loader import load_cboe_putcall


_GOOD_CSV = (
    b"DATE,P/C Ratio\n"
    b"2024-01-02,0.85\n"
    b"2024-01-03,0.92\n"
    b"2024-01-04,0.78\n"
)


def _yfin_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    return pd.DataFrame({"Close": [0.85, 0.92, 0.78]}, index=idx)


def test_cboe_direct_happy_path(tmp_path: Path, mocker, caplog) -> None:
    mock_resp = MagicMock(status_code=200, content=_GOOD_CSV)
    mocker.patch.object(cboe_loader.requests, "get", return_value=mock_resp)
    mocker.patch.object(cboe_loader.yf, "download")  # must NOT be called

    with caplog.at_level("INFO"):
        s = load_cboe_putcall(date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert len(s) == 3
    assert s.name == "putcall_ratio"
    cboe_loader.yf.download.assert_not_called()
    assert any("cboe_direct" in rec.message for rec in caplog.records)


def test_falls_back_to_yfinance_when_direct_fails(tmp_path: Path, mocker, caplog) -> None:
    mocker.patch.object(cboe_loader.requests, "get", return_value=MagicMock(status_code=500, content=b""))
    mocker.patch.object(cboe_loader.yf, "download", return_value=_yfin_df())

    with caplog.at_level("WARNING"):
        s = load_cboe_putcall(date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert len(s) == 3
    assert any("falling back" in rec.message for rec in caplog.records)


def test_both_sources_fail_raises(tmp_path: Path, mocker) -> None:
    mocker.patch.object(cboe_loader.requests, "get", return_value=MagicMock(status_code=500, content=b""))
    mocker.patch.object(cboe_loader.yf, "download", return_value=pd.DataFrame())

    with pytest.raises(DataLoadError, match="no data"):
        load_cboe_putcall(date(2024, 1, 1), date(2024, 1, 31), tmp_path)


def test_cache_round_trip(tmp_path: Path, mocker) -> None:
    mock_resp = MagicMock(status_code=200, content=_GOOD_CSV)
    mock_get = mocker.patch.object(cboe_loader.requests, "get", return_value=mock_resp)

    s1 = load_cboe_putcall(date(2024, 1, 1), date(2024, 1, 31), tmp_path)
    s2 = load_cboe_putcall(date(2024, 1, 1), date(2024, 1, 31), tmp_path)

    assert mock_get.call_count == 1  # second call hits cache
    pd.testing.assert_series_equal(s1, s2)


def test_filters_to_requested_window(tmp_path: Path, mocker) -> None:
    mock_resp = MagicMock(status_code=200, content=_GOOD_CSV)
    mocker.patch.object(cboe_loader.requests, "get", return_value=mock_resp)

    s = load_cboe_putcall(date(2024, 1, 3), date(2024, 1, 3), tmp_path)
    assert len(s) == 1
    assert s.index[0] == pd.Timestamp("2024-01-03")
