"""Tests for src.data.loaders.fred_loader.

Includes the mandatory real-API sanity check that CPI for January 2024
was published on ~2024-02-13. Marked `integration` so CI can opt out.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.exceptions import DataLoadError
from src.data.loaders import fred_loader
from src.data.loaders.fred_loader import load_fred_release_dates, load_fred_value_series


def _fake_releases() -> pd.DataFrame:
    """Mimic fredapi.get_series_all_releases output: date, realtime_start, value.
    Includes a revision (later realtime_start) for the Jan period to verify dedup.
    """
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-01",  # Jan ref + a revision
            "2024-02-01",
            "2024-03-01",
        ]),
        "realtime_start": pd.to_datetime([
            "2024-02-13", "2024-03-12",  # original Jan release + revision
            "2024-03-12",
            "2024-04-10",
        ]),
        "value": [307.0, 307.5, 308.4, 309.0],
    })


def test_keeps_only_first_release_per_period(tmp_path: Path, mocker) -> None:
    fake_fred = mocker.MagicMock()
    fake_fred.get_series_all_releases.return_value = _fake_releases()
    mocker.patch.object(fred_loader, "Fred", return_value=fake_fred)

    df = load_fred_release_dates("CPIAUCSL", date(2024, 1, 1), date(2024, 3, 31), tmp_path, api_key="x")

    assert len(df) == 3
    jan = df[df["reference_date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert jan["release_date"] == pd.Timestamp("2024-02-13")
    assert jan["available_from"] == pd.Timestamp("2024-02-14")  # next BDay


def test_filters_reference_window(tmp_path: Path, mocker) -> None:
    fake_fred = mocker.MagicMock()
    fake_fred.get_series_all_releases.return_value = _fake_releases()
    mocker.patch.object(fred_loader, "Fred", return_value=fake_fred)

    df = load_fred_release_dates("CPIAUCSL", date(2024, 2, 1), date(2024, 2, 28), tmp_path, api_key="x")
    assert len(df) == 1
    assert df["reference_date"].iloc[0] == pd.Timestamp("2024-02-01")


def test_empty_response_raises(tmp_path: Path, mocker) -> None:
    fake_fred = mocker.MagicMock()
    fake_fred.get_series_all_releases.return_value = pd.DataFrame()
    mocker.patch.object(fred_loader, "Fred", return_value=fake_fred)
    mocker.patch.object(fred_loader.time, "sleep")

    with pytest.raises(DataLoadError, match="no releases"):
        load_fred_release_dates("XYZ", date(2024, 1, 1), date(2024, 3, 31), tmp_path, api_key="x")


def test_missing_api_key_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="FRED_API_KEY"):
        load_fred_release_dates("CPIAUCSL", date(2024, 1, 1), date(2024, 3, 31), tmp_path, api_key="")


def test_cache_round_trip(tmp_path: Path, mocker) -> None:
    fake_fred = mocker.MagicMock()
    fake_fred.get_series_all_releases.return_value = _fake_releases()
    mocker.patch.object(fred_loader, "Fred", return_value=fake_fred)

    df1 = load_fred_release_dates("CPIAUCSL", date(2024, 1, 1), date(2024, 3, 31), tmp_path, api_key="x")
    df2 = load_fred_release_dates("CPIAUCSL", date(2024, 1, 1), date(2024, 3, 31), tmp_path, api_key="x")

    assert fake_fred.get_series_all_releases.call_count == 1
    pd.testing.assert_frame_equal(df1, df2)


def test_value_series_stub_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        load_fred_value_series()


@pytest.mark.integration
def test_fred_release_dates_known_value(tmp_path: Path) -> None:
    """Real FRED call: CPI January 2024 was published ~2024-02-13.

    Tolerance: ±1 day to absorb minor FRED metadata differences.
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        pytest.skip("FRED_API_KEY not set in environment")

    df = load_fred_release_dates(
        "CPIAUCSL",
        date(2024, 1, 1),
        date(2024, 3, 31),
        tmp_path,
        api_key=api_key,
        refresh=True,
    )

    jan_rows = df[df["reference_date"] == pd.Timestamp("2024-01-01")]
    assert len(jan_rows) == 1, f"Expected exactly one Jan-2024 row, got {len(jan_rows)}"
    release = jan_rows.iloc[0]["release_date"]
    expected = pd.Timestamp("2024-02-13")
    delta = abs((release - expected).days)
    assert delta <= 1, f"CPI Jan-2024 release_date {release.date()} differs from expected 2024-02-13 by {delta} days"

    available = jan_rows.iloc[0]["available_from"]
    assert available > release, "available_from must be strictly after release_date"
