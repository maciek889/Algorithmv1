"""Tests for src.data.validators QA report generator."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.validators import generate_qa_report


def _master(n: int = 250, year: int = 2024, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{year}-01-02", periods=n, freq="B")
    close = 15000 + rng.normal(0, 100, size=n).cumsum()
    return pd.DataFrame({
        "us100_open":   close - 10,
        "us100_high":   close + 20,
        "us100_low":    close - 20,
        "us100_close":  close,
        "us100_volume": rng.integers(1_000_000, 5_000_000, size=n),
        "vix_open":     [15.0] * n,
        "vix_high":     [16.0] * n,
        "vix_low":      [14.0] * n,
        "vix_close":    [15.5] * n,
        "spx_close":    close * 0.3,
        "dxy_close":    [100.0] * n,
        "putcall_ratio": [0.85] * n,
    }, index=idx)


def _calendars(complete: bool = True) -> dict[str, pd.DataFrame]:
    fomc = pd.DataFrame({"meeting_date": pd.to_datetime([
        f"2024-{m:02d}-15" for m in range(1, 9 if complete else 4)])})
    cpi = pd.DataFrame({"release_date": pd.to_datetime([
        f"2024-{m:02d}-13" for m in range(1, 13 if complete else 7)])})
    nfp = pd.DataFrame({"release_date": pd.to_datetime([
        f"2024-{m:02d}-05" for m in range(1, 13 if complete else 7)])})
    return {"fomc": fomc, "cpi": cpi, "nfp": nfp}


def test_report_writes_all_sections(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    generate_qa_report(_master(), _calendars(), out, training_start=date(2024, 1, 1), putcall_source="cboe_direct")

    text = out.read_text(encoding="utf-8")
    for section in ("Master overview", "NaN counts", "OHLC sanity", "Volume",
                    "Put/Call coverage", "Calendars", "Distribution sanity"):
        assert section in text, f"missing section: {section}"


def test_report_flags_high_below_low(tmp_path: Path) -> None:
    df = _master()
    df.iloc[5, df.columns.get_loc("us100_high")] = 0.0  # high < low
    out = tmp_path / "report.md"
    generate_qa_report(df, _calendars(), out, training_start=date(2024, 1, 1))

    text = out.read_text(encoding="utf-8")
    assert "high < low" in text
    assert "❌" in text


def test_report_flags_zero_volume(tmp_path: Path) -> None:
    df = _master()
    df.iloc[10, df.columns.get_loc("us100_volume")] = 0
    out = tmp_path / "report.md"
    generate_qa_report(df, _calendars(), out, training_start=date(2024, 1, 1))

    text = out.read_text(encoding="utf-8")
    assert "Days with zero volume: **1**" in text


def test_report_flags_missing_calendar_year(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    generate_qa_report(_master(), _calendars(complete=False), out, training_start=date(2024, 1, 1))
    text = out.read_text(encoding="utf-8")
    assert "⚠️" in text  # incomplete years should warn


def test_report_flags_nan_in_core_post_training(tmp_path: Path) -> None:
    df = _master()
    df.iloc[20, df.columns.get_loc("us100_close")] = np.nan
    out = tmp_path / "report.md"
    generate_qa_report(df, _calendars(), out, training_start=date(2024, 1, 1))
    text = out.read_text(encoding="utf-8")
    assert "us100_close" in text and "❌" in text
