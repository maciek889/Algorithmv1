"""Tests for src.data.loaders.fomc_loader (synthetic CSV)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.exceptions import DataLoadError
from src.data.loaders.fomc_loader import load_fomc_dates


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["meeting_date,decision_date,is_unscheduled"]
    lines.extend(f"{a},{b},{c}" for a, b, c in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _full_csv_rows(year_start: int = 2007, year_end: int = 2024) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for y in range(year_start, year_end + 1):
        for m in (1, 3, 5, 6, 7, 9, 11, 12):  # 8 meetings/yr
            d = f"{y}-{m:02d}-15"
            rows.append((d, d, "False"))
    return rows


def test_happy_path(tmp_path: Path) -> None:
    csv = tmp_path / "fomc.csv"
    _write_csv(csv, _full_csv_rows())

    df = load_fomc_dates(csv, date(2007, 1, 1), date(2024, 12, 31))
    assert len(df) == 8 * 18
    assert set(df.columns) == {"meeting_date", "decision_date", "is_unscheduled"}
    assert df["meeting_date"].is_monotonic_increasing


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError, match="not found"):
        load_fomc_dates(tmp_path / "missing.csv", date(2007, 1, 1), date(2024, 12, 31))


def test_missing_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    csv.write_text("meeting_date,decision_date\n2024-01-15,2024-01-15\n", encoding="utf-8")
    with pytest.raises(DataLoadError, match="missing columns"):
        load_fomc_dates(csv, date(2024, 1, 1), date(2024, 12, 31))


def test_too_few_rows_raises(tmp_path: Path) -> None:
    csv = tmp_path / "small.csv"
    _write_csv(csv, [("2024-01-15", "2024-01-15", "False")])
    with pytest.raises(DataLoadError, match="only 1 rows"):
        load_fomc_dates(csv, date(2024, 1, 1), date(2024, 12, 31))


def test_duplicate_meeting_date_raises(tmp_path: Path) -> None:
    csv = tmp_path / "dup.csv"
    rows = _full_csv_rows()
    rows.append(("2007-01-15", "2007-01-15", "False"))  # duplicate
    _write_csv(csv, rows)
    with pytest.raises(DataLoadError, match="duplicate"):
        load_fomc_dates(csv, date(2007, 1, 1), date(2024, 12, 31))


def test_filters_to_window(tmp_path: Path) -> None:
    csv = tmp_path / "fomc.csv"
    _write_csv(csv, _full_csv_rows())

    df = load_fomc_dates(csv, date(2020, 1, 1), date(2020, 12, 31))
    assert len(df) == 8


def test_unscheduled_truthy_variants(tmp_path: Path) -> None:
    csv = tmp_path / "fomc.csv"
    rows = _full_csv_rows()
    rows.append(("2008-03-11", "2008-03-11", "True"))
    rows.append(("2020-03-23", "2020-03-23", "yes"))
    _write_csv(csv, rows)

    df = load_fomc_dates(csv, date(2007, 1, 1), date(2024, 12, 31))
    assert bool(df.loc[df["meeting_date"] == pd.Timestamp("2008-03-11"), "is_unscheduled"].iloc[0])
    assert bool(df.loc[df["meeting_date"] == pd.Timestamp("2020-03-23"), "is_unscheduled"].iloc[0])
