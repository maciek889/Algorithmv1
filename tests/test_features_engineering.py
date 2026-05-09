"""Tests for Stage 1.3 feature engineering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.engineering import (
    build_feature_frame,
    compute_bollinger_bands,
    compute_bollinger_position,
    compute_macro_3d_flag,
    compute_obv_direction,
    compute_wilder_atr,
)


def test_compute_wilder_atr_constant_true_range() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    high = pd.Series([11.0] * 20, index=idx)
    low = pd.Series([9.0] * 20, index=idx)
    close = pd.Series([10.0] * 20, index=idx)

    atr = compute_wilder_atr(high, low, close, period=14)

    assert atr.iloc[:14].isna().all()
    assert atr.iloc[14] == 2.0
    assert atr.iloc[19] == 2.0


def test_bollinger_position_is_clipped_to_zero_one() -> None:
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    close = pd.Series([100.0] * 20 + [150.0, 90.0, 100.0, 100.0, 100.0], index=idx)
    lower, upper = compute_bollinger_bands(close, window=20, num_std=2.0)

    position = compute_bollinger_position(close, lower, upper)

    assert position.dropna().between(0.0, 1.0).all()


def test_obv_direction_uses_five_day_lookback() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="B")
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 12.0, 13.0, 12.0], index=idx)
    volume = pd.Series([100.0] * 7, index=idx)

    out = compute_obv_direction(close, volume, lookback=5)

    assert out.iloc[:5].isna().all()
    assert out.iloc[5] == 1.0
    assert out.iloc[6] == 1.0


def test_macro_3d_flag_uses_next_three_calendar_days(tmp_path: Path) -> None:
    pd.DataFrame({"decision_date": pd.to_datetime(["2024-01-05"])}).to_parquet(
        tmp_path / "fomc_dates.parquet"
    )
    pd.DataFrame({"release_date": pd.to_datetime(["2024-01-10"])}).to_parquet(
        tmp_path / "cpi_release_dates.parquet"
    )
    pd.DataFrame({"release_date": pd.to_datetime(["2024-01-15"])}).to_parquet(
        tmp_path / "nfp_release_dates.parquet"
    )
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])

    out = compute_macro_3d_flag(pd.DatetimeIndex(idx), tmp_path)

    assert out.loc[pd.Timestamp("2024-01-02")] == 1
    assert out.loc[pd.Timestamp("2024-01-03")] == 1
    assert out.loc[pd.Timestamp("2024-01-05")] == 0
    assert out.loc[pd.Timestamp("2024-01-08")] == 1


def test_build_feature_frame_handles_zero_range_as_nan(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=205, freq="B")
    master = pd.DataFrame(
        {
            "us100_open": [100.0] * 205,
            "us100_high": [101.0] * 205,
            "us100_low": [99.0] * 205,
            "us100_close": [100.0] * 205,
            "us100_volume": [1_000_000.0] * 205,
            "vix_close": [15.0] * 205,
            "spx_close": [4000.0] * 205,
            "dxy_close": [100.0] * 205,
        },
        index=idx,
    )
    master.iloc[-1, master.columns.get_loc("us100_high")] = 100.0
    master.iloc[-1, master.columns.get_loc("us100_low")] = 100.0
    pd.DataFrame({"decision_date": pd.to_datetime([])}).to_parquet(tmp_path / "fomc_dates.parquet")
    pd.DataFrame({"release_date": pd.to_datetime([])}).to_parquet(tmp_path / "cpi_release_dates.parquet")
    pd.DataFrame({"release_date": pd.to_datetime([])}).to_parquet(tmp_path / "nfp_release_dates.parquet")

    features = build_feature_frame(master, tmp_path)

    assert pd.isna(features.iloc[-1]["body_ratio"])
    assert pd.isna(features.iloc[-1]["close_position"])
