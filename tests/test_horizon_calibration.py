"""Tests for Stage 1.2 horizon calibration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.calibrate_horizon import (
    calibrate_horizon,
    compute_atr_wilder,
    simulate_triple_barrier,
)


def test_compute_atr_wilder_constant_true_range() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    high = pd.Series([11.0] * 20, index=idx)
    low = pd.Series([9.0] * 20, index=idx)
    close = pd.Series([10.0] * 20, index=idx)

    atr = compute_atr_wilder(high, low, close, period=14)

    assert atr.iloc[:14].isna().all()
    assert atr.iloc[14] == 2.0
    assert atr.iloc[19] == 2.0


def test_simulate_triple_barrier_hits_tp_on_day_3() -> None:
    df = pd.DataFrame(
        {
            "high": [100.5, 101.0, 102.0, 103.1],
            "low": [99.5, 99.0, 99.0, 99.0],
            "close": [100.0, 100.5, 101.0, 102.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="B"),
    )

    result = simulate_triple_barrier(df, entry_idx=0, atr_value=1.0, max_lookout=60)

    assert result["time_to_first_barrier"] == 3
    assert result["first_barrier"] == "tp"
    assert result["no_barrier_hit"] is False


def test_simulate_triple_barrier_uses_tp_first_same_day_convention() -> None:
    df = pd.DataFrame(
        {
            "high": [100.5, 103.5],
            "low": [99.5, 98.0],
            "close": [100.0, 100.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="B"),
    )

    result = simulate_triple_barrier(df, entry_idx=0, atr_value=1.0, max_lookout=60)

    assert result["time_to_first_barrier"] == 1
    assert result["first_barrier"] == "tp"


def test_simulate_triple_barrier_no_hit_within_60_days() -> None:
    idx = pd.date_range("2024-01-01", periods=61, freq="B")
    df = pd.DataFrame(
        {
            "high": [100.5] * 61,
            "low": [99.5] * 61,
            "close": [100.0] * 61,
        },
        index=idx,
    )

    result = simulate_triple_barrier(df, entry_idx=0, atr_value=1.0, max_lookout=60)

    assert result["no_barrier_hit"] is True
    assert result["time_to_first_barrier"] is None
    assert result["first_barrier"] == "none"


def test_calibrate_horizon_real_master_dataset_sanity() -> None:
    master_path = Path("data/processed/master_dataset.parquet")
    master_df = pd.read_parquet(master_path)

    result = calibrate_horizon(master_df)

    assert result["n_simulations"] > 400
    assert 3 <= result["horizon_days"] <= 30
