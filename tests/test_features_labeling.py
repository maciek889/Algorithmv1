"""Tests for Stage 1.3 triple-barrier labeling."""

from __future__ import annotations

import pandas as pd

from src.features.labeling import HorizonConfig, build_target_labels, label_one_entry


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="B"))


def test_label_one_entry_hits_tp_before_sl() -> None:
    high = _series([100.0, 101.0, 103.1])
    low = _series([100.0, 99.5, 100.0])
    close = _series([100.0, 100.5, 102.0])

    label = label_one_entry(high, low, close, 0, 1.0, 3, 3.0, 1.5)

    assert label == 1


def test_label_one_entry_hits_sl_before_tp() -> None:
    high = _series([100.0, 101.0, 104.0])
    low = _series([100.0, 98.4, 99.0])
    close = _series([100.0, 99.0, 101.0])

    label = label_one_entry(high, low, close, 0, 1.0, 3, 3.0, 1.5)

    assert label == 0


def test_label_one_entry_same_day_breach_is_pessimistic() -> None:
    high = _series([100.0, 103.1])
    low = _series([100.0, 98.4])
    close = _series([100.0, 100.0])

    label = label_one_entry(high, low, close, 0, 1.0, 3, 3.0, 1.5)

    assert label == 0


def test_label_one_entry_no_hit_expires_as_failure() -> None:
    high = _series([100.0, 101.0, 101.0, 101.0])
    low = _series([100.0, 99.0, 99.0, 99.0])
    close = _series([100.0, 100.0, 100.0, 100.0])

    label = label_one_entry(high, low, close, 0, 1.0, 3, 3.0, 1.5)

    assert label == 0


def test_build_target_labels_leaves_last_horizon_rows_unresolved() -> None:
    idx = pd.date_range("2024-01-01", periods=25, freq="B")
    master = pd.DataFrame(
        {
            "us100_high": [101.0] * 25,
            "us100_low": [99.0] * 25,
            "us100_close": [100.0] * 25,
        },
        index=idx,
    )
    config = HorizonConfig(horizon_days=3, tp_multiplier=3.0, sl_multiplier=1.5, atr_period=14)

    labels = build_target_labels(master, config)

    assert labels.iloc[-3:].isna().all()
    assert labels.iloc[:14].isna().all()
    assert labels.iloc[14:-3].notna().all()
