"""Tests for ATR trailing-stop backtest exits."""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import resolve_trade


def _master(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.date_range("2024-01-01", periods=len(rows), freq="B"))


def test_resolve_trade_has_no_fixed_take_profit_exit() -> None:
    df = _master(
        [
            {"us100_open": 100.0, "us100_high": 100.5, "us100_low": 99.5, "us100_close": 100.0},
            {"us100_open": 101.0, "us100_high": 104.5, "us100_low": 100.0, "us100_close": 104.0},
            {"us100_open": 104.0, "us100_high": 104.2, "us100_low": 103.5, "us100_close": 104.0},
            {"us100_open": 104.0, "us100_high": 104.2, "us100_low": 103.5, "us100_close": 104.0},
        ]
    )

    trade = resolve_trade(
        master_df=df,
        entry_date=df.index[0],
        atr_value=1.0,
        horizon_days=3,
        sl_multiplier=1.5,
    )

    assert trade is not None
    assert trade.exit_reason == "time"
    assert trade.holding_days == 3
    assert trade.trailing_stop_activated is True


def test_resolve_trade_exits_on_trailing_stop_after_activation() -> None:
    df = _master(
        [
            {"us100_open": 100.0, "us100_high": 100.5, "us100_low": 99.5, "us100_close": 100.0},
            {"us100_open": 101.0, "us100_high": 103.0, "us100_low": 100.0, "us100_close": 102.5},
            {"us100_open": 102.6, "us100_high": 102.8, "us100_low": 101.4, "us100_close": 102.0},
        ]
    )

    trade = resolve_trade(
        master_df=df,
        entry_date=df.index[0],
        atr_value=1.0,
        horizon_days=2,
        sl_multiplier=1.5,
    )

    assert trade is not None
    assert trade.exit_reason == "tsl"
    assert trade.final_stop_level == 101.5
    assert trade.peak_price == 103.0


def test_resolve_trade_exits_on_initial_stop_before_activation() -> None:
    df = _master(
        [
            {"us100_open": 100.0, "us100_high": 100.5, "us100_low": 99.5, "us100_close": 100.0},
            {"us100_open": 99.0, "us100_high": 100.0, "us100_low": 98.5, "us100_close": 99.5},
        ]
    )

    trade = resolve_trade(
        master_df=df,
        entry_date=df.index[0],
        atr_value=1.0,
        horizon_days=1,
        sl_multiplier=1.5,
    )

    assert trade is not None
    assert trade.exit_reason == "sl"
    assert trade.trailing_stop_activated is False
