"""Trade simulator: trailing-stop resolution with realistic CFD costs.

The simulator receives ML entry signals and resolves each long trade against
future OHLC data using an ATR-based trailing stop loss.

Cost Model (US100 CFD):
  - Spread: 1.0 index point round-trip
  - Slippage: normal exits use 0.5 points; stop-loss exits use 2.0 points
  - Overnight swap: annualized financing applied on full position value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


SPREAD_POINTS = 1.0
SLIPPAGE_NORMAL = 0.5
SLIPPAGE_SL = 2.0
SWAP_RATE_ANNUAL = 0.06
TSL_ACTIVATION_MULTIPLIER = 1.5
TSL_TRAILING_DISTANCE_MULTIPLIER = 2.0


@dataclass
class TradeResult:
    """One resolved trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    effective_entry: float
    exit_price: float
    exit_reason: str
    holding_days: int
    pnl_points: float
    spread_cost: float
    swap_cost: float
    atr_at_entry: float
    sl_level: float
    final_stop_level: float
    peak_price: float
    trailing_stop_activated: bool


@dataclass
class PortfolioState:
    """Tracks equity, position sizing and trade accounting."""

    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    point_value: float = 1.0
    equity: float = field(init=False)
    trades: list[TradeResult] = field(default_factory=list, init=False)
    equity_curve: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.equity = self.initial_capital


def resolve_trade(
    master_df: pd.DataFrame,
    entry_date: pd.Timestamp,
    atr_value: float,
    horizon_days: int,
    sl_multiplier: float,
    tsl_activation_multiplier: float = TSL_ACTIVATION_MULTIPLIER,
    trailing_distance_multiplier: float | None = TSL_TRAILING_DISTANCE_MULTIPLIER,
) -> TradeResult | None:
    """Resolve a long trade using an ATR-based trailing stop.

    Initial risk is set at ``sl_multiplier`` ATR below the effective entry.
    The trailing stop activates after price reaches ``tsl_activation_multiplier``
    ATR of open profit, then trails ``trailing_distance_multiplier`` ATR below
    the highest observed high since entry.  The stop only tightens.

    With daily OHLC data, the existing stop is checked before the day's high
    can update the trailing stop.  This avoids same-bar lookahead.
    """
    if atr_value <= 0 or not np.isfinite(atr_value):
        return None
    if horizon_days < 1:
        return None
    if sl_multiplier <= 0 or not np.isfinite(sl_multiplier):
        return None
    if tsl_activation_multiplier <= 0 or not np.isfinite(tsl_activation_multiplier):
        return None

    if trailing_distance_multiplier is None:
        trailing_distance_multiplier = TSL_TRAILING_DISTANCE_MULTIPLIER
    if trailing_distance_multiplier <= 0 or not np.isfinite(trailing_distance_multiplier):
        return None

    idx = master_df.index
    try:
        entry_pos = idx.get_loc(entry_date)
    except KeyError:
        return None

    entry_close = float(master_df.iloc[entry_pos]["us100_close"])
    effective_entry = entry_close + (SPREAD_POINTS / 2.0) + SLIPPAGE_NORMAL

    initial_sl_level = effective_entry - (sl_multiplier * atr_value)
    stop_level = initial_sl_level
    activation_level = effective_entry + (tsl_activation_multiplier * atr_value)
    trailing_distance = trailing_distance_multiplier * atr_value
    peak_price = effective_entry
    trailing_stop_activated = False

    max_pos = min(entry_pos + horizon_days, len(master_df) - 1)
    if entry_pos + 1 > max_pos:
        return None

    exit_price = None
    exit_reason = None
    exit_pos = None

    for future_pos in range(entry_pos + 1, max_pos + 1):
        day_open = float(master_df.iloc[future_pos]["us100_open"])
        day_high = float(master_df.iloc[future_pos]["us100_high"])
        day_low = float(master_df.iloc[future_pos]["us100_low"])

        if day_open <= stop_level:
            exit_price = day_open
            exit_reason = "tsl" if trailing_stop_activated else "sl"
            exit_pos = future_pos
            break

        if day_low <= stop_level:
            exit_price = stop_level
            exit_reason = "tsl" if trailing_stop_activated else "sl"
            exit_pos = future_pos
            break

        peak_price = max(peak_price, day_high)
        if peak_price >= activation_level:
            trailing_stop_activated = True
            stop_level = max(stop_level, peak_price - trailing_distance)

    if exit_price is None:
        exit_pos = max_pos
        exit_price = float(master_df.iloc[exit_pos]["us100_close"])
        exit_reason = "time"

    exit_date = master_df.index[exit_pos]
    holding_days = exit_pos - entry_pos
    calendar_days = (exit_date - entry_date).days

    spread_cost = SPREAD_POINTS / 2.0
    slippage_cost = SLIPPAGE_SL if exit_reason in {"sl", "tsl"} else SLIPPAGE_NORMAL
    swap_cost = calendar_days * (effective_entry * SWAP_RATE_ANNUAL / 365.0)

    effective_exit = exit_price - spread_cost - slippage_cost
    net_pnl = effective_exit - effective_entry - swap_cost

    return TradeResult(
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_close,
        effective_entry=effective_entry,
        exit_price=effective_exit,
        exit_reason=exit_reason,
        holding_days=holding_days,
        pnl_points=net_pnl,
        spread_cost=SPREAD_POINTS / 2.0 + spread_cost,
        swap_cost=swap_cost,
        atr_at_entry=atr_value,
        sl_level=initial_sl_level,
        final_stop_level=stop_level,
        peak_price=peak_price,
        trailing_stop_activated=trailing_stop_activated,
    )


def run_portfolio_simulation(
    trades: list[TradeResult],
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.01,
    point_value: float = 1.0,
) -> dict[str, Any]:
    """Simulate portfolio equity progression using fixed fractional sizing.

    Position sizing risks ``risk_per_trade_pct`` of current equity against the
    initial stop distance.  Later trailing-stop movement does not resize the
    trade.
    """
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    equity_curve: list[dict[str, Any]] = []
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:
        sl_distance = trade.effective_entry - trade.sl_level
        if sl_distance <= 0:
            continue

        risk_amount = equity * risk_per_trade_pct
        contracts = risk_amount / (sl_distance * point_value)
        dollar_pnl = trade.pnl_points * contracts * point_value

        equity += dollar_pnl

        if dollar_pnl > 0:
            gross_profit += dollar_pnl
        else:
            gross_loss += abs(dollar_pnl)

        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown = drawdown

        equity_curve.append(
            {
                "date": trade.exit_date.strftime("%Y-%m-%d"),
                "equity": round(equity, 2),
                "trade_pnl": round(dollar_pnl, 2),
                "contracts": round(contracts, 4),
                "exit_reason": trade.exit_reason,
            }
        )

    return {
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "net_profit": round(equity - initial_capital, 2),
        "net_profit_pct": round((equity / initial_capital - 1) * 100, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
        "peak_equity": round(peak_equity, 2),
        "equity_curve": equity_curve,
    }
